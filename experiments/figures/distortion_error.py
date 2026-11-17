from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Iterator, Literal

import cv2
import numpy as np
import orjson
import torch

from kineo.annotations.bundle_adjustment_keypoints import (
    BundleAdjustmentKeypointsAnnotations,
)
from kineo.annotations.camera_intrinsics import (
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotations,
)
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.geometry.transformations import distort_points, undistort_points

# Error map size, values in full-resolution pixels
GRID_HW = (540, 960)

# Newton tolerance from AnyCalib's undist_tol
UNDISTORT_ITERATIONS = 20
UNDISTORT_TOLERANCE = 1e-5

REGIONS = ("observed", "background", "full")
STATS = ("mean", "median", "p95", "max")

# Older runs wrote the singular name
INTRINSICS_FILENAMES = ("cameras_intrinsics.pkl", "camera_intrinsics.pkl")

MaskSource = Literal["bundle_adjustment", "detections"]


@dataclass(frozen=True)
class ViewDistortionError:
    sequence_name: str
    view_id: str
    images_dir: str
    error: np.ndarray  # (GRID_HW), pixels
    observed: np.ndarray  # (GRID_HW), bool
    vectors: np.ndarray  # (GRID_HW, 2), pixels
    focal_scale: float
    aligned_vectors: np.ndarray  # (GRID_HW, 2), pixels

    @property
    def aligned_error(self) -> np.ndarray:
        return np.linalg.norm(self.aligned_vectors, axis=-1)


def _radial_coefficients(D: torch.Tensor, distortion_model: str) -> tuple:
    if distortion_model == "brown_conrady":
        return D[0], D[1], D[4]
    return D[0], D[1], D[2], D[3]


def _radial_scale(squared: torch.Tensor, coefficients: tuple) -> torch.Tensor:
    return 1 + sum(
        k * squared ** (order + 1) for order, k in enumerate(coefficients)
    )


def _radial_derivative(squared: torch.Tensor, coefficients: tuple) -> torch.Tensor:
    return 1 + sum(
        (2 * order + 3) * k * squared ** (order + 1)
        for order, k in enumerate(coefficients)
    )


def _solve_fisheye_angle(
    distorted_angle: torch.Tensor, D: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    coefficients = _radial_coefficients(D, "opencv_fisheye")
    angle = distorted_angle.clone()

    for _ in range(UNDISTORT_ITERATIONS):
        squared = angle**2
        residual = angle * _radial_scale(squared, coefficients) - distorted_angle
        angle = angle - residual / _radial_derivative(squared, coefficients)

    return angle, residual.abs() < UNDISTORT_TOLERANCE


def _unproject(
    points: torch.Tensor, intrinsics: CameraIntrinsicsAnnotation
) -> tuple[torch.Tensor, torch.Tensor]:
    K, D = intrinsics.K, intrinsics.distortion_coefficients
    distortion_model = intrinsics.distortion_model.value
    normalized = (points - K[:2, 2]) / K.diagonal()[:2]

    if distortion_model == "opencv_fisheye":
        distorted_angle = torch.linalg.norm(normalized, dim=-1)
        angle, converged = _solve_fisheye_angle(distorted_angle, D)
        eps = torch.finfo(points.dtype).eps
        direction = normalized / distorted_angle.clamp_min(eps)[..., None]
        bearings = torch.cat(
            [direction * torch.sin(angle)[..., None], torch.cos(angle)[..., None]],
            dim=-1,
        )
        return bearings, converged

    undistorted = undistort_points(
        points, K=K, D=D, distortion_model=distortion_model
    )
    plane = (undistorted - K[:2, 2]) / K.diagonal()[:2]
    bearings = torch.cat([plane, torch.ones_like(plane[..., :1])], dim=-1)
    return bearings, torch.isfinite(bearings).all(dim=-1)


def _project(
    bearings: torch.Tensor, intrinsics: CameraIntrinsicsAnnotation
) -> tuple[torch.Tensor, torch.Tensor]:
    K, D = intrinsics.K, intrinsics.distortion_coefficients
    distortion_model = intrinsics.distortion_model.value
    eps = torch.finfo(bearings.dtype).eps

    plane = bearings[..., :2] / bearings[..., 2:].clamp_min(eps)
    pixels = distort_points(
        plane * K.diagonal()[:2] + K[:2, 2],
        K=K,
        D=D,
        distortion_model=distortion_model,
    )

    radius = torch.linalg.norm(plane, dim=-1)
    if distortion_model != "brown_conrady":
        radius = torch.atan(radius)
    monotonic = _radial_derivative(
        radius**2, _radial_coefficients(D, distortion_model)
    )
    return pixels, (bearings[..., 2] > 0) & (monotonic > 0)


def grid_points(image_hw: tuple[int, int]) -> np.ndarray:
    image_h, image_w = image_hw
    ys, xs = np.meshgrid(
        (np.arange(GRID_HW[0]) + 0.5) * image_h / GRID_HW[0],
        (np.arange(GRID_HW[1]) + 0.5) * image_w / GRID_HW[1],
        indexing="ij",
    )
    return np.stack([xs, ys], axis=-1).astype(np.float32)


def distortion_error_vectors(
    gt_intrinsics: CameraIntrinsicsAnnotation,
    pred_intrinsics: CameraIntrinsicsAnnotation,
) -> np.ndarray:
    points = torch.from_numpy(grid_points(gt_intrinsics.resolution_hw)).reshape(-1, 2)

    bearings, unprojected = _unproject(points, gt_intrinsics)
    predicted, projected = _project(bearings, pred_intrinsics)

    valid = (unprojected & projected)[:, None]
    vectors = torch.where(valid, predicted - points, torch.nan)
    return vectors.reshape(*GRID_HW, 2).numpy().astype(np.float32)


def align_focal(
    vectors: np.ndarray,
    observed: np.ndarray,
    image_hw: tuple[int, int],
    principal_point: np.ndarray,
) -> tuple[float, np.ndarray]:
    points = grid_points(image_hw)
    predicted = points + vectors - principal_point
    target = points - principal_point
    fit = observed & np.isfinite(vectors).all(axis=-1)
    scale = float((predicted[fit] * target[fit]).sum() / (predicted[fit] ** 2).sum())
    return scale, principal_point + scale * predicted - points


def keypoints_mask(
    xy: np.ndarray,
    scores: np.ndarray,
    image_hw: tuple[int, int],
    radius: int = 25,
    score_threshold: float = 0.0,
) -> np.ndarray:
    scale = np.array([GRID_HW[1] / image_hw[1], GRID_HW[0] / image_hw[0]])
    kept = (scores > score_threshold) & np.isfinite(xy).all(axis=-1)
    mask = np.zeros(GRID_HW, dtype=np.uint8)

    for x, y in np.round(xy[kept] * scale).astype(int):
        cv2.circle(
            mask, (x, y), max(round(radius * scale.mean()), 1), 255, thickness=-1
        )

    return mask > 0


def error_stats(error: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    values_by_region = {
        "observed": error[observed],
        "background": error[~observed],
        "full": error.reshape(-1),
    }
    stats = {
        "observed_coverage": float(observed.mean()),
        "valid_fraction": float(np.isfinite(error).mean()),
    }

    for region, values in values_by_region.items():
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            stats.update({f"{region}_{stat}": float("nan") for stat in STATS})
            continue
        stats[f"{region}_mean"] = float(np.mean(finite))
        stats[f"{region}_median"] = float(np.median(finite))
        stats[f"{region}_p95"] = float(np.percentile(finite, 95))
        stats[f"{region}_max"] = float(np.max(finite))

    return stats


def load_sequences(dataset_dir: str) -> list[dict]:
    with open(os.path.join(dataset_dir, "egohumans_sequences.json"), "rb") as f:
        return orjson.loads(f.read())


def prediction_intrinsics_path(predictions_dir: str) -> str:
    for filename in INTRINSICS_FILENAMES:
        path = os.path.join(predictions_dir, filename)
        if os.path.exists(path):
            return path
    names = " or ".join(INTRINSICS_FILENAMES)
    raise FileNotFoundError(f"No {names} in {predictions_dir}")


def load_gt_intrinsics(
    dataset_dir: str, sequence: dict
) -> CameraIntrinsicsAnnotations:
    path = os.path.join(dataset_dir, sequence["annotations"]["cameras_intrinsics"])
    with open(path, "rb") as f:
        return CameraIntrinsicsAnnotations.from_dict(orjson.loads(f.read()))


def _load_prediction(path: str, annotations_class):
    with open(path, "rb") as f:
        return annotations_class.from_dict(pickle.load(f))


def _observed_keypoints(
    predictions_dir: str, mask_source: MaskSource
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if mask_source == "bundle_adjustment":
        keypoints = _load_prediction(
            os.path.join(predictions_dir, "bundle_adjustment_keypoints.pkl"),
            BundleAdjustmentKeypointsAnnotations,
        ).first_or_default()
        return {
            view_id: (
                keypoints.kps_2d_xy[i].numpy(),
                keypoints.kps_2d_scores[i].numpy(),
            )
            for i, view_id in enumerate(keypoints.view_ids)
        }

    keypoints = _load_prediction(
        os.path.join(predictions_dir, "keypoints_2d.pkl"), Keypoints2DAnnotations
    )
    per_view: dict[str, tuple[list, list]] = {}
    for annotation in keypoints:
        xy, scores = per_view.setdefault(annotation.view_id, ([], []))
        xy.append(annotation.xy.numpy())
        scores.append(annotation.scores.numpy())

    return {
        view_id: (np.concatenate(xy), np.concatenate(scores))
        for view_id, (xy, scores) in per_view.items()
    }


def iter_view_errors(
    dataset_dir: str,
    pred_annotations_dir: str,
    sequences: list[dict],
    mask_source: MaskSource = "bundle_adjustment",
    views: set[tuple[str, str]] | None = None,
    radius: int = 25,
    score_threshold: float = 0.0,
) -> Iterator[ViewDistortionError]:
    for sequence in sequences:
        sequence_name = sequence["sequence_name"]
        predictions_dir = os.path.join(pred_annotations_dir, sequence_name)

        try:
            gt_intrinsics = load_gt_intrinsics(dataset_dir, sequence)
            pred_intrinsics = _load_prediction(
                prediction_intrinsics_path(predictions_dir),
                CameraIntrinsicsAnnotations,
            )
            observed_keypoints = _observed_keypoints(predictions_dir, mask_source)
        except FileNotFoundError as error:
            print(f"skip {sequence_name}: {error}")
            continue

        for view_id in sequence["views"]:
            if views is not None and (sequence_name, view_id) not in views:
                continue

            gt_view = gt_intrinsics.filter_by_view_id(view_id).first_or_default()
            pred_view = pred_intrinsics.filter_by_view_id(view_id).first_or_default()
            if gt_view is None or pred_view is None:
                continue
            if view_id not in observed_keypoints:
                continue

            xy, scores = observed_keypoints[view_id]
            vectors = distortion_error_vectors(gt_view, pred_view)
            observed = keypoints_mask(
                xy,
                scores,
                image_hw=gt_view.resolution_hw,
                radius=radius,
                score_threshold=score_threshold,
            )
            focal_scale, aligned_vectors = align_focal(
                vectors,
                observed,
                gt_view.resolution_hw,
                pred_view.K[:2, 2].numpy(),
            )
            yield ViewDistortionError(
                sequence_name=sequence_name,
                view_id=view_id,
                images_dir=sequence["views"][view_id]["images_dir"],
                error=np.linalg.norm(vectors, axis=-1),
                observed=observed,
                vectors=vectors,
                focal_scale=focal_scale,
                aligned_vectors=aligned_vectors,
            )
