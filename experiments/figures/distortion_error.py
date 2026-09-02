"""Intrinsics error of the predictions against the EgoHumans ground truth.

The ground truth uses OpenCV's fisheye model and the pipeline estimates a
Brown-Conrady one, so their coefficients are not comparable term by term. What
is comparable is the mapping they induce, which is what single-view calibration
work reports: the parameters are partially coupled, so the displacement of the
whole map is the meaningful quantity and the estimated focal cannot be held out
of it.

The protocol follows AnyCalib (Tirado-Garin and Civera, ICCV 2025), whose
evaluation unprojects a uniform image grid with the ground-truth camera and
reprojects the bearings with the predicted one, keeping the samples both
cameras call valid and averaging the displacement over them:

    bearings, valid_u = cam_gt.unproject(intrins_gt, im_coords_grid)
    im_coords_pred, valid_p = cam.project(intrins, bearings)
    errors = torch.linalg.norm(im_coords_grid - im_coords_pred, dim=-1)
    errors = errors[valid_u & valid_p]

    -- siclib/eval/simple_pipeline_rays.py

Validity is the cameras' own: a bearing the predicted camera images in front of
itself, and an unprojection that converged. Rejecting samples past a radius the
predicted radial polynomial folds back on is ours; AnyCalib projects radial
models under the cheirality check alone, which does not catch a fold.
"""

from __future__ import annotations

import glob
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

# Maps are rasterized at this resolution; errors stay in full-resolution pixels.
GRID_HW = (540, 960)

# Newton solve of the fisheye angle, to AnyCalib's undist_tol.
UNDISTORT_ITERATIONS = 20
UNDISTORT_TOLERANCE = 1e-5

REGIONS = ("observed", "background", "full")
STATS = ("mean", "median", "p95", "max")

# Runs predating the pluralized annotation key carry the singular name, and the
# published benchmarks are among them.
INTRINSICS_FILENAMES = ("cameras_intrinsics.pkl", "camera_intrinsics.pkl")

MaskSource = Literal["bundle_adjustment", "detections"]


@dataclass(frozen=True)
class ViewDistortionError:
    """One view's error map and the region it observed people in."""

    sequence_name: str
    view_id: str
    images_dir: str
    error: np.ndarray  # (GRID_HW), pixels
    observed: np.ndarray  # (GRID_HW), bool


def _radial_coefficients(D: torch.Tensor, distortion_model: str) -> tuple:
    """The coefficients acting on even powers of the model's radial variable."""
    if distortion_model == "brown_conrady":
        return D[0], D[1], D[4]
    return D[0], D[1], D[2], D[3]


def _radial_scale(squared: torch.Tensor, coefficients: tuple) -> torch.Tensor:
    """The factor `1 + k1 u^2 + k2 u^4 + ...` a radial model scales by."""
    return 1 + sum(
        k * squared ** (order + 1) for order, k in enumerate(coefficients)
    )


def _radial_derivative(squared: torch.Tensor, coefficients: tuple) -> torch.Tensor:
    """Derivative in `u` of `u * _radial_scale(u^2)`."""
    return 1 + sum(
        (2 * order + 3) * k * squared ** (order + 1)
        for order, k in enumerate(coefficients)
    )


def _solve_fisheye_angle(
    distorted_angle: torch.Tensor, D: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Newton solve of the field angle a fisheye model distorted to this one."""
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
    """Bearings a camera sees at each pixel, and whether they were recovered."""
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
    """Pixels a camera images those bearings at, and whether it can.

    Invalid where the bearing falls behind the camera, or past the radius its
    radial polynomial folds back on and stops being a projection at all.
    """
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


def distortion_error_map(
    gt_intrinsics: CameraIntrinsicsAnnotation,
    pred_intrinsics: CameraIntrinsicsAnnotation,
) -> np.ndarray:
    """Pixel displacement between the ground-truth and predicted projections.

    Sampled at the centres of a uniform `GRID_HW` grid over the ground-truth
    image, NaN at the samples either camera rejects.
    """
    image_h, image_w = gt_intrinsics.resolution_hw
    ys, xs = torch.meshgrid(
        (torch.arange(GRID_HW[0]) + 0.5) * image_h / GRID_HW[0],
        (torch.arange(GRID_HW[1]) + 0.5) * image_w / GRID_HW[1],
        indexing="ij",
    )
    points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    bearings, unprojected = _unproject(points, gt_intrinsics)
    predicted, projected = _project(bearings, pred_intrinsics)

    error = torch.linalg.norm(points - predicted, dim=-1)
    error = torch.where(unprojected & projected, error, torch.nan)
    return error.reshape(GRID_HW).numpy().astype(np.float32)


def keypoints_mask(
    xy: np.ndarray,
    scores: np.ndarray,
    image_hw: tuple[int, int],
    radius: int = 25,
    score_threshold: float = 0.0,
) -> np.ndarray:
    """Union of discs around the keypoints a view observed.

    `xy` and `radius` are in the full-resolution frame `image_hw`; the mask is
    rasterized at `GRID_HW`.
    """
    scale = np.array([GRID_HW[1] / image_hw[1], GRID_HW[0] / image_hw[0]])
    kept = (scores > score_threshold) & np.isfinite(xy).all(axis=-1)
    mask = np.zeros(GRID_HW, dtype=np.uint8)

    for x, y in np.round(xy[kept] * scale).astype(int):
        cv2.circle(
            mask, (x, y), max(round(radius * scale.mean()), 1), 255, thickness=-1
        )

    return mask > 0


def error_stats(error: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Statistics of an error map, inside, outside and over the whole frame.

    The mean is the statistic AnyCalib takes within an image; the others are
    reported next to it because a fold or a failed calibration reaches it. Also
    reports the mask coverage and the share of samples both cameras called
    valid, so an exclusion never passes for full coverage.
    """
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


def overlay_error(
    frame_rgb: np.ndarray,
    error: np.ndarray,
    observed: np.ndarray,
    vmax: float,
    alpha: float = 0.55,
) -> np.ndarray:
    """Blends an error map over a frame and outlines the observed region.

    The frame is resized to the error's resolution, and `vmax` is the error the
    color scale saturates at.
    """
    frame = cv2.resize(frame_rgb, (error.shape[1], error.shape[0]))
    normalized = np.clip(np.nan_to_num(error, nan=vmax, posinf=vmax) / vmax, 0, 1)
    color = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    blended = (
        alpha * cv2.cvtColor(color, cv2.COLOR_BGR2RGB) + (1 - alpha) * frame
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        observed.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return cv2.drawContours(blended, contours, -1, (255, 255, 255), 2)


def load_sequences(dataset_dir: str) -> list[dict]:
    """Reads the EgoHumans sequence index."""
    with open(os.path.join(dataset_dir, "egohumans_sequences.json"), "rb") as f:
        return orjson.loads(f.read())


def load_middle_frame(dataset_dir: str, images_dir: str) -> np.ndarray:
    """Reads the frame halfway through a view, as an RGB uint8 array."""
    paths = sorted(glob.glob(os.path.join(dataset_dir, images_dir, "*.jpg")))
    if not paths:
        raise FileNotFoundError(f"No images in {images_dir}")
    return cv2.cvtColor(cv2.imread(paths[len(paths) // 2]), cv2.COLOR_BGR2RGB)


def prediction_intrinsics_path(predictions_dir: str) -> str:
    """Path to a run's exported intrinsics, under either name it was written.

    Raises:
        FileNotFoundError: If neither name is present.
    """
    for filename in INTRINSICS_FILENAMES:
        path = os.path.join(predictions_dir, filename)
        if os.path.exists(path):
            return path
    names = " or ".join(INTRINSICS_FILENAMES)
    raise FileNotFoundError(f"No {names} in {predictions_dir}")


def _load_gt_intrinsics(
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
    """Keypoints each view observed, as (xy, scores) arrays per view id.

    The bundle-adjustment source holds the points that actually constrained the
    calibration, the detection source every 2D detection whether it did or not.
    """
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
    """Yields the error of every predicted view, one sequence at a time.

    `views` keeps only the given (sequence name, view id) pairs, or all of them
    when None. Sequences whose predictions are missing are reported and skipped.
    """
    for sequence in sequences:
        sequence_name = sequence["sequence_name"]
        predictions_dir = os.path.join(pred_annotations_dir, sequence_name)

        try:
            gt_intrinsics = _load_gt_intrinsics(dataset_dir, sequence)
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
            yield ViewDistortionError(
                sequence_name=sequence_name,
                view_id=view_id,
                images_dir=sequence["views"][view_id]["images_dir"],
                error=distortion_error_map(gt_view, pred_view),
                observed=keypoints_mask(
                    xy,
                    scores,
                    image_hw=gt_view.resolution_hw,
                    radius=radius,
                    score_threshold=score_threshold,
                ),
            )
