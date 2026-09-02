"""Intrinsics error of the predictions against the EgoHumans ground truth.

The ground truth uses OpenCV's fisheye model and the pipeline estimates a
Brown-Conrady one, so their coefficients are not comparable term by term. What
is comparable is the mapping they induce: undistorting a pixel with the ground
truth model and re-distorting it with the predicted one lands somewhere else,
and the displacement between the two, in pixels, is the error.

Each side projects under its own K, which is how single-view calibration work
reports intrinsic accuracy: the parameters are partially coupled, so the
displacement of the whole map is the meaningful quantity and the estimated
focal cannot be held out of it.
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

# Error maps and masks are rasterized at this resolution. The error itself stays
# expressed in full-resolution pixels, whatever the grid.
GRID_HW = (540, 960)

# A ray this far off-axis is at the horizon of the ground-truth camera; no
# lens images it, and the models diverge there whatever their coefficients.
MAX_FIELD_ANGLE_DEG = 89.0

REGIONS = ("observed", "background", "full")
STATS = ("mean", "median", "p95", "max")

MaskSource = Literal["bundle_adjustment", "detections"]


@dataclass(frozen=True)
class ViewDistortionError:
    """The distortion error of one view, and where that view saw people."""

    sequence_name: str
    view_id: str
    images_dir: str
    error: np.ndarray  # (GRID_HW), pixels
    observed: np.ndarray  # (GRID_HW), bool


def _radial_map_is_monotonic(
    radius: torch.Tensor, D: torch.Tensor, distortion_model: str
) -> torch.Tensor:
    """Whether a distortion model still grows with radius at each sample.

    Past the radius where the derivative turns negative the polynomial folds
    back on itself and stops being a projection at all, so the displacement it
    reports there is meaningless rather than large.

    Args:
        radius: Normalized radial distance of each ray, (N,).
        D: Distortion coefficients of the model.
        distortion_model: Name of the model the coefficients belong to.

    Returns:
        A boolean tensor of the shape of `radius`.
    """
    if distortion_model == "brown_conrady":
        k1, k2, k3 = D[0], D[1], D[4]
        r2 = radius**2
        return 1 + 3 * k1 * r2 + 5 * k2 * r2**2 + 7 * k3 * r2**3 > 0

    theta2 = torch.atan(radius) ** 2
    k1, k2, k3, k4 = D[0], D[1], D[2], D[3]
    return (
        1
        + 3 * k1 * theta2
        + 5 * k2 * theta2**2
        + 7 * k3 * theta2**3
        + 9 * k4 * theta2**4
        > 0
    )


def distortion_error_map(
    gt_intrinsics: CameraIntrinsicsAnnotation,
    pred_intrinsics: CameraIntrinsicsAnnotation,
    grid_hw: tuple[int, int] = GRID_HW,
) -> np.ndarray:
    """Pixel displacement between the ground-truth and predicted distortions.

    Args:
        gt_intrinsics: Ground-truth intrinsics, whose resolution defines the
            image frame the error is measured in.
        pred_intrinsics: Intrinsics the pipeline estimated, K included.
        grid_hw: Resolution the error is sampled at.

    Returns:
        The error magnitude in pixels, shape `grid_hw`, NaN at the samples no
        lens images or the predicted model has folded over.
    """
    image_h, image_w = gt_intrinsics.resolution_hw
    grid_h, grid_w = grid_hw
    K_gt, K_pred = gt_intrinsics.K, pred_intrinsics.K
    D_pred = pred_intrinsics.distortion_coefficients
    model_pred = pred_intrinsics.distortion_model.value

    ys, xs = torch.meshgrid(
        torch.linspace(0, image_h - 1, grid_h),
        torch.linspace(0, image_w - 1, grid_w),
        indexing="ij",
    )
    points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    undistorted = undistort_points(
        points,
        K=K_gt,
        D=gt_intrinsics.distortion_coefficients,
        distortion_model=gt_intrinsics.distortion_model.value,
    )
    # The ray the ground-truth camera sees at this pixel. Both projections must
    # start from it, so the predicted one is fed the pixel that ray falls on in
    # its own frame rather than in the ground truth's.
    rays = torch.stack(
        [
            (undistorted[..., 0] - K_gt[0, 2]) / K_gt[0, 0],
            (undistorted[..., 1] - K_gt[1, 2]) / K_gt[1, 1],
        ],
        dim=-1,
    )
    predicted = distort_points(
        torch.stack(
            [
                rays[..., 0] * K_pred[0, 0] + K_pred[0, 2],
                rays[..., 1] * K_pred[1, 1] + K_pred[1, 2],
            ],
            dim=-1,
        ),
        K=K_pred,
        D=D_pred,
        distortion_model=model_pred,
    )

    radius = torch.linalg.norm(rays, dim=-1)
    visible = torch.rad2deg(torch.atan(radius)) <= MAX_FIELD_ANGLE_DEG
    visible &= _radial_map_is_monotonic(radius, D_pred, model_pred)

    error = torch.linalg.norm(points - predicted, dim=-1)
    error = torch.where(visible, error, torch.nan)
    return error.reshape(grid_h, grid_w).numpy().astype(np.float32)


def keypoints_mask(
    xy: np.ndarray,
    scores: np.ndarray,
    image_hw: tuple[int, int],
    grid_hw: tuple[int, int] = GRID_HW,
    radius: int = 25,
    score_threshold: float = 0.0,
) -> np.ndarray:
    """Union of discs around the keypoints a view observed.

    Args:
        xy: Keypoint pixel coordinates in the full-resolution frame, (N, 2).
        scores: Confidence of each keypoint, (N,).
        image_hw: Resolution `xy` is expressed in.
        grid_hw: Resolution the mask is rasterized at.
        radius: Disc radius, in full-resolution pixels.
        score_threshold: Keypoints at or below this score are ignored.

    Returns:
        A boolean mask of shape `grid_hw`.
    """
    scale_y = grid_hw[0] / image_hw[0]
    scale_x = grid_hw[1] / image_hw[1]
    scale = np.array([scale_x, scale_y])

    kept = (scores > score_threshold) & np.isfinite(xy).all(axis=-1)
    mask = np.zeros(grid_hw, dtype=np.uint8)

    for x, y in np.round(xy[kept] * scale).astype(int):
        cv2.circle(
            mask, (x, y), max(round(radius * scale.mean()), 1), 255, thickness=-1
        )

    return mask > 0


def error_stats(error: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Statistics of an error map, inside, outside and over the whole frame.

    Args:
        error: Error map in pixels.
        observed: Mask of the region where keypoints were observed.

    Returns:
        One entry per region and statistic, the mask coverage, and the fraction
        of samples that were scored at all, the rest being the ones the
        visibility filter rejected.
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

    Args:
        frame_rgb: Frame to draw on, resized to the error's resolution.
        error: Error map in pixels.
        observed: Mask of the region where keypoints were observed.
        vmax: Error the color scale saturates at, in pixels.
        alpha: Opacity of the colormap over the frame.

    Returns:
        An RGB uint8 image of the error map's shape.
    """
    grid_h, grid_w = error.shape
    frame = cv2.resize(frame_rgb, (grid_w, grid_h))

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


def _load_gt_intrinsics(
    dataset_dir: str, sequence: dict
) -> CameraIntrinsicsAnnotations:
    path = os.path.join(dataset_dir, sequence["annotations"]["cameras_intrinsics"])
    with open(path, "rb") as f:
        return CameraIntrinsicsAnnotations.from_dict(orjson.loads(f.read()))


# Runs made before the exported annotation key was pluralized carry the
# singular name, and the published benchmarks are among them.
_INTRINSICS_FILENAMES = ("cameras_intrinsics.pkl", "camera_intrinsics.pkl")


def prediction_intrinsics_path(predictions_dir: str) -> str:
    """Path to a run's exported intrinsics, under either name it was written.

    Args:
        predictions_dir: Directory holding one sequence's exported annotations.

    Returns:
        The path that exists.

    Raises:
        FileNotFoundError: If neither name is present.
    """
    for filename in _INTRINSICS_FILENAMES:
        path = os.path.join(predictions_dir, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"No {' or '.join(_INTRINSICS_FILENAMES)} in {predictions_dir}"
    )


def _load_prediction(path: str, annotations_class):
    with open(path, "rb") as f:
        return annotations_class.from_dict(pickle.load(f))


def _observed_keypoints(
    predictions_dir: str, mask_source: MaskSource
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Keypoints each view observed, as (xy, scores) arrays per view id.

    The bundle-adjustment source holds the points that actually constrained the
    calibration; the detection source holds every 2D detection, constraining or
    not.
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
    """Yields the distortion error of every predicted view.

    Args:
        dataset_dir: Directory holding the preprocessed EgoHumans dataset.
        pred_annotations_dir: Directory holding one subdirectory of exported
            annotations per sequence.
        sequences: Sequences to walk, as read by `load_sequences`.
        mask_source: Which keypoints stand for the observed region.
        views: (sequence name, view id) pairs to keep, or None for all of them.
        radius: Disc radius around each keypoint, in full-resolution pixels.
        score_threshold: Keypoints at or below this score are ignored.

    Yields:
        One `ViewDistortionError` per view. Sequences whose predictions are
        missing are reported and skipped.
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
