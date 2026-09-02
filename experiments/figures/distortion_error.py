"""Distortion error of estimated intrinsics against the EgoHumans ground truth.

The ground truth uses OpenCV's fisheye model and the pipeline estimates a
Brown-Conrady one, so their coefficients are not comparable term by term. What
is comparable is the mapping they induce: undistorting a pixel with the ground
truth model and re-distorting it with the predicted one lands somewhere else,
and the displacement between the two, in pixels, is the error.

Both projections use the ground-truth K, so the error is the one of the
distortion coefficients alone; the focal and principal point are reported
separately as vfov_error.
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


def distortion_error_map(
    gt_intrinsics: CameraIntrinsicsAnnotation,
    pred_intrinsics: CameraIntrinsicsAnnotation,
    grid_hw: tuple[int, int] = GRID_HW,
) -> np.ndarray:
    """Pixel displacement between the ground-truth and predicted distortions.

    Args:
        gt_intrinsics: Ground-truth intrinsics, whose K and resolution define
            the image frame the error is measured in.
        pred_intrinsics: Intrinsics the pipeline estimated, of which only the
            distortion coefficients and model are read.
        grid_hw: Resolution the error is sampled at.

    Returns:
        The error magnitude in pixels, shape `grid_hw`.
    """
    image_h, image_w = gt_intrinsics.resolution_hw
    grid_h, grid_w = grid_hw

    ys, xs = torch.meshgrid(
        torch.linspace(0, image_h - 1, grid_h),
        torch.linspace(0, image_w - 1, grid_w),
        indexing="ij",
    )
    points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)

    K = gt_intrinsics.K
    rays = undistort_points(
        points,
        K=K,
        D=gt_intrinsics.distortion_coefficients,
        distortion_model=gt_intrinsics.distortion_model.value,
    )
    predicted = distort_points(
        rays,
        K=K,
        D=pred_intrinsics.distortion_coefficients,
        distortion_model=pred_intrinsics.distortion_model.value,
    )

    error = torch.linalg.norm(points - predicted, dim=-1)
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
        One entry per region and statistic, plus the mask coverage. Pixels
        where undistortion diverges are dropped rather than averaged in.
    """
    values_by_region = {
        "observed": error[observed],
        "background": error[~observed],
        "full": error.reshape(-1),
    }
    stats = {"observed_coverage": float(observed.mean())}

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
                os.path.join(predictions_dir, "cameras_intrinsics.pkl"),
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
