"""
Shared utilities to analyze the spatial distribution of the distortion error of
estimated camera intrinsics on the EgoHumans benchmark.

Error definition (distortion-only, GT K):
    For each pixel p in the GT-K image frame:
        n      = undistort_points(p, K_gt, D_gt)      # ideal ray in GT-K frame
        p_pred = distort_points(n, K_gt, D_pred)       # where pred distortion sends it
        err(p) = || p - p_pred ||                       # pixel displacement

Using GT K for both projections isolates the *distortion coefficients* error from
the focal/principal-point (K) error, even though the pipeline also estimates K.
"""

import os
import glob
import pickle

import cv2
import numpy as np
import orjson
import torch

from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.geometry.transformations import distort_points, undistort_points


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_gt_camera_intrinsics(
    dataset_dir: str, sequence: dict
) -> CameraIntrinsicsAnnotations:
    f = os.path.join(dataset_dir, sequence["annotations"]["cameras_intrinsics"])
    with open(f, "rb") as fh:
        return CameraIntrinsicsAnnotations.from_dict(orjson.loads(fh.read()))


def load_pred_camera_intrinsics(
    pred_annotations_dir: str, sequence: dict
) -> CameraIntrinsicsAnnotations:
    f = os.path.join(
        pred_annotations_dir, sequence["sequence_name"], "camera_intrinsics.pkl"
    )
    with open(f, "rb") as fh:
        return CameraIntrinsicsAnnotations.from_dict(pickle.load(fh))


def load_pred_keypoints_2d(
    pred_annotations_dir: str, sequence: dict
) -> Keypoints2DAnnotations:
    f = os.path.join(
        pred_annotations_dir, sequence["sequence_name"], "keypoints_2d.pkl"
    )
    with open(f, "rb") as fh:
        return Keypoints2DAnnotations.from_dict(pickle.load(fh))


def load_ba_keypoints(pred_annotations_dir: str, sequence: dict) -> dict:
    """Raw bundle-adjustment keypoints dict: the 2D points that actually
    constrained the calibration (K, D, Rt) in bundle adjustment."""
    f = os.path.join(
        pred_annotations_dir,
        sequence["sequence_name"],
        "bundle_adjustment_keypoints.pkl",
    )
    with open(f, "rb") as fh:
        return pickle.load(fh)


def load_middle_frame_rgb(dataset_dir: str, sequence: dict, view_id: str) -> np.ndarray:
    """Return the middle frame of a view as an HxWx3 RGB uint8 numpy array."""
    images_dir = sequence["views"][view_id]["images_dir"]
    imgs = sorted(glob.glob(os.path.join(dataset_dir, images_dir, "*.jpg")))
    if not imgs:
        raise FileNotFoundError(f"No images found in {images_dir}")
    bgr = cv2.imread(imgs[len(imgs) // 2])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# Distortion error map
# --------------------------------------------------------------------------- #
def compute_distortion_error_map(
    gt_ci,  # single CameraIntrinsicsAnnotation
    pred_ci,  # single CameraIntrinsicsAnnotation
    img_w: int,
    img_h: int,
    grid_w: int = 960,
    grid_h: int = 540,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Compute the per-pixel distortion displacement error (in full-resolution
    pixels) on a coarse grid, then resize to (img_h, img_w).

    Returns:
        np.ndarray (img_h, img_w) float32, error magnitude in pixels.
    """
    K_gt = gt_ci.K.to(device)
    D_gt = gt_ci.distortion_coefficients.to(device)
    D_pred = pred_ci.distortion_coefficients.to(device)
    model_gt = gt_ci.distortion_model.value
    model_pred = pred_ci.distortion_model.value

    xs = torch.linspace(0, img_w - 1, grid_w, device=device)
    ys = torch.linspace(0, img_h - 1, grid_h, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    p = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # (N, 2)

    # Ideal ray in GT-K frame, then re-distort with predicted coefficients.
    n = undistort_points(p, K=K_gt, D=D_gt, distortion_model=model_gt)
    p_pred = distort_points(n, K=K_gt, D=D_pred, distortion_model=model_pred)

    err = torch.linalg.norm(p - p_pred, dim=-1).reshape(grid_h, grid_w)
    err = err.detach().cpu().numpy().astype(np.float32)

    if (grid_h, grid_w) != (img_h, img_w):
        err = cv2.resize(err, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    return err


# --------------------------------------------------------------------------- #
# People / observed-region mask (keypoints union)
# --------------------------------------------------------------------------- #
def build_keypoints_mask(
    keypoints_2d: Keypoints2DAnnotations,
    view_id: str,
    img_w: int,
    img_h: int,
    radius: int = 25,
    score_thr: float = 0.3,
) -> np.ndarray:
    """
    Union of dilated discs around every 2D keypoint (score > thr) across ALL
    frames of the view. Represents image regions where people were observed.

    Returns:
        np.ndarray (img_h, img_w) bool.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    view_kp = keypoints_2d.filter_by_view_id(view_id)
    for ann in view_kp:
        xy = ann.xy.detach().cpu().numpy()
        scores = ann.scores.detach().cpu().numpy()
        annotated = ann.annotated.detach().cpu().numpy()
        for (x, y), s, a in zip(xy, scores, annotated):
            if not a or s < score_thr:
                continue
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, -1)
    return mask > 0


def build_ba_keypoints_mask(
    ba_keypoints: dict,
    view_id: str,
    img_w: int,
    img_h: int,
    radius: int = 25,
    score_thr: float = 0.0,
) -> np.ndarray:
    """
    Union of discs around the bundle-adjustment 2D keypoints for a view: the
    points that actually constrained the calibration (K, D, Rt). This is the
    region where the estimated distortion is genuinely observed.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    ann = ba_keypoints["annotations"][0]
    view_ids = list(ann["view_ids"])
    if view_id not in view_ids:
        return mask > 0
    i = view_ids.index(view_id)
    xy = np.asarray(ann["kps_2d_xy"][i])
    scores = np.asarray(ann["kps_2d_scores"][i])
    for (x, y), s in zip(xy, scores):
        if s <= score_thr or not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(mask, (int(round(x)), int(round(y))), radius, 255, -1)
    return mask > 0


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def colorize_error(
    err: np.ndarray, vmax: float | None = None, cmap: int = cv2.COLORMAP_TURBO
) -> tuple[np.ndarray, float]:
    """Map an error map to an RGB uint8 image. Returns (rgb, vmax_used)."""
    finite = err[np.isfinite(err)]
    if vmax is None:
        vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    err = np.nan_to_num(err, nan=vmax, posinf=vmax, neginf=vmax)
    norm = np.clip(err / vmax, 0.0, 1.0)
    bgr = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), vmax


def overlay_rgb(base: np.ndarray, color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return (alpha * color + (1 - alpha) * base).astype(np.uint8)


def draw_mask_contour(
    img: np.ndarray, mask: np.ndarray, color=(255, 255, 255), thickness: int = 3
) -> np.ndarray:
    out = img.copy()
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
def error_stats(err: np.ndarray, mask: np.ndarray) -> dict:
    """Distortion error stats over the full image, masked (people) and unmasked."""
    m = mask.astype(bool)

    def stats(a: np.ndarray) -> dict:
        a = a[np.isfinite(a)]  # drop pixels where undistortion diverges (wide FOV edges)
        if a.size == 0:
            return {"mean": float("nan"), "median": float("nan"),
                    "p95": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(np.max(a)),
        }

    full = stats(err.reshape(-1))
    masked = stats(err[m])
    unmasked = stats(err[~m])
    return {
        "full_mean": full["mean"], "full_median": full["median"],
        "full_p95": full["p95"], "full_max": full["max"],
        "masked_mean": masked["mean"], "masked_median": masked["median"],
        "masked_p95": masked["p95"], "masked_max": masked["max"],
        "unmasked_mean": unmasked["mean"], "unmasked_median": unmasked["median"],
        "unmasked_p95": unmasked["p95"], "unmasked_max": unmasked["max"],
        "mask_coverage": float(m.mean()),
    }
