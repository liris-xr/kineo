# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import itertools
from typing import Any

import torch
from kineo.geometry.transformations import (
    compute_similarity_transform,
    inverse_Rt,
    apply_similarity_transform_to_Rt,
)
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
import roma


def _compute_vfov(K: torch.Tensor, resolution_hw: torch.Tensor) -> torch.Tensor:
    """
    Compute the vertical field of view (in degrees) of a camera from its
    intrinsics matrix and resolution.
    """
    fy = K[..., 1, 1]
    h = resolution_hw[..., 0]
    return torch.rad2deg(2 * torch.atan(h / (2 * fy)))


def _compute_camera_metrics_for_frame(
    frame_idx: int,
    gt_cam_intrinsics_annotations: CameraIntrinsicsAnnotations,
    gt_cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
    pred_cam_intrinsics_annotations: CameraIntrinsicsAnnotations,
    pred_cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
) -> dict[str, torch.Tensor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_gt_cam_intrinsics_annotations = (
        gt_cam_intrinsics_annotations.get_closest_by_frame_idx(frame_idx)
    )
    frame_pred_cam_intrinsics_annotations = (
        pred_cam_intrinsics_annotations.get_closest_by_frame_idx(frame_idx)
    )

    frame_gt_cam_extrinsics_annotations = (
        gt_cam_extrinsics_annotations.filter_active_by_frame_idx(frame_idx)
    )
    frame_pred_cam_extrinsics_annotations = (
        pred_cam_extrinsics_annotations.get_closest_by_frame_idx(frame_idx)
    )

    if (
        frame_gt_cam_intrinsics_annotations.views_ids
        != frame_pred_cam_intrinsics_annotations.views_ids
    ):
        raise ValueError(
            f"Views ids in frame {frame_idx} are different in ground truth and prediction"
        )

    if (
        frame_gt_cam_extrinsics_annotations.views_ids
        != frame_pred_cam_extrinsics_annotations.views_ids
    ):
        raise ValueError(
            f"Views ids in frame {frame_idx} are different in ground truth and prediction"
        )

    if (
        frame_gt_cam_intrinsics_annotations.views_ids
        != frame_gt_cam_extrinsics_annotations.views_ids
    ):
        raise ValueError(
            f"Views ids in frame {frame_idx} are different in ground truth intrinsics and extrinsics"
        )

    cam_ids = frame_gt_cam_intrinsics_annotations.views_ids

    out = []

    gt_Rt = torch.stack(
        [
            frame_gt_cam_extrinsics_annotations.filter_by_view_id(cam_id)
            .first_or_default()
            .Rt.to(device)
            for cam_id in cam_ids
        ]
    )

    pred_Rt = torch.stack(
        [
            frame_pred_cam_extrinsics_annotations.filter_by_view_id(cam_id)
            .first_or_default()
            .Rt.to(device)
            for cam_id in cam_ids
        ]
    )

    gt_cam2world = inverse_Rt(gt_Rt)
    pred_cam2world = inverse_Rt(pred_Rt)

    gt_cam_pos = gt_cam2world[:, :3, 3]
    pred_cam_pos = pred_cam2world[:, :3, 3]

    # Camera position accuracy, with scale alignment.
    R, t, s = compute_similarity_transform(
        X=pred_cam_pos,
        Y=gt_cam_pos,
        estimate_scale=True,
    )
    pred_Rt_scaled_aligned = apply_similarity_transform_to_Rt(pred_Rt, R, t, s)
    pred_cam2world_scaled_aligned = inverse_Rt(pred_Rt_scaled_aligned)
    pred_cam_pos_scaled_aligned = pred_cam2world_scaled_aligned[:, :3, 3]

    # Camera position accuracy, without scale alignment.
    R, t, s = compute_similarity_transform(
        X=pred_cam_pos,
        Y=gt_cam_pos,
        estimate_scale=False,
    )
    pred_Rt_unscaled_aligned = apply_similarity_transform_to_Rt(pred_Rt, R, t, s)
    pred_cam2world_unscaled_aligned = inverse_Rt(pred_Rt_unscaled_aligned)
    pred_cam_pos_unscaled_aligned = pred_cam2world_unscaled_aligned[:, :3, 3]

    # Gauged on rotations, not centres: two centres leave the rotation about
    # their baseline to numerical noise, flipping half the two-view sequences.
    pred_R_rotation_aligned = _align_rotations(
        gt_cam2world[:, :3, :3], pred_cam2world[:, :3, :3]
    )

    # Equation (7) in RelPose++.
    scene_scale = torch.max(
        torch.linalg.norm(
            gt_cam_pos - torch.mean(gt_cam_pos, dim=0, keepdim=True), dim=-1
        )
    )

    for cam_idx, cam_id in enumerate(cam_ids):
        gt_K = (
            frame_gt_cam_intrinsics_annotations.filter_by_view_id(cam_id)
            .first_or_default()
            .K
        )
        pred_K = (
            frame_pred_cam_intrinsics_annotations.filter_by_view_id(cam_id)
            .first_or_default()
            .K
        )
        cam_resolution_hw = torch.tensor(
            frame_gt_cam_intrinsics_annotations.filter_by_view_id(cam_id)
            .first_or_default()
            .resolution_hw
        )
        gt_vfov = _compute_vfov(gt_K, cam_resolution_hw)
        pred_vfov = _compute_vfov(pred_K, cam_resolution_hw)
        vfov_error = torch.abs(gt_vfov - pred_vfov)

        # Similarity alignment has 7 DOF against 3n constraints, so under three
        # cameras it fits any prediction exactly and s-TE is 0 regardless of
        # quality. Rigid alignment cannot change the inter-centre distance.
        if len(cam_ids) < 3:
            s_TE = torch.tensor(torch.nan, device=device)
        else:
            s_TE = torch.norm(
                (gt_cam_pos[cam_idx] - pred_cam_pos_scaled_aligned[cam_idx]), dim=-1
            )
        TE = torch.norm(
            (gt_cam_pos[cam_idx] - pred_cam_pos_unscaled_aligned[cam_idx]), dim=-1
        )

        # Camera center accuracy, with scale alignment. NaN < x is False, which
        # would report an undefined s-TE as a failed threshold.
        def _s_CCA(fraction: float) -> torch.Tensor:
            if torch.isnan(s_TE):
                return s_TE
            return (s_TE < (fraction * scene_scale)).to(torch.float32)

        s_CCA05 = _s_CCA(0.05)
        s_CCA10 = _s_CCA(0.10)
        s_CCA15 = _s_CCA(0.15)
        s_CCA20 = _s_CCA(0.20)
        s_CCA25 = _s_CCA(0.25)
        s_CCA30 = _s_CCA(0.30)

        # Camera center accuracy, without scale alignment.
        CCA05 = (TE < (0.05 * scene_scale)).to(torch.float32)
        CCA10 = (TE < (0.10 * scene_scale)).to(torch.float32)
        CCA15 = (TE < (0.15 * scene_scale)).to(torch.float32)
        CCA20 = (TE < (0.20 * scene_scale)).to(torch.float32)
        CCA25 = (TE < (0.25 * scene_scale)).to(torch.float32)
        CCA30 = (TE < (0.30 * scene_scale)).to(torch.float32)

        # Orientation accuracy, against the rotation-aligned prediction.
        angular_distance = roma.rotmat_geodesic_distance(
            gt_cam2world[cam_idx, :3, :3], pred_R_rotation_aligned[cam_idx]
        )

        # Camera angle error
        AE = torch.rad2deg(angular_distance)

        out.append(
            {
                "view_id": cam_id,
                "vfov_error": vfov_error.item(),
                "s-TE": s_TE.item(),
                "TE": TE.item(),
                "s-CCA05": s_CCA05.item(),
                "s-CCA10": s_CCA10.item(),
                "s-CCA15": s_CCA15.item(),
                "s-CCA20": s_CCA20.item(),
                "s-CCA25": s_CCA25.item(),
                "s-CCA30": s_CCA30.item(),
                "CCA05": CCA05.item(),
                "CCA10": CCA10.item(),
                "CCA15": CCA15.item(),
                "CCA20": CCA20.item(),
                "CCA25": CCA25.item(),
                "CCA30": CCA30.item(),
                "AE": AE.item(),
            }
        )

    return {
        "views": out,
        "pairs": _compute_pairwise_rra(gt_cam2world, pred_cam2world, cam_ids),
    }


def _align_rotations(
    gt_R: torch.Tensor, pred_R: torch.Tensor
) -> torch.Tensor:
    """Rotate predicted orientations onto the ground truth ones.

    A reconstruction is only determined up to a global rotation, so the gauge
    has to be fixed before absolute orientations can be compared. Uses the
    chordal L2 mean: the G minimising sum_i ||R_i_gt - G R_i_pred||_F is the
    projection of sum_i R_i_gt R_i_pred^T onto SO(3).

    Args:
        gt_R: Ground truth camera-to-world rotations. Shape (C, 3, 3).
        pred_R: Predicted camera-to-world rotations. Shape (C, 3, 3).

    Returns:
        The predicted rotations after alignment. Shape (C, 3, 3).
    """
    G = roma.special_procrustes(torch.einsum("nij,nkj->ik", gt_R, pred_R))
    return G @ pred_R


def _compute_pairwise_rra(
    gt_cam2world: torch.Tensor, pred_cam2world: torch.Tensor, cam_ids: list[str]
) -> list[dict[str, Any]]:
    """Relative Rotation Accuracy over every camera pair.

    RRA as defined in RelPose (Zhang et al., 2022) and used by HSfM. A
    reconstruction is only determined up to a global rotation G, which acts on
    camera-to-world as G @ R and so cancels in R_i^T R_j; comparing pairs
    therefore needs no alignment. Absolute orientation error does need one, and
    with two cameras that alignment is underdetermined. Must be camera-to-world:
    G multiplies world-to-camera on the right and would only conjugate.

    Args:
        gt_cam2world: Ground truth camera-to-world matrices. Shape (C, 3, 4).
        pred_cam2world: Predicted camera-to-world matrices. Shape (C, 3, 4).
        cam_ids: View id of each camera, in the same order.

    Returns:
        One entry per unordered pair, holding the angular error in degrees and
        the accuracy indicators at each threshold.
    """
    gt_R = gt_cam2world[:, :3, :3]
    pred_R = pred_cam2world[:, :3, :3]

    out = []
    for i, j in itertools.combinations(range(len(cam_ids)), 2):
        gt_R_ij = gt_R[i].transpose(-1, -2) @ gt_R[j]
        pred_R_ij = pred_R[i].transpose(-1, -2) @ pred_R[j]
        error_deg = torch.rad2deg(
            roma.rotmat_geodesic_distance(gt_R_ij, pred_R_ij)
        )

        entry = {
            "pair_ids": (cam_ids[i], cam_ids[j]),
            "pair_deg_error": error_deg.item(),
        }
        for threshold in (5, 10, 15, 20, 25, 30):
            entry[f"RRA{threshold:02d}"] = float(error_deg.item() < threshold)
        out.append(entry)

    return out

def compute_camera_metrics(
    gt_cam_intrinsics_annotations: CameraIntrinsicsAnnotations,
    gt_cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
    pred_cam_intrinsics_annotations: CameraIntrinsicsAnnotations,
    pred_cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
) -> list[dict[str, torch.Tensor]]:
    # Evaluate at the GT extrinsics onset frames: {0} for static sequences
    # (single-frame path unchanged), {0, onset...} for non-static cameras.
    frames = gt_cam_extrinsics_annotations.frames

    out = {}
    for frame in frames:
        out[frame] = _compute_camera_metrics_for_frame(
            frame,
            gt_cam_intrinsics_annotations,
            gt_cam_extrinsics_annotations,
            pred_cam_intrinsics_annotations,
            pred_cam_extrinsics_annotations,
        )
    return out


def flatten_camera_metrics(
    camera_metrics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out = {}

    all_vfov_error = []
    all_s_TE = []
    all_TE = []
    all_s_CCA05 = []
    all_s_CCA10 = []
    all_s_CCA15 = []
    all_s_CCA20 = []
    all_s_CCA25 = []
    all_s_CCA30 = []
    all_CCA05 = []
    all_CCA10 = []
    all_CCA15 = []
    all_CCA20 = []
    all_CCA25 = []
    all_CCA30 = []
    all_AE = []
    all_RRA05 = []
    all_RRA10 = []
    all_RRA15 = []
    all_RRA20 = []
    all_RRA25 = []
    all_RRA30 = []

    all_pair_deg_error = []

    for frame_metrics in camera_metrics.values():
        for pair_metric in frame_metrics["pairs"]:
            all_pair_deg_error.append(pair_metric["pair_deg_error"])
            all_RRA05.append(pair_metric["RRA05"])
            all_RRA10.append(pair_metric["RRA10"])
            all_RRA15.append(pair_metric["RRA15"])
            all_RRA20.append(pair_metric["RRA20"])
            all_RRA25.append(pair_metric["RRA25"])
            all_RRA30.append(pair_metric["RRA30"])

        for metric in frame_metrics["views"]:
            all_vfov_error.append(metric["vfov_error"])
            all_s_TE.append(metric["s-TE"])
            all_TE.append(metric["TE"])
            all_s_CCA05.append(metric["s-CCA05"])
            all_s_CCA10.append(metric["s-CCA10"])
            all_s_CCA15.append(metric["s-CCA15"])
            all_s_CCA20.append(metric["s-CCA20"])
            all_s_CCA25.append(metric["s-CCA25"])
            all_s_CCA30.append(metric["s-CCA30"])
            all_CCA05.append(metric["CCA05"])
            all_CCA10.append(metric["CCA10"])
            all_CCA15.append(metric["CCA15"])
            all_CCA20.append(metric["CCA20"])
            all_CCA25.append(metric["CCA25"])
            all_CCA30.append(metric["CCA30"])
            all_AE.append(metric["AE"])

    out = {
        "vfov_error": all_vfov_error,
        "s-TE": all_s_TE,
        "TE": all_TE,
        "s-CCA05": all_s_CCA05,
        "s-CCA10": all_s_CCA10,
        "s-CCA15": all_s_CCA15,
        "s-CCA20": all_s_CCA20,
        "s-CCA25": all_s_CCA25,
        "s-CCA30": all_s_CCA30,
        "CCA05": all_CCA05,
        "CCA10": all_CCA10,
        "CCA15": all_CCA15,
        "CCA20": all_CCA20,
        "CCA25": all_CCA25,
        "CCA30": all_CCA30,
        "AE": all_AE,
        "pair_deg_error": all_pair_deg_error,
        "RRA05": all_RRA05,
        "RRA10": all_RRA10,
        "RRA15": all_RRA15,
        "RRA20": all_RRA20,
        "RRA25": all_RRA25,
        "RRA30": all_RRA30,
    }

    return out
