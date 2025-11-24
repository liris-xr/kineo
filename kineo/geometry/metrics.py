# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import torch
from kineo.geometry.conversions import convert_points_to_homogeneous
from kineo.geometry.camera import (
    project_points_from_camera_to_image,
    transform_points_from_world_to_camera,
)
from kineo.geometry.transformations import (
    apply_similarity_transform_to_points,
    compute_similarity_transform,
)
from kineo.maths import weighted_median, weighted_mean
from kineo.torch_utils import check_shape
from typing import Literal


def compute_height_from_h36m_keypoints(
    kps_3d: torch.Tensor,
    kps_3d_scores: torch.Tensor,
    aggregation_method: Literal["mean", "median"] = "mean",
) -> torch.Tensor:
    """
    Compute the height of the subject from the H3.6M skeleton.

    Args:
        kps_3d: torch.Tensor of shape (F, K, 3)
        kps_3d_scores: torch.Tensor of shape (F, K)

    Returns:
        torch.Tensor of shape (F,)
    """
    assert kps_3d.ndim in [
        2,
        3,
    ], f"Expected kps_3d to be of shape (K, 3) or (F, K, 3), got {kps_3d.shape}"
    assert kps_3d_scores.ndim in [
        1,
        2,
    ], f"Expected kps_3d_scores to be of shape (K) or (F, K), got {kps_3d_scores.shape}"

    assert kps_3d_scores.ndim == kps_3d.ndim - 1, (
        f"Expected kps_3d_scores to be of shape (K) or (F, K), got {kps_3d_scores.shape}"
    )

    if kps_3d.ndim == 2:
        kps_3d = kps_3d.unsqueeze(0)
        kps_3d_scores = kps_3d_scores.unsqueeze(0)

    pelvis, pelvis_score = kps_3d[..., 0, :], kps_3d_scores[..., 0]
    lhip, lhip_score = kps_3d[..., 4, :], kps_3d_scores[..., 4]
    lknee, lknee_score = kps_3d[..., 5, :], kps_3d_scores[..., 5]
    lfoot, lfoot_score = kps_3d[..., 6, :], kps_3d_scores[..., 6]
    rhip, rhip_score = kps_3d[..., 1, :], kps_3d_scores[..., 1]
    rknee, rknee_score = kps_3d[..., 2, :], kps_3d_scores[..., 2]
    rfoot, rfoot_score = kps_3d[..., 3, :], kps_3d_scores[..., 3]
    spine, spine_score = kps_3d[..., 7, :], kps_3d_scores[..., 7]
    neck, neck_score = kps_3d[..., 8, :], kps_3d_scores[..., 8]
    head, head_score = kps_3d[..., 10, :], kps_3d_scores[..., 10]

    lupper_leg_score = torch.sqrt(lhip_score * lknee_score)
    rupper_leg_score = torch.sqrt(rhip_score * rknee_score)
    llower_leg_score = torch.sqrt(lknee_score * lfoot_score)
    rlower_leg_score = torch.sqrt(rknee_score * rfoot_score)
    lower_spine_score = torch.sqrt(spine_score * pelvis_score)
    upper_spine_score = torch.sqrt(neck_score * spine_score)
    head_score = torch.sqrt(neck_score * head_score)

    lupper_leg_length = (lhip - lknee).norm(dim=-1)
    rupper_leg_length = (rhip - rknee).norm(dim=-1)
    llower_leg_length = (lknee - lfoot).norm(dim=-1)
    rlower_leg_length = (rknee - rfoot).norm(dim=-1)
    lower_spine_length = (spine - pelvis).norm(dim=-1)
    upper_spine_length = (neck - spine).norm(dim=-1)
    head_length = (head - neck).norm(dim=-1)

    if aggregation_method == "mean":
        lupper_leg_length_agg = weighted_mean(lupper_leg_length, lupper_leg_score)
        rupper_leg_length_agg = weighted_mean(rupper_leg_length, rupper_leg_score)
        llower_leg_length_agg = weighted_mean(llower_leg_length, llower_leg_score)
        rlower_leg_length_agg = weighted_mean(rlower_leg_length, rlower_leg_score)
        lower_spine_length_agg = weighted_mean(lower_spine_length, lower_spine_score)
        upper_spine_length_agg = weighted_mean(upper_spine_length, upper_spine_score)
        head_length_agg = weighted_mean(head_length, head_score)
    elif aggregation_method == "median":
        lupper_leg_length_agg, _ = weighted_median(lupper_leg_length, lupper_leg_score)
        rupper_leg_length_agg, _ = weighted_median(rupper_leg_length, rupper_leg_score)
        llower_leg_length_agg, _ = weighted_median(llower_leg_length, llower_leg_score)
        rlower_leg_length_agg, _ = weighted_median(rlower_leg_length, rlower_leg_score)
        lower_spine_length_agg, _ = weighted_median(
            lower_spine_length, lower_spine_score
        )
        upper_spine_length_agg, _ = weighted_median(
            upper_spine_length, upper_spine_score
        )
        head_length_agg, _ = weighted_median(head_length, head_score)
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    total_height = (
        (lupper_leg_length_agg + rupper_leg_length_agg) / 2
        + (llower_leg_length_agg + rlower_leg_length_agg) / 2
        + lower_spine_length_agg
        + upper_spine_length_agg
        + head_length_agg
    )

    return total_height


def compute_calibrated_height_from_h36m_keypoints(
    kps_3d: torch.Tensor, kps_3d_scores: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute the calibrated height of the subject from the H3.6M 3D keypoints and applies a calibration factor to get a more accurate height measurement.
    Note: To see how the height calibration was computed, please refer to the `experiments/compute_height_calibration.py` file.
    """
    if kps_3d_scores is None:
        kps_3d_scores = torch.ones_like(kps_3d[..., 0])
    uncalibrated_height = compute_height_from_h36m_keypoints(kps_3d, kps_3d_scores)
    calibrated_height = 1.1026 * uncalibrated_height - 0.0232
    return calibrated_height


def sampson_distance(
    points_i: torch.Tensor, points_j: torch.Tensor, F: torch.Tensor
) -> torch.Tensor:
    """
    Compute the Sampson distance for point correspondences.
    This measures how well points satisfy the epipolar constraint: x'^T @ F @ x = 0
    It is the first order approximation of the geometric distance.
        sd = (x' @ F @ x)^2 / (||F @ x||^2 + ||x' @ F.T||^2)

    Args:
        points_i: torch.Tensor of shape (P, 2)
        points_j: torch.Tensor of shape (P, 2)
        F: torch.Tensor of shape (3, 3)

    Returns:
        torch.Tensor of shape (P,)
    """
    assert points_i.shape == points_j.shape
    assert points_i.ndim == 2 and points_i.shape[1] == 2
    assert F.shape == (3, 3)

    points_i = convert_points_to_homogeneous(points_i)
    points_j = convert_points_to_homogeneous(points_j)

    # Compute F @ x
    F_x1 = torch.einsum("ij,pj->pi", F, points_i)

    # Compute F.T @ x'
    Ft_x2 = torch.einsum("ij,pj->pi", F.T, points_j)

    # Compute x'^T @ F @ x
    x2t_F_x1 = torch.einsum("ip,pi->p", points_j.T, F_x1)

    # sd = (x' @ F @ x)^2 / (||F @ x||^2 + ||x' @ F.T||^2)
    sd = (x2t_F_x1**2) / (
        torch.sum(F_x1[..., [0, 1]] ** 2, dim=-1)
        + torch.sum(Ft_x2[..., [0, 1]] ** 2, dim=-1)
    )

    return sd


def compute_reprojection_residuals(
    kps_3d: torch.Tensor,
    kps_2d: torch.Tensor,
    Ks: torch.Tensor,
    Rts: torch.Tensor,
    Ds: torch.Tensor,
    distortion_model: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the reprojection error between 3D points projected to 2D and the target 2D points.

    Args:
        kps_3d: 3D points in world space. Shape (*, P, 3).
        kps_2d: Target 2D points. Shape (*, C, P, 2).
        Ks: Camera intrinsics. Shape (*, C, 3, 3).
        Rts: Camera extrinsics. Shape (*, C, 3, 4).
        Ds: Camera distortion coefficients. Shape (*, C, D).
        distortion_model: Distortion model (e.g. "brown_conrady" or "opencv_fisheye")

    Returns:
        Reprojection error. Shape (*, C, P).
        Depth of the projected points. Shape (*, C, P).
    """
    check_shape(kps_3d, ("*", "P", "3"))
    P = kps_3d.shape[-2]

    check_shape(Ks, ("*", "C", "3", "3"))
    C = Ks.shape[-3]
    check_shape(Rts, ("*", "C", "3", "4"))
    check_shape(Ds, ("*", C, "D"))

    check_shape(kps_2d, ("*", C, P, "2"))

    if distortion_model == "brown_conrady":
        check_shape(Ds, ("*", C, "5"))
    elif distortion_model == "opencv_fisheye":
        check_shape(Ds, ("*", C, "4"))
    else:
        raise ValueError(f"Invalid distortion model: {distortion_model}")

    kps_3d_cam = transform_points_from_world_to_camera(kps_3d, Rts)
    proj_kps_2d, kps_2d_depth = project_points_from_camera_to_image(
        kps_3d_cam, Ks, Ds, distortion_model
    )

    return (proj_kps_2d - kps_2d).norm(dim=-1), kps_2d_depth


def compute_normalized_reprojection_residuals(
    kps_3d: torch.Tensor,
    kps_2d: torch.Tensor,
    Ks: torch.Tensor,
    Rts: torch.Tensor,
    Ds: torch.Tensor,
    distortion_model: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the normalized reprojection error between 3D points projected to 2D and the target 2D points.
    Note: The output is not pixel units like the classical reprojection error.

    Args:
        kps_3d: 3D points to project. Shape (*, P, 3).
        kps_2d: Target 2D points. Shape (*, C, P, 2).
        Ks: Camera intrinsics. Shape (*, C, 3, 3).
        Rts: Camera extrinsics. Shape (*, C, 3, 4).
        Ds: Camera distortion coefficients. Shape (*, C, D).
        distortion_model: Distortion model (e.g. "brown_conrady" or "opencv_fisheye")

    Returns:
        Tensor of shape (*, C, P) containing the normalized reprojection in each view
    """

    check_shape(kps_3d, ("*", "P", "3"))
    P = kps_3d.shape[-2]

    check_shape(Ks, ("*", "C", "3", "3"))
    C = Ks.shape[-3]
    check_shape(Rts, ("*", "C", "3", "4"))
    check_shape(Ds, ("*", C, "D"))

    check_shape(kps_2d, ("*", C, P, "2"))

    if distortion_model == "brown_conrady":
        check_shape(Ds, ("*", C, "5"))
    elif distortion_model == "opencv_fisheye":
        check_shape(Ds, ("*", C, "4"))
    else:
        raise ValueError(f"Invalid distortion model: {distortion_model}")

    e, kps_2d_depth = compute_reprojection_residuals(
        kps_3d, kps_2d, Ks, Rts, Ds, distortion_model
    )
    fx = Ks[..., 0, 0]
    fy = Ks[..., 1, 1]
    f = (fx + fy) / 2
    e_normalized = e / (f.unsqueeze(-1) + torch.finfo(e.dtype).eps)
    return e_normalized, kps_2d_depth


def pairwise_reprojection_consensus_score(
    kps_3d: torch.Tensor,
    kps_2d: torch.Tensor,
    kps_2d_scores: torch.Tensor,
    Rts: torch.Tensor,
    Ks: torch.Tensor,
    Ds: torch.Tensor,
    distortion_model: str,
    lambda_exp_decay: float = 1.0,
) -> torch.Tensor:
    """
    Compute the pairwise reprojection consensus score between all pairs of views.
    This defines how well the 3D points agree between two views based on the reprojection error.
    The greater the number of pairs agreeing on a point, the higher the score.
    This function takes into account the 2D keypoints scores.

    Args:
        kps_3d: 3D points to project. Shape (*, P, 3).
        kps_2d: Target 2D points. Shape (*, C, P, 2).
        kps_2d_scores: 2D keypoints scores. Shape (*, C, P).
        Rts: Camera extrinsics. Shape (*, C, 3, 4).
        Ks: Camera intrinsics. Shape (*, C, 3, 3).
        Ds: Camera distortion coefficients. Shape (*, C, D).
        distortion_model: Distortion model (e.g. "brown_conrady" or "opencv_fisheye")

    Returns:
        Tensor of shape (*, P) containing the pairwise reprojection consensus score for each point.
    """
    check_shape(kps_3d, ("*", "P", "3"))
    P = kps_3d.shape[-2]

    check_shape(Ks, ("*", "C", "3", "3"))
    C = Ks.shape[-3]
    check_shape(Rts, ("*", "C", "3", "4"))
    check_shape(Ds, ("*", C, "D"))

    check_shape(kps_2d, ("*", C, P, "2"))
    check_shape(kps_2d_scores, ("*", C, P))

    if distortion_model == "brown_conrady":
        check_shape(Ds, ("*", C, "5"))
    elif distortion_model == "opencv_fisheye":
        check_shape(Ds, ("*", C, "4"))

    assert lambda_exp_decay > 0, f"Expected positive value, got {lambda_exp_decay}"

    scores = []
    weights = []

    n_views = Rts.shape[0]

    for view_i in range(n_views - 1):
        for view_j in range(view_i + 1, n_views):
            w_i = kps_2d_scores[..., view_i, :]
            w_j = kps_2d_scores[..., view_j, :]

            e_normalized, _ = compute_normalized_reprojection_residuals(
                kps_3d=kps_3d,
                kps_2d=kps_2d[..., [view_i, view_j], :, :],
                Ks=Ks[..., [view_i, view_j], :, :],
                Rts=Rts[..., [view_i, view_j], :, :],
                Ds=Ds[..., [view_i, view_j], :],
                distortion_model=distortion_model,
            )

            pair_weight = torch.sqrt(w_i * w_j)

            pair_score = torch.sqrt(
                torch.exp(-lambda_exp_decay * e_normalized).prod(dim=-2)
            )

            scores.append(pair_score)
            weights.append(pair_weight)

    scores = torch.stack(scores, dim=-1)
    weights = torch.stack(weights, dim=-1)

    # n_pairs = (n_views * (n_views - 1)) // 2
    total_score = torch.mean(scores * weights, dim=-1)
    # total_score = weighted_mean(scores, weights, dim=-1)

    assert total_score.shape[-1] == P, f"Expected (*, P), got {total_score.shape}"
    return total_score


def mpjpe_error(pred_joints_3d: torch.Tensor, target_joints_3d: torch.Tensor):
    """
    Compute the mean per-joint position error (MPJPE) between the predicted and target 3D keypoints.

    Args:
        pred_joints_3d: Predicted 3D keypoints. Shape (*, 3).
        target_joints_3d: Target 3D keypoints. Shape (*, 3).
        reduction: Reduction method.

    Returns:
        MPJPE error.
    """

    assert pred_joints_3d.shape[-1] == 3, "Last dimension must be 3 (x, y, z)"
    assert pred_joints_3d.shape == target_joints_3d.shape, "Input shapes must match"

    distances = torch.sqrt(((pred_joints_3d - target_joints_3d) ** 2).sum(dim=-1))

    # Mean over joints
    mpjpe = torch.mean(distances, dim=-1)
    # Mean over frames
    mpjpe = torch.mean(mpjpe)
    return mpjpe


def mpjpe_pa_error(pred_joints_3d: torch.Tensor, target_joints_3d: torch.Tensor):
    R, T, s = compute_similarity_transform(
        pred_joints_3d, target_joints_3d, estimate_scale=True
    )
    pred_joints_3d_hat = apply_similarity_transform_to_points(pred_joints_3d, R, T, s)

    return mpjpe_error(pred_joints_3d_hat, target_joints_3d)
