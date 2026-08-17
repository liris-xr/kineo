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
from kineo.torch_utils import check_shape


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
    total_score = torch.mean(scores * weights, dim=-1)

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
