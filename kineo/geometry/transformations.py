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
from typing import Literal

from kineo.torch_utils import check_shape

# Cubing a larger radius overflows float32. A downstream finiteness mask
# cannot undo that: it zeroes the incoming gradient, but the local
# derivative is already infinite, so the backward pass computes 0 * inf.
MAX_DISTORTION_RADIUS_SQ = 1e8
from kornia.utils.grid import create_meshgrid
from kornia.geometry.transform.imgwarp import remap
import kornia
import cv2
import numpy as np


def inverse_Rt(Rt: torch.Tensor) -> torch.Tensor:
    """
    Inverse the camera extrinsic matrix.
    """
    assert Rt.ndim >= 2 and Rt.shape[-2:] == (3, 4)
    return inverse_affine_transform_matrix(Rt, orthogonal_linear_part=True)


def compute_similarity_transform(
    X: torch.Tensor,
    Y: torch.Tensor,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finds a similarity transformation (R, T, s) between two given sets of corresponding points sets X and Y.

    Args:
        X (torch.Tensor): Tensor of shape (B, P, 3) containing points
        Y (torch.Tensor): Tensor of shape (B, P, 3) containing points
        estimate_scale (bool): Whether to estimate the scale of the transformation
        allow_reflection (bool): Whether to allow reflections in the transformation

    Returns:
        R (torch.Tensor): Rotation matrix of shape (B, 3, 3)
        T (torch.Tensor): Translation vector of shape (B, 3)
        s (torch.Tensor): Scale factor of shape (B,)
    References:
        [1] Shinji Umeyama: Least-Squares Estimation of Transformation Parameters Between Two Point Patterns
    """
    check_shape(X, [("B", "P", 3), ("P", 3)])
    batch_dims = X.shape[:-2]
    n_points = X.shape[-2]
    check_shape(Y, (*batch_dims, n_points, 3))

    has_batch_dim = X.ndim == 3

    if not has_batch_dim:
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)

    B, P, C = X.shape

    # 1. Remove mean.
    Xmu = X.mean(dim=1, keepdim=True)
    Ymu = Y.mean(dim=1, keepdim=True)

    # Mean-center the points.
    Xc = X - Xmu
    Yc = Y - Ymu

    # Compute the covariance between the point sets Xc, Yc.
    XY_cov = torch.bmm(Xc.transpose(2, 1), Yc)

    # Decompose the covariance matrix.
    U, S, V = torch.svd(XY_cov)

    # Identity matrix used for fixing reflections.
    E = torch.eye(C, dtype=XY_cov.dtype, device=XY_cov.device)[None].repeat(B, 1, 1)

    if not allow_reflection:
        # Reflection test:
        #   Checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # Find the rotation matrix.
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # Estimate scale
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        X_cov = (Xc * Xc).sum((1, 2))
        s = trace_ES / torch.clamp(X_cov, torch.finfo(X_cov.dtype).eps)
        # Recover translation
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        # Recover translation
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]
        # Unit scaling since we do not estimate scale
        s = T.new_ones(B)

    if not has_batch_dim:
        R = R.squeeze(0)
        T = T.squeeze(0)
        s = s.squeeze(0)

    return R, T, s


def compute_weighted_similarity_transform(
    X: torch.Tensor,
    Y: torch.Tensor,
    weights: torch.Tensor = None,
    estimate_scale: bool = False,
    allow_reflection: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Finds a similarity transformation (R, T, s) between two given sets of corresponding points sets X and Y.

    Args:
        X (torch.Tensor): Tensor of shape (B, P, 3) containing points
        Y (torch.Tensor): Tensor of shape (B, P, 3) containing points
        weights (torch.Tensor): Tensor of shape (B, P) or (P,) containing weights for each point.
                               If None, uniform weights are used.
        estimate_scale (bool): Whether to estimate the scale of the transformation
        allow_reflection (bool): Whether to allow reflections in the transformation

    Returns:
        R (torch.Tensor): Rotation matrix of shape (B, 3, 3)
        T (torch.Tensor): Translation vector of shape (B, 3)
        s (torch.Tensor): Scale factor of shape (B,)
    References:
        [1] Shinji Umeyama: Least-Squares Estimation of Transformation Parameters Between Two Point Patterns
    """
    check_shape(X, [("B", "P", 3), ("P", 3)])
    batch_dims = X.shape[:-2]
    n_points = X.shape[-2]
    check_shape(Y, (*batch_dims, n_points, 3))

    has_batch_dim = X.ndim == 3

    if not has_batch_dim:
        X = X.unsqueeze(0)
        Y = Y.unsqueeze(0)

    B, P, C = X.shape

    # Handle weights
    if weights is None:
        weights = torch.ones(B, P, dtype=X.dtype, device=X.device)
    else:
        if weights.ndim == 1:
            weights = weights.unsqueeze(0).expand(B, -1)
        elif weights.ndim == 2 and not has_batch_dim:
            weights = weights.unsqueeze(0)

        check_shape(weights, (B, P))

    # Normalize weights so they sum to 1 for each batch
    weights_normalized = weights / weights.sum(dim=1, keepdim=True).clamp(
        min=torch.finfo(weights.dtype).eps
    )

    # Add dimension for broadcasting: (B, P, 1)
    w = weights_normalized.unsqueeze(-1)

    # 1. Compute weighted means
    Xmu = (w * X).sum(dim=1, keepdim=True)  # (B, 1, 3)
    Ymu = (w * Y).sum(dim=1, keepdim=True)  # (B, 1, 3)

    # Mean-center the points.
    Xc = X - Xmu  # (B, P, 3)
    Yc = Y - Ymu  # (B, P, 3)

    # Compute the weighted covariance between the point sets Xc, Yc.
    # XY_cov = sum_i w_i * Xc_i^T * Yc_i
    # Since weights are normalized, we can compute: sum_i w_i * Xc_i^T * Yc_i
    Xc_weighted = Xc * w  # (B, P, 3)
    XY_cov = torch.bmm(Xc_weighted.transpose(2, 1), Yc)  # (B, 3, 3)

    # Decompose the covariance matrix.
    U, S, V = torch.svd(XY_cov)

    # Identity matrix used for fixing reflections.
    E = torch.eye(C, dtype=XY_cov.dtype, device=XY_cov.device)[None].repeat(B, 1, 1)

    if not allow_reflection:
        # Reflection test:
        #   Checks whether the estimated rotation has det==1,
        #   if not, finds the nearest rotation s.t. det==1 by
        #   flipping the sign of the last singular vector U
        R_test = torch.bmm(U, V.transpose(2, 1))
        E[:, -1, -1] = torch.det(R_test)

    # Find the rotation matrix.
    R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))

    if estimate_scale:
        # Estimate scale using weighted computation
        trace_ES = (torch.diagonal(E, dim1=1, dim2=2) * S).sum(1)
        # Weighted variance of X: sum_i w_i * ||Xc_i||^2
        X_weighted_var = (w * (Xc * Xc).sum(dim=-1, keepdim=True)).sum(dim=1)[
            :, 0
        ]  # (B,)
        s = trace_ES / torch.clamp(
            X_weighted_var, torch.finfo(X_weighted_var.dtype).eps
        )
        # Recover translation
        T = Ymu[:, 0, :] - s[:, None] * torch.bmm(Xmu, R)[:, 0, :]
    else:
        # Recover translation
        T = Ymu[:, 0, :] - torch.bmm(Xmu, R)[:, 0, :]
        # Unit scaling since we do not estimate scale
        s = T.new_ones(B)

    if not has_batch_dim:
        R = R.squeeze(0)
        T = T.squeeze(0)
        s = s.squeeze(0)
        if weights is not None:
            weights = weights.squeeze(0)

    return R, T, s


def apply_similarity_transform_to_points(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Apply a similarity transform to a set of points.

    Args:
        X (torch.Tensor): Tensor of shape (B, P, 3) or (P, 3) containing points
        R (torch.Tensor): Rotation matrix of shape (3, 3) or (B, 3, 3)
        T (torch.Tensor): Translation vector of shape (3,) or (B, 3)
        s (torch.Tensor): Scale factor of shape () or (B,)

    Returns:
        X_transformed (torch.Tensor): Tensor of shape (B, P, 3) or (P, 3) containing the transformed points
    """

    assert X.ndim in [2, 3] and X.shape[-1] == 3, (
        f"Expected X to have shape (B, P, 3) or (P, 3), got {X.shape}"
    )

    original_shape = X.shape

    batch_dims = X.shape[:-2]

    assert R.shape in [
        (*batch_dims, 3, 3),
        (1, 3, 3),
        (3, 3),
    ], (
        f"Expected R to have shape {batch_dims + (3, 3)}, (1, 3, 3) or (3, 3), got {R.shape}"
    )
    assert T.shape in [
        (*batch_dims, 3),
        (1, 3),
        (3,),
    ], f"Expected T to have shape {batch_dims + (3,)}, (1, 3) or (3,), got {T.shape}"
    assert s.shape in [
        (*batch_dims,),
        (1,),
        (),
    ], f"Expected s to have shape {batch_dims}, (1,) or (), got {s.shape}"

    # Make broadcastable
    if len(batch_dims) > 0 and R.ndim == 2:
        R = R.expand((*batch_dims, -1, -1))
    if len(batch_dims) > 0 and T.ndim == 1:
        T = T.expand((*batch_dims, -1))
    if len(batch_dims) > 0 and s.ndim == 0:
        s = s.expand((*batch_dims,))

    X = torch.einsum("...nm,...mp->...np", X, s[..., None, None] * R) + T[..., None, :]

    return X.reshape(original_shape)


def apply_similarity_transform_to_Rt(
    Rt: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    """
    Apply a similarity transform to a camera Rt matrix.

    Args:
        Rt (torch.Tensor): Tensor of shape (B, 3, 4) or (3, 4) containing the camera Rt matrix
        R (torch.Tensor): Tensor of shape (B, 3, 3) or (3, 3) containing the rotation matrix
        T (torch.Tensor): Tensor of shape (B, 3) or (3,) containing the translation vector
        s (torch.Tensor): Tensor of shape (B,) or () containing the scale factor

    Returns:
        Rt_transformed (torch.Tensor): Tensor of shape (B, 3, 4) or (3, 4) containing the transformed camera Rt matrix
    """
    assert Rt.ndim in [3, 2] and Rt.shape[-2:] == (3, 4)

    original_shape = Rt.shape
    batch_dims = Rt.shape[:-2]

    assert R.shape in [
        (*batch_dims, 3, 3),
        (1, 3, 3),
        (3, 3),
    ], f"Expected R to have shape {batch_dims + (3, 3)} or (3, 3), got {R.shape}"
    assert T.shape in [
        (*batch_dims, 3),
        (1, 3),
        (3,),
    ], f"Expected T to have shape {batch_dims + (3,)} or (3,), got {T.shape}"
    assert s.shape in [
        (*batch_dims,),
        (1,),
        (),
    ], f"Expected s to have shape {batch_dims}, (1,) or (), got {s.shape}"

    Rt_inv = inverse_Rt(Rt)
    cam_pose = Rt_inv[..., :3, 3]
    cam_pose_right = cam_pose + Rt_inv[..., :3, 0]
    cam_pose_up = cam_pose + Rt_inv[..., :3, 1]
    cam_pose_forward = cam_pose + Rt_inv[..., :3, 2]

    cam_pose_right = apply_similarity_transform_to_points(
        cam_pose_right.reshape(-1, 3), R, T, s
    ).reshape(-1, 3)
    cam_pose_up = apply_similarity_transform_to_points(
        cam_pose_up.reshape(-1, 3), R, T, s
    ).reshape(-1, 3)
    cam_pose_forward = apply_similarity_transform_to_points(
        cam_pose_forward.reshape(-1, 3), R, T, s
    ).reshape(-1, 3)
    cam_pose = apply_similarity_transform_to_points(
        cam_pose.reshape(-1, 3), R, T, s
    ).reshape(-1, 3)

    cam_right = cam_pose_right - cam_pose
    cam_up = cam_pose_up - cam_pose
    cam_forward = cam_pose_forward - cam_pose

    cam_right = torch.nn.functional.normalize(cam_right, dim=-1)
    cam_up = torch.nn.functional.normalize(cam_up, dim=-1)
    cam_forward = torch.nn.functional.normalize(cam_forward, dim=-1)

    Rt_transformed_inv = torch.zeros((*batch_dims, 3, 4), device=Rt.device)
    Rt_transformed_inv[..., :3, 0] = cam_right
    Rt_transformed_inv[..., :3, 1] = cam_up
    Rt_transformed_inv[..., :3, 2] = cam_forward
    Rt_transformed_inv[..., :3, 3] = cam_pose

    Rt_transformed = inverse_Rt(Rt_transformed_inv)

    return Rt_transformed.reshape(original_shape)


def inverse_affine_transform_matrix(
    M: torch.Tensor, orthogonal_linear_part: bool = False
) -> torch.Tensor:
    """
    Inverse an affine transform matrix.
        y = A @ x + b
        x = A^-1 @ (y - b)
        x = M^-1 @ y
        M^-1 = [A^-1 | -A^-1 @ b]
    If ortho_linear is True, the linear transformation matrix is assumed to be an orthogonal matrix and the inverse is computed using the transpose.

    Args:
        M (torch.Tensor): Affine transform matrix. Shape (*, D, D + 1).

    Returns:
        M_inv (torch.Tensor): Inverse of the affine transform matrix. Shape (*, D, D + 1).
    """
    assert M.ndim >= 2
    D = M.shape[-2]
    assert M.shape[-1] == D + 1
    A = M[..., :D, :D]
    b = M[..., :D, D : D + 1]
    if orthogonal_linear_part:
        A_inv = A.transpose(-1, -2)
    else:
        A_inv = torch.linalg.inv(A)
    return torch.cat([A_inv, -A_inv @ b], dim=-1)


def distort_points(
    points: torch.Tensor,
    K: torch.Tensor,
    D: torch.Tensor,
    distortion_model: Literal["brown_conrady", "opencv_fisheye"] = "brown_conrady",
) -> torch.Tensor:
    if distortion_model == "brown_conrady":
        return distort_points_brown_conrady(points, K=K, D=D)
    elif distortion_model == "opencv_fisheye":
        return distort_points_fisheye(points, K=K, D=D)
    elif not isinstance(distortion_model, str):
        raise ValueError(f"Expected distortion model to be a string, got {distortion_model}")
    else:
        raise NotImplementedError(
            f"Distortion model {distortion_model} not implemented"
        )


def distort_points_brown_conrady(
    points: torch.Tensor,
    K: torch.Tensor,
    D: torch.Tensor,
    new_K: torch.Tensor | None = None,
) -> torch.Tensor:
    check_shape(points, ("*", "P", 2))
    check_shape(K, ("*", 3, 3))
    check_shape(D, ("*", 5))

    if new_K is not None:
        check_shape(new_K, ("*", 3, 3))

    if new_K is None:
        new_K = K

    # Convert 2D points from pixels to normalized camera coordinates
    new_cx: torch.Tensor = new_K[..., 0:1, 2]  # princial point in x (Bx1)
    new_cy: torch.Tensor = new_K[..., 1:2, 2]  # princial point in y (Bx1)
    new_fx: torch.Tensor = new_K[..., 0:1, 0]  # focal in x (Bx1)
    new_fy: torch.Tensor = new_K[..., 1:2, 1]  # focal in y (Bx1)

    # This is equivalent to K^-1 [u,v,1]^T
    x: torch.Tensor = (
        points[..., 0] - new_cx
    ) / new_fx  # (BxN - Bx1)/Bx1 -> BxN or (N,)
    y: torch.Tensor = (
        points[..., 1] - new_cy
    ) / new_fy  # (BxN - Bx1)/Bx1 -> BxN or (N,)

    # Distort points
    r2 = (x * x + y * y).clamp(max=MAX_DISTORTION_RADIUS_SQ)

    k1, k2, p1, p2, k3 = D[..., 0:1], D[..., 1:2], D[..., 2:3], D[..., 3:4], D[..., 4:5]

    rad_poly = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2**3
    xd = x * rad_poly + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad_poly + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # Convert points from normalized camera coordinates to pixel coordinates
    cx: torch.Tensor = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy: torch.Tensor = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx: torch.Tensor = K[..., 0:1, 0]  # focal in x (Bx1)
    fy: torch.Tensor = K[..., 1:2, 1]  # focal in y (Bx1)

    x = fx * xd + cx
    y = fy * yd + cy

    return torch.stack([x, y], dim=-1)

def distort_points_fisheye(
    points: torch.Tensor,
    K: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    check_shape(points, ("*", "P", 2))
    check_shape(K, ("*", 3, 3))
    check_shape(D, ("*", 4))

    normalized_points = kornia.geometry.normalize_points_with_intrinsics(points, K)
    x, y = normalized_points[..., 0], normalized_points[..., 1]

    r = torch.sqrt(x**2 + y**2 + torch.finfo(x.dtype).eps)
    theta = torch.atan(r)

    theta2 = theta**2
    theta_d = theta * (1 + D[..., 0:1] * theta2 + D[..., 1:2] * theta2**2 + D[..., 2:3] * theta2**3 + D[..., 3:4] * theta2**4)
    scale = theta_d / r
    distorted_points = torch.stack([x * scale, y * scale], dim=-1)

    distorted_points = kornia.geometry.denormalize_points_with_intrinsics(distorted_points, K)
    return distorted_points

def undistort_points(
    points: torch.Tensor,
    K: torch.Tensor,
    D: torch.Tensor,
    num_iters: int = 5,
    distortion_model: Literal["brown_conrady", "opencv_fisheye"] = "brown_conrady",
) -> torch.Tensor:
    """
    Undistort points with according to the distortion model.

    Args:
        points: Input image points with shape (*, N, 2).
        K: Intrinsic matrix.
        D: Distortion coefficients.
        new_K: New intrinsic matrix.
        num_iters: Number of iterations.
        distortion_model: Distortion model to use.

    Returns:
        Undistorted points with the same shape as input
    """
    # TODO: replace all of this with pytorch versions to preserve gradients
    if distortion_model == "brown_conrady":
        if K.ndim == 2:
            assert K.shape == (3, 3)
            assert D.shape == (5,)
            points_undistorted = cv2.undistortPoints(
                points.reshape(-1, 1, 2).cpu().numpy(),
                K.cpu().numpy(),
                D.cpu().numpy(),
                None,
                None,
                K.cpu().numpy(),
            ).reshape(-1, 2)
            points_undistorted = torch.from_numpy(points_undistorted).to(points.device)
            return points_undistorted
        else:
            n_points = points.shape[-2]
            K = K.reshape(-1, 3, 3)
            D = D.reshape(-1, 5)
            points_undistorted = torch.zeros_like(points).reshape(-1, n_points, 2)

            for i in range(len(K)):
                p = cv2.undistortPoints(
                    points[i].reshape(n_points, 1, 2).cpu().numpy(),
                    K[i].cpu().numpy(),
                    D[i].cpu().numpy(),
                )
                p = torch.from_numpy(p).to(points.device).reshape(n_points, 2)
                points_undistorted[i] = p
            return points_undistorted.reshape_as(points)

        # TODO: kornia undistort_points is invalid
        # return kornia.geometry.undistort_points(
        #     points, K=K, dist=D, num_iters=num_iters
        # )
    elif distortion_model == "opencv_fisheye":
        if K.ndim == 2:
            assert K.shape == (3, 3)
            assert D.shape == (4,)
            points_undistorted = cv2.fisheye.undistortPoints(
                distorted=points.reshape(-1, 1, 2).cpu().numpy(),
                K=K.cpu().numpy(),
                D=D.cpu().numpy(),
                R=np.eye(3),
                P=K.cpu().numpy(),
            ).reshape(-1, 2)
            points_undistorted = torch.from_numpy(points_undistorted).to(points.device)
            points_undistorted = points_undistorted.reshape_as(points)
            return points_undistorted
        else:
            n_points = points.shape[-2]
            K = K.reshape(-1, 3, 3)
            D = D.reshape(-1, 5)
            points_undistorted = torch.zeros_like(points).reshape(-1, n_points, 2)

            for i in range(len(K)):
                p = cv2.fisheye.undistortPoints(
                    distorted=points[i].reshape(n_points, 1, 2).cpu().numpy(),
                    K=K[i].cpu().numpy(),
                    D=D[i].cpu().numpy(),
                )
                p = torch.from_numpy(p).to(points.device).reshape(n_points, 2)
                points_undistorted[i] = p
            return points_undistorted.reshape_as(points)
    else:
        raise NotImplementedError(
            f"Distortion model {distortion_model} not implemented"
        )


def compute_optimal_K(
    img_hw: tuple[int, int],
    K: torch.Tensor,
    D: torch.Tensor,
    distortion_model: Literal["brown_conrady", "opencv_fisheye"] = "brown_conrady",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the optimal intrinsics for a given distortion model.

    Args:
        img_hw (tuple[int, int]): Image height and width.
        K (torch.Tensor): Intrinsic matrix.
        D (torch.Tensor): Distortion coefficients.
        distortion_model (Literal["brown_conrady", "opencv_fisheye"]): Distortion model to use.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: New intrinsic matrix and crop region in xyxy format.
    """
    h, w = img_hw

    if distortion_model == "brown_conrady":
        distorted_points = torch.tensor(
            [
                [0, 0],  # x=left y=top
                [0, h / 2],  # x=left y=center
                [0, h - 1],  # x=left y=bottom
                [w / 2, 0],  # x=center y=top
                [w / 2, h / 2],  # x=center y=center
                [w / 2, h - 1],  # x=center y=bottom
                [w - 1, 0],  # x=right y=top
                [w - 1, h / 2],  # x=right y=center
                [w - 1, h],  # x=right y=bottom
            ]
        ).reshape((-1, 2))
        distorted_points = distorted_points.to(K.device)

        undistorted_points = undistort_points(
            distorted_points, K=K, D=D, distortion_model=distortion_model
        )

        crop_x1 = undistorted_points[[0, 1, 2], 0].max()  # all the left points
        crop_x2 = undistorted_points[[6, 7, 8], 0].min()  # all the right points
        crop_y1 = undistorted_points[[0, 3, 6], 1].max()  # all the top points
        crop_y2 = undistorted_points[[2, 5, 8], 1].min()  # all the bottom points
        crop_x1 = crop_x1.ceil().int()
        crop_x2 = crop_x2.floor().int()
        crop_y1 = crop_y1.ceil().int()
        crop_y2 = crop_y2.floor().int()
        crop_x1 = max(crop_x1, 0)
        crop_y1 = max(crop_y1, 0)
        crop_x2 = min(crop_x2, w)
        crop_y2 = min(crop_y2, h)
        roi_xyxy = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2], device=K.device)

        # Crop the intrinsics to the new ROI
        new_cx = K[0, 2] - crop_x1
        new_cy = K[1, 2] - crop_y1
        new_K = K.clone()
        new_K[0, 2] = new_cx
        new_K[1, 2] = new_cy
        return new_K, roi_xyxy

    elif distortion_model == "opencv_fisheye":
        raise NotImplementedError(
            "OpenCV fisheye compute_optimal_intrinsics not implemented"
        )
    else:
        raise NotImplementedError(
            f"Distortion model {distortion_model} not implemented"
        )


def undistort_image(
    img: torch.Tensor,
    K: torch.Tensor,
    D: torch.Tensor,
    distortion_model: Literal["brown_conrady", "opencv_fisheye"] = "brown_conrady",
) -> torch.Tensor:
    """
    Undistort an image.

    Args:
        img (torch.Tensor): Image to undistort (C, H, W)
        K (torch.Tensor): Intrinsic matrix
        D (torch.Tensor | None): Distortion coefficients
        distortion_model (Literal["brown_conrady", "opencv_fisheye"]): Distortion model to use

    Returns:
        Undistorted image (C, H, W)
    """

    K = K.to(img.device)
    D = D.to(img.device)

    if distortion_model == "brown_conrady":
        dtype = img.dtype
        if dtype == torch.uint8:
            img = img.float() / 255.0
        undistorted_img = kornia.geometry.undistort_image(img, K=K, dist=D)
        if dtype == torch.uint8:
            undistorted_img = (undistorted_img * 255.0).to(torch.uint8)
        return undistorted_img
    elif distortion_model == "opencv_fisheye":
        # TODO: Implement OpenCV fisheye undistort_image using pytorch only
        # Convert to HWC BGR format
        img_np = img.flip(0).permute(1, 2, 0).cpu().numpy()
        img_undistorted_np = cv2.fisheye.undistortImage(
            img_np, K=K.cpu().numpy(), D=D.cpu().numpy(), Knew=K.cpu().numpy()
        )
        img_undistorted = (
            torch.from_numpy(img_undistorted_np).permute(2, 0, 1).flip(0).to(img.device)
        )
        return img_undistorted
    else:
        raise NotImplementedError(
            f"Distortion model {distortion_model} not implemented"
        )


def distort_image(
    img: torch.Tensor,
    K: torch.Tensor,
    D: torch.Tensor,
    distortion_model: Literal["brown_conrady", "opencv_fisheye"] = "brown_conrady",
) -> torch.Tensor:
    """
    Distort an image.

    Args:
        img (torch.Tensor): Image to undistort (C, H, W)
        K (torch.Tensor): Intrinsic matrix
        D (torch.Tensor | None): Distortion coefficients
        distortion_model (Literal["brown_conrady", "opencv_fisheye"]): Distortion model to use

    Returns:
        Distorted image (C, H, W)
    """

    distorted_img = None

    K = K.to(img.device)
    D = D.to(img.device)

    dtype = img.dtype
    if dtype == torch.uint8:
        img = img.float() / 255.0

    if distortion_model == "brown_conrady":
        distorted_img = distort_image_brown_conrady(img, K=K, dist=D)
    elif distortion_model == "opencv_fisheye":
        distorted_img = distort_image_opencv_fisheye(img, K=K, dist=D)
    else:
        raise NotImplementedError(
            f"Distortion model {distortion_model} not implemented"
        )

    if dtype == torch.uint8:
        distorted_img = (distorted_img * 255.0).to(torch.uint8)

    return distorted_img


def distort_image_opencv_fisheye(
    image: torch.Tensor, K: torch.Tensor, dist: torch.Tensor
) -> torch.Tensor:
    r"""Compensate an image for lens distortion.

    Radial :math:`(k_1, k_2, k_3, k_4)`

    Args:
        image: Input image with shape :math:`(*, C, H, W)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,k_3,k_4)`. This is
            a vector with 5 elements.

    Returns:
        Distorted image with shape :math:`(*, C, H, W)`.

    """

    K = K.to(image.device)
    dist = dist.to(image.device)

    if len(image.shape) < 3:
        raise ValueError(f"Image shape is invalid. Got: {image.shape}.")

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K matrix shape is invalid. Got {K.shape}.")

    if dist.shape[-1] != 4:
        raise ValueError(
            f"Invalid number of distortion coefficients. Got {dist.shape[-1]}."
        )

    if not image.is_floating_point():
        raise ValueError(
            f"Invalid input image data type. Input should be float. Got {image.dtype}."
        )

    if image.shape[:-3] != K.shape[:-2] or image.shape[:-3] != dist.shape[:-1]:
        # Input with image shape (1, C, H, W), K shape (3, 3), dist shape (4)
        # allowed to avoid a breaking change.
        if not all(
            (image.shape[:-3] == (1,), K.shape[:-2] == (), dist.shape[:-1] == ())
        ):
            raise ValueError(
                "Input shape is invalid. Input batch dimensions should match. "
                f"Got {image.shape[:-3]}, {K.shape[:-2]}, {dist.shape[:-1]}."
            )

    channels, rows, cols = image.shape[-3:]
    B = image.numel() // (channels * rows * cols)

    # Create point coordinates for each pixel of the image
    xy_grid: torch.Tensor = create_meshgrid(
        rows, cols, False, image.device, dtype=torch.float32
    )
    pts = xy_grid.reshape(-1, 2)  # (rows*cols)x2 matrix of pixel coordinates

    # Undistort points and define maps
    # pts_normalized = normalize_points_with_intrinsics(pts, K)
    pts_distorted = cv2.fisheye.undistortPoints(
        pts.reshape(-1, 1, 2).cpu().numpy(),
        # pts_normalized.reshape(-1, 1, 2).cpu().numpy(),
        K=K.cpu().numpy(),
        D=dist.cpu().numpy(),
        R=np.eye(3),
        P=K.cpu().numpy(),
    ).reshape(-1, 2)

    pts_distorted = torch.from_numpy(pts_distorted).to(pts.device)

    mapx: torch.Tensor = pts_distorted[..., 0].reshape(
        B, rows, cols
    )  # B x rows x cols, float
    mapy: torch.Tensor = pts_distorted[..., 1].reshape(
        B, rows, cols
    )  # B x rows x cols, float

    # Remap image to undistort
    out = remap(image.reshape(B, channels, rows, cols), mapx, mapy, align_corners=True)

    return out.view_as(image)


def distort_image_brown_conrady(
    image: torch.Tensor, K: torch.Tensor, dist: torch.Tensor
) -> torch.Tensor:
    r"""Compensate an image for lens distortion.

    Radial :math:`(k_1, k_2, k_3)`,
    tangential :math:`(p_1, p_2)`
    distortion models are considered in this function.

    Args:
        image: Input image with shape :math:`(*, C, H, W)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2,k_3)`. This is
            a vector with 5 elements.

    Returns:
        Distorted image with shape :math:`(*, C, H, W)`.

    """

    K = K.to(image.device)
    dist = dist.to(image.device)

    if len(image.shape) < 3:
        raise ValueError(f"Image shape is invalid. Got: {image.shape}.")

    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K matrix shape is invalid. Got {K.shape}.")

    if dist.shape[-1] != 5:
        raise ValueError(
            f"Invalid number of distortion coefficients. Got {dist.shape[-1]}."
        )

    if not image.is_floating_point():
        raise ValueError(
            f"Invalid input image data type. Input should be float. Got {image.dtype}."
        )

    if image.shape[:-3] != K.shape[:-2] or image.shape[:-3] != dist.shape[:-1]:
        # Input with image shape (1, C, H, W), K shape (3, 3), dist shape (4)
        # allowed to avoid a breaking change.
        if not all(
            (image.shape[:-3] == (1,), K.shape[:-2] == (), dist.shape[:-1] == ())
        ):
            raise ValueError(
                "Input shape is invalid. Input batch dimensions should match. "
                f"Got {image.shape[:-3]}, {K.shape[:-2]}, {dist.shape[:-1]}."
            )

    channels, rows, cols = image.shape[-3:]
    B = image.numel() // (channels * rows * cols)

    # Create point coordinates for each pixel of the image
    xy_grid: torch.Tensor = create_meshgrid(
        rows, cols, False, image.device, dtype=torch.float32
    )
    pts = xy_grid.reshape(-1, 2)  # (rows*cols)x2 matrix of pixel coordinates

    # Undistort points and define maps
    pts_undistorted: torch.Tensor = undistort_points(pts, K, dist)  # Bx(rows*cols)x2
    mapx: torch.Tensor = pts_undistorted[..., 0].reshape(
        B, rows, cols
    )  # B x rows x cols, float
    mapy: torch.Tensor = pts_undistorted[..., 1].reshape(
        B, rows, cols
    )  # B x rows x cols, float

    # Remap image to undistort
    out = remap(image.reshape(B, channels, rows, cols), mapx, mapy, align_corners=True)

    return out.view_as(image)
