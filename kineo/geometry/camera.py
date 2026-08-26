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
from typing import Optional
from kineo.geometry.transformations import (
    inverse_affine_transform_matrix,
    distort_points,
)
import kornia
from kineo.geometry.conversions import (
    convert_points_from_homogeneous,
    convert_points_to_homogeneous,
)
import cv2
from kineo.torch_utils import check_shape
from math import radians, tan, atan, degrees
from typing import Literal

# Points closer than this to the camera plane have no usable projection. Bounding
# the perspective divide keeps them differentiable; callers reject them on the
# exact depth that projection returns alongside the 2D points.
MIN_PROJECTION_DEPTH = 1e-6


def create_default_K(
    hfov: float, resolution_hw: tuple[int, int], device: torch.device = "cpu"
) -> torch.Tensor:
    """
    Create a default camera intrinsic matrix for each camera.

    Args:
        hfov: Horizontal field of view in degrees.
        resolution_hw: Resolution of the image in (height, width).
        device: Device to create the camera intrinsic matrix on.
    """
    f = focal_length_from_hfov(hfov, resolution_hw[1])
    K = torch.zeros(3, 3, device=device)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = resolution_hw[1] / 2
    K[1, 2] = resolution_hw[0] / 2
    K[2, 2] = 1.0
    return K


def focal_length_from_hfov(hfov: float, width: int) -> float:
    """
    Compute the focal length from the horizontal field of view and the image resolution.

    Args:
        hfov: Horizontal field of view in degrees.
        width: Width of the image in pixels.
    """
    hfov_rad = radians(hfov)
    return width / (2 * tan(hfov_rad / 2))


def hfov_from_focal_length(focal_length: float, width: int) -> float:
    """
    Compute the horizontal field of view from the focal length and the image resolution.

    Args:
        focal_length: Focal length in pixels.
        width: Width of the image in pixels.
    """
    return 2 * degrees(atan(width / (2 * focal_length)))


def cross_product_matrix(v: torch.Tensor) -> torch.Tensor:
    """
    Given a 3D vector v, compute the cross product [v]_x matrix representation.
    Args:
        v: A vector or a batch of vectors of shape (..., 3).
    Returns:
        A Bx3x3 matrix corresponding to the cross product matrix of the input vector.

    Example:
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([4, 5, 6])
        a_x = cross_product_matrix(v)
        c = a_x @ b
    """

    return torch.tensor(
        [
            [0, -v[..., 2], v[..., 1]],
            [v[..., 2], 0, -v[..., 0]],
            [-v[..., 1], v[..., 0], 0],
        ],
        device=v.device,
    )


def relative_Rt(Rt1: torch.Tensor, Rt2: torch.Tensor) -> torch.Tensor:
    """
    Given two rigid transformation matrices, compute the relative transformation to convert coordinates from the first frame to the second frame.

    Args:
        RT1: The first 3x4 rigid transformation matrix.
        RT2: The second 3x4 rigid transformation matrix.
    Returns:
        A 3x4 matrix corresponding to the relative transformation from the first frame to the second frame.

    Example:
        world_to_cam1 = ... # 3x4 matrix representing the transformation from world to camera 1
        world_to_cam2 = ... # 3x4 matrix representing the transformation from world to camera 2
        cam1_to_cam2 = relative_Rt(world_to_cam1, world_to_cam2)
    """
    assert Rt1.shape[-2:] == (3, 4)
    assert Rt2.shape[-2:] == (3, 4)
    return multiply_Rt(Rt2, inverse_Rt(Rt1))


def fundamental_matrix_from_essential_matrix(
    E: torch.Tensor, K1: torch.Tensor, K2: torch.Tensor
) -> torch.Tensor:
    """
    Given the camera intrinsics and essential matrix of two cameras, compute the fundamental matrix.
    The fundamental matrix is defined as F = K2^-T * E * K1^-1.

    Args:
        E: The 3x3 essential matrix.
        K1: The 3x3 camera intrinsics matrix for the first camera.
        K2: The 3x3 camera intrinsics matrix for the second camera.

    Returns:
        A 3x3 matrix corresponding to the fundamental matrix between the two cameras.

    Example:
        world_to_cam1 = ... # 3x4 matrix representing the transformation from world to camera 1
        world_to_cam2 = ... # 3x4 matrix representing the transformation from world to camera 2
        K1 = ... # 3x3 camera intrinsics matrix for camera 1
        K2 = ... # 3x3 camera intrinsics matrix for camera 2
        F = fundamental_matrix(K1, K2, world_to_cam1, world_to_cam2)
    """
    F = torch.linalg.inv(K2.T) @ E @ torch.linalg.inv(K1)
    # Normalize the fundamental matrix so that F[2, 2] = 1
    F = F / F[..., 2, 2]
    return F


def essential_matrix_from_Rt(Rt_rel: torch.Tensor) -> torch.Tensor:
    """
    Given two rigid transformation matrices, compute the essential matrix between the two frames.

    Args:
        Rt_rel: The relative 3x4 rigid transformation matrix to go from camera 1 to camera 2.

    Returns:
        A 3x3 matrix corresponding to the essential matrix between the two frames.

    Example:
        world_to_cam1 = ... # 3x4 matrix representing the transformation from world to camera 1
        world_to_cam2 = ... # 3x4 matrix representing the transformation from world to camera 2
        E = essential_matrix(world_to_cam1, world_to_cam2)
    """
    R = Rt_rel[..., :3, :3]
    T = Rt_rel[..., :3, 3]
    E = cross_product_matrix(T) @ R
    return E


def compute_focal_length_rnorm_from_hfov(hfov: float) -> float:
    """
    Compute the normalized focal length from the horizontal field of view.
        Note: the normalized focal length is the focal length divided by the image resolution. If you
        need the focal length in pixels, you need to multiply the result by the image resolution.

    Args:
        hfov: Horizontal field of view in degrees.

    Returns:
        Resolution-independent focal length.
    """
    hfov_rad = radians(hfov)
    return 1 / (2 * tan(hfov_rad / 2))


def multiply_Rt(RT1: torch.Tensor, RT2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two rigid transformation matrices together.
    RT = RT1 @ RT2

    Args:
        RT1: A 3x4 rigid transformation matrix.
        RT2: A 3x4 rigid transformation matrix.

    Returns:
        A 3x4 matrix corresponding to the product of the two input rigid transformation matrices.
    """
    assert RT1.shape[-2:] == (3, 4)
    assert RT2.shape[-2:] == (3, 4)

    # Create new tensors instead of modifying in place
    R = RT1[..., :3, :3] @ RT2[..., :3, :3]
    t = RT1[..., :3, :3] @ RT2[..., :3, 3] + RT1[..., :3, 3]

    # Stack the results instead of modifying in place
    return torch.cat([R, t.unsqueeze(-1)], dim=-1)


def inverse_Rt(Rt: torch.Tensor) -> torch.Tensor:
    """
    Inverse the camera extrinsic matrix.
    """
    assert Rt.ndim >= 2 and Rt.shape[-2:] == (3, 4)
    return inverse_affine_transform_matrix(Rt, orthogonal_linear_part=True)


def inverse_K(K: torch.Tensor) -> torch.Tensor:
    """
    Inverse the camera intrinsic matrix.

    Args:
        K: Camera intrinsic matrix. Shape (*, 3, 3).

    Returns:
        K_inv: Inverse of the camera intrinsic matrix. Shape (*, 3, 3).
    """
    assert K.ndim >= 2 and K.shape[-2:] == (3, 3)
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    K_inv = torch.empty_like(K)
    K_inv[..., 0, 0] = 1.0 / fx
    K_inv[..., 1, 1] = 1.0 / fy
    K_inv[..., 0, 2] = -cx / fx
    K_inv[..., 1, 2] = -cy / fy
    return K_inv


def transform_vectors_from_camera_to_world(
    vectors_cam: torch.Tensor, Rt: torch.Tensor
) -> torch.Tensor:
    """
    Transform 3D vectors expressed in the camera frame to the world frame.
    """
    check_shape(vectors_cam, ("*", "P", 3))
    batch_dims = vectors_cam.shape[:-2]
    check_shape(Rt, (*batch_dims, 3, 4))

    return torch.einsum("pi,ij->pj", vectors_cam, Rt[:3, :3])


def transform_points_from_camera_to_world(
    points_3d_cam: torch.Tensor, Rt: torch.Tensor
) -> torch.Tensor:
    """
    Transform 3D points expressed in the camera frame to the world frame.
        X_world = Rt^-1 @ [X_cam, 1]

    Args:
        points_3d_cam (torch.Tensor): 3D points to convert. Shape (*, P, 3).
        Rt (torch.Tensor): Camera extrinsic matrix. Shape (*, 3, 4).

    Returns:
        points_3d_world (torch.Tensor): 3D points in the world frame. Shape (*, P, 3).
    """
    check_shape(points_3d_cam, ("*", "P", 3))
    batch_dims = points_3d_cam.shape[:-2]
    check_shape(Rt, [(3, 4), (*batch_dims, 3, 4)])

    # Expand Rt to be broadcastable with points_3d_cam
    if Rt.ndim == 2:
        Rt = Rt.view(*[1 for _ in batch_dims], 3, 4)

    Rt_inv = inverse_Rt(Rt)

    points_h = convert_points_to_homogeneous(points_3d_cam)
    points_transformed = torch.einsum("...ij,...pj->...pi", Rt_inv, points_h)

    return points_transformed


def transform_points_from_world_to_camera(
    points_3d_world: torch.Tensor, Rt: torch.Tensor
) -> torch.Tensor:
    """
    Transform 3D points expressed in the world frame to the camera frame.
        X_cam = Rt @ [X_world, 1]

    Args:
        points_3d_world: 3D points to convert. Shape (P, 3) or (F, P, 3) with F being the number of frames and P being the number of points.
        Rt: Camera extrinsic matrix. Shape (3, 4), (C, 3, 4) or (F, C, 3, 4) with C being the number of cameras.

    Returns:
        points_3d_cam: 3D points in the camera frame. Shape (P, 3), (C, P, 3) or (F, C, P, 3).
    """
    check_shape(points_3d_world, [("P", 3), ("F", "P", 3)])
    points_has_frame_dim = points_3d_world.ndim == 3

    if not points_has_frame_dim:
        points_3d_world = points_3d_world.unsqueeze(0)

    F, P, _ = points_3d_world.shape

    Rt_has_camera_dim = Rt.ndim >= 3
    Rt_has_frame_dim = Rt.ndim == 4

    if not Rt_has_camera_dim:
        Rt = Rt.unsqueeze(0)
    if not Rt_has_frame_dim:
        Rt = Rt.unsqueeze(0)

    # Ensure that types are broadcastable
    check_shape(
        Rt,
        [(1, 1, 3, 4), (1, "C", 3, 4), (F, "C", 3, 4)],
    )

    points_h = convert_points_to_homogeneous(points_3d_world)
    points_transformed = torch.einsum("fcij,fpj->fcpi", Rt, points_h)

    if not points_has_frame_dim and not Rt_has_frame_dim:
        points_transformed = points_transformed.squeeze(-4)
    if not Rt_has_camera_dim:
        points_transformed = points_transformed.squeeze(-3)

    return points_transformed


def camera_centers_from_extrinsics(Rts: torch.Tensor) -> torch.Tensor:
    r"""Camera centers in the world frame.

    Inverts :math:`X_{cam} = R X_{world} + t` at :math:`X_{cam} = 0`, giving
    :math:`C = -R^{T} t`.

    Args:
        Rts: World-to-camera extrinsics with shape :math:`(*, C, 3, 4)`.

    Returns:
        Camera centers with shape :math:`(*, C, 3)`.
    """
    check_shape(Rts, ["*", "C", "3", "4"])
    return -(Rts[..., :3].transpose(-2, -1) @ Rts[..., 3:]).squeeze(-1)


def positive_depth_mask(
    points_3d: torch.Tensor,
    Rts: torch.Tensor,
    min_depth: float = MIN_PROJECTION_DEPTH,
) -> torch.Tensor:
    """Flags the points lying in front of each camera.

    A point behind a camera still projects, to a mirrored image position, so
    the cheirality test is what separates a real observation from one the
    pinhole model merely produces a number for.

    Args:
        points_3d: Points in the world frame with shape (n_points, 3).
        Rts: World-to-camera extrinsics with shape (n_views, 3, 4).
        min_depth: Lower bound on the camera-frame depth. Points below it sit
            on the focal plane, where the projection is singular.

    Returns:
        Bool tensor of shape (n_views, n_points), True where the point is in
        front of that view.
    """
    check_shape(points_3d, ["N", "3"])
    check_shape(Rts, ["C", "3", "4"])
    return transform_points_from_world_to_camera(points_3d, Rts)[..., 2] > min_depth


def project_points_from_camera_to_image(
    points_3d_cam: torch.Tensor,
    K: torch.Tensor,
    D: torch.Tensor,
    distortion_model: Literal["brown_conrady", "opencv_fisheye"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points from the camera frame to the image plane and undistort them if coefficients are provided.
    The 2D points coordinates are in the range [0, W] x [0, H] (expressed in pixels).
        X_img = K @ [X_cam, 1]

    Args:
        points_3d_cam: 3D points to project. Shape (P, 3) or (C, P, 3) or (F, C, P, 3).
        K: Camera intrinsic matrix. Shape (3, 3), (C, 3, 3), (F, C, 3, 3), (1, C, 3, 3) or (1, 1, 3, 3).
        D: Distortion coefficients. Shape (D,), (C, D), (F, C, D), (1, C, D) or (1, 1, D).
        distortion_model: Distortion model (either "brown_conrady" or "opencv_fisheye").

    Returns:
        points_2d: Projected 2D points. Shape (P, 2) or (C, P, 2) or (F, C, P, 2).
        points_depth: Depth of the projected points. Shape (P, 1) or (C, P, 1) or (F, C, P, 1).
    """
    check_shape(points_3d_cam, [("P", 3), ("C", "P", 3), ("F", "C", "P", 3)])
    points_has_camera_dim = points_3d_cam.ndim >= 3
    points_has_frame_dim = points_3d_cam.ndim == 4

    if not points_has_camera_dim:
        points_3d_cam = points_3d_cam.unsqueeze(0)
    if not points_has_frame_dim:
        points_3d_cam = points_3d_cam.unsqueeze(0)

    F, C, P, _ = points_3d_cam.shape

    K_has_camera_dim = K.ndim >= 3
    K_has_frame_dim = K.ndim == 4

    if not K_has_camera_dim:
        K = K.unsqueeze(0)
    if not K_has_frame_dim:
        K = K.unsqueeze(0)

    K = K.to(points_3d_cam.device)
    D = D.to(points_3d_cam.device)

    # Ensure that types are broadcastable
    check_shape(K, [(1, 1, 3, 3), (1, "C", 3, 3), (F, "C", 3, 3)])

    projected_points = torch.einsum("fcij,fcpj->fcpi", K, points_3d_cam)
    projected_points, projected_depth = convert_points_from_homogeneous(
        projected_points, eps=MIN_PROJECTION_DEPTH
    )

    if D is not None:
        projected_points = distort_points(
            projected_points, K=K, D=D, distortion_model=distortion_model
        )

    if not points_has_frame_dim and not K_has_frame_dim:
        # Remove the F dimension
        projected_points = projected_points.squeeze(-4)
        projected_depth = projected_depth.squeeze(-4)
    if not K_has_camera_dim:
        # Remove the C dimension
        projected_points = projected_points.squeeze(-3)
        projected_depth = projected_depth.squeeze(-3)

    return projected_points, projected_depth


def unproject_points_from_image_to_camera(
    points_2d: torch.Tensor,
    points_depth: torch.Tensor,
    K: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    distortion_model: Literal["brown_conrady", "opencv_fisheye"] = "brown_conrady",
) -> torch.Tensor:
    # Expects points_2d to be normalized
    check_shape(points_2d, ("P", 2))
    check_shape(points_depth, ("P", 1))
    check_shape(K, (3, 3))

    if D is not None:
        check_shape(D, ("D",))

    if D is not None and distortion_model == "brown_conrady":
        points_2d = kornia.geometry.undistort_points(
            points_2d,
            K=K,
            dist=D,
        )
    elif D is not None and distortion_model == "opencv_fisheye":
        points_2d = cv2.fisheye.undistortPoints(
            points_2d.cpu().numpy(),
            K=K.cpu().numpy(),
            D=D.cpu().numpy(),
            Knew=K.cpu().numpy(),
        )

    points_3d_cam = kornia.geometry.unproject_points(
        points_2d, points_depth, camera_matrix=K, normalize=True
    )

    return points_3d_cam
