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

CANONICAL_BASIS = torch.tensor(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=torch.float32,
)

# Left-handed +Y-up basis is +Y-up, +X-right, +Z-forward (into the screen)
#
# Given the canonical basis of R^3:
# +Z
#  ^
#  |  7 +Y
#  | /
#  |/
#  +--------> +X
#
# LH +Y-up
#
# +Y_w
#  ^
#  |  7 +Z_w
#  | /
#  |/
#  +--------> +X_w
LH_POS_Y_UP_BASIS = torch.tensor(
    [
        # x_w, y_w, z_w
        [1, 0, 0],  # x
        [0, 0, 1],  # y
        [0, 1, 0],  # z
    ],
    dtype=torch.float32,
)

# Left-handed +Z-up basis is +Z-up, +X-right, -Y-forward (into the screen)
#
# Given the canonical basis of R^3:
# +Z
#  ^
#  |  7 +Y
#  | /
#  |/
#  +--------> +X
#
# LH +Z-up
#
#    +Z_w
#     ^
#     |
#     |
#     |
#     +--------> +X_w
#    /
#   /
#  v
#  +Y_w
LH_POS_Z_UP_BASIS = torch.tensor(
    [
        # x_w, y_w, z_w
        [1, 0, 0],  # x
        [0, -1, 0],  # y
        [0, 0, 1],  # z
    ],
    dtype=torch.float32,
)

# Left-handed -Y-up basis is -Y-up, +X-right, -Z-forward (into the screen)
#
# Given the canonical basis of R^3:
# +Z
#  ^
#  |  7 +Y
#  | /
#  |/
#  +--------> +X
#
# LH -Y-up
#
#        +--------> +X_w
#       /|
#      / |
#     v  |
# +Z_w   V
#        +Y_w
LH_NEG_Y_UP_BASIS = torch.tensor(
    [
        # x_w, y_w, z_w
        [1, 0, 0],  # x
        [0, 0, -1],  # y
        [0, -1, 0],  # z
    ],
    dtype=torch.float32,
)

# Left-handed -Z-up basis is -Z-up, +X-right, +Y-forward (into the screen)
#
# Given the canonical basis of R^3:
# +Z
#  ^
#  |  7 +Y
#  | /
#  |/
#  +--------> +X
#
# LH -Z-up
#
#    7 +Y_w
#   /
#  /
# +--------> +X_w
# |
# |
# |
# V
# +Z_w
LH_NEG_Z_UP_BASIS = torch.tensor(
    [
        # x_w, y_w, z_w
        [1, 0, 0],  # x
        [0, 1, 0],  # y
        [0, 0, -1],  # z
    ],
    dtype=torch.float32,
)

# Right-handed Y-up basis is +Y-up, +X-right, -Z-forward (into the screen)
#
# Given the canonical basis of R^3:
# +Z
#  ^
#  |  7 +Y
#  | /
#  |/
#  +--------> +X
#
# RH +Y-up
#
#    +Y_w
#     ^
#     |
#     |
#     |
#     +--------> +X_w
#    /
#   /
#  v
#  +Z_w
RH_POS_Y_UP_BASIS = torch.tensor(
    [
        # x_w, y_w, z_w
        [1, 0, 0],  # x
        [0, 0, -1],  # y
        [0, 1, 0],  # z
    ],
    dtype=torch.float32,
)

# Right-handed -Y-up basis is -Y-up, +X-right, +Z-forward (into the screen)
#
# Given the canonical basis of R^3:
# +Z
#  ^
#  |  7 +Y
#  | /
#  |/
#  +--------> +X
#
# RH -Y-up
#
#     7 +Z_w
#    /
#   /
#  +--------> +X_w
#  |
#  |
#  |
#  V +Y_w
RH_NEG_Y_UP_BASIS = torch.tensor(
    [
        # x_w, y_w, z_w
        [1, 0, 0],  # x
        [0, 0, 1],  # y
        [0, -1, 0],  # z
    ],
    dtype=torch.float32,
)

# Right-handed +Z-up basis is +Z-up, +X-right, +Y-forward (into the screen)
#
# Given the canonical basis of R^3:
# +Z
#  ^
#  |  7 +Y
#  | /
#  |/
#  +--------> +X
#
# RH +Z-up
#
#  +Z_w
#  ^
#  |  7 +Y_w
#  | /
#  |/
#  +--------> +X_w
RH_POS_Z_UP_BASIS = torch.tensor(
    [
        # x_w, y_w, z_w
        [1, 0, 0],  # x
        [0, 1, 0],  # y
        [0, 0, 1],  # z
    ],
    dtype=torch.float32,
)

# Right-handed -Z-up basis is -Z-up, +X-right, -Y-forward (into the screen)
#
# Given the canonical basis of R^3:
# +Z
#  ^
#  |  7 +Y
#  | /
#  |/
#  +--------> +X
#
# RH -Z-up
#
#        +--------> +X_w
#       /|
#      / |
#     v  |
# +Y_w   V
#        +Z_w
RH_NEG_Z_UP_BASIS = torch.tensor(
    [
        # x_w, y_w, z_w
        [1, 0, 0],  # x
        [0, -1, 0],  # y
        [0, 0, -1],  # z
    ],
    dtype=torch.float32,
)

# OpenGL camera basis is right-handed +Y-up, +X-right, -Z-forward (into the screen)
OPENGL_CAMERA_BASIS = RH_POS_Y_UP_BASIS.clone()

# OpenGL world basis is right-handed +Y-up, +X-right, -Z-forward (into the screen)
OPENGL_WORLD_BASIS = RH_POS_Y_UP_BASIS.clone()

# OpenCV camera basis is right-handed -Y-up, +X-right, +Z-forward (into the screen)
OPENCV_CAMERA_BASIS = RH_NEG_Y_UP_BASIS.clone()

# OpenCV world basis is right-handed +Z-up, +X-right, +Y-forward (into the screen)
OPENCV_WORLD_BASIS = RH_POS_Z_UP_BASIS.clone()

OPENCV_WORLD_UP_VECTOR = torch.tensor([0, 0, 1], dtype=torch.float32)
OPENCV_CAMERA_UP_VECTOR = torch.tensor([0, -1, 0], dtype=torch.float32)


def resample_keypoints(
    kps: torch.Tensor,
    kps_scores: torch.Tensor,
    timestamps: torch.Tensor,
    target_timestamps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resample keypoints and their confidence scores to new timestamps using linear interpolation.

    This function takes keypoints and their scores at given timestamps and interpolates them to
    new target timestamps. The interpolation is done using linear interpolation (lerp) between
    the nearest available frames.

    Args:
        kps: Keypoints to resample. Shape (N, K, 2) where N is number of frames,
            K is number of keypoints, and 2 represents (x,y) coordinates.
        kps_scores: Confidence scores for each keypoint. Shape (N, K).
        timestamps: Original timestamps for the keypoints. Shape (N,).
        target_timestamps: Target timestamps to resample to. Shape (M,).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Resampled keypoints. Shape (M, K, 2).
            - Resampled keypoint scores. Shape (M, K).
    """
    start_idx = torch.searchsorted(timestamps, target_timestamps, right=True) - 1
    start_idx = torch.clamp(start_idx, 0, len(timestamps) - 2)
    end_idx = start_idx + 1
    end_idx = torch.clamp(end_idx, 0, len(timestamps) - 1)

    start_times = timestamps[start_idx]
    end_times = timestamps[end_idx]
    weights = (target_timestamps - start_times) / (
        end_times - start_times + torch.finfo(start_times.dtype).eps
    )

    start_kps = kps[start_idx]
    end_kps = kps[end_idx]
    start_kps_scores = kps_scores[start_idx]
    end_kps_scores = kps_scores[end_idx]

    resampled_kps = torch.lerp(start_kps, end_kps, weights.unsqueeze(-1).unsqueeze(-1))
    resampled_kps_scores = torch.lerp(
        start_kps_scores, end_kps_scores, weights.unsqueeze(-1)
    )

    return resampled_kps, resampled_kps_scores


def create_common_uniform_time_grid(
    frame_timestamps: list[torch.Tensor],
    frame_rate: float,
) -> torch.Tensor:
    """
    Get a uniform common time grid for multiple cameras at the given frame rate.

    Args:
        frame_timestamps: List of timestamps for each camera's frames. Each element is a tensor of shape (N_i,).
            These timestamps must be in the global time reference (i.e., the data is already synchronized).
        frame_rate: Target frame rate (in Hz) for the resampled data.

    Returns:
        torch.Tensor: Resampled timestamps of shape (M,) where M is the number of frames in the common time range.
    """
    time_range_min = max([cam_timestamps[0] for cam_timestamps in frame_timestamps])
    time_range_max = min([cam_timestamps[-1] for cam_timestamps in frame_timestamps])

    timestamps = torch.arange(
        time_range_min,
        time_range_max,
        1 / frame_rate,
        device=frame_timestamps[0].device,
        dtype=frame_timestamps[0].dtype,
    )
    return timestamps


def resample_multicamera_keypoints(
    kps_2d: list[torch.Tensor],
    kps_scores: list[torch.Tensor],
    frame_timestamps: list[torch.Tensor],
    frame_rate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Resample 2D keypoints and their scores from multiple cameras to a uniform time grid.

    This function assumes the input data is already synchronized (frame_timestamps are given in the global time reference)
    and performs the following steps:
    1. Find the common time range where all cameras have data
    2. Create a uniform time grid at the specified frame rate within this common range
    3. Linearly interpolate keypoints and scores for each camera to these new timestamps

    Args:
        kps_2d: List of 2D keypoints for each camera. Each element is a tensor of shape (N_i, K, 2)
            where N_i is the number of frames for camera i, K is the number of keypoints,
            and 2 represents (x,y) coordinates.
        kps_scores: List of confidence scores for each keypoint. Each element is a tensor of shape (N_i, K).
        frame_timestamps: List of timestamps for each camera's frames. Each element is a tensor of shape (N_i,).
            These timestamps must be in the global time reference (i.e., the data is already synchronized).
        frame_rate: Target frame rate (in Hz) for the resampled data.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Resampled timestamps of shape (M,) where M is the number of frames in the common time range
            - Resampled keypoints of shape (M, n_cameras, K, 2)
            - Resampled keypoint scores of shape (M, n_cameras, K)
    """
    assert len(kps_2d) == len(kps_scores) == len(frame_timestamps)
    n_cameras = len(kps_2d)

    resampled_timestamps = create_common_uniform_time_grid(frame_timestamps, frame_rate)

    n_frames = len(resampled_timestamps)
    n_keypoints = kps_2d[0].shape[1]

    resampled_kps = torch.empty(
        (n_frames, n_cameras, n_keypoints, 2),
        device=frame_timestamps[0].device,
        dtype=frame_timestamps[0].dtype,
    )
    resampled_kps_scores = torch.empty(
        (n_frames, n_cameras, n_keypoints),
        device=frame_timestamps[0].device,
        dtype=frame_timestamps[0].dtype,
    )

    for cam_idx in range(n_cameras):
        resampled_kps[:, cam_idx], resampled_kps_scores[:, cam_idx] = (
            resample_keypoints(
                kps_2d[cam_idx],
                kps_scores[cam_idx],
                frame_timestamps[cam_idx],
                resampled_timestamps,
            )
        )

    return resampled_timestamps, resampled_kps, resampled_kps_scores


def convert_points_basis(
    points: torch.Tensor, src_basis: torch.Tensor, dst_basis: torch.Tensor
) -> torch.Tensor:
    """
    Convert points from src_basis to dst_basis.

    Args:
        points: Points to convert. Shape (*, 3).

    Returns:
        torch.Tensor: Points in dst_basis. Shape (*, 3).
    """
    return change_vector3d_basis(points, src_basis, dst_basis)


def convert_R_basis(
    R: torch.Tensor, src_basis: torch.Tensor, dst_basis: torch.Tensor
) -> torch.Tensor:
    """
    Convert a rotation matrix from src_basis to dst_basis.
    """
    batch_dims = R.shape[:-2]
    P_h = torch.eye(4, device=R.device).repeat(*batch_dims, 1, 1)
    P_h[..., :3, :3] = compute_transition_matrix(dst_basis, src_basis)

    R_h = torch.eye(4, device=R.device).repeat(*batch_dims, 1, 1)
    R_h[..., :3, :3] = R

    # World to camera (dst_basis)
    R_h_gl = torch.einsum("...ij,...jk->...ik", R_h, P_h)
    R_gl = R_h_gl[..., :3, :3]
    return R_gl


def convert_Rt_basis(
    Rt: torch.Tensor, src_basis: torch.Tensor, dst_basis: torch.Tensor
) -> torch.Tensor:
    """
    Convert a camera extrinsic matrix from src_basis to dst_basis.

    Args:
        Rt: Camera extrinsic matrix. Shape (*, 3, 4).

    Returns:
        torch.Tensor: Camera extrinsic matrix in dst_basis. Shape (*, 3, 4).
    """
    batch_dims = Rt.shape[:-2]
    P_h = torch.eye(4, device=Rt.device).repeat(*batch_dims, 1, 1)
    P_h[..., :3, :3] = compute_transition_matrix(dst_basis, src_basis)

    # World to camera (src_basis)
    Rt_h = torch.eye(4, device=Rt.device).repeat(*batch_dims, 1, 1)
    Rt_h[..., :3, :] = Rt

    # World to camera (dst_basis)
    Rt_h_gl = torch.einsum("...ij,...jk->...ik", Rt_h, P_h)
    Rt_gl = Rt_h_gl[..., :3, :]
    return Rt_gl


def convert_world_points_from_opencv_to_opengl(points: torch.Tensor) -> torch.Tensor:
    """
    Convert points from OpenCV to OpenGL coordinate system.
    """
    return convert_points_basis(points, OPENCV_WORLD_BASIS, OPENGL_WORLD_BASIS)


def convert_R_from_opencv_to_opengl(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix from OpenCV to OpenGL coordinate system.
    """
    return convert_R_basis(R, OPENCV_WORLD_BASIS, OPENGL_WORLD_BASIS)


def convert_Rt_from_opencv_to_opengl(Rt: torch.Tensor) -> torch.Tensor:
    """
    Convert a camera extrinsic matrix from OpenCV to OpenGL coordinate system.

    Args:
        Rt: Camera extrinsic matrix. Shape (*, 3, 4).

    Returns:
        torch.Tensor: Camera extrinsic matrix in OpenGL coordinate system. Shape (*, 3, 4).
    """
    return convert_Rt_basis(Rt, OPENCV_WORLD_BASIS, OPENGL_WORLD_BASIS)


def compute_transition_matrix(
    old_basis: torch.Tensor,
    new_basis: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the transition matrix P to convert from old_basis to new_basis.

    Args:
        old_basis: Vectors of the old basis. Shape (*, D, D). Vectors are columns.
            This also corresponds to the change of basis matrix from the old basis to the canonical basis of R^D.
        new_basis: Vectors of the new basis. Shape (*, D, D). Vectors are columns.
            This also corresponds to the change of basis matrix from the new basis to the canonical basis of R^D.

    Returns:
        P: Transition matrix. Shape (*, D, D).
    """
    P = torch.linalg.inv(new_basis) @ old_basis
    return P


def change_endomorphism_basis(
    M: torch.Tensor, src_basis: torch.Tensor, dst_basis: torch.Tensor
) -> torch.Tensor:
    """
    Convert the square matrix M of an endomorphism from src_basis to dst_basis.
        M_dst = P @ M_src @ P^-1 where P is the transition matrix from src_basis to dst_basis.
        or written differently:
        M_dst = P_src2dst @ M_src @ P_dst2src
        Note that points using this new matrix will need to be expressed in the destination basis.

    Args:
        M: Square matrix of an endomorphism. Shape (*, D, D).
        src_basis: Basis vectors of the source basis. Shape (*, D, D). Vectors are columns.
        dst_basis: Basis vectors of the destination basis. Shape (*, D, D). Vectors are columns.

    Returns:
        M_transformed: Endomorphism in the destination basis. Shape (*, D, D).
    """
    P = compute_transition_matrix(src_basis, dst_basis)
    P_h = torch.eye(4, device=P.device)
    P_h[:3, :3] = P
    return P_h @ M @ torch.linalg.inv(P_h)


def change_Rt_basis(
    Rt: torch.Tensor, src_basis: torch.Tensor, dst_basis: torch.Tensor
) -> torch.Tensor:
    """
    Change the basis of a rigid transformation.

    Args:
        Rt: Rigid transformation expressed in src_basis. Shape (*, 3, 4).
        src_basis: Basis vectors of the source basis. Shape (*, 3, 3). Vectors are columns.
        dst_basis: Basis vectors of the destination basis. Shape (*, 3, 3). Vectors are columns.

    Returns:
        Rt_transformed: Rigid transformation in the destination basis. Shape (*, 3, 4).
    """
    # Make Rt square
    batch_dims = Rt.shape[:-2]
    Rt = torch.cat([Rt, torch.zeros(batch_dims + (1, 4), device=Rt.device)], dim=-2)
    Rt[..., 3, 3] = 1.0

    # Change basis
    Rt_dst = change_endomorphism_basis(
        Rt, src_basis.to(Rt.device), dst_basis.to(Rt.device)
    )
    return Rt_dst[..., :3, :]


def change_vector3d_basis(
    vector: torch.Tensor, src_basis: torch.Tensor, dst_basis: torch.Tensor
) -> torch.Tensor:
    """
    Change the basis of a vector from src_basis to dst_basis.

    Args:
        vector: Vector to change. Shape (*, 3).
        src_basis: Basis vectors of the source frame. Shape (*, 3, 3). Vectors are columns.
        dst_basis: Basis vectors of the destination frame. Shape (*, 3, 3). Vectors are columns.

    Returns:
        vector_transformed: Vector in the destination frame. Shape (*, 3).
    """
    assert vector.shape[-1] == 3
    P = compute_transition_matrix(src_basis, dst_basis).to(vector.device)
    vector_dst = torch.einsum("...ij,...j->...i", P, vector)
    return vector_dst


def convert_points_to_homogeneous(pts: torch.Tensor) -> torch.Tensor:
    """
    Convert points to homogeneous coordinates.

    Args:
        points (torch.Tensor): points to convert. Shape (*, D).

    Returns:
        torch.Tensor: points in homogeneous coordinates. Shape (*, D + 1).
    """
    batch_dims = pts.shape[:-1]
    return torch.cat([pts, torch.ones(batch_dims + (1,), device=pts.device)], dim=-1)


def convert_points_from_homogeneous(
    pts: torch.Tensor,
    eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Drop the homogeneous dimension of points.

    Args:
        points (torch.Tensor): points to convert. Shape (*, D + 1).
        eps (float): lower bound on the magnitude of the divisor. Dividing by a
            near-zero coordinate gives +-inf, whose NaN gradient survives a
            downstream finiteness mask (0 * NaN = NaN). Defaults to no bound, so
            callers that detect degeneracy by testing finiteness keep doing so.

    Returns:
        torch.Tensor: points in Euclidean coordinates. Shape (*, D).
        torch.Tensor: value of the dropped dimension, never bounded. Shape (*, 1).
    """
    divisor = pts[..., -1:]
    bounded_divisor = torch.where(
        divisor.abs() < eps, torch.full_like(divisor, eps), divisor
    )
    return pts[..., :-1] / bounded_divisor, divisor


def normalize_points_with_intrinsics(
    points: torch.Tensor, K: torch.Tensor
) -> torch.Tensor:
    """
    Normalize points with intrinsics.
    """
    cxcy = K[..., :2, 2]
    fxfy = K[..., :2, :2].diagonal(dim1=-2, dim2=-1)
    if len(cxcy.shape) < len(points.shape):
        cxcy, fxfy = cxcy.unsqueeze(-2), fxfy.unsqueeze(-2)
    xy = (points - cxcy) / fxfy
    return xy.reshape_as(points)


def unnormalize_points_with_intrinsics(
    points: torch.Tensor, K: torch.Tensor
) -> torch.Tensor:
    """
    Unnormalize points with intrinsics.
    """
    x_coord = points[..., 0]
    y_coord = points[..., 1]

    # unpack intrinsics
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    if len(cx.shape) < len(x_coord.shape):
        cx, cy, fx, fy = (
            cx.unsqueeze(-1),
            cy.unsqueeze(-1),
            fx.unsqueeze(-1),
            fy.unsqueeze(-1),
        )

    u_coord = x_coord * fx + cx
    v_coord = y_coord * fy + cy

    return torch.stack([u_coord, v_coord], dim=-1)


def normalize_points_with_resolution(
    points: torch.Tensor, size_hw: tuple[int, int]
) -> torch.Tensor:
    """
    Normalize points to the range [-1, 1]
    """
    height, width = size_hw

    assert (
        points.shape[-1] == 2 and points.ndim == 2
    ), f"Expected points to be of shape (P, 2), got {points.shape}"
    points = points.clone()
    points[..., 0] = (points[..., 0] - width / 2) / (width / 2)
    points[..., 1] = (points[..., 1] - height / 2) / (height / 2)
    return points


def unnormalize_points_with_resolution(
    points: torch.Tensor, size_hw: tuple[int, int]
) -> torch.Tensor:
    """
    Unnormalize points from the range [-1, 1]
    """
    height, width = size_hw

    assert (
        points.shape[-1] == 2 and points.ndim == 2
    ), f"Expected points to be of shape (P, 2), got {points.shape}"
    points = points.clone()
    points[..., 0] = points[..., 0] * (width / 2) + width / 2
    points[..., 1] = points[..., 1] * (height / 2) + height / 2
    return points
