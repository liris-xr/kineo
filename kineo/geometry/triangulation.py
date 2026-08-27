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
import torch
from math import radians
from kineo.geometry.camera import (
    MIN_PROJECTION_DEPTH,
    camera_centers_from_extrinsics,
    positive_depth_mask,
)
from kineo.geometry.conversions import convert_points_from_homogeneous
from kineo.geometry.metrics import compute_normalized_reprojection_residuals
from kineo.torch_utils import check_shape


def triangulation_parallax_angles(
    points_3d: torch.Tensor,
    camera_centers: torch.Tensor,
    points_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Widest angle each point subtends between a pair of cameras.

    Depth uncertainty of a triangulated point grows as :math:`1/\sin\theta`
    with :math:`\theta` the angle between the rays that fixed it, so this is
    the conditioning of the triangulation.

    Args:
        points_3d: Triangulated points with shape :math:`(*, N, 3)`.
        camera_centers: Camera centers in the world frame, shape
            :math:`(*, C, 3)`.
        points_weights: Per-view weights with shape :math:`(*, C, N)`. Views
            weighting a point at zero contributed no ray to it and are left
            out of its angle.

    Returns:
        Angles in radians with shape :math:`(*, N)`. Zero where fewer than two
        cameras contributed.
    """
    check_shape(points_3d, ["*", "N", "3"])
    check_shape(camera_centers, ["*", "C", "3"])
    C = camera_centers.shape[-2]
    N = points_3d.shape[-2]
    if points_weights is not None:
        check_shape(points_weights, ["*", C, N])

    rays = points_3d.unsqueeze(-3) - camera_centers.unsqueeze(-2)
    rays = rays / rays.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # Widest angle == smallest cosine. The unit diagonal never wins that
    # minimum, so self-pairs need no masking.
    cosines = torch.einsum("...cnk,...dnk->...cdn", rays, rays)

    if points_weights is not None:
        contributes = points_weights > 0
        pair_contributes = contributes.unsqueeze(-2) & contributes.unsqueeze(-3)
        cosines = torch.where(pair_contributes, cosines, torch.ones_like(cosines))

    return torch.arccos(cosines.amin(dim=(-3, -2)).clamp(-1.0, 1.0))


def triangulate_points(
    Ps: torch.Tensor,
    points: torch.Tensor,
    points_weights: torch.Tensor | None = None,
    use_eigendecomposition: bool = True,
) -> torch.Tensor:
    r"""Reconstructs a bunch of points by triangulation.

    Triangulates the 3d position of 2d correspondences between several images.
    Reference: Internally it uses DLT method from Hartley/Zisserman 12.2 pag.312

    The input points are assumed to be in homogeneous coordinate system and being inliers
    correspondences. The method does not perform any robust estimation.

    Args:
        Ps: The projection matrices for the cameras with shape :math:`(*, C, 3, 4)`.
        points: The set of points seen from each camera with shape :math:`(*, C, N, 2)`.
        points_weights: The weights of the points with shape :math:`(*, C, N)`.
    Returns:
        The reconstructed 3d points in the world frame with shape :math:`(*, N, 3)`.
    """
    check_shape(Ps, ["*", "C", "3", "4"])
    C = Ps.shape[-3]
    check_shape(points, ["*", C, "N", "2"])
    N = points.shape[-2]

    if points_weights is not None:
        check_shape(points_weights, ["*", C, N])
    else:
        points_weights = torch.ones(
            points.shape[:-1], dtype=points.dtype, device=points.device
        )

    pairs = list(itertools.combinations(range(C), 2))
    n_pairs = len(pairs)
    batch_dims = points.shape[:-3]

    # Pair axis carried as a batch axis; laid out as (*, N, n_pairs, 4, 4)
    # directly, which is what the reshape below wants.
    view_i = torch.tensor([i for i, _ in pairs], device=points.device)
    view_j = torch.tensor([j for _, j in pairs], device=points.device)

    pair_points1 = points[..., view_i, :, :].transpose(-3, -2)
    pair_points2 = points[..., view_j, :, :].transpose(-3, -2)
    pair_P1 = Ps[..., view_i, :, :]
    pair_P2 = Ps[..., view_j, :, :]
    pair_weights = torch.sqrt(
        points_weights[..., view_i, :] * points_weights[..., view_j, :]
    ).transpose(-2, -1).unsqueeze(-1)

    P1_row0 = pair_P1[..., 0, :].unsqueeze(-3)
    P1_row1 = pair_P1[..., 1, :].unsqueeze(-3)
    P1_row2 = pair_P1[..., 2, :].unsqueeze(-3)
    P2_row0 = pair_P2[..., 0, :].unsqueeze(-3)
    P2_row1 = pair_P2[..., 1, :].unsqueeze(-3)
    P2_row2 = pair_P2[..., 2, :].unsqueeze(-3)

    X = torch.stack(
        [
            pair_weights * (pair_points1[..., 0:1] * P1_row2 - P1_row0),
            pair_weights * (pair_points1[..., 1:2] * P1_row2 - P1_row1),
            pair_weights * (pair_points2[..., 0:1] * P2_row2 - P2_row0),
            pair_weights * (pair_points2[..., 1:2] * P2_row2 - P2_row1),
        ],
        dim=-2,
    )

    X = X.reshape(-1, n_pairs * 4, 4)

    if use_eigendecomposition:
        # eigh on the 4x4 normal matrix: invalid (zero-weight) pairs contribute
        # nothing, and it always converges (unlike SVD on the zero-padded X, whose
        # repeated singular values make cusolver diverge for occluded points).
        XtX = X.transpose(-2, -1) @ X
        try:
            _, eigvecs = torch.linalg.eigh(XtX)
        except torch.linalg.LinAlgError:
            # cusolver eigh diverges on rank-deficient XtX (occluded points); CPU
            # LAPACK is robust, and XtX is tiny (4x4) so the transfer is cheap.
            _, eigvecs = torch.linalg.eigh(XtX.cpu())
            eigvecs = eigvecs.to(X.device)
        points3d_h = eigvecs[..., :, 0]
    else:
        _, _, V = torch.svd(X)
        points3d_h = V[..., -1]

    points3d, _ = convert_points_from_homogeneous(points3d_h)
    points3d = points3d.reshape((*batch_dims, N, 3))

    # Points seen by fewer than 2 views are untriangulatable; mark them invalid.
    valid_views = (points_weights > 0).sum(dim=-2)
    points3d = points3d.masked_fill((valid_views < 2).unsqueeze(-1), float("nan"))
    return points3d


def triangulate_points_in_chunks(
    Ps: torch.Tensor,
    points: torch.Tensor,
    points_weights: torch.Tensor | None = None,
    max_chunk_bytes: int = 128 * 1024 * 1024,
    use_eigendecomposition: bool = True,
) -> torch.Tensor:
    r"""Triangulates in slices bounded by memory rather than by point count.

    :func:`triangulate_points` materializes a design matrix holding
    :math:`4\binom{C}{2}` rows of 4 for every point, so its footprint grows
    with the square of the view count. A point budget would therefore mean
    something different on every rig; a byte budget does not, and it keeps a
    wider rig from silently quadrupling its peak.

    Args:
        Ps: Projection matrices with shape :math:`(*, C, 3, 4)`.
        points: Points seen from each camera, shape :math:`(*, C, N, 2)`.
        points_weights: Per-view weights with shape :math:`(*, C, N)`.
        max_chunk_bytes: Ceiling on the design matrix of a single slice. Peak
            usage runs to roughly twice this, since the matrix is stacked
            before it is reshaped. Non-positive means no slicing.
        use_eigendecomposition: Forwarded to :func:`triangulate_points`.

    Returns:
        Reconstructed points in the world frame, shape :math:`(*, N, 3)`.
    """
    n_views = points.shape[-3]
    n_points = points.shape[-2]

    if max_chunk_bytes <= 0 or n_points == 0:
        return triangulate_points(
            Ps, points, points_weights, use_eigendecomposition
        )

    n_pairs = n_views * (n_views - 1) // 2
    n_batch = points.numel() // (n_views * n_points * 2)
    # 4 rows of 4 per pair, for every batch element, for every point.
    bytes_per_point = max(
        16 * n_pairs * n_batch * points.element_size(), 1
    )
    chunk_size = max(1, max_chunk_bytes // bytes_per_point)

    if n_points <= chunk_size:
        return triangulate_points(
            Ps, points, points_weights, use_eigendecomposition
        )

    chunks = []

    for start in range(0, n_points, chunk_size):
        chunk = slice(start, start + chunk_size)
        chunks.append(
            triangulate_points(
                Ps,
                points[..., chunk, :],
                None if points_weights is None else points_weights[..., chunk],
                use_eigendecomposition,
            )
        )

    return torch.cat(chunks, dim=-2)


def triangulation_quality_mask(
    points_3d: torch.Tensor,
    points_2d: torch.Tensor,
    Ks: torch.Tensor,
    Rts: torch.Tensor,
    Ds: torch.Tensor,
    distortion_model: str,
    observations_mask: torch.Tensor,
    min_parallax_deg: float | None = None,
    max_reproj_error_focal_ratio: float | None = None,
    reject_negative_depth: bool = False,
    min_depth: float = MIN_PROJECTION_DEPTH,
) -> torch.Tensor:
    """Narrows an observation mask to the geometrically trustworthy entries.

    Applies the three conditions a triangulated point is normally required to
    meet before it is allowed to constrain a bundle adjustment: it lies in
    front of the cameras that saw it, it reprojects near its own measurements,
    and the rays that fixed it met at a workable angle. Each is optional, and a
    condition left at its default is not evaluated at all.

    Rejections compose in that order, so the parallax angle is measured over
    the views that survived the first two rather than over every view that
    nominally observed the point.

    Args:
        points_3d: Triangulated points in the world frame, shape (n_points, 3).
        points_2d: Measured image-space keypoints, still distorted, with shape
            (n_views, n_points, 2).
        Ks: Camera intrinsics with shape (n_views, 3, 3).
        Rts: World-to-camera extrinsics with shape (n_views, 3, 4).
        Ds: Distortion coefficients with shape (n_views, n_coefficients).
        distortion_model: Distortion model name, e.g. ``"brown_conrady"``.
        observations_mask: Bool tensor of shape (n_views, n_points) flagging
            the entries that are real observations to begin with.
        min_parallax_deg: Reject a point whose widest ray pair subtends less
            than this angle. ``None`` disables the test.
        max_reproj_error_focal_ratio: Reject an observation whose reprojection
            lands further than this from its measurement, as a fraction of the
            view's focal length. Pixels are not comparable across a rig whose
            views differ in focal length, and dividing by it leaves an angular
            error. ``None`` disables the test.
        reject_negative_depth: Whether to reject observations whose point sits
            behind the camera.
        min_depth: Depth bound used by the cheirality test.

    Returns:
        Bool tensor of shape (n_views, n_points), the input mask with the
        rejected entries cleared. Points failing the per-point parallax test
        have all of their observations cleared, so a caller can recover the
        surviving points with ``mask.sum(dim=0) >= 2``.
    """
    check_shape(points_3d, ["N", "3"])
    check_shape(observations_mask, ["C", "N"])

    mask = observations_mask

    if reject_negative_depth:
        mask = mask & positive_depth_mask(points_3d, Rts, min_depth)

    if max_reproj_error_focal_ratio is not None:
        residuals, _ = compute_normalized_reprojection_residuals(
            kps_3d=points_3d,
            kps_2d=points_2d,
            Ks=Ks,
            Rts=Rts,
            Ds=Ds,
            distortion_model=distortion_model,
        )
        # A non-finite residual fails the comparison, which is the intent.
        mask = mask & (residuals <= max_reproj_error_focal_ratio)

    if min_parallax_deg is not None:
        angles = triangulation_parallax_angles(
            points_3d=points_3d,
            camera_centers=camera_centers_from_extrinsics(Rts),
            points_weights=mask.to(points_3d.dtype),
        )
        mask = mask & (angles >= radians(min_parallax_deg)).unsqueeze(0)

    return mask
