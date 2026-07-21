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
from tqdm import tqdm
from kineo.geometry.conversions import convert_points_from_homogeneous
from kineo.torch_utils import check_shape


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
    X = torch.zeros(
        (*batch_dims, N, n_pairs, 4, 4), dtype=points.dtype, device=points.device
    )

    for pair_idx, (view_i, view_j) in tqdm(enumerate(pairs), total=n_pairs, desc="Building linear system", leave=False):
        pair_points1 = points[..., view_i, :, :]
        pair_points2 = points[..., view_j, :, :]
        pair_P1 = Ps[..., view_i, :, :]
        pair_P2 = Ps[..., view_j, :, :]
        pair_weights = torch.sqrt(
            points_weights[..., view_i, :] * points_weights[..., view_j, :]
        )

        for i in range(4):
            X[..., pair_idx, 0, i] = pair_weights * (
                pair_points1[..., 0] * pair_P1[..., 2:3, i] - pair_P1[..., 0:1, i]
            )
            X[..., pair_idx, 1, i] = pair_weights * (
                pair_points1[..., 1] * pair_P1[..., 2:3, i] - pair_P1[..., 1:2, i]
            )
            X[..., pair_idx, 2, i] = pair_weights * (
                pair_points2[..., 0] * pair_P2[..., 2:3, i] - pair_P2[..., 0:1, i]
            )
            X[..., pair_idx, 3, i] = pair_weights * (
                pair_points2[..., 1] * pair_P2[..., 2:3, i] - pair_P2[..., 1:2, i]
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
