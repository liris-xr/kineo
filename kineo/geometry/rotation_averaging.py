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


def geodesic_angle(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    """Geodesic (rotation) angle between two rotation matrices.

    Args:
        R_a: Rotation matrices of shape (*, 3, 3).
        R_b: Rotation matrices broadcastable to R_a.

    Returns:
        Angle in radians of shape (*,).
    """
    rel = R_a.transpose(-1, -2) @ R_b
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.arccos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))


def project_to_so3(M: torch.Tensor) -> torch.Tensor:
    """Nearest rotation matrix to M (special orthogonal Procrustes).

    Args:
        M: Matrices of shape (*, 3, 3).

    Returns:
        Rotation matrices of shape (*, 3, 3) with det = +1.
    """
    U, _, Vh = torch.linalg.svd(M)
    det = torch.linalg.det(U @ Vh)
    D = torch.eye(3, device=M.device, dtype=M.dtype).expand(
        M.shape[:-2] + (3, 3)
    ).clone()
    D[..., 2, 2] = det
    return U @ D @ Vh


def edge_closure_rates(
    node_pairs: torch.Tensor,
    rel_rotations: torch.Tensor,
    n_views: int,
    thresh_rad: float,
) -> torch.Tensor:
    """Fraction of each edge's triplets whose rotation loop closes.

    For every triplet (i, j, k) with all three directed edges present, the loop
    R_ki @ R_jk @ R_ij is compared to identity; the triplet closes when its
    angle is below thresh_rad. An edge's rate is the fraction of its triplets
    that close. Outlier edges break most of their loops and score low. The rate
    is shared by both directions of an undirected pair.

    Args:
        node_pairs: Directed edges as (E, 2) long tensor of (source, target).
        rel_rotations: (E, 3, 3) relative rotation per edge; edge (i, j) maps
            view i coordinates to view j (R_ij = R_j R_iᵀ).
        n_views: Number of views.
        thresh_rad: Loop-closure tolerance in radians.

    Returns:
        (E,) closure rate in [0, 1]; edges in no complete triplet score 1.
    """
    lookup = {
        (int(a), int(b)): e for e, (a, b) in enumerate(node_pairs.tolist())
    }
    good: dict[frozenset, float] = {}
    total: dict[frozenset, float] = {}

    for i, j, k in itertools.combinations(range(n_views), 3):
        edges = [(i, j), (j, k), (k, i)]
        if not all(e in lookup for e in edges):
            continue
        R_ij, R_jk, R_ki = (rel_rotations[lookup[e]] for e in edges)
        R_loop = R_ki @ R_jk @ R_ij
        angle = geodesic_angle(
            torch.eye(3, device=R_loop.device, dtype=R_loop.dtype), R_loop
        )
        closed = float(angle < thresh_rad)
        for a, b in edges:
            key = frozenset((a, b))
            good[key] = good.get(key, 0.0) + closed
            total[key] = total.get(key, 0.0) + 1.0

    rates = torch.ones(
        len(node_pairs), device=rel_rotations.device, dtype=rel_rotations.dtype
    )
    for e, (a, b) in enumerate(node_pairs.tolist()):
        key = frozenset((int(a), int(b)))
        if key in total:
            rates[e] = good[key] / total[key]
    return rates
