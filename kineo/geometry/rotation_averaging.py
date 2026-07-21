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


def average_rotations(
    node_pairs: torch.Tensor,
    rel_rotations: torch.Tensor,
    R_seed: torch.Tensor,
    n_l1_iters: int = 20,
    n_irls_iters: int = 20,
    huber_delta_rad: float = 0.0873,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Robustly average relative rotations into absolute orientations.

    Block-coordinate descent: each sweep resets every non-anchor view to the
    robust mean of the predictions from its incident edges (edge (i, j)
    predicts R_i = R_ijᵀ R_j). The L1 (Weiszfeld) phase tolerates a rough
    seed and gross outliers; the Huber IRLS phase refines. View 0 is the
    gauge anchor, held fixed at R_seed[0]. Assumes a well-connected graph
    (redundant edges let good edges outvote a bad seed).

    Args:
        node_pairs: Directed edges as (E, 2) long tensor of (source, target).
        rel_rotations: (E, 3, 3) relative rotations; edge (i, j) maps view i
            to j.
        R_seed: (N, 3, 3) initial absolute rotations; row 0 is the fixed
            anchor.
        n_l1_iters: Weiszfeld sweeps.
        n_irls_iters: Huber IRLS sweeps.
        huber_delta_rad: Huber threshold (radians) for the IRLS phase.
        eps: Floor on residual angle, avoids division by zero in Weiszfeld.

    Returns:
        (N, 3, 3) absolute rotations.
    """
    n_views = R_seed.shape[0]
    R = R_seed.clone()

    incident: list[list[tuple[int, int]]] = [[] for _ in range(n_views)]
    for e, (a, b) in enumerate(node_pairs.tolist()):
        incident[int(a)].append((e, int(b)))

    def sweep(use_huber: bool) -> None:
        for i in range(1, n_views):
            edges = incident[i]
            if not edges:
                continue
            preds = torch.stack(
                [rel_rotations[e].transpose(-1, -2) @ R[b] for e, b in edges]
            )
            angles = geodesic_angle(preds, R[i].expand_as(preds))
            if use_huber:
                w = torch.where(
                    angles <= huber_delta_rad,
                    torch.ones_like(angles),
                    huber_delta_rad / angles.clamp_min(eps),
                )
            else:
                w = 1.0 / angles.clamp_min(eps)
            R[i] = project_to_so3((w.view(-1, 1, 1) * preds).sum(0))

    for _ in range(n_l1_iters):
        sweep(use_huber=False)
    for _ in range(n_irls_iters):
        sweep(use_huber=True)

    return R
