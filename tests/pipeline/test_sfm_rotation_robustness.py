import math
import networkx as nx
import torch

from kineo.geometry.rotation_averaging import geodesic_angle, project_to_so3
from kineo.pipeline.stages.sfm_camera_extrinsics_initialization import (
    _average_rotations,
    _compute_absolute_Rts,
    _compute_relative_scale_factors,
)


def _complete_graph(R_gt: torch.Tensor, corrupt: set) -> nx.DiGraph:
    n = R_gt.shape[0]
    g = nx.DiGraph()
    for v in range(n):
        g.add_node(v, K=torch.eye(3))
        g.add_edge(v, v, cost=0.0, Rt=torch.eye(4)[:3, :])
    for i in range(n):
        for j in range(i + 1, n):
            R_ij = R_gt[j] @ R_gt[i].transpose(-1, -2)
            if frozenset((i, j)) in corrupt:
                R_ij = project_to_so3(torch.randn(3, 3)) @ R_ij
            t = torch.tensor([[1.0], [0.0], [0.0]])
            Rt = torch.cat([R_ij, t], dim=-1)
            Rt_inv = torch.cat(
                [R_ij.transpose(-1, -2), -R_ij.transpose(-1, -2) @ t], dim=-1
            )
            g.add_edge(i, j, cost=0.01, Rt=Rt)
            g.add_edge(j, i, cost=0.01, Rt=Rt_inv)
    return g


def test_average_rotations_adapter_recovers_node_rotations():
    torch.manual_seed(0)
    R_gt = project_to_so3(torch.randn(7, 3, 3))
    # Put node 0's frame as the world gauge: express all rotations relative to it.
    R_gt = R_gt @ R_gt[0].transpose(-1, -2)
    # Two gross-outlier undirected edges; averaging must down-weight (not reject)
    # them via the cycle-consistency seed + Huber IRLS and still recover GT.
    corrupt = {frozenset((0, 2)), frozenset((3, 5))}
    g = _complete_graph(R_gt, corrupt)
    g = _average_rotations(g, n_iters=30, huber_delta_rad=math.radians(5.0))
    R_out = torch.stack([g.nodes[v]["Rt"][:3, :3] for v in range(7)])
    ang = geodesic_angle(R_out, R_gt)
    assert float(ang.max()) < math.radians(3.0)
    assert torch.allclose(g.nodes[0]["Rt"][:3, 3], torch.zeros(3), atol=1e-6)


def test_average_rotations_keeps_noisy_inliers():
    # Regression guard for the legoassemble failure: clean-but-noisy edges (a few
    # degrees of pairwise-rotation noise, NO gross outliers) must all be kept.
    # The old hard triplet-closure reject mistook this noise for outliers and
    # dropped ~half the edges, wrecking the seed (AE 0.12 deg -> 32 deg).
    torch.manual_seed(3)
    n = 8
    R_gt = project_to_so3(torch.randn(n, 3, 3))
    R_gt = R_gt @ R_gt[0].transpose(-1, -2)
    g = nx.DiGraph()
    for v in range(n):
        g.add_node(v, K=torch.eye(3))
        g.add_edge(v, v, cost=0.0, Rt=torch.eye(4)[:3, :])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            noise = project_to_so3(torch.eye(3) + 0.05 * torch.randn(3, 3))
            R_ij = noise @ (R_gt[j] @ R_gt[i].transpose(-1, -2))
            t = torch.tensor([[1.0], [0.0], [0.0]])
            g.add_edge(i, j, cost=0.01, Rt=torch.cat([R_ij, t], dim=-1))
    g = _average_rotations(g, n_iters=30, huber_delta_rad=math.radians(5.0))
    R_out = torch.stack([g.nodes[v]["Rt"][:3, :3] for v in range(n)])
    # Averaging over the full redundant set of noisy edges recovers GT to within
    # the input noise (~4-5 deg). If edges were dropped as "outliers" the seed
    # would collapse and this would blow up to tens of degrees (the regression).
    assert float(geodesic_angle(R_out, R_gt).max()) < math.radians(8.0)


def _perfect_graph(R_gt: torch.Tensor, t_gt: torch.Tensor) -> nx.DiGraph:
    """Complete outlier-free graph consistent with GT poses (world = node 0).

    Edge (i, j) stores R_ij = R_j R_iᵀ and the unit direction of the true
    relative translation t_ij = t_j - R_ij t_i, matching the pipeline's edge
    convention. All edge costs are equal so no edge is a rotation outlier.
    """
    n = R_gt.shape[0]
    g = nx.DiGraph()
    for v in range(n):
        g.add_node(v, K=torch.eye(3))
        g.add_edge(v, v, cost=0.0, Rt=torch.eye(4)[:3, :])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            R_ij = R_gt[j] @ R_gt[i].transpose(-1, -2)
            t_ij = t_gt[j] - R_ij @ t_gt[i]
            t_hat = (t_ij / t_ij.norm()).view(3, 1)
            g.add_edge(i, j, cost=0.01, Rt=torch.cat([R_ij, t_hat], dim=-1))
    return g


def test_robust_init_recovers_poses_on_perfect_graph():
    # End-to-end average -> scale -> absolute on a clean synthetic graph: the
    # only automated coverage for the scale/translation solve.
    torch.manual_seed(0)
    n = 5
    R_gt = project_to_so3(torch.randn(n, 3, 3))
    R_gt = R_gt @ R_gt[0].transpose(-1, -2)  # node 0 = world gauge
    t_gt = torch.randn(n, 3)
    t_gt[0] = torch.zeros(3)

    g = _perfect_graph(R_gt, t_gt)
    g = _average_rotations(g, n_iters=30, huber_delta_rad=math.radians(5.0))
    g = _compute_relative_scale_factors(g)
    g = _compute_absolute_Rts(g)

    R_out = torch.stack([g.nodes[v]["Rt"][:3, :3] for v in range(n)])
    assert float(geodesic_angle(R_out, R_gt).max()) < math.radians(0.5)

    # Translations recovered up to a single global scale (gauge freedom).
    t_out = torch.stack([g.nodes[v]["Rt"][:3, 3] for v in range(n)])
    scale = (t_out * t_gt).sum() / (t_gt * t_gt).sum()
    rel_err = (t_out - scale * t_gt).norm() / (scale * t_gt).norm()
    assert float(rel_err) < 0.05


def test_scale_solve_handles_missing_edges():
    # An edge can be absent when pairwise pose estimation failed; the scale solve
    # must enumerate only present pairs/triplets (else KeyError on a missing pair).
    torch.manual_seed(0)
    n = 5
    R_gt = project_to_so3(torch.randn(n, 3, 3))
    R_gt = R_gt @ R_gt[0].transpose(-1, -2)
    t_gt = torch.randn(n, 3)
    t_gt[0] = torch.zeros(3)

    g = _perfect_graph(R_gt, t_gt)
    g.remove_edge(1, 3)  # simulate a failed pairwise pose estimation
    g.remove_edge(3, 1)
    assert not g.has_edge(1, 3)

    g = _average_rotations(g, n_iters=30, huber_delta_rad=math.radians(5.0))
    g = _compute_relative_scale_factors(g)
    g = _compute_absolute_Rts(g)

    R_out = torch.stack([g.nodes[v]["Rt"][:3, :3] for v in range(n)])
    assert float(geodesic_angle(R_out, R_gt).max()) < math.radians(0.5)
    t_out = torch.stack([g.nodes[v]["Rt"][:3, 3] for v in range(n)])
    scale = (t_out * t_gt).sum() / (t_gt * t_gt).sum()
    rel_err = (t_out - scale * t_gt).norm() / (scale * t_gt).norm()
    assert float(rel_err) < 0.05
