import math
import networkx as nx
import torch

from kineo.geometry.rotation_averaging import geodesic_angle, project_to_so3
from kineo.pipeline.stages.sfm_camera_extrinsics_initialization import (
    _average_rotations,
    _reject_rotation_outlier_edges,
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


def test_reject_drops_outlier_edges_and_keeps_connected():
    torch.manual_seed(0)
    R_gt = project_to_so3(torch.randn(6, 3, 3))
    corrupt = {frozenset((0, 1)), frozenset((2, 4))}
    g = _complete_graph(R_gt, corrupt)
    g = _reject_rotation_outlier_edges(g, thresh_deg=5.0, min_close_rate=0.5)
    assert not g.has_edge(0, 1)
    assert not g.has_edge(2, 4)
    assert g.has_edge(1, 2)  # a good edge survives
    assert nx.is_connected(g.to_undirected())


def test_reject_never_disconnects_even_if_all_edges_look_bad():
    torch.manual_seed(0)
    R_gt = project_to_so3(torch.randn(4, 3, 3))
    # Corrupt every edge -> connectivity guard must keep a spanning tree.
    corrupt = {frozenset((i, j)) for i in range(4) for j in range(i + 1, 4)}
    g = _complete_graph(R_gt, corrupt)
    g = _reject_rotation_outlier_edges(g, thresh_deg=5.0, min_close_rate=0.5)
    assert nx.is_connected(g.to_undirected())


def test_average_rotations_adapter_recovers_node_rotations():
    torch.manual_seed(0)
    R_gt = project_to_so3(torch.randn(7, 3, 3))
    # Put node 0's frame as the world gauge: express all rotations relative to it.
    R_gt = R_gt @ R_gt[0].transpose(-1, -2)
    corrupt = {frozenset((0, 2)), frozenset((3, 5))}
    g = _complete_graph(R_gt, corrupt)
    g = _reject_rotation_outlier_edges(g, thresh_deg=5.0, min_close_rate=0.5)
    g = _average_rotations(g, n_iters=30, huber_delta_rad=math.radians(5.0))
    R_out = torch.stack([g.nodes[v]["Rt"][:3, :3] for v in range(7)])
    ang = geodesic_angle(R_out, R_gt)
    assert float(ang.max()) < math.radians(3.0)
    assert torch.allclose(g.nodes[0]["Rt"][:3, 3], torch.zeros(3), atol=1e-6)
