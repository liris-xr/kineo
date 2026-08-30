import networkx as nx
import pytest
import torch

from kineo.pipeline.stages.sfm_camera_extrinsics_initialization_rotation_averaging import (
    _refine_camera_extrinsics,
)


def _graph_with_degenerate_pairs(n_views: int, degenerate: set) -> nx.DiGraph:
    """Build a refinement-ready graph, some pairs lacking correspondences.

    Mirrors what _initialize_graph emits: pairs with fewer than 5
    correspondences become placeholder edges carrying an infinite cost, an
    identity relative pose and no point attributes.
    """
    torch.manual_seed(0)
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

    graph = nx.DiGraph()
    for view_idx in range(n_views):
        Rt = torch.eye(4)[:3, :]
        Rt[:3, 3] = torch.tensor([float(view_idx), 0.2 * view_idx, 0.0])
        graph.add_node(view_idx, view_idx=view_idx, K=K, Rt=Rt)
        graph.add_edge(view_idx, view_idx, cost=0.0, Rt=torch.eye(4)[:3, :])

    for view_i in range(n_views):
        for view_j in range(view_i + 1, n_views):
            rel_Rt = torch.eye(4)[:3, :]
            rel_Rt[:3, 3] = torch.tensor([1.0, 0.0, 0.0])

            if frozenset((view_i, view_j)) in degenerate:
                graph.add_edge(
                    view_i, view_j, cost=torch.tensor(float("inf")), Rt=rel_Rt
                )
                graph.add_edge(
                    view_j, view_i, cost=torch.tensor(float("inf")), Rt=rel_Rt
                )
                continue

            points1 = torch.rand(20, 2) * torch.tensor([640.0, 480.0])
            points2 = points1 + torch.randn(20, 2)
            graph.add_edge(
                view_i, view_j, cost=0.01, Rt=rel_Rt, points1=points1, points2=points2
            )
            graph.add_edge(
                view_j, view_i, cost=0.01, Rt=rel_Rt, points1=points2, points2=points1
            )

    return graph


def test_refinement_skips_pairs_without_correspondences():
    # Regression guard for volleyball_005 at 4 views: cam04 and cam11 shared only
    # 3 correspondences, so their edge held no points and refinement died with
    # KeyError: 'points1' instead of ignoring the pair.
    graph = _graph_with_degenerate_pairs(4, degenerate={frozenset((1, 3))})

    graph = _refine_camera_extrinsics(graph, n_iters=2)

    Rts = torch.stack([graph.nodes[v]["Rt"] for v in range(4)])
    assert torch.isfinite(Rts).all()


def test_refinement_returns_unrefined_when_every_pair_is_degenerate():
    degenerate = {frozenset((i, j)) for i in range(3) for j in range(i + 1, 3)}
    graph = _graph_with_degenerate_pairs(3, degenerate=degenerate)
    Rts_before = torch.stack([graph.nodes[v]["Rt"].clone() for v in range(3)])

    with pytest.warns(UserWarning, match="no usable correspondences"):
        graph = _refine_camera_extrinsics(graph, n_iters=2)

    Rts_after = torch.stack([graph.nodes[v]["Rt"] for v in range(3)])
    assert torch.equal(Rts_before, Rts_after)
