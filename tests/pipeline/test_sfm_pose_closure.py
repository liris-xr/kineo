"""Full-pose loop closure: one rate per edge, covering rotation and scale.

The rotation-only rate cannot see whether an edge's scale is observable, so a
scale-degenerate edge scores near-perfect and lands on the tree translations
chain along. Testing the whole SE(3) loop instead gives a single rate in
[0, 1], scored by the same -log convention.
"""
import itertools
import math

import torch

from kineo.pipeline.stages.sfm_camera_extrinsics_initialization import (
    edge_pose_closure_rates,
)


def _ring(n_views, scales, rotations=None):
    """A consistent rig: cameras on a circle, exact relative poses.

    Returns (node_R, edges) where edges maps (i, j) -> (R_ij, t_hat_ij, scale).
    """
    centres = torch.stack([
        torch.tensor([math.cos(2 * math.pi * v / n_views),
                      math.sin(2 * math.pi * v / n_views), 0.0]) * 5.0
        for v in range(n_views)
    ])
    R = {v: torch.eye(3) for v in range(n_views)}
    if rotations:
        R.update(rotations)

    edges = {}
    for i, j in itertools.permutations(range(n_views), 2):
        # World-to-camera: t = -R C. Relative translation i -> j.
        t = -R[j] @ centres[j] + R[j] @ R[i].T @ (R[i] @ centres[i])
        norm = t.norm().clamp_min(1e-12)
        key = (min(i, j), max(i, j))
        edges[(i, j)] = (R[j] @ R[i].T, t / norm,
                         scales.get(key, float(norm)))
    return R, edges


def test_a_consistent_rig_closes_every_loop():
    node_R, edges = _ring(5, {})

    rates = edge_pose_closure_rates(node_R, edges, n_views=5, rel_thresh=0.10)

    assert all(r == 1.0 for r in rates.values()), rates


def test_a_collapsed_scale_breaks_its_loops():
    """The failure the rotation-only rate is blind to."""
    node_R, edges = _ring(5, {})
    _, degenerate = _ring(5, {(0, 1): 1e-9})

    healthy = edge_pose_closure_rates(node_R, edges, n_views=5, rel_thresh=0.10)
    broken = edge_pose_closure_rates(
        node_R, degenerate, n_views=5, rel_thresh=0.10
    )

    assert healthy[frozenset((0, 1))] == 1.0
    assert broken[frozenset((0, 1))] < 0.5


def test_a_wrong_but_plausible_scale_also_breaks_its_loops():
    """Not just collapse: any wrong scale must fail, however ordinary it looks.

    This is what a rotation-only rate and a collapse-detector both miss once a
    prior stops the scale from going to zero.
    """
    _, edges = _ring(5, {})
    node_R, wrong = _ring(5, {(0, 1): 2.5})

    rates = edge_pose_closure_rates(node_R, wrong, n_views=5, rel_thresh=0.10)

    assert rates[frozenset((0, 1))] < 0.5


def test_rate_is_a_fraction_in_the_unit_interval():
    node_R, edges = _ring(6, {(0, 1): 1e-9, (2, 3): 5.0})

    rates = edge_pose_closure_rates(node_R, edges, n_views=6, rel_thresh=0.10)

    assert all(0.0 <= r <= 1.0 for r in rates.values())


def test_edges_in_no_complete_triplet_score_one():
    """Same convention as the rotation rate: absent evidence is not guilt."""
    node_R, edges = _ring(2, {})

    rates = edge_pose_closure_rates(node_R, edges, n_views=2, rel_thresh=0.10)

    assert rates[frozenset((0, 1))] == 1.0
