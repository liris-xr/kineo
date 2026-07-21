import math
import torch

from kineo.geometry.rotation_averaging import (
    edge_closure_rates,
    geodesic_angle,
    project_to_so3,
)


def _random_rotations(n: int) -> torch.Tensor:
    torch.manual_seed(0)
    return project_to_so3(torch.randn(n, 3, 3))


def test_geodesic_angle_zero_and_known():
    R = _random_rotations(1)[0]
    assert torch.allclose(geodesic_angle(R, R), torch.zeros(()), atol=1e-5)
    Rz = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )  # +90 deg about z
    assert abs(float(geodesic_angle(torch.eye(3), Rz)) - math.pi / 2) < 1e-5


def test_project_to_so3_is_rotation_and_idempotent():
    R = project_to_so3(torch.randn(4, 3, 3))
    eye = torch.eye(3).expand(4, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(4), atol=1e-5)
    assert torch.allclose(project_to_so3(R), R, atol=1e-5)


def test_edge_closure_rates_flags_the_outlier_edge():
    n = 5
    R_gt = _random_rotations(n)
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    node_pairs = torch.tensor(pairs, dtype=torch.long)
    rel = torch.stack([R_gt[j] @ R_gt[i].transpose(-1, -2) for i, j in pairs])
    # Corrupt one undirected edge {0,1} in both directions with a 40 deg error.
    bad = project_to_so3(torch.randn(3, 3))
    for e, (i, j) in enumerate(pairs):
        if {i, j} == {0, 1}:
            rel[e] = bad @ rel[e]
    rates = edge_closure_rates(node_pairs, rel, n, math.radians(5.0))
    rate_by_edge = {tuple(sorted(p)): float(r) for p, r in zip(pairs, rates)}
    # The outlier edge breaks all its triplets -> lowest rate, below the 0.5
    # rejection threshold. Edges sharing a triplet with it dip (~2/3 at n=5) but
    # stay above 0.5, so the pre-filter drops only the true outlier.
    assert rate_by_edge[(0, 1)] < 0.5
    assert rate_by_edge[(0, 1)] == min(rate_by_edge.values())
    others = [v for k, v in rate_by_edge.items() if k != (0, 1)]
    assert min(others) > 0.5
