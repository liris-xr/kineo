import torch

from kineo.optimization.manifolds import (
    UnitDirection,
    rotation_from_6d,
    tangent_basis,
)


def test_zero_tangent_returns_the_base_direction():
    direction = torch.tensor([[3.0, 0.0, 4.0], [0.0, -1.0, 0.0]])
    unit = UnitDirection(direction)

    expected = direction / direction.norm(dim=-1, keepdim=True)
    assert torch.allclose(unit(), expected, atol=1e-6)


def test_the_direction_stays_on_the_unit_sphere():
    unit = UnitDirection(torch.randn(8, 3))
    with torch.no_grad():
        unit.tangent.copy_(torch.randn(8, 2) * 0.7)

    norms = unit().norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_two_numbers_carry_two_freedoms():
    unit = UnitDirection(torch.tensor([[0.0, 0.0, 1.0]]))

    assert unit.tangent.shape == (1, 2)
    unit().sum().backward()
    assert unit.tangent.grad is not None


def test_the_basis_is_orthonormal_and_tangent():
    direction = torch.randn(16, 3)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    first, second = tangent_basis(direction)

    zero = torch.zeros(16)
    assert torch.allclose((first * direction).sum(-1), zero, atol=1e-6)
    assert torch.allclose((second * direction).sum(-1), zero, atol=1e-6)
    assert torch.allclose((first * second).sum(-1), zero, atol=1e-6)
    assert torch.allclose(first.norm(dim=-1), torch.ones(16), atol=1e-6)


def test_the_basis_survives_axis_aligned_directions():
    # A naive cross product with a fixed axis degenerates when the direction is
    # parallel to it, so each axis is exercised.
    direction = torch.eye(3)
    first, second = tangent_basis(direction)

    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert torch.allclose(first.norm(dim=-1), torch.ones(3), atol=1e-6)


def test_rotation_from_6d_is_a_rotation():
    R = rotation_from_6d(torch.randn(4, 6))

    identity = torch.eye(3).expand(4, 3, 3)
    assert torch.allclose(R @ R.transpose(-1, -2), identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(R), torch.ones(4), atol=1e-5)
