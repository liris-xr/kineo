# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Minimal parameterizations for quantities that live on a manifold.

An optimizer that carries more parameters than the quantity has degrees of
freedom leaves flat directions in the objective. The gradient along them is
zero, so the search never explores them, but they still enter the quasi-Newton
curvature pairs, where ``s^T y -> 0`` degrades the inverse-Hessian estimate.
These wrappers store exactly as many numbers as the quantity has freedoms.
"""

from __future__ import annotations

import torch
from roma import special_gramschmidt

from kineo.torch_utils import check_shape


def rotation_from_6d(rot6d: torch.Tensor) -> torch.Tensor:
    """Rotation matrices from the 6D representation.

    Args:
        rot6d: Shape (B, 6), two unconstrained 3-vectors per rotation.

    Returns:
        Rotation matrices with shape (B, 3, 3).
    """
    check_shape(rot6d, ["B", "6"])
    return special_gramschmidt(rot6d.view(-1, 3, 2))


class Rotation6D(torch.nn.Module):
    """A rotation carried as the 6D representation.

    Six numbers for three freedoms, mapped to SO(3) by Gram-Schmidt. The
    redundancy is deliberate: minimal parameterizations of rotation either wrap
    around or gimbal-lock, and the extra numbers cost nothing here because the
    map is global rather than anchored at a base point.
    """

    def __init__(self, batch_size: int, device=None, requires_grad: bool = True):
        """
        Args:
            batch_size: Number of rotations carried.
            device: Device the parameter lives on.
            requires_grad: Whether the rotation is optimized.
        """
        super().__init__()
        self._batch_size = batch_size
        self._rot6d = torch.nn.Parameter(
            torch.eye(3, device=device, dtype=torch.float32)
            .repeat(batch_size, 1, 1)[..., :3, :2]
            .reshape((batch_size, 6)),
            requires_grad=requires_grad,
        )

    @property
    def rot6d(self) -> torch.Tensor:
        """The six stored numbers, shape (B, 6)."""
        return self._rot6d

    @rot6d.setter
    def rot6d(self, rot6d: torch.Tensor) -> None:
        check_shape(rot6d, [(self._batch_size, 6), (1, 6)])
        self._rot6d.data = rot6d.expand(self._batch_size, -1).to(
            self._rot6d.device
        )

    def forward(self) -> torch.Tensor:
        """The rotation matrices, shape (B, 3, 3)."""
        return rotation_from_6d(self._rot6d)


def tangent_basis(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """An orthonormal basis of the plane tangent to the sphere at `direction`.

    The first basis vector comes from crossing `direction` with whichever axis
    it is least aligned with, so the cross product is never degenerate.

    Args:
        direction: Unit vectors with shape (B, 3).

    Returns:
        Two tensors of shape (B, 3), orthonormal and both orthogonal to
        `direction`.
    """
    check_shape(direction, ["B", "3"])
    axis = torch.zeros_like(direction)
    axis.scatter_(1, direction.abs().argmin(dim=1, keepdim=True), 1.0)
    first = torch.cross(direction, axis, dim=-1)
    first = first / first.norm(dim=-1, keepdim=True)
    second = torch.cross(direction, first, dim=-1)
    return first, second


class UnitDirection(torch.nn.Module):
    """A unit 3-vector carried as the two freedoms it actually has.

    The direction is the exponential map of a tangent vector at a fixed base
    point, so the two stored numbers are coordinates in the tangent plane. At
    the origin the map returns the base point exactly, and it stays a bijection
    out to a right angle from it, which is far beyond what a refinement moves.
    """

    def __init__(self, direction: torch.Tensor, requires_grad: bool = True):
        """
        Args:
            direction: Initial directions with shape (B, 3). Need not be unit
                length; only the direction is kept.
            requires_grad: Whether the tangent coordinates are optimized.
        """
        super().__init__()
        check_shape(direction, ["B", "3"])

        base = direction / direction.norm(dim=-1, keepdim=True).clamp_min(
            torch.finfo(direction.dtype).eps
        )
        first, second = tangent_basis(base)
        self.register_buffer("_base", base)
        self.register_buffer("_first", first)
        self.register_buffer("_second", second)

        self._tangent = torch.nn.Parameter(
            torch.zeros(
                (direction.shape[0], 2),
                dtype=direction.dtype,
                device=direction.device,
            ),
            requires_grad=requires_grad,
        )

    @property
    def tangent(self) -> torch.Tensor:
        """The two tangent-plane coordinates, shape (B, 2)."""
        return self._tangent

    def forward(self) -> torch.Tensor:
        """The unit direction, shape (B, 3), differentiable in the tangent."""
        offset = (
            self._tangent[:, 0:1] * self._first
            + self._tangent[:, 1:2] * self._second
        )
        angle = offset.norm(dim=-1, keepdim=True)
        # sin(a)/a -> 1 as a -> 0; torch.sinc(x) is sin(pi x)/(pi x), which is
        # exact at zero and keeps the gradient finite there.
        return (
            torch.cos(angle) * self._base
            + torch.sinc(angle / torch.pi) * offset
        )
