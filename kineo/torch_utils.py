# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import torch
from typing import Sequence


def _check_shape(
    shape_to_check: torch.Size,
    shape: Sequence[str | int],
    raises: bool = True,
    error: str | None = None,
) -> bool:
    """Check that the `shape_to_check` matches the `shape` template.

    The shape can be specified with a implicit or explicit list of strings.
    The guard also check whether the variable is a type `Tensor`.

    Args:
        shape_to_check: the shape to check.
        shape: a list with strings with the expected shape.
        raises: bool indicating whether an exception should be raised upon failure.

    Raises:
        Exception: if the input tensor is has not the expected shape and raises is True.

    Example:
        >>> x = torch.rand(2, 3, 4, 4)
        >>> check_shape(x, ["B", "C", "H", "W"])  # implicit
        True

        >>> x = torch.rand(2, 3, 4, 4)
        >>> check_shape(x, ["2", "3", "H", "W"])  # explicit
        True

    Adapted from Kornia: https://github.com/kornia/kornia/blob/604e3aa6de140520041a01e4a06b2aa0789c16e1/kornia/core/check.py#L49
    """

    if error is None:
        error = f"Expected shape to be {tuple(shape)}. Got {tuple(shape_to_check)}"

    # Disable when using torchscript
    if torch.jit.is_scripting():
        return True

    if shape.count("*") > 1:
        raise ValueError("A shape can only contain one wildcard *")

    if len(shape) == 0 and len(shape_to_check) == 0:
        return True

    if "*" in shape:  # Wildcard is in the middle of the shape
        n_leading_fixed_dims = shape.index("*")
        n_trailing_fixed_dims = len(shape) - n_leading_fixed_dims - 1
        leading_dims_shape_to_check = shape_to_check[:n_leading_fixed_dims]
        trailing_dims_shape_to_check = shape_to_check[-n_trailing_fixed_dims:]
        leading_dims_shape = shape[:n_leading_fixed_dims]
        trailing_dims_shape = shape[-n_trailing_fixed_dims:]

        leading_dims_check = _check_shape(
            leading_dims_shape_to_check, leading_dims_shape, raises=False
        )
        trailing_dims_check = _check_shape(
            trailing_dims_shape_to_check, trailing_dims_shape, raises=False
        )

        if raises and (not leading_dims_check or not trailing_dims_check):
            raise ValueError(error)

        return leading_dims_check and trailing_dims_check

    if len(shape_to_check) != len(shape):
        if raises:
            raise ValueError(error)
        return False

    # Check that dimensions with explicit names have the correct size
    # If a dimension with an implicit name is already known, check that it matches the previous one
    known_implicit_dims: dict[str, int] = {}

    for i in range(len(shape_to_check)):
        dim = shape[i]
        is_implicit = isinstance(dim, str)

        if is_implicit:
            if (
                dim in known_implicit_dims
                and known_implicit_dims[dim] != shape_to_check[i]
            ):
                # Shape is not consistent (same implicit dimension is used with different sizes)
                raise ValueError(error)
            known_implicit_dims[dim] = shape_to_check[i]
            continue

        if shape_to_check[i] != dim:
            if raises:
                raise ValueError(error)
            else:
                return False

    return True


def check_shape(
    x: torch.Tensor,
    shape: Sequence[str | int] | Sequence[Sequence[str | int]],
    raises: bool = True,
    error: str | None = None,
) -> bool:
    if error is None:
        error = f"Expected shape to be {tuple(shape)}. Got {tuple(x.shape)}"

    # Check for multiple possible shapes
    if (
        isinstance(shape, Sequence)
        and len(shape) > 0
        and not isinstance(shape[0], (int, str))
    ):
        at_least_one_shape_matches = False
        for s in shape:
            matches = _check_shape(x.shape, s, raises=False)
            at_least_one_shape_matches = at_least_one_shape_matches or matches

        if not at_least_one_shape_matches and raises:
            raise ValueError(error)

        return at_least_one_shape_matches
    else:
        return _check_shape(x.shape, shape, raises, error)
