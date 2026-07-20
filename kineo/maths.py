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

def clamp(value: int, min_value: int, max_value: int) -> int:
    """
    Clamp a value between a minimum and maximum value.
    """
    return max(min_value, min(value, max_value))

def median(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the median of values along a specified dimension.

    This implementation is necessary for deterministic results because built-in torch.median doesn't provide support for a deterministic algorithm.
    """
    sorted_vals, indices = torch.sort(values, dim=dim)
    n = values.size(dim)

    if n % 2 == 1:
        idx = n // 2
        median_val = sorted_vals.select(dim, idx)
        median_idx = indices.select(dim, idx)
    else:
        idx1 = n // 2 - 1
        idx2 = n // 2
        median_val = (sorted_vals.select(dim, idx1) + sorted_vals.select(dim, idx2)) / 2
        median_idx = indices.select(dim, idx1)

    return median_val, median_idx


def weighted_mean(
    values: torch.Tensor, weights: torch.Tensor, dim: int | None = None, keepdim: bool = False
) -> torch.Tensor:
    """
    Compute the weighted mean of values with associated weights.
    """
    return (values * weights).sum(dim=dim, keepdim=keepdim) / weights.sum(
        dim=dim, keepdim=keepdim
    ).clamp_min(torch.finfo(weights.dtype).eps)


def weighted_variance(
    values: torch.Tensor, weights: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """
    Compute the weighted variance of values with associated weights.

    \sigma^2_W = \frac{1}{W} \sum_{i=1}^n w_i (x_i - \hat{x}_W)^2
    """
    w_mean = weighted_mean(values, weights, dim=dim)
    return (weights * (values - w_mean) ** 2).sum(dim=dim) / weights.sum(dim=dim)


def weighted_quantile(
    values: torch.Tensor, weights: torch.Tensor, q: float, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the weighted quantile of values with associated weights.
    """
    sorted_vals, indices = torch.sort(values, dim=dim)
    sorted_weights = torch.gather(weights, dim, indices)
    weight_cumsum = torch.cumsum(sorted_weights, dim=dim)
    weight_cumsum = weight_cumsum / weight_cumsum.sum(dim=dim, keepdim=True)

    # Find the index where cumulative weight crosses q
    mask = weight_cumsum >= q
    idx = mask.float().argmax(dim=dim, keepdim=True)

    quantile = torch.gather(sorted_vals, dim, idx).squeeze(dim)
    quantile_idx = torch.gather(indices, dim, idx).squeeze(dim)
    return quantile, quantile_idx


def weighted_median(
    values: torch.Tensor, weights: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the weighted median along a dimension.

    Args:
        values: Tensor of shape (..., N)
        weights: Tensor of same shape (..., N), must be non-negative.
        dim: Dimension along which to compute the weighted median.

    Returns:
        median: Tensor of shape (...,) containing weighted medians.
        indices: Tensor of shape (...,) containing indices of median values (in the original order).
    """
    sorted_vals, indices = torch.sort(values, dim=dim)
    sorted_weights = torch.gather(weights, dim, indices)

    # Compute cumulative weight sum normalized to [0, 1]
    weight_cumsum = torch.cumsum(sorted_weights, dim=dim)
    total_weight = torch.sum(sorted_weights, dim=dim, keepdim=True)
    weight_cumsum = weight_cumsum / total_weight

    # Find the index where cumulative weight crosses 0.5
    mask = weight_cumsum >= 0.5
    idx = mask.float().argmax(dim=dim, keepdim=True)

    median = torch.gather(sorted_vals, dim, idx).squeeze(dim)
    median_idx = torch.gather(indices, dim, idx).squeeze(dim)

    return median, median_idx
