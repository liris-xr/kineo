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


def farthest_point_sampling(
    features: torch.Tensor,
    n_samples: int,
    generator: torch.Generator,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run greedy farthest-point sampling, batched over rows."""
    n_rows, n_points, _ = features.shape
    n_samples = min(max(n_samples, 0), n_points)
    device = features.device

    if valid_mask is None:
        valid_mask = torch.ones(n_rows, n_points, dtype=torch.bool, device=device)

    selected = torch.zeros(n_rows, n_samples, dtype=torch.long, device=device)
    selected_valid = torch.arange(n_samples, device=device).unsqueeze(
        0
    ) < valid_mask.sum(dim=1, keepdim=True)

    if n_samples == 0:
        return selected, selected_valid

    for row in range(n_rows):
        row_points = torch.where(valid_mask[row])[0]

        if len(row_points) == 0:
            continue

        selected[row, 0] = row_points[
            torch.randint(len(row_points), (1,), generator=generator, device=device)
        ]

    # Negative so argmax skips invalid and selected points
    min_sq_dists = torch.where(
        valid_mask,
        torch.full_like(features[..., 0], float("inf")),
        torch.full_like(features[..., 0], -1.0),
    )
    rows = torch.arange(n_rows, device=device)

    for i in range(1, n_samples):
        last = features[rows, selected[:, i - 1]]
        sq_dists = ((features - last.unsqueeze(1)) ** 2).sum(dim=-1)
        min_sq_dists = torch.minimum(min_sq_dists, sq_dists)
        min_sq_dists[rows, selected[:, i - 1]] = -1.0
        selected[:, i] = min_sq_dists.argmax(dim=1)

    return selected, selected_valid


def normalized_uv(
    kps_xy: torch.Tensor,
    resolutions_hw: torch.Tensor,
) -> torch.Tensor:
    heights = resolutions_hw[:, 0].unsqueeze(1)
    widths = resolutions_hw[:, 1].unsqueeze(1)

    return torch.stack([kps_xy[..., 0] / widths, kps_xy[..., 1] / heights], dim=-1)
