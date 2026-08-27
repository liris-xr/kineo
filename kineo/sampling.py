# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Candidate selection primitives shared by the sampling stages.

The sampler is deliberately blind to what its axes mean: callers assemble their
own feature array, weights already folded into the columns, and the distance is
a plain Euclidean sum over whatever they stacked.
"""

import torch


def farthest_point_sampling(
    features: torch.Tensor,
    n_samples: int,
    generator: torch.Generator,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy farthest-point sampling, batched over the leading axis.

    The first point of each row is drawn uniformly at random; every subsequent
    one maximizes the distance to that row's already selected set. Rows advance
    together, so the greedy loop costs one round of kernels per sample rather
    than one per sample *per row* — the loop is short and launch-bound, so that
    is the difference between the two.

    Rows share the ``features`` axis and are separated only by ``valid_mask``,
    which lets callers whose rows see overlapping subsets of one candidate set
    skip building a ragged batch. Masked-out entries are pinned below every real
    distance, so they can never win the argmax and never perturb another row.

    Args:
        features: Point features of shape (n_rows, n_points, n_dims).
        n_samples: Points to select per row. Clamped to ``n_points``.
        generator: Seeded generator, drawn from once per row, in row order.
        valid_mask: Bool tensor of shape (n_rows, n_points) flagging the points
            each row may select. Defaults to all of them.

    Returns:
        selected: Long tensor (n_rows, n_samples) of indices into the point
            axis, in selection order.
        selected_valid: Bool tensor (n_rows, n_samples), False where a row held
            fewer valid points than the budget and its tail is padding.
    """
    n_rows, n_points, _ = features.shape
    n_samples = min(max(n_samples, 0), n_points)
    device = features.device

    if valid_mask is None:
        valid_mask = torch.ones(
            n_rows, n_points, dtype=torch.bool, device=device
        )

    selected = torch.zeros(n_rows, n_samples, dtype=torch.long, device=device)
    selected_valid = (
        torch.arange(n_samples, device=device).unsqueeze(0)
        < valid_mask.sum(dim=1, keepdim=True)
    )

    if n_samples == 0:
        return selected, selected_valid

    # Drawn per row and in row order so the batch reproduces what a sequential
    # per-row run sharing this generator would have picked.
    for row in range(n_rows):
        row_points = torch.where(valid_mask[row])[0]

        if len(row_points) == 0:
            continue

        selected[row, 0] = row_points[
            torch.randint(
                len(row_points), (1,), generator=generator, device=device
            )
        ]

    # Squared distance from every point to the closest selected one. Invalid and
    # already selected points are driven negative so argmax cannot return them;
    # torch.minimum keeps them there, since a real distance is never negative.
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


def uniform_point_sampling(
    valid_mask: torch.Tensor,
    n_samples: int,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Uniform draw without replacement per row, over the FPS candidate sets.

    The ablation's other arm: same rows, same ``valid_mask``, no coverage
    criterion. Rows are drawn in order from the shared generator, so the result
    is reproducible from the seed alone.

    Returned per row rather than stacked, since rows hold different numbers of
    valid points and there is no selection order to pad against.

    Args:
        valid_mask: Bool tensor of shape (n_rows, n_points) flagging the points
            each row may select.
        n_samples: Points to draw per row. A row holding fewer valid points
            yields all of them.
        generator: Seeded generator, drawn from once per row, in row order.

    Returns:
        One long tensor of point indices per row, in row order.
    """
    picked = []

    for row_mask in valid_mask:
        row_points = torch.where(row_mask)[0]
        order = torch.randperm(
            len(row_points), generator=generator, device=row_points.device
        )[:n_samples]
        picked.append(row_points[order])

    return picked


def normalized_uv(
    kps_xy: torch.Tensor,
    resolutions_hw: torch.Tensor,
) -> torch.Tensor:
    """Rescales keypoints into their own view's unit square.

    Each view is normalized by its own resolution, since a rig's cameras do not
    share a sensor size.

    Args:
        kps_xy: Keypoint positions of shape (n_views, n_points, 2).
        resolutions_hw: Per-view (height, width) of shape (n_views, 2).

    Returns:
        Float tensor of shape (n_views, n_points, 2), inside ``[0, 1]^2``
        exactly when the keypoint is inside its view's frame.
    """
    heights = resolutions_hw[:, 0].unsqueeze(1)
    widths = resolutions_hw[:, 1].unsqueeze(1)

    return torch.stack(
        [kps_xy[..., 0] / widths, kps_xy[..., 1] / heights], dim=-1
    )


def valid_observations_mask(
    kps_xy: torch.Tensor,
    resolutions_hw: torch.Tensor,
) -> torch.Tensor:
    """Flags the keypoints that are finite and inside their own view's frame.

    Args:
        kps_xy: Keypoint positions of shape (n_views, n_candidates, 2).
        resolutions_hw: Per-view (height, width) of shape (n_views, 2).

    Returns:
        Bool tensor of shape (n_views, n_candidates), False where a keypoint is
        non-finite or falls outside ``[0, W] x [0, H]`` of its own view.
    """
    uvs = normalized_uv(kps_xy, resolutions_hw)

    # A non-finite keypoint normalizes to nan or inf, which fails the bounds
    # test on its own; the explicit check keeps that from being incidental.
    in_frame = ((uvs >= 0.0) & (uvs <= 1.0)).all(dim=-1)
    return in_frame & torch.isfinite(kps_xy).all(dim=-1)
