# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Alignment of two frame timelines for metric computation.

Metrics compare an annotation with a prediction, which agree on when something
happened but not on how frames are numbered: ground truth counts frames of the
annotated sequence while a prediction counts frames of whatever grid the
pipeline resampled onto. Matching them by frame index only works when the two
grids happen to coincide, which is a property of the dataset rather than
something to assume, so the index is resolved through the timestamps instead.
"""

from __future__ import annotations

import torch


def resolve_timestamps_to_frame_indices(
    query_timestamps: torch.Tensor,
    frame_timestamps: torch.Tensor,
) -> torch.Tensor:
    """Finds the frame being shown at each query timestamp.

    A frame stays on screen from its own timestamp until the next one, so the
    frame a query lands on is the last one at or before it, never the nearest
    one: rounding to the nearest would name a frame that has not been shown
    yet.

    Args:
        query_timestamps: Timestamps to resolve, in seconds. Shape (Q,).
        frame_timestamps: Timestamps of the frames to resolve against, in
            seconds, ascending. Shape (F,).

    Returns:
        The frame index shown at each query timestamp, of shape (Q,) and dtype
        long. Queries falling before the first frame resolve to -1, having no
        frame to name.

    Raises:
        ValueError: If `frame_timestamps` is empty or not ascending.
    """
    if frame_timestamps.ndim != 1 or query_timestamps.ndim != 1:
        raise ValueError(
            f"Expected 1-D timestamps, got shapes {tuple(query_timestamps.shape)} "
            f"and {tuple(frame_timestamps.shape)}."
        )

    if frame_timestamps.numel() == 0:
        raise ValueError("Cannot resolve timestamps against an empty timeline.")

    if bool(torch.any(frame_timestamps[1:] < frame_timestamps[:-1])):
        raise ValueError("Expected frame_timestamps to be ascending.")

    frames = frame_timestamps.to(torch.float64)
    queries = query_timestamps.to(torch.float64).to(frames.device)

    return torch.searchsorted(frames, queries, right=True) - 1


def build_slots_by_prediction_frame(
    gt_frame_indices: list[int],
    gt_frame_timestamps: torch.Tensor | None = None,
    pred_frame_timestamps: torch.Tensor | None = None,
) -> dict[int, list[int]]:
    """Maps each predicted frame to the ground-truth slots it answers for.

    Without timestamps the two timelines are taken to be the same one, which is
    what a capture whose views are already synchronized and untrimmed gives, and
    a predicted frame answers for the ground-truth frame of equal index. With
    timestamps each ground-truth slot is resolved onto the predicted frame shown
    at its instant, so one predicted frame may answer for several slots when the
    prediction runs at the lower rate.

    Args:
        gt_frame_indices: Frame index of each ground-truth slot, in slot order.
        gt_frame_timestamps: Timestamps of the ground-truth timeline, indexed by
            ground-truth frame index. `None` to match by index.
        pred_frame_timestamps: Timestamps of the prediction timeline, indexed by
            predicted frame index. `None` to match by index.

    Returns:
        Maps a predicted frame index to the ground-truth slots it fills.
    """
    if gt_frame_timestamps is None or pred_frame_timestamps is None:
        return {frame_idx: [slot] for slot, frame_idx in enumerate(gt_frame_indices)}

    gt_timestamps = gt_frame_timestamps[
        torch.as_tensor(gt_frame_indices, dtype=torch.long)
    ]
    pred_frames = resolve_timestamps_to_frame_indices(
        gt_timestamps, pred_frame_timestamps
    )

    slots_by_frame: dict[int, list[int]] = {}
    for slot, pred_frame in enumerate(pred_frames.tolist()):
        if pred_frame < 0:
            continue
        slots_by_frame.setdefault(pred_frame, []).append(slot)

    return slots_by_frame
