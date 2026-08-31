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

Ground truth counts frames of the annotated sequence, a prediction counts
frames of whatever grid the pipeline resampled onto. Matching by frame index
holds only when the two grids coincide, so the index is resolved through the
timestamps instead.
"""

from __future__ import annotations

import torch

from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation


def resolve_timestamps_to_frame_indices(
    query_timestamps: torch.Tensor,
    frame_timestamps: torch.Tensor,
) -> torch.Tensor:
    """Finds the frame being shown at each query timestamp.

    A frame stays on screen from its own timestamp until the next one, so a
    query lands on the last frame at or before it, never the nearest one:
    rounding to the nearest would name a frame not yet shown.

    Args:
        query_timestamps: Timestamps to resolve, in seconds. Shape (Q,).
        frame_timestamps: Ascending timestamps of the frames to resolve
            against, in seconds. Shape (F,).

    Returns:
        The frame index shown at each query timestamp, of shape (Q,) and dtype
        long. Queries before the first frame resolve to -1.

    Raises:
        ValueError: If `frame_timestamps` is empty or not ascending.
    """
    if frame_timestamps.numel() == 0:
        raise ValueError("Cannot resolve timestamps against an empty timeline.")

    if bool(torch.any(frame_timestamps[1:] < frame_timestamps[:-1])):
        raise ValueError("Expected frame_timestamps to be ascending.")

    frames = frame_timestamps.to(torch.float64)
    queries = query_timestamps.to(torch.float64).to(frames.device)

    return torch.searchsorted(frames, queries, right=True) - 1


def timestamps_on_view(
    reference: GlobalTimeReferenceAnnotation, view_id: str
) -> torch.Tensor:
    """Restates a timeline's instants on one view's local clock.

    A pipeline resamples onto the clock of the view it pins to a zero offset,
    so annotations have to be read on that same view's clock to meet its
    predictions. Both clocks being uniform and of the same rate, which holds for
    a constant-frame-rate recording, an instant's local frame index is all that
    is needed to restate it.

    Args:
        reference: Timeline to restate, holding the local frame of each view
            shown at each of its instants.
        view_id: View whose clock to restate the instants on.

    Returns:
        The instants, in seconds on `view_id`'s clock.

    Raises:
        ValueError: If the timeline is too short to read a rate off, or does not
            cover `view_id`.
    """
    if reference.timestamps.numel() < 2:
        raise ValueError("Cannot read a rate off fewer than two instants.")

    if view_id not in reference.closest_local_frame_idx:
        raise ValueError(
            f"Timeline does not cover view {view_id}: "
            f"{sorted(reference.closest_local_frame_idx)}."
        )

    frame_duration = reference.timestamps[1] - reference.timestamps[0]

    return reference.closest_local_frame_idx[view_id] * frame_duration


def build_slots_by_prediction_frame(
    gt_frame_indices: list[int],
    gt_frame_timestamps: torch.Tensor | None = None,
    pred_frame_timestamps: torch.Tensor | None = None,
) -> dict[int, list[int]]:
    """Maps each predicted frame to the ground-truth slots it answers for.

    Without both timelines the two are taken to be the same one and a
    predicted frame answers for the ground-truth frame of equal index, which
    is what an already-synchronized, untrimmed capture gives. With them, each
    slot resolves onto the predicted frame shown at its instant, so one
    predicted frame may answer for several slots when the prediction runs at
    the lower rate.

    Args:
        gt_frame_indices: Frame index of each ground-truth slot, in slot order.
        gt_frame_timestamps: Ground-truth timeline, indexed by ground-truth
            frame index. `None` to match by index.
        pred_frame_timestamps: Prediction timeline, indexed by predicted frame
            index. `None` to match by index.

    Returns:
        Maps a predicted frame index to the ground-truth slots it fills.
    """
    if gt_frame_timestamps is None or pred_frame_timestamps is None:
        return {frame_idx: [slot] for slot, frame_idx in enumerate(gt_frame_indices)}

    pred_frames = resolve_timestamps_to_frame_indices(
        gt_frame_timestamps[torch.as_tensor(gt_frame_indices, dtype=torch.long)],
        pred_frame_timestamps,
    )

    slots_by_frame: dict[int, list[int]] = {}
    for slot, pred_frame in enumerate(pred_frames.tolist()):
        if pred_frame >= 0:
            slots_by_frame.setdefault(pred_frame, []).append(slot)

    return slots_by_frame
