# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from __future__ import annotations

import torch

from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation


def resolve_timestamps_to_frame_indices(
    query_timestamps: torch.Tensor,
    frame_timestamps: torch.Tensor,
) -> torch.Tensor:
    """Find the last frame at or before each query timestamp, -1 if none."""
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
    """Express the reference timestamps on the view's clock."""
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
