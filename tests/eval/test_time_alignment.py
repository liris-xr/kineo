import pytest
import torch

from kineo.annotations.global_time_reference import (
    GlobalTimeReferenceAnnotation,
)
from kineo.eval import time_alignment
from kineo.eval.time_alignment import (
    build_slots_by_prediction_frame,
    resolve_timestamps_to_frame_indices,
)


def _timeline(n: int, fps: float, start: float = 0.0) -> torch.Tensor:
    return start + torch.arange(n, dtype=torch.float64) / fps


def test_query_on_a_frame_timestamp_resolves_to_that_frame():
    frames = _timeline(5, 10.0)
    resolved = resolve_timestamps_to_frame_indices(frames.clone(), frames)
    assert resolved.tolist() == [0, 1, 2, 3, 4]


def test_query_between_frames_resolves_to_the_frame_being_shown():
    frames = _timeline(4, 10.0)  # 0.0, 0.1, 0.2, 0.3
    queries = torch.tensor([0.05, 0.1, 0.19, 0.2, 0.35])
    assert resolve_timestamps_to_frame_indices(queries, frames).tolist() == [
        0,
        1,
        1,
        2,
        3,
    ]


def test_query_before_the_first_frame_resolves_to_minus_one():
    frames = _timeline(3, 10.0, start=1.0)
    queries = torch.tensor([0.0, 0.999, 1.0])
    assert resolve_timestamps_to_frame_indices(queries, frames).tolist() == [
        -1,
        -1,
        0,
    ]


def test_query_after_the_last_frame_holds_the_last_frame():
    frames = _timeline(3, 10.0)
    assert resolve_timestamps_to_frame_indices(
        torch.tensor([99.0]), frames
    ).tolist() == [2]


def test_empty_timeline_is_rejected():
    with pytest.raises(ValueError, match="empty timeline"):
        resolve_timestamps_to_frame_indices(torch.tensor([0.0]), torch.tensor([]))


def test_descending_timeline_is_rejected():
    with pytest.raises(ValueError, match="ascending"):
        resolve_timestamps_to_frame_indices(
            torch.tensor([0.0]), torch.tensor([1.0, 0.0])
        )


def test_slots_without_timestamps_match_by_frame_index():
    slots = build_slots_by_prediction_frame([3, 4, 7])
    assert slots == {3: [0], 4: [1], 7: [2]}


def test_slots_with_identical_timelines_match_by_frame_index():
    timeline = _timeline(6, 60.0)
    slots = build_slots_by_prediction_frame([0, 2, 5], timeline, timeline)
    assert slots == {0: [0], 2: [1], 5: [2]}


def test_slots_map_a_shifted_prediction_timeline_onto_ground_truth():
    # The prediction starts a quarter of a frame late, so each ground-truth
    # instant is still answered by the frame shown just before it.
    gt = _timeline(4, 10.0)
    pred = _timeline(4, 10.0, start=-0.025)
    assert build_slots_by_prediction_frame([0, 1, 2, 3], gt, pred) == {
        0: [0],
        1: [1],
        2: [2],
        3: [3],
    }


def test_a_slower_prediction_answers_for_several_ground_truth_slots():
    gt = _timeline(4, 20.0)  # 0.00, 0.05, 0.10, 0.15
    pred = _timeline(2, 10.0)  # 0.00, 0.10
    assert build_slots_by_prediction_frame([0, 1, 2, 3], gt, pred) == {
        0: [0, 1],
        1: [2, 3],
    }


def test_ground_truth_before_the_prediction_starts_is_dropped():
    gt = _timeline(3, 10.0)  # 0.0, 0.1, 0.2
    pred = _timeline(2, 10.0, start=0.15)  # 0.15, 0.25
    assert build_slots_by_prediction_frame([0, 1, 2], gt, pred) == {0: [2]}


def _reference(timestamps, closest_local_frame_idx):
    return GlobalTimeReferenceAnnotation(
        timestamps=torch.tensor(timestamps),
        closest_local_frame_idx={
            view_id: torch.tensor(frames)
            for view_id, frames in closest_local_frame_idx.items()
        },
    )


def test_timestamps_on_view_restates_instants_on_that_views_clock():
    reference = _reference(
        timestamps=[10 / 50, 11 / 50, 12 / 50],
        closest_local_frame_idx={"c01": [10, 11, 12], "c02": [12, 13, 14]},
    )

    torch.testing.assert_close(
        time_alignment.timestamps_on_view(reference, "c02"),
        torch.tensor([12 / 50, 13 / 50, 14 / 50]),
    )


def test_timestamps_on_the_view_they_were_built_against_are_unchanged():
    reference = _reference(
        timestamps=[10 / 50, 11 / 50, 12 / 50],
        closest_local_frame_idx={"c01": [10, 11, 12], "c02": [12, 13, 14]},
    )

    torch.testing.assert_close(
        time_alignment.timestamps_on_view(reference, "c01"), reference.timestamps
    )


def test_timestamps_on_an_unknown_view_is_an_error():
    reference = _reference(
        timestamps=[0.0, 0.02], closest_local_frame_idx={"c01": [0, 1]}
    )

    with pytest.raises(ValueError, match="c09"):
        time_alignment.timestamps_on_view(reference, "c09")
