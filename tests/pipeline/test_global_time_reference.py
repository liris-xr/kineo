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

from kineo.annotations.camera_temporal import (
    CameraTemporalAnnotation,
    CameraTemporalAnnotations,
    CameraTemporalAnnotationsMetadata,
)
from kineo.datasets.annotations_io import build_synchronized_camera_temporal
from kineo.pipeline.stages import global_time_reference

CPU = torch.device("cpu")


class _FakeFrameLoader:
    def __init__(self, n_frames: int, fps: float):
        self.n_frames = n_frames
        self.frame_timestamps_local = torch.arange(n_frames) / fps


def _views(frame_counts: dict[str, int], fps: float) -> list[dict]:
    return [
        {"view_id": view_id, "frame_loader": _FakeFrameLoader(n_frames, fps)}
        for view_id, n_frames in frame_counts.items()
    ]


def _temporal(offsets: dict[str, float]) -> CameraTemporalAnnotations:
    return CameraTemporalAnnotations(
        metadata=CameraTemporalAnnotationsMetadata(),
        annotations=[
            CameraTemporalAnnotation(
                view_id=view_id, frame_idx=0, time_offset=offset
            )
            for view_id, offset in offsets.items()
        ],
    )


def test_synchronized_views_are_read_at_their_own_rate():
    # What EgoHumans and H3.6M give: no offsets, equal frame counts.
    views = _views({"cam01": 100, "cam02": 100}, fps=20.0)

    reference = global_time_reference.build_global_time_reference(
        views=views,
        camera_temporal=build_synchronized_camera_temporal(["cam01", "cam02"]),
        target_fps=5,
        device=CPU,
    ).first_or_default()

    assert reference.timestamps.numel() == 100
    assert global_time_reference.is_pass_through(reference)
    for view_id in ("cam01", "cam02"):
        torch.testing.assert_close(
            reference.closest_local_frame_idx[view_id], torch.arange(100)
        )


def test_synchronized_views_keep_every_frame_worth_inferring():
    views = _views({"cam01": 100, "cam02": 100}, fps=20.0)
    annotations = {
        "global_time_reference": global_time_reference.build_global_time_reference(
            views=views,
            camera_temporal=build_synchronized_camera_temporal(["cam01", "cam02"]),
            target_fps=5,
            device=CPU,
        )
    }

    frames = global_time_reference.build_inference_frames(annotations, views)

    assert frames == {"cam01": list(range(100)), "cam02": list(range(100))}


def test_offset_views_are_read_on_a_grid_over_what_they_share():
    views = _views({"cam01": 100, "cam02": 100}, fps=20.0)

    reference = global_time_reference.build_global_time_reference(
        views=views,
        camera_temporal=_temporal({"cam01": 0.0, "cam02": 1.0}),
        target_fps=20,
        device=CPU,
    ).first_or_default()

    # cam02 starts a second later, so they share all but that second.
    assert reference.timestamps.numel() == 80
    assert not global_time_reference.is_pass_through(reference)
    torch.testing.assert_close(
        reference.closest_local_frame_idx["cam01"], 20 + torch.arange(80)
    )
    torch.testing.assert_close(
        reference.closest_local_frame_idx["cam02"], torch.arange(80)
    )


def test_a_grid_below_the_recorded_rate_skips_frames():
    views = _views({"cam01": 100, "cam02": 100}, fps=60.0)

    annotations = {
        "global_time_reference": global_time_reference.build_global_time_reference(
            views=views,
            camera_temporal=_temporal({"cam01": 0.0, "cam02": 0.1}),
            target_fps=20,
            device=CPU,
        )
    }

    frames = global_time_reference.build_inference_frames(annotations, views)

    assert len(frames["cam01"]) < 100
    assert frames["cam01"] == sorted(set(frames["cam01"]))


def test_no_timeline_yet_leaves_the_caller_alone():
    views = _views({"cam01": 10}, fps=20.0)

    assert global_time_reference.build_inference_frames({}, views) is None
