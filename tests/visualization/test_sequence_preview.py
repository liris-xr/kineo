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

from kineo.annotations.bboxes_2d import (
    BBox2DAnnotation,
    BBox2DAnnotations,
    BBox2DAnnotationsMetadata,
)
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.visualization.sequence_preview import (
    local_frame_indices,
    rebase_on_global_frames,
)


def build_time_reference() -> GlobalTimeReferenceAnnotation:
    return GlobalTimeReferenceAnnotation(
        timestamps=torch.arange(3) / 60.0,
        closest_local_frame_idx={
            "c01": torch.tensor([775, 776, 777]),
            "c02": torch.tensor([784, 785, 786]),
        },
    )


def build_bboxes(frames_by_view: dict[str, list[int]]) -> BBox2DAnnotations:
    return BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(),
        annotations=[
            BBox2DAnnotation(
                view_id=view_id,
                frame_idx=frame_idx,
                subject_id="s0",
                category_id=0,
                xyxy=torch.zeros(4),
                score=1.0,
            )
            for view_id, frames in frames_by_view.items()
            for frame_idx in frames
        ],
    )


def test_views_without_a_time_reference_are_read_as_frame_aligned():
    torch.testing.assert_close(
        local_frame_indices("c01", n_frames=4, time_reference=None),
        torch.arange(4),
    )


def test_a_time_reference_places_the_steps_in_each_recording():
    torch.testing.assert_close(
        local_frame_indices(
            "c02", n_frames=1480, time_reference=build_time_reference()
        ),
        torch.tensor([784, 785, 786]),
    )


def test_one_instant_lands_on_one_step_in_every_view():
    bboxes = build_bboxes({"c01": [775, 776], "c02": [784, 785]})

    rebased = rebase_on_global_frames(bboxes, build_time_reference())

    assert sorted(annotation.frame_idx for annotation in rebased) == [0, 0, 1, 1]


def test_annotations_outside_the_annotated_window_are_dropped():
    bboxes = build_bboxes({"c01": [774, 775, 778]})

    rebased = rebase_on_global_frames(bboxes, build_time_reference())

    assert [annotation.frame_idx for annotation in rebased] == [0]
