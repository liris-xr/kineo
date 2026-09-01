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
from kineo.annotations.camera_intrinsics import (
    CameraDistortionModel,
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotations,
    CameraIntrinsicsAnnotationsMetadata,
)
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.annotations.keypoints_2d import (
    Keypoints2DAnnotation,
    Keypoints2DAnnotations,
    Keypoints2DAnnotationsMetadata,
)
from kineo.annotations import COCO_17_KEYPOINTS_FORMAT
from kineo.visualization.sequence_preview import (
    local_frame_indices,
    preview_scale,
    rebase_on_global_frames,
    scale_pixel_space,
)


def build_intrinsics(resolution_hw=(2160, 3840)) -> CameraIntrinsicsAnnotations:
    return CameraIntrinsicsAnnotations(
        metadata=CameraIntrinsicsAnnotationsMetadata(),
        annotations=[
            CameraIntrinsicsAnnotation(
                view_id="c01",
                frame_idx=0,
                K=torch.tensor(
                    [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
                ),
                distortion_coefficients=torch.zeros(5),
                distortion_model=CameraDistortionModel.BROWN_CONRADY,
                resolution_hw=resolution_hw,
            )
        ],
    )


class FakeLoader:
    def __init__(self, resolution_hw):
        self.resolution_hw = resolution_hw


def build_time_reference() -> GlobalTimeReferenceAnnotation:
    return GlobalTimeReferenceAnnotation(
        timestamps=torch.arange(3) / 60.0,
        closest_local_frame_idx={
            "c01": torch.tensor([775, 776, 777]),
            "c02": torch.tensor([784, 785, 786]),
        },
    )


def build_keypoints_2d(xy: torch.Tensor) -> Keypoints2DAnnotations:
    return Keypoints2DAnnotations(
        metadata=Keypoints2DAnnotationsMetadata(
            formats=[COCO_17_KEYPOINTS_FORMAT]
        ),
        annotations=[
            Keypoints2DAnnotation(
                view_id="c01",
                frame_idx=0,
                subject_id="s0",
                xy=xy,
                scores=torch.ones(xy.shape[0]),
                format=COCO_17_KEYPOINTS_FORMAT.name,
            )
        ],
    )


def build_bboxes(
    frames_by_view: dict[str, list[int]],
    xyxy: torch.Tensor | None = None,
) -> BBox2DAnnotations:
    return BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(),
        annotations=[
            BBox2DAnnotation(
                view_id=view_id,
                frame_idx=frame_idx,
                subject_id="s0",
                category_id=0,
                xyxy=torch.zeros(4) if xyxy is None else xyxy,
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


def test_the_widest_view_sets_the_preview_scale():
    views = [
        {"view_id": "c01", "frame_loader": FakeLoader((2160, 3840))},
        {"view_id": "c02", "frame_loader": FakeLoader((1080, 1920))},
    ]

    assert preview_scale(views, max_side=480) == 480 / 3840


def test_a_view_smaller_than_the_preview_is_left_alone():
    views = [{"view_id": "c01", "frame_loader": FakeLoader((240, 320))}]

    assert preview_scale(views, max_side=480) == 1.0


def test_resizing_moves_boxes_and_keypoints_with_the_footage():
    annotations = {
        "bboxes_2d": build_bboxes(
            {"c01": [0]}, xyxy=torch.tensor([10.0, 20.0, 30.0, 40.0])
        ),
        "keypoints_2d": build_keypoints_2d(xy=torch.full((17, 2), 100.0)),
    }

    resized = scale_pixel_space(annotations, 0.5)

    torch.testing.assert_close(
        resized["keypoints_2d"].annotations[0].xy, torch.full((17, 2), 50.0)
    )
    torch.testing.assert_close(
        resized["bboxes_2d"].annotations[0].xyxy,
        torch.tensor([5.0, 10.0, 15.0, 20.0]),
    )


def test_resizing_moves_the_intrinsics_with_the_footage():
    resized = scale_pixel_space({"cameras_intrinsics": build_intrinsics()}, 0.5)

    intrinsics = resized["cameras_intrinsics"].annotations[0]

    torch.testing.assert_close(
        intrinsics.K,
        torch.tensor(
            [[500.0, 0.0, 480.0], [0.0, 500.0, 270.0], [0.0, 0.0, 1.0]]
        ),
    )
    assert intrinsics.resolution_hw == (1080, 1920)


def test_a_preview_at_the_dataset_size_changes_nothing():
    annotations = {"cameras_intrinsics": build_intrinsics()}

    assert scale_pixel_space(annotations, 1.0) is annotations
