import math

import torch

from kineo.annotations.camera_extrinsics import (
    CameraExtrinsicsAnnotation,
    CameraExtrinsicsAnnotations,
    CameraExtrinsicsAnnotationsMetadata,
)


def _ann(view_id: str, frame_idx: int, tx: float) -> CameraExtrinsicsAnnotation:
    return CameraExtrinsicsAnnotation(
        view_id=view_id,
        frame_idx=frame_idx,
        R=torch.eye(3),
        t=torch.tensor([tx, 0.0, 0.0]),
    )


def _annotations(anns) -> CameraExtrinsicsAnnotations:
    return CameraExtrinsicsAnnotations(
        metadata=CameraExtrinsicsAnnotationsMetadata(),
        annotations=anns,
    )


def test_is_view_static_reflects_annotation_count():
    anns = _annotations([_ann("cam01", 0, 0.0), _ann("cam02", 10, 1.0), _ann("cam02", 100, 2.0)])
    assert anns.is_view_static("cam01") is True
    assert anns.is_view_static("cam02") is False


def _rot_z(degrees: float) -> torch.Tensor:
    angle = math.radians(degrees)
    cos, sin = math.cos(angle), math.sin(angle)
    return torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def test_is_view_static_treats_near_identical_poses_as_static():
    # A pose re-estimated per segment for a camera that never actually moved:
    # several annotations, all within tolerance, so the view is still static.
    anns = _annotations([_ann("cam01", 0, 0.0), _ann("cam01", 100, 0.005)])

    assert anns.is_view_static("cam01") is True


def test_is_view_static_compares_against_a_reference_not_consecutively():
    # Each step is under tolerance but the camera ends up 4.5 cm from where it
    # started. Comparing consecutive poses would call this static; it is not.
    anns = _annotations(
        [
            _ann("cam01", 0, 0.0),
            _ann("cam01", 50, 0.015),
            _ann("cam01", 100, 0.030),
            _ann("cam01", 150, 0.045),
        ]
    )

    assert anns.is_view_static("cam01") is False


def test_is_view_static_detects_rotation_with_no_translation():
    # A panning camera stays at the same point, so translation alone misses it.
    anns = _annotations(
        [
            CameraExtrinsicsAnnotation(
                view_id="cam01", frame_idx=0, R=torch.eye(3), t=torch.zeros(3)
            ),
            CameraExtrinsicsAnnotation(
                view_id="cam01", frame_idx=100, R=_rot_z(5.0), t=torch.zeros(3)
            ),
        ]
    )

    assert anns.is_view_static("cam01") is False


def test_is_view_static_tolerances_are_overridable():
    anns = _annotations([_ann("cam01", 0, 0.0), _ann("cam01", 100, 0.5)])

    assert anns.is_view_static("cam01") is False
    assert anns.is_view_static("cam01", translation_tolerance=1.0) is True
