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
