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


def test_filter_active_picks_most_recent_onset_with_earliest_fallback():
    # cam01 static (one pose); cam02 non-static: segment A onset 10, B onset 100.
    anns = _annotations(
        [
            _ann("cam01", 0, 0.0),
            _ann("cam02", 10, 1.0),
            _ann("cam02", 100, 2.0),
        ]
    )

    def cam02_tx(frame_idx):
        active = anns.filter_active_by_frame_idx(frame_idx)
        return active.filter_by_view_id("cam02").first_or_default().t[0].item()

    assert cam02_tx(0) == 1.0  # before first onset -> earliest (A)
    assert cam02_tx(10) == 1.0  # at A onset
    assert cam02_tx(50) == 1.0  # between onsets -> most recent (A)
    assert cam02_tx(100) == 2.0  # at B onset
    assert cam02_tx(200) == 2.0  # after B onset -> B

    active = anns.filter_active_by_frame_idx(50)
    assert active.views_ids == ["cam01", "cam02"]  # one pose per view
    assert len(active.annotations) == 2


def test_is_view_static_reflects_annotation_count():
    anns = _annotations([_ann("cam01", 0, 0.0), _ann("cam02", 10, 1.0), _ann("cam02", 100, 2.0)])
    assert anns.is_view_static("cam01") is True
    assert anns.is_view_static("cam02") is False
