import os

import orjson
import pytest
import torch

from kineo.annotations import camera_extrinsics
from kineo.datasets import annotations_io


class _FakeAnnotations:
    """Minimal stand-in exposing the to_dict() contract the writer relies on."""

    def __init__(self, payload: dict):
        self._payload = payload

    def to_dict(self) -> dict:
        return self._payload


def test_build_synchronized_camera_temporal_emits_one_zero_offset_per_view():
    temporal = annotations_io.build_synchronized_camera_temporal(
        ["cam01", "cam02"]
    )

    assert [a.view_id for a in temporal.annotations] == ["cam01", "cam02"]
    assert all(a.frame_idx == 0 for a in temporal.annotations)
    assert all(a.time_offset == 0.0 for a in temporal.annotations)
    assert all(isinstance(a.time_offset, float) for a in temporal.annotations)


def test_write_sequence_annotations_writes_files_and_returns_posix_relpaths(
    tmp_path,
):
    relpaths = annotations_io.write_sequence_annotations(
        dataset_dir=str(tmp_path),
        annotations_reldir=os.path.join("seq", "annotations"),
        annotations={
            "keypoints_3d": _FakeAnnotations({"a": 1}),
            "bboxes_2d": _FakeAnnotations({"b": 2}),
        },
    )

    assert relpaths == {
        "keypoints_3d": "seq/annotations/keypoints_3d.json",
        "bboxes_2d": "seq/annotations/bboxes_2d.json",
    }
    written = tmp_path / "seq" / "annotations" / "keypoints_3d.json"
    assert orjson.loads(written.read_bytes()) == {"a": 1}


def test_write_sequence_annotations_returns_canonical_key_order(tmp_path):
    relpaths = annotations_io.write_sequence_annotations(
        dataset_dir=str(tmp_path),
        annotations_reldir="annotations",
        annotations={
            "cameras_extrinsics": _FakeAnnotations({}),
            "keypoints_2d": _FakeAnnotations({}),
            "cameras_temporal": _FakeAnnotations({}),
        },
    )

    assert list(relpaths) == [
        "keypoints_2d",
        "cameras_temporal",
        "cameras_extrinsics",
    ]


def test_write_sequence_annotations_omits_absent_kinds(tmp_path):
    relpaths = annotations_io.write_sequence_annotations(
        dataset_dir=str(tmp_path),
        annotations_reldir="annotations",
        annotations={"keypoints_3d": _FakeAnnotations({})},
    )

    assert "cameras_temporal" not in relpaths
    assert not (tmp_path / "annotations" / "cameras_temporal.json").exists()


def test_write_sequence_annotations_rejects_unknown_key_before_writing(tmp_path):
    with pytest.raises(KeyError):
        annotations_io.write_sequence_annotations(
            dataset_dir=str(tmp_path),
            annotations_reldir="annotations",
            annotations={
                "keypoints_3d": _FakeAnnotations({}),
                "keypoints_4d": _FakeAnnotations({}),
            },
        )

    assert not (tmp_path / "annotations" / "keypoints_3d.json").exists()


def test_write_sequence_annotations_serializes_tensors(tmp_path):
    annotations_io.write_sequence_annotations(
        dataset_dir=str(tmp_path),
        annotations_reldir="annotations",
        annotations={
            "keypoints_3d": _FakeAnnotations({"xyz": torch.zeros(2, 3)})
        },
    )

    written = tmp_path / "annotations" / "keypoints_3d.json"
    assert orjson.loads(written.read_bytes()) == {
        "xyz": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    }


def test_single_pose_per_view_reads_back_as_static():
    # The H3.6M fallback: one extrinsics annotation per view is exactly the
    # degenerate case of the moving-camera format, no flag needed.
    extrinsics = camera_extrinsics.CameraExtrinsicsAnnotations(
        metadata=camera_extrinsics.CameraExtrinsicsAnnotationsMetadata(),
        annotations=[
            camera_extrinsics.CameraExtrinsicsAnnotation(
                view_id="54138969",
                frame_idx=0,
                R=torch.eye(3),
                t=torch.zeros(3),
            )
        ],
    )

    assert extrinsics.is_view_static("54138969")


def test_h36m_annotation_kinds_match_egohumans():
    # Both datasets emit the same six kinds; h36m's cameras are synchronized, so
    # it uses the shared builder rather than skipping the kind.
    from kineo.datasets.h36m import h36m_preprocess  # noqa: F401

    assert set(annotations_io.ANNOTATION_FILENAMES) == {
        "keypoints_2d",
        "keypoints_3d",
        "bboxes_2d",
        "cameras_temporal",
        "cameras_intrinsics",
        "cameras_extrinsics",
    }
