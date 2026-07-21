import numpy as np
import torch

from kineo.datasets.egohumans.egohumans_smpl_gt import (
    load_egohumans_smpl_keypoints_3d,
)


def _write_frame(smpl_dir, basename, subjects):
    data = {
        sid: {"joints": np.arange(45 * 3).reshape(45, 3).astype(np.float32) + off}
        for sid, off in subjects.items()
    }
    np.save(f"{smpl_dir}/{basename}.npy", np.array(data, dtype=object))


def test_loader_builds_smpl_22_annotations(tmp_path):
    smpl_dir = tmp_path / "smpl"
    smpl_dir.mkdir()
    _write_frame(str(smpl_dir), "00001", {"aria01": 0.0, "aria02": 100.0})
    _write_frame(str(smpl_dir), "00002", {"aria01": 1.0, "aria02": 101.0})

    anns = load_egohumans_smpl_keypoints_3d(str(smpl_dir))

    assert anns.metadata.formats[0].name == "smpl_22"
    assert set(anns.subjects_ids) == {"aria01", "aria02"}
    # Frame index is basename - 1.
    assert sorted(anns.frames) == [0, 1]
    ann = anns.filter_by_subject_id("aria01").filter_by_frame_idx(0).first_or_default()
    assert ann.xyz.shape == (22, 3)
    assert ann.xyz.dtype == torch.float32


def test_loader_filters_invalid_subjects(tmp_path):
    smpl_dir = tmp_path / "smpl"
    smpl_dir.mkdir()
    _write_frame(str(smpl_dir), "00001", {"aria01": 0.0, "aria02": 100.0})
    anns = load_egohumans_smpl_keypoints_3d(
        str(smpl_dir), valid_subject_ids=["aria01"]
    )
    assert anns.subjects_ids == ["aria01"]
