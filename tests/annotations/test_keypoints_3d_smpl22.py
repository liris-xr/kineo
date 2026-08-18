import torch

from kineo.annotations.keypoints_3d import (
    Keypoints3DAnnotation,
    Keypoints3DAnnotations,
    Keypoints3DAnnotationsMetadata,
)
from kineo.pipeline.stages.nlf.skeleton_keypoints_format import (
    SMPL_22_KEYPOINTS_FORMAT,
    SMPL_24_KEYPOINTS_FORMAT,
)


def test_smpl_22_format_is_prefix_of_smpl_24():
    assert SMPL_22_KEYPOINTS_FORMAT.n_keypoints == 22
    assert (
        SMPL_22_KEYPOINTS_FORMAT.keypoints_names
        == SMPL_24_KEYPOINTS_FORMAT.keypoints_names[:22]
    )


def test_convert_smpl_24_to_smpl_22_drops_hand_joints():
    xyz = torch.arange(24 * 3, dtype=torch.float32).reshape(24, 3)
    ann = Keypoints3DAnnotations(
        metadata=Keypoints3DAnnotationsMetadata(
            formats=[SMPL_24_KEYPOINTS_FORMAT]
        ),
        annotations=[
            Keypoints3DAnnotation(
                frame_idx=0,
                subject_id="aria01",
                xyz=xyz,
                scores=torch.ones(24, dtype=torch.float32),
                format="smpl_24",
            )
        ],
    )
    out = ann.convert_to_format(SMPL_22_KEYPOINTS_FORMAT)
    out_ann = out.first_or_default()
    assert out_ann.format == "smpl_22"
    assert out_ann.xyz.shape == (22, 3)
    assert torch.equal(out_ann.xyz, xyz[:22])
