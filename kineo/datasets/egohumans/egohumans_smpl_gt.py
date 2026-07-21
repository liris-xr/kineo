# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import glob
import os

import numpy as np
import torch

from kineo.annotations.keypoints_3d import Keypoints3DAnnotation
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotationsMetadata
from kineo.pipeline.stages.nlf.skeleton_keypoints_format import (
    SMPL_22_KEYPOINTS_FORMAT,
)


def load_egohumans_smpl_keypoints_3d(
    smpl_dir: str,
    valid_subject_ids: list[str] | None = None,
) -> Keypoints3DAnnotations:
    """Load EgoHumans GT SMPL joints as smpl_22 3D keypoint annotations.

    Reads the per-frame SMPL fits in `smpl_dir` (`<frame>.npy`, each a dict of
    subject id -> {"joints": (45, 3), ...}), keeps the first 22 SMPL kinematic
    joints (HSfM parity), and returns them in world coordinates.

    Args:
        smpl_dir: Directory with EgoHumans `processed_data/smpl/*.npy` files.
        valid_subject_ids: If given, only these subjects are kept.

    Returns:
        Keypoints3DAnnotations in SMPL_22_KEYPOINTS_FORMAT. Frame index is the
        npy basename minus one.

    Raises:
        FileNotFoundError: If no npy files are found in smpl_dir.
    """
    smpl_files = sorted(glob.glob(os.path.join(smpl_dir, "*.npy")))
    if not smpl_files:
        raise FileNotFoundError(f"No SMPL files found in {smpl_dir}")

    annotations: list[Keypoints3DAnnotation] = []
    for smpl_file in smpl_files:
        frame_idx = int(os.path.basename(smpl_file).split(".")[0]) - 1
        data = np.load(smpl_file, allow_pickle=True).item()

        for subject_id, subject_data in data.items():
            if valid_subject_ids is not None and subject_id not in valid_subject_ids:
                continue
            joints = torch.from_numpy(
                np.asarray(subject_data["joints"][:22], dtype=np.float32)
            )
            annotations.append(
                Keypoints3DAnnotation(
                    frame_idx=frame_idx,
                    subject_id=subject_id,
                    xyz=joints,
                    annotated=torch.ones(22, dtype=torch.bool),
                    scores=torch.ones(22, dtype=torch.float32),
                    format=SMPL_22_KEYPOINTS_FORMAT.name,
                )
            )

    return Keypoints3DAnnotations(
        metadata=Keypoints3DAnnotationsMetadata(
            formats=[SMPL_22_KEYPOINTS_FORMAT]
        ),
        annotations=annotations,
    )
