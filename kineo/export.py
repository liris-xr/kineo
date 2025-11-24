# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import csv
from kineo.annotations import Keypoints3DAnnotations, KeypointsFormat
import os


def export_keypoints_to_csv(
    csv_path: str,
    kps_3d: Keypoints3DAnnotations,
    fps: float,
):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time (s)", "joint_name", "x (m)", "y (m)", "z (m)", "confidence"]
        )

        first_subject_id = list(kps_3d.subjects_ids)[0]
        kps_3d = kps_3d.filter_by_subject_id(first_subject_id)
        kps_format: KeypointsFormat = next(
            f
            for f in kps_3d.metadata.formats
            if f.name == kps_3d.first_or_default().format
        )
        kps_names = kps_format.keypoints_names

        for annot in kps_3d:
            scores = annot.scores.cpu().numpy()
            frame_idx = annot.frame_idx
            timestamp = frame_idx / fps
            xyz = annot.xyz.cpu().numpy()
            for joint_id, joint_name in enumerate(kps_names):
                x, y, z = xyz[joint_id]
                score = scores[joint_id]
                writer.writerow([timestamp, joint_name, x, y, z, score])
