# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import os
import orjson
import glob
import torch
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.bboxes_2d import BBox2DAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.visualization.viz_3d import show_keypoints_and_cameras
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.io.frame_sequence_loader import ImagesLoader

from kineo.geometry.transformations import inverse_Rt

FPS = 20

SEQUENCES = [
    {
        "directory": "01_tagging",
        "name": "tagging",
        "n_subsequences": 14,
    },
    {
        "directory": "02_lego",
        "name": "legoassemble",
        "n_subsequences": 6,
    },
    {
        "directory": "03_fencing",
        "name": "fencing",
        "n_subsequences": 14,
    },
    {
        "directory": "04_basketball",
        "name": "basketball",
        "n_subsequences": 14,
        "excluded_subsequences": [9],  # 010_basketball is missing processed_data
    },
    {
        "directory": "05_volleyball",
        "name": "volleyball",
        "n_subsequences": 11,
    },
    {
        "directory": "06_badminton",
        "name": "badminton",
        "n_subsequences": 61,
    },
    {
        "directory": "07_tennis",
        "name": "tennis",
        "n_subsequences": 13,
    },
]


def visualize_egohumans(dataset_dir: str):
    for sequence in SEQUENCES:
        sequence_dir = sequence["directory"]
        sequence_name = sequence["name"]
        n_subsequences = sequence["n_subsequences"]

        for subsequence_idx in range(n_subsequences):
            if sequence_name != "fencing" or subsequence_idx != 0:
                continue

            print(f"Visualizing {sequence_name} {subsequence_idx + 1:03d}")

            subsequence_dir = os.path.join(
                sequence_dir, f"{subsequence_idx + 1:03d}_{sequence_name}"
            )
            subsequence_abs_dir = os.path.join(dataset_dir, subsequence_dir)

            kps_2d_annotations_relpath = os.path.join(
                subsequence_dir, "annotations", "keypoints_2d.json"
            )
            kps_2d_annotations_abspath = os.path.join(
                dataset_dir, kps_2d_annotations_relpath
            )

            with open(kps_2d_annotations_abspath, "rb") as f:
                kps_2d_annotations = Keypoints2DAnnotations.from_dict(
                    orjson.loads(f.read())
                )

            bboxes_2d_annotations_relpath = os.path.join(
                subsequence_dir, "annotations", "bboxes_2d.json"
            )
            bboxes_2d_annotations_abspath = os.path.join(
                dataset_dir, bboxes_2d_annotations_relpath
            )

            with open(bboxes_2d_annotations_abspath, "rb") as f:
                bboxes_2d_annotations = BBox2DAnnotations.from_dict(
                    orjson.loads(f.read())
                )

            kps_3d_annotations_relpath = os.path.join(
                subsequence_dir, "annotations", "keypoints_3d.json"
            )
            kps_3d_annotations_abspath = os.path.join(
                dataset_dir, kps_3d_annotations_relpath
            )
            with open(kps_3d_annotations_abspath, "rb") as f:
                kps_3d_annotations = Keypoints3DAnnotations.from_dict(
                    orjson.loads(f.read())
                )

            cam_intrinsics_annotations_relpath = os.path.join(
                subsequence_dir, "annotations", "cameras_intrinsics.json"
            )
            cam_intrinsics_annotations_abspath = os.path.join(
                dataset_dir, cam_intrinsics_annotations_relpath
            )
            with open(cam_intrinsics_annotations_abspath, "rb") as f:
                cam_intrinsics_annotations = CameraIntrinsicsAnnotations.from_dict(
                    orjson.loads(f.read())
                )

            cam_extrinsics_annotations_relpath = os.path.join(
                subsequence_dir, "annotations", "cameras_extrinsics.json"
            )
            cam_extrinsics_annotations_abspath = os.path.join(
                dataset_dir, cam_extrinsics_annotations_relpath
            )
            with open(cam_extrinsics_annotations_abspath, "rb") as f:
                cam_extrinsics_annotations = CameraExtrinsicsAnnotations.from_dict(
                    orjson.loads(f.read())
                )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            views = []

            for view_id in cam_extrinsics_annotations.views_ids:
                img_paths = sorted(
                    glob.glob(
                        os.path.join(
                            subsequence_abs_dir, "exo", view_id, "images", "*.jpg"
                        )
                    )
                )
                n_frames = len(img_paths)
                assert n_frames > 0, f"No images found for view {view_id}"
                frame_timestamps_local = torch.arange(n_frames) / FPS

                views.append(
                    ViewInput(
                        name=view_id,
                        frame_loader=ImagesLoader(
                            img_paths=img_paths,
                            frame_timestamps_local=frame_timestamps_local,
                            device=device,
                        ),
                        audio_loader=None,
                    )
                )

            # for view_id in cam_extrinsics_annotations.views_ids:
            #     show_bboxes_and_keypoints(
            #         views=[v for v in views if v["name"] == view_id],
            #         bboxes_2d=bboxes_2d_annotations.filter_by_view_id(view_id),
            #         keypoints_2d=kps_2d_annotations.filter_by_view_id(view_id),
            #         fps=FPS,
            #     )

            pos = {}

            for view_idx, view_id in enumerate(cam_extrinsics_annotations.views_ids):
                camera_extrinsics = (
                    cam_extrinsics_annotations.filter_by_view_id(view_id)
                    .first_or_default()
                    .Rt
                )
                cam_pose = inverse_Rt(camera_extrinsics)[:3, 3]
                pos[view_idx] = tuple(cam_pose[:2].tolist())

            print(pos)

            show_keypoints_and_cameras(
                keypoints_3d=kps_3d_annotations,
                camera_extrinsics=cam_extrinsics_annotations,
                camera_intrinsics=cam_intrinsics_annotations,
                fps=FPS,
            )
