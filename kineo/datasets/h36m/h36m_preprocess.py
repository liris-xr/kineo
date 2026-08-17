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
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import orjson
import roma
import roma.euler
import torch
from spacepy import pycdf
from tqdm import tqdm

from kineo.annotations import KeypointsFormat
from kineo.annotations.keypoints_3d import (
    Keypoints3DAnnotation,
    Keypoints3DAnnotations,
    Keypoints3DAnnotationsMetadata,
)
from kineo.annotations.camera_extrinsics import (
    CameraExtrinsicsAnnotation,
    CameraExtrinsicsAnnotations,
    CameraExtrinsicsAnnotationsMetadata,
)
from kineo.annotations.camera_intrinsics import (
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotations,
    CameraIntrinsicsAnnotationsMetadata,
    CameraDistortionModel,
)

from kineo.annotations.keypoints_utils import generate_kps2d_from_kps3d_and_cameras
from kineo.annotations.bboxes_utils import generate_bboxes2d_from_kps2d
from kineo.datasets import annotations_io
from kineo.geometry.camera import inverse_Rt
from kineo.io.file import extract_tar_with_progress

BLACKLISTED_SEQUENCES = [
    ("S9", "Greeting"),
    ("S9", "SittingDown"),
    ("S9", "Waiting"),
]

DOWNLOADED_SUBJECTS = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

PROTOCOL1_TRAIN_SUBJECTS = ["S1", "S5", "S6", "S7", "S8"]
PROTOCOL1_VAL_SUBJECTS = ["S9", "S11"]
PROTOCOL1_SELECTED_CAMERAS = ["54138969", "55011271", "58860488", "60457274"]

KEYPOINTS_FORMAT = KeypointsFormat.from_mmpose_dataset("h36m")
# Mapping from the original H3.6M format to MMPose's documented format
# References:
# - https://github.com/qxcv/pose-prediction/blob/master/H36M-NOTES.md#joint-information
# - https://mmpose.readthedocs.io/en/latest/dataset_zoo/3d_body_keypoint.html#human3-6m
H36M_JOINTS_MAPPING = [11, 1, 2, 3, 6, 7, 8, 12, 24, 14, 15, 17, 18, 19, 25, 26, 27]


def _get_annotations_rel_filepath(
    subject_name: str, subaction_name: str, split: str
) -> str:
    return f"annotations/protocol1/{split}/{subject_name}_{subaction_name}.json"


def _get_video_rel_filepath(
    subject_name: str, subaction_name: str, camera_name: str
) -> str:
    return f"{subject_name}/Videos/{subaction_name}.{camera_name}.mp4"


def _serializer_fallback(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, range):
        return {
            "type": "range",
            "start": obj.start,
            "stop": obj.stop,
            "step": obj.step,
        }
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def preprocess_h36m(
    dataset_dir: str,
    skip_extract: bool = False,
    skip_annotations_generation: bool = False,
):
    """
    Preprocess the Human3.6M dataset for use in Kineo.

    This function extracts and processes the Human3.6M dataset, creating annotations in JSON format following Protocol 1.
    It also creates train/val splits for Protocol 1.

    Args:
        raw_dataset_dir: Path to the directory containing the downloaded H3.6M dataset files
        process_dataset_dir: Path where the processed data will be saved
        skip_extract: If True, skip the extraction of archive files (useful if already extracted)
    """

    if not skip_extract:
        _extract_data(dataset_dir)

    actions = _parse_actions_metadata(dataset_dir, dataset_dir)
    _generate_protocol1_annotations(dataset_dir, actions, skip_annotations_generation)


def _generate_protocol1_annotations(
    dataset_dir: str,
    actions: list[dict[str, Any]],
    skip_annotations_generation: bool = False,
):
    pbar = tqdm(total=len(actions), desc="Generating annotations")

    sequences_infos = []

    for action in actions:
        subject_name = action["subject_name"]
        action_name = action["action_name"]
        subaction_name = action["subaction_name"]

        if (
            subject_name not in PROTOCOL1_VAL_SUBJECTS
            and subject_name not in PROTOCOL1_TRAIN_SUBJECTS
        ):
            pbar.update(1)
            continue

        pbar.set_postfix_str(f"{subject_name} {subaction_name}")

        subject_dir_relpath = os.path.join(subject_name)
        annotations_relpath = os.path.join(
            subject_name, "annotations_protocol1", subaction_name
        )
        annotations_abspath = os.path.join(dataset_dir, annotations_relpath)

        os.makedirs(annotations_abspath, exist_ok=True)

        cameras_names = action["cameras_names"]
        cameras_n_frames = action["cameras_n_frames"]
        cameras_intrinsics = action["cameras_intrinsics"]
        cameras_distortion_coefficients = action["cameras_distortion_coefficients"]
        cameras_extrinsics = action["cameras_extrinsics"]
        cameras_resolution_hw = action["cameras_resolution_hw"]

        n_cameras = len(cameras_names)

        keypoints_3d_world = _get_keypoints_3d(
            dataset_dir, subject_name, subaction_name
        )

        # Downsample to 10Hz
        n_frames = min(keypoints_3d_world.shape[0], min(cameras_n_frames)) // 5
        keypoints_3d_world = keypoints_3d_world[::5][:n_frames]
        n_frames = keypoints_3d_world.shape[0]

        # Only used when generation is skipped: list the annotation files
        # already on disk, whatever a past run happened to write.
        annotations_relpaths = {
            key: Path(os.path.join(annotations_relpath, filename)).as_posix()
            for key, filename in annotations_io.ANNOTATION_FILENAMES.items()
            if os.path.exists(os.path.join(annotations_abspath, filename))
        }

        camera_extrinsics_annotations = CameraExtrinsicsAnnotations(
            metadata=CameraExtrinsicsAnnotationsMetadata(),
            annotations=[
                CameraExtrinsicsAnnotation(
                    view_id=cameras_names[camera_idx],
                    frame_idx=0,
                    R=cameras_extrinsics[camera_idx][:3, :3],
                    t=cameras_extrinsics[camera_idx][:3, 3],
                )
                for camera_idx in range(n_cameras)
            ],
        )

        if not skip_annotations_generation:
            kps3d_annotations = Keypoints3DAnnotations(
                metadata=Keypoints3DAnnotationsMetadata(
                    formats=[
                        KEYPOINTS_FORMAT,
                    ]
                ),
                annotations=[
                    Keypoints3DAnnotation(
                        frame_idx=frame_idx,
                        subject_id=subject_name,
                        xyz=keypoints_3d_world[frame_idx],
                        scores=torch.ones_like(keypoints_3d_world[frame_idx][..., 0]),
                        annotated=torch.ones_like(
                            keypoints_3d_world[frame_idx][..., 0],
                            dtype=torch.bool,
                        ),
                        format=KEYPOINTS_FORMAT.name,
                    )
                    for frame_idx in range(n_frames)
                ],
            )

            camera_intrinsics_annotations = CameraIntrinsicsAnnotations(
                metadata=CameraIntrinsicsAnnotationsMetadata(),
                annotations=[
                    CameraIntrinsicsAnnotation(
                        view_id=cameras_names[camera_idx],
                        frame_idx=0,
                        K=cameras_intrinsics[camera_idx],
                        distortion_coefficients=cameras_distortion_coefficients[
                            camera_idx
                        ],
                        distortion_model=CameraDistortionModel.BROWN_CONRADY,
                        resolution_hw=cameras_resolution_hw[camera_idx],
                    )
                    for camera_idx in range(n_cameras)
                ],
            )

            kps2d_annotations = generate_kps2d_from_kps3d_and_cameras(
                kps_annotations=kps3d_annotations,
                cam_intrinsics=camera_intrinsics_annotations,
                cam_extrinsics=camera_extrinsics_annotations,
            )

            bboxes2d_annotations = generate_bboxes2d_from_kps2d(
                kps2d_annotations=kps2d_annotations,
                views_resolution_hw={
                    camera_name: resolution_hw
                    for camera_name, resolution_hw in zip(
                        cameras_names, cameras_resolution_hw
                    )
                },
                category_id=0,
            )

            annotations_relpaths = annotations_io.write_sequence_annotations(
                dataset_dir=dataset_dir,
                annotations_reldir=annotations_relpath,
                annotations={
                    "keypoints_2d": kps2d_annotations,
                    "keypoints_3d": kps3d_annotations,
                    "bboxes_2d": bboxes2d_annotations,
                    "cameras_intrinsics": camera_intrinsics_annotations,
                    "cameras_extrinsics": camera_extrinsics_annotations,
                },
            )

        split = "train" if subject_name in PROTOCOL1_TRAIN_SUBJECTS else "val"

        sequences_infos.append(
            {
                "sequence_name": f"{subject_name}_{subaction_name}",
                "action_name": action_name,
                "subject_name": subject_name,
                "subaction_name": subaction_name,
                "split": split,
                "annotations": annotations_relpaths,
                "views": {
                    view_id: {
                        "video_path": (Path(subject_dir_relpath) / "Videos" / f"{subaction_name}.{view_id}.mp4").as_posix(),
                        "selected_frames": range(0, n_frames * 5, 5),
                        "is_static": camera_extrinsics_annotations.is_view_static(
                            view_id
                        ),
                    }
                    for view_id in cameras_names
                },
            }
        )

        pbar.update(1)

    sequences_annotations_filename = os.path.join(
        dataset_dir,
        "h36m_protocol1_sequences.json",
    )

    with open(sequences_annotations_filename, "wb") as f:
        f.write(
            orjson.dumps(
                sequences_infos,
                default=_serializer_fallback,
                option=orjson.OPT_INDENT_2,
            )
        )
    print(f'Saved sequences annotations to "{sequences_annotations_filename}"')

    pbar.close()


def _extract_data(raw_dataset_dir: str):
    for subject_name in tqdm(DOWNLOADED_SUBJECTS, desc="Extracting data"):
        videos_tar_path = os.path.join(raw_dataset_dir, f"Videos_{subject_name}.tgz")
        meshes_tar_path = os.path.join(raw_dataset_dir, f"Meshes_{subject_name}.tgz")
        poses_tar_path = os.path.join(
            raw_dataset_dir, f"Poses_D3_Positions_{subject_name}.tgz"
        )

        extract_tar_with_progress(
            videos_tar_path,
            raw_dataset_dir,
            desc=f"Extracting '{subject_name}' videos",
            leave=False,
        )

        extract_tar_with_progress(
            poses_tar_path,
            raw_dataset_dir,
            desc=f"Extracting '{subject_name}' poses",
            leave=False,
        )
        extract_tar_with_progress(
            meshes_tar_path,
            raw_dataset_dir,
            desc=f"Extracting '{subject_name}' meshes",
            leave=False,
        )


def _parse_actions_metadata(
    raw_dataset_dir: str, processed_dataset_dir: str
) -> list[dict[str, Any]]:
    code_zip = os.path.join(raw_dataset_dir, "code-v1.2.zip")

    if not os.path.exists(code_zip):
        raise FileNotFoundError(f"Code zip file not found: {code_zip}")

    zipfile = ZipFile(code_zip, "r")

    with zipfile.open("Release-v1.2/metadata.xml") as f:
        root = ET.fromstring(f.read())

    mapping = list(root.find("mapping"))

    camera_names = [elem.text for elem in root.find("dbcameras/index2id")]
    # Subject names from S1 to S11
    subjects_name = [elem.text for elem in mapping[0]][2:]

    # Parse all the actions/subactions
    actions: list[dict[str, Any]] = []
    action_names = [elem.text for elem in root.find("actionnames")]

    # Skip header and first two rows (_ALL actions)
    for action_row in mapping[3:33]:
        action_id = int(action_row[0].text) - 1
        action_name = action_names[action_id]

        for subject_idx, action_column in enumerate(action_row[2:]):
            subject_name = subjects_name[subject_idx]
            subaction_name = action_column.text

            actions.append(
                {
                    "subject_name": subject_name,
                    "action_name": action_name,
                    "subaction_name": subaction_name,
                    "cameras_names": camera_names,
                }
            )

    n_cameras = len(camera_names)
    n_subjects = len(subjects_name)

    # The layout of the metadata is as follows:
    # - The first 264 elements are the camera extrinsics organized as [-x, y, -z, tx, ty, tz] * n_cameras * n_subjects
    # - The next 36 elements are the camera intrinsics organized as [fx, fy, cx, cy, k1, k2, p1, p2, k3] * n_cameras
    cameras_params = torch.from_numpy(
        np.fromstring(root.find("w0").text.strip("[]"), sep=" ")
    ).to(torch.float32)
    cameras_extrinsics_params = cameras_params[:264].reshape(n_cameras, n_subjects, 6)
    cameras_intrinsics_params = cameras_params[264:].reshape(n_cameras, 9)

    cameras_intrinsics = torch.zeros(n_cameras, 3, 3)
    cameras_intrinsics[..., 0, 0] = cameras_intrinsics_params[..., 0]  # fx
    cameras_intrinsics[..., 1, 1] = cameras_intrinsics_params[..., 1]  # fy
    cameras_intrinsics[..., 0, 2] = cameras_intrinsics_params[..., 2]  # cx
    cameras_intrinsics[..., 1, 2] = cameras_intrinsics_params[..., 3]  # cy
    cameras_intrinsics[..., 2, 2] = 1.0
    cameras_distortion_coefficients = torch.zeros(n_cameras, 5)
    cameras_distortion_coefficients[..., 0] = cameras_intrinsics_params[..., 4]  # k1
    cameras_distortion_coefficients[..., 1] = cameras_intrinsics_params[..., 5]  # k2
    cameras_distortion_coefficients[..., 2] = cameras_intrinsics_params[..., 6]  # p1
    cameras_distortion_coefficients[..., 3] = cameras_intrinsics_params[..., 7]  # p2
    cameras_distortion_coefficients[..., 4] = cameras_intrinsics_params[..., 8]  # k3

    # Each subject has different extrinsics
    cameras_extrinsics: dict[str, torch.Tensor] = {}
    for subject_idx in range(n_subjects):
        subject_name = subjects_name[subject_idx]

        # Camera parameters are stored as cam2world in left-handed coordinates
        # we negate angles to flip coordinate system (left-handed to right-handed coordinate system)
        angles = -cameras_extrinsics_params[..., subject_idx, :3].clone()
        cam_rot = roma.euler.euler_to_rotmat(convention="xyz", angles=angles)
        cam_pose = cameras_extrinsics_params[..., subject_idx, 3:6].unsqueeze(-1)
        cam_pose = cam_pose / 1000.0  # Convert from mm to m

        cam2world = torch.cat((cam_rot, cam_pose), dim=-1)
        world2cam = inverse_Rt(cam2world)

        cameras_extrinsics[subject_name] = world2cam

    # Remove blacklisted sequences and only keep S1, S5, S6, S7, S8, S9, S11
    actions = [
        a
        for a in actions
        if (a["subject_name"], a["action_name"]) not in BLACKLISTED_SEQUENCES
        and a["subject_name"] in DOWNLOADED_SUBJECTS
    ]

    for action in actions:
        subject_name = action["subject_name"]
        subaction_name = action["subaction_name"]
        cameras_names = action["cameras_names"]

        cameras_info = [
            _get_video_info(
                os.path.join(
                    raw_dataset_dir,
                    subject_name,
                    "Videos",
                    f"{subaction_name}.{camera_name}.mp4",
                )
            )
            for camera_name in cameras_names
        ]

        action["cameras_intrinsics"] = cameras_intrinsics
        action["cameras_distortion_coefficients"] = cameras_distortion_coefficients
        action["cameras_extrinsics"] = cameras_extrinsics[subject_name]
        action["cameras_resolution_hw"] = [
            (camera_info["height"], camera_info["width"])
            for camera_info in cameras_info
        ]
        action["cameras_n_frames"] = [
            camera_info["n_frames"] for camera_info in cameras_info
        ]

    return actions


def _get_video_info(
    video_path: str,
) -> dict[str, Any]:
    assert os.path.exists(video_path), f"Video {video_path} does not exist"
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {
        "width": width,
        "height": height,
        "n_frames": n_frames,
    }


def _get_subject_height(subject_data_dir: str) -> float:
    mesh_fp = os.path.join(subject_data_dir, "Scan/Scan.obj")

    # read all the lines of the file starting with 'v '
    with open(mesh_fp, "r") as f:
        line = f.readline()

        while not line.startswith("v "):
            line = f.readline()

        # The scan are oriented with the x axis as the up axis
        min_x = float("inf")
        max_x = float("-inf")

        while line.startswith("v "):
            line = line.replace("v ", "")
            x = float(line.split()[0])
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            line = f.readline()

    return max_x - min_x


def _get_keypoints_3d(
    dataset_dir: str, subject_name: str, subaction_name: str
) -> torch.Tensor:
    poses_cdf_fp = os.path.join(
        dataset_dir,
        subject_name,
        "MyPoseFeatures/D3_Positions/",
        f"{subaction_name}.cdf",
    )

    keypoints_3d_world = pycdf.CDF(poses_cdf_fp)["Pose"][0]
    keypoints_3d_world = torch.from_numpy(keypoints_3d_world).to(torch.float32)
    n_frames = keypoints_3d_world.shape[0]

    # Convert from mm to meters
    keypoints_3d_world = keypoints_3d_world.reshape(n_frames, -1, 3) / 1000.0
    # Map original H3.6M joints to MMPose joints
    keypoints_3d_world = keypoints_3d_world[:, H36M_JOINTS_MAPPING]
    return keypoints_3d_world.reshape(n_frames, -1, 3)
