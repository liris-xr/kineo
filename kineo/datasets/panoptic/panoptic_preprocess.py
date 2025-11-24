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
import torch
import cv2
from typing import Any
from collections import namedtuple
import warnings

try:
    from orjson import loads as json_loads
except ImportError:
    warnings.warn("orjson is not installed, using json instead. This will be slower.")
    from json import loads as json_loads

from copy import deepcopy
import glob
import tarfile
from kineo.datasets.keypoints_sequence_data import (
    KeypointsSequenceMultiviewData,
    KeypointsSequenceMonocularData,
    KeypointsMetadata,
)
from kineo.datasets.preprocess_utils import (
    compute_keypoints_3d_cam,
    compute_keypoints_2d,
    compute_bboxes_xywh,
    standardize_keypoints,
    suppress_overlapping_bboxes,
)
from kineo.geometry.conversions import (
    RH_NEG_Y_UP_BASIS,
    OPENCV_WORLD_BASIS,
    convert_Rt_basis,
)
from tqdm import tqdm

DOWNLOADED_CAMERAS = ["00_03", "00_06", "00_12", "00_13", "00_23"]

FPS = 30000 / 1001

Sequence = namedtuple(
    "Sequence",
    ["sequence_name", "subsequence_name", "start_frame_idx", "end_frame_idx"],
)

# TODO
# TRAINING_SEQUENCES = [
#     Sequence("160422_ultimatum1", "160422_ultimatum1_01", 0, -1),
#     Sequence("160224_haggling1", "160224_haggling1_01", 0, -1),
#     Sequence("160226_haggling1", "160226_haggling1_01", 0, -1),
#     Sequence("161202_haggling1", "161202_haggling1_01", 0, -1),
#     Sequence("160906_ian1", "160906_ian1_01", 0, -1),
#     Sequence("160906_ian2", "160906_ian2_01", 0, -1),
#     Sequence("160906_ian3", "160906_ian3_01", 0, -1),
#     Sequence("160906_band1", "160906_band1_01", 0, -1),
#     Sequence("160906_band2", "160906_band2_01", 0, -1),
# ]

MULTI_PERSON_TESTING_SEQUENCES = [
    # Multi-person
    Sequence(
        "160906_pizza1",
        "160906_pizza1_01",
        int(12 * FPS),
        int(42 * FPS),
    ),
    Sequence(
        "160906_pizza1",
        "160906_pizza1_02",
        int((1 * 60 + 1) * FPS),
        -1,
    ),
    Sequence(
        "160422_haggling1",
        "160422_haggling1_01",
        int(15 * FPS),
        int((60 + 17) * FPS),
    ),
    Sequence(
        "160422_haggling1",
        "160422_haggling1_02",
        int((60 + 38) * FPS),
        int((2 * 60 + 29) * FPS),
    ),
    Sequence(
        "160422_haggling1",
        "160422_haggling1_03",
        int((2 * 60 + 53) * FPS),
        int((3 * 60 + 48) * FPS),
    ),
    Sequence(
        "160422_haggling1",
        "160422_haggling1_04",
        int((4 * 60 + 9) * FPS),
        int((4 * 60 + 54) * FPS),
    ),
    Sequence(
        "160422_haggling1",
        "160422_haggling1_05",
        int((5 * 60 + 8) * FPS),
        int((6 * 60 + 1) * FPS),
    ),
    Sequence(
        "160422_haggling1",
        "160422_haggling1_06",
        int((6 * 60 + 22) * FPS),
        int((7 * 60 + 14) * FPS),
    ),
    Sequence("160906_ian5", "160906_ian5", 0, -1),
    Sequence("160906_band4", "160906_band4", 0, -1),
]

# Single-person
SINGLE_PERSON_TESTING_SEQUENCES = [
    ############### 171204_pose1 ###############
    Sequence(
        "171204_pose1", "171204_pose1_01", int(10 * FPS), int((2 * 60 + 23) * FPS)
    ),
    Sequence(
        "171204_pose1",
        "171204_pose1_02",
        int((2 * 60 + 42) * FPS),
        int((4 * 60 + 55) * FPS),
    ),
    Sequence(
        "171204_pose1",
        "171204_pose1_03",
        int((5 * 60 + 23) * FPS),
        int((7 * 60 + 34) * FPS),
    ),
    Sequence(
        "171204_pose1",
        "171204_pose1_04",
        int((7 * 60 + 52) * FPS),
        int((10 * 60 + 5) * FPS),
    ),
    Sequence(
        "171204_pose1",
        "171204_pose1_05",
        int((10 * 60 + 30) * FPS),
        int((12 * 60 + 43) * FPS),
    ),
    Sequence(
        "171204_pose1",
        "171204_pose1_06",
        int((13 * 60 + 8) * FPS),
        int((15 * 60 + 21) * FPS),
    ),
    Sequence(
        "171204_pose1",
        "171204_pose1_06",
        int((13 * 60 + 8) * FPS),
        int((15 * 60 + 21) * FPS),
    ),
    ############### 171204_pose2 ###############
    Sequence(
        "171204_pose2", "171204_pose2_01", int(12 * FPS), int((2 * 60 + 24) * FPS)
    ),
    Sequence(
        "171204_pose2",
        "171204_pose2_02",
        int((2 * 60 + 40) * FPS),
        int((4 * 60 + 53) * FPS),
    ),
    Sequence(
        "171204_pose2",
        "171204_pose2_03",
        int((5 * 60 + 21) * FPS),
        int((7 * 60 + 34) * FPS),
    ),
    Sequence(
        "171204_pose2",
        "171204_pose2_04",
        int((8 * 60 + 5) * FPS),
        int((10 * 60 + 18) * FPS),
    ),
    Sequence(
        "171204_pose2",
        "171204_pose2_05",
        int((10 * 60 + 55) * FPS),
        int((13 * 60 + 8) * FPS),
    ),
    Sequence(
        "171204_pose2",
        "171204_pose2_06",
        int((13 * 60 + 28) * FPS),
        int((15 * 60 + 42) * FPS),
    ),
    Sequence(
        "171204_pose2",
        "171204_pose2_07",
        int((16 * 60 + 1) * FPS),
        int((18 * 60 + 14) * FPS),
    ),
    Sequence(
        "171204_pose2",
        "171204_pose2_08",
        int((18 * 60 + 46) * FPS),
        int((21 * 60) * FPS),
    ),
    ############### 171204_pose3 ###############
    Sequence(
        "171204_pose3", "171204_pose3_01", int(14 * FPS), int((2 * 60 + 27) * FPS)
    ),
    Sequence(
        "171204_pose3",
        "171204_pose3_02",
        int((2 * 60 + 57) * FPS),
        int((5 * 60 + 2) * FPS),
    ),
    ############### 171204_pose4 ###############
    Sequence(
        "171204_pose4", "171204_pose4_01", int(11 * FPS), int((2 * 60 + 23) * FPS)
    ),
    Sequence(
        "171204_pose4",
        "171204_pose4_02",
        int((2 * 60 + 40) * FPS),
        int((4 * 60 + 53) * FPS),
    ),
    Sequence(
        "171204_pose4",
        "171204_pose4_03",
        int((5 * 60 + 20) * FPS),
        int((7 * 60 + 24) * FPS),
    ),
    Sequence(
        "171204_pose4",
        "171204_pose4_04",
        int((7 * 60 + 44) * FPS),
        int((9 * 60 + 57) * FPS),
    ),
    Sequence(
        "171204_pose4",
        "171204_pose4_05",
        int((10 * 60 + 21) * FPS),
        int((12 * 60 + 33) * FPS),
    ),
    Sequence(
        "171204_pose4",
        "171204_pose4_06",
        int((12 * 60 + 50) * FPS),
        int((15 * 60 + 2) * FPS),
    ),
    Sequence("171204_pose4", "171204_pose4_07", int((15 * 60 + 30) * FPS), -1),
    ############### 171204_pose5 ###############
    Sequence(
        "171204_pose5", "171204_pose5_01", int(12 * FPS), int((2 * 60 + 23) * FPS)
    ),
    Sequence(
        "171204_pose5",
        "171204_pose5_02",
        int((2 * 60 + 45) * FPS),
        int((4 * 60 + 57) * FPS),
    ),
    Sequence(
        "171204_pose5",
        "171204_pose5_03",
        int((5 * 60 + 16) * FPS),
        int((7 * 60 + 28) * FPS),
    ),
    Sequence(
        "171204_pose5",
        "171204_pose5_04",
        int((7 * 60 + 51) * FPS),
        int((10 * 60 + 2) * FPS),
    ),
    Sequence(
        "171204_pose5",
        "171204_pose5_05",
        int((10 * 60 + 22) * FPS),
        int((12 * 60 + 35) * FPS),
    ),
    Sequence("171204_pose5", "171204_pose5_06", int((13 * 60 + 3) * FPS), -1),
    ############### 171204_pose6 ###############
    Sequence(
        "171204_pose6", "171204_pose6_01", int(18 * FPS), int((2 * 60 + 28) * FPS)
    ),
    Sequence(
        "171204_pose6",
        "171204_pose6_02",
        int((2 * 60 + 51) * FPS),
        int((5 * 60 + 3) * FPS),
    ),
    Sequence(
        "171204_pose6",
        "171204_pose6_03",
        int((5 * 60 + 28) * FPS),
        int((7 * 60 + 39) * FPS),
    ),
    Sequence(
        "171204_pose6",
        "171204_pose6_04",
        int((7 * 60 + 59) * FPS),
        int((10 * 60 + 11) * FPS),
    ),
    Sequence(
        "171204_pose6",
        "171204_pose6_05",
        int((10 * 60 + 33) * FPS),
        int((12 * 60 + 45) * FPS),
    ),
    ############### 171026_pose1 ###############
    Sequence("171026_pose1", "171026_pose1_01", int(9 * FPS), int((4 * 60 + 5) * FPS)),
    Sequence(
        "171026_pose1",
        "171026_pose1_02",
        int((4 * 60 + 23) * FPS),
        int((8 * 60 + 17) * FPS),
    ),
    Sequence(
        "171026_pose1",
        "171026_pose1_03",
        int((8 * 60 + 34) * FPS),
        int((12 * 60 + 26) * FPS),
    ),
    ############### 171026_pose2 ###############
    Sequence(
        "171026_pose2", "171026_pose2_01", int(18 * FPS), int((4 * 60 + 10) * FPS)
    ),
    Sequence(
        "171026_pose2",
        "171026_pose2_02",
        int((4 * 60 + 24) * FPS),
        int((8 * 60 + 19) * FPS),
    ),
    ############### 171026_pose3 ###############
    Sequence("171026_pose3", "171026_pose3_01", int(10 * FPS), int((4 * 60 + 3) * FPS)),
]
# Follows the format of CMU Panoptic as described in MMPose's documentation:
# https://mmpose.readthedocs.io/en/latest/dataset_zoo/3d_body_keypoint.html#cmu-panoptic
PANOPTIC_BODY_KEYPOINTS_METADATA = KeypointsMetadata.from_mmpose_dataset(
    "panoptic_body3d"
)


def _save_sequences(
    all_sequences: list[KeypointsSequenceMultiviewData],
    sequences_info: list[Sequence],
    filepath: str,
):
    saved_sequences = []

    for seq_info in sequences_info:
        matching_seq = next(
            (s for s in all_sequences if s.sequence_name == seq_info.sequence_name),
            None,
        )
        if matching_seq is None:
            raise ValueError(f"Sequence {seq_info.sequence_name} not found.")

        n_frames = len(matching_seq.gt_frame_timestamps_world)

        subsequence_name = seq_info.subsequence_name
        start_frame_idx = seq_info.start_frame_idx
        end_frame_idx = seq_info.end_frame_idx

        if end_frame_idx == -1:
            end_frame_idx = n_frames - 1

        subseq = deepcopy(matching_seq)
        subseq.sequence_name = subsequence_name
        subseq.gt_frame_timestamps_world = subseq.gt_frame_timestamps_world[
            start_frame_idx : end_frame_idx + 1
        ]
        subseq.gt_keypoints_3d_world = subseq.gt_keypoints_3d_world[
            start_frame_idx : end_frame_idx + 1
        ]
        subseq.gt_keypoints_3d_world_scores = subseq.gt_keypoints_3d_world_scores[
            start_frame_idx : end_frame_idx + 1
        ]

        for subseq_view in subseq.views:
            subseq_view._video_frames_indices = subseq_view._video_frames_indices[
                start_frame_idx : end_frame_idx + 1
            ]
            subseq_view.frame_timestamps_local = subseq_view.frame_timestamps_local[
                start_frame_idx : end_frame_idx + 1
            ]
            subseq_view.gt_bboxes_xywh = subseq_view.gt_bboxes_xywh[
                start_frame_idx : end_frame_idx + 1
            ]
            subseq_view.gt_keypoints_2d = subseq_view.gt_keypoints_2d[
                start_frame_idx : end_frame_idx + 1
            ]
            subseq_view.gt_keypoints_2d_scores = subseq_view.gt_keypoints_2d_scores[
                start_frame_idx : end_frame_idx + 1
            ]
            subseq_view.gt_keypoints_3d_cam = subseq_view.gt_keypoints_3d_cam[
                start_frame_idx : end_frame_idx + 1
            ]
            subseq_view.gt_keypoints_3d_cam_scores = (
                subseq_view.gt_keypoints_3d_cam_scores[
                    start_frame_idx : end_frame_idx + 1
                ]
            )

        subseq.check_validity()
        saved_sequences.append(subseq)

    print(f"Saving {len(saved_sequences)} sequences to {filepath}...")
    with open(filepath, "wb") as f:
        torch.save(saved_sequences, f)
    print(f"Saved {len(saved_sequences)} sequences to {filepath}")


def preprocess_panoptic(
    dataset_dir: str,
    skip_extract: bool = False,
):
    all_sequences = _parse_sequences(
        dataset_dir=dataset_dir,
        skip_extract=skip_extract,
    )

    _save_sequences(
        all_sequences=all_sequences,
        sequences_info=MULTI_PERSON_TESTING_SEQUENCES,
        filepath=os.path.join(dataset_dir, "cmu_panoptic_testing_multiperson.pt"),
    )

    _save_sequences(
        all_sequences=all_sequences,
        sequences_info=SINGLE_PERSON_TESTING_SEQUENCES,
        filepath=os.path.join(dataset_dir, "cmu_panoptic_testing_singleperson.pt"),
    )


def _parse_sequences(
    dataset_dir: str,
    skip_extract: bool = False,
) -> list[KeypointsSequenceMultiviewData]:
    # glob all sequences in the dataset_dir, note that the sequences should be named like "160422_ultimatum1"
    all_sequences_dir = glob.glob(os.path.join(dataset_dir, "[0-9]*_*"))
    all_sequences_dir = [d for d in all_sequences_dir if os.path.isdir(d)]

    sequences: list[KeypointsSequenceMultiviewData] = []

    pbar = tqdm(
        total=len(all_sequences_dir),
        desc="Parsing sequences",
        unit="sequence",
        leave=False,
    )

    for sequence_dir in all_sequences_dir:
        sequence_name = os.path.basename(sequence_dir)

        pbar_prefix = f"Parsing {sequence_name}"

        calibration_file = os.path.join(
            sequence_dir, f"calibration_{sequence_name}.json"
        )

        if not os.path.exists(calibration_file):
            warnings.warn(
                f"Calibration file {calibration_file} not found for sequence {sequence_name}, skipping..."
            )
            continue

        if not skip_extract:
            pbar.set_description(f"{pbar_prefix} | Extracting poses")
            _extract_body_keypoints(dataset_dir, sequence_name)

        pbar.set_description(f"{pbar_prefix} | Parsing cameras")
        cameras = _get_cameras_from_file(calibration_file)

        pbar.set_description(f"{pbar_prefix} | Parsing body keypoints")

        (
            body_kps_start_frame_idx,
            body_kps_end_frame_idx,
            subjects_body_kps,
            subjects_body_kps_scores,
        ) = _get_body_keypoints_from_files(dataset_dir, sequence_name)

        has_body_kps = subjects_body_kps is not None

        all_subjects_ids = list(
            set(subjects_body_kps.keys()) if has_body_kps else set()
        )

        n_subjects = len(all_subjects_ids)

        if n_subjects == 0:
            warnings.warn(f"No subjects found in sequence {sequence_name}, skipping...")
            continue

        n_frames = body_kps_end_frame_idx + 1 if has_body_kps else 0

        assert n_frames > 0, f"Expected at least one frame in sequence {sequence_name}"

        fps = 30000 / 1001
        frame_timestamps_world = torch.arange(0, n_frames) / fps

        body_kps_3d_world = None
        body_kps_3d_world_scores = None

        if has_body_kps:
            body_kps_3d_world = torch.zeros(
                n_frames, n_subjects, PANOPTIC_BODY_KEYPOINTS_METADATA.n_keypoints, 3
            )
            body_kps_3d_world_scores = torch.zeros(
                n_frames, n_subjects, PANOPTIC_BODY_KEYPOINTS_METADATA.n_keypoints
            )

            for subject_id in all_subjects_ids:
                subject_idx = all_subjects_ids.index(subject_id)
                body_kps_3d_world[
                    body_kps_start_frame_idx : body_kps_end_frame_idx + 1,
                    subject_idx,
                ] = subjects_body_kps[subject_id]
                body_kps_3d_world_scores[
                    body_kps_start_frame_idx : body_kps_end_frame_idx + 1,
                    subject_idx,
                ] = subjects_body_kps_scores[subject_id]

        sequence_views: list[KeypointsSequenceMonocularData] = []

        for camera in cameras:
            camera_type = camera["type"]
            camera_name = camera["name"]

            pbar.set_description(f"{pbar_prefix} | Processing camera {camera_name}")

            if camera_type == "hd":
                video_path = os.path.join(
                    sequence_name, f"hdVideos/hd_{camera_name}.mp4"
                )
            else:
                raise NotImplementedError

            video_info = _get_video_info(os.path.join(dataset_dir, video_path))
            resolution_hw = (video_info["height"], video_info["width"])

            # For CMU Panoptic, all timestamps are synchronized, so we can use the same timestamps for all views
            frame_timestamps_local = frame_timestamps_world.clone()
            video_frames_indices = torch.arange(0, n_frames)

            K = camera["K"]
            distortion_coefficients = camera["distortion_coefficients"]
            Rt = camera["Rt"]

            body_kps_3d_cam = None
            body_kps_3d_cam_scores = None
            body_kps_2d = None
            body_bboxes_xywh = None
            body_kps_2d_scores = None

            if has_body_kps:
                body_kps_3d_cam = compute_keypoints_3d_cam(
                    keypoints_3d_world=body_kps_3d_world, Rt=Rt
                )

                body_kps_3d_cam_scores = body_kps_3d_world_scores.clone()

                body_kps_2d = compute_keypoints_2d(
                    keypoints_3d_world=body_kps_3d_world,
                    Rt=Rt,
                    K=K,
                    distortion_coefficients=distortion_coefficients,
                )

                body_kps_2d_scores = torch.where(
                    (body_kps_2d[..., 0] > 0)
                    & (body_kps_2d[..., 1] > 0)
                    & (body_kps_2d[..., 0] < resolution_hw[1])
                    & (body_kps_2d[..., 1] < resolution_hw[0]),
                    body_kps_3d_world_scores,
                    torch.zeros_like(body_kps_3d_world_scores),
                )

                body_bboxes_xywh = compute_bboxes_xywh(
                    poses_2d=body_kps_2d,
                    padding_h=70,
                    padding_v=80,
                    resolution_hw=resolution_hw,
                )

                # bboxes_distances = body_kps_3d_cam[..., 2, 2]
                # keep_idxs = suppress_overlapping_bboxes(
                #     bboxes_xywh=body_bboxes_xywh,
                #     distance=bboxes_distances,
                #     overlap_threshold=0.75,
                # )
                # # Where the bbox was not kept, set the kps 2d score to 0
                # body_kps_2d_scores[~keep_idxs] = 0
                # body_kps_3d_cam_scores[~keep_idxs] = 0

            sequence_view = KeypointsSequenceMonocularData(
                name=camera["type"] + "_" + camera["name"],
                keypoints_metadata=PANOPTIC_BODY_KEYPOINTS_METADATA,
                _video_path=video_path,
                _video_frames_indices=video_frames_indices,
                static_extrinsics=True,
                static_intrinsics=True,
                resolution_hw=resolution_hw,
                frame_timestamps_local=frame_timestamps_local,
                gt_n_subjects=n_subjects,
                # Panoptic cameras are synchronized, so the time offset is 0
                gt_time_offset=0.0,
                gt_bboxes_xywh=body_bboxes_xywh,
                gt_keypoints_2d=body_kps_2d,
                gt_keypoints_2d_scores=body_kps_2d_scores,
                gt_keypoints_3d_cam=body_kps_3d_cam,
                gt_keypoints_3d_cam_scores=body_kps_3d_cam_scores,
                gt_Rt=Rt,
                gt_K=K,
                gt_distortion_coefficients=distortion_coefficients,
            )

            # Preview
            # for frame_idx in [400, 800, 1200]:
            #     import matplotlib.pyplot as plt
            #     sequence_view._base_path = dataset_dir
            #     frame_rgb_np = sequence_view.load_frame_at(frame_idx).permute(1, 2, 0).cpu().numpy()
            #     frame_rgb_np = cv2.cvtColor(frame_rgb_np, cv2.COLOR_RGB2BGR)

            #     for subject_idx, bbox in enumerate(body_bboxes_xywh[frame_idx]):
            #         is_subject_visible = torch.any(body_kps_2d_scores[frame_idx, subject_idx] > 0, dim=-1)
            #         if not is_subject_visible:
            #             continue

            #         x, y, w, h = bbox.int().cpu().numpy()
            #         x1, y1 = x, y
            #         x2, y2 = x + w, y + h
            #         cv2.rectangle(frame_rgb_np, (x1, y1), (x2, y2), (0, 0, 255), 2)

            #     plt.imshow(cv2.cvtColor(frame_rgb_np, cv2.COLOR_BGR2RGB))
            #     plt.show()

            sequence_views.append(sequence_view)

        sequence = KeypointsSequenceMultiviewData(
            sequence_name=sequence_name,
            keypoints_metadata=PANOPTIC_BODY_KEYPOINTS_METADATA,
            gt_n_subjects=n_subjects,
            gt_frame_timestamps_world=frame_timestamps_world,
            gt_keypoints_3d_world=body_kps_3d_world.reshape(
                -1, n_subjects, PANOPTIC_BODY_KEYPOINTS_METADATA.n_keypoints, 3
            ),
            gt_keypoints_3d_world_scores=body_kps_3d_world_scores.reshape(
                -1, n_subjects, PANOPTIC_BODY_KEYPOINTS_METADATA.n_keypoints
            ),
            views=sequence_views,
        )

        sequences.append(sequence)

        pbar.update(1)

    pbar.close()

    return sequences


def _get_video_info(video_path: str) -> dict[str, Any]:
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


def _extract_body_keypoints(dataset_dir: str, sequence_name: str):
    poses_filepath = os.path.join(
        dataset_dir, sequence_name, "hdPose3d_stage1_coco19.tar"
    )
    poses_output_dir = os.path.join(dataset_dir, sequence_name)
    os.makedirs(poses_output_dir, exist_ok=True)

    if not os.path.exists(poses_filepath):
        raise FileNotFoundError(f"Poses file {poses_filepath} not found")

    with tarfile.open(poses_filepath, "r") as tar:
        tar.extractall(poses_output_dir)


def _get_body_keypoints_from_files(
    dataset_dir: str, sequence_name: str
) -> tuple[int, int, dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    body_kps_dir = os.path.join(dataset_dir, sequence_name, "hdPose3d_stage1_coco19")

    if not os.path.exists(body_kps_dir):
        return None, None, None, None

    # It's possible that some poses are nested in a "hd" folder like in "160224_haggling1",
    # in that case, use this folder instead for finding the *.json files
    if os.path.exists(os.path.join(body_kps_dir, "hd")):
        body_kps_dir = os.path.join(body_kps_dir, "hd")

    def _get_frame_number(poses_file: str) -> int:
        return int(os.path.basename(poses_file).split("_")[1].split(".")[0])

    body_kps_files = glob.glob(os.path.join(body_kps_dir, "*.json"))
    body_kps_files = sorted(body_kps_files, key=lambda x: _get_frame_number(x))

    start_frame_idx = _get_frame_number(body_kps_files[0])
    end_frame_idx = _get_frame_number(body_kps_files[-1])

    n_frames = end_frame_idx - start_frame_idx + 1

    subjects_body_kps: dict[int, torch.Tensor] = {}
    subjects_body_kps_scores: dict[int, torch.Tensor] = {}

    pbar = tqdm(
        total=len(body_kps_files),
        desc=f"Parsing {sequence_name} body keypoints",
        unit="frame",
        leave=False,
    )

    for frame_idx, body_kps_file in enumerate(body_kps_files):
        try:
            with open(body_kps_file, "rb") as f:
                body_kps = json_loads(f.read())
        except Exception as e:
            print(
                f"Failed to parse poses file {body_kps_file}. Assuming it is empty (no detections). {e}"
            )

            # Didn't find the file, so we skip this frame
            pbar.update(1)
            continue

        for body in body_kps["bodies"]:
            body_id = int(body["id"])

            if body_id not in subjects_body_kps:
                subjects_body_kps[body_id] = torch.zeros(
                    (n_frames, PANOPTIC_BODY_KEYPOINTS_METADATA.n_keypoints, 3)
                )

            if body_id not in subjects_body_kps_scores:
                subjects_body_kps_scores[body_id] = torch.zeros(
                    (n_frames, PANOPTIC_BODY_KEYPOINTS_METADATA.n_keypoints)
                )

            joints19 = (
                torch.as_tensor(body["joints19"])
                .to(torch.float32)
                .view(PANOPTIC_BODY_KEYPOINTS_METADATA.n_keypoints, 4)
            )

            subject_body_kps = joints19[..., :3]
            subject_body_kps_scores = joints19[..., 3]
            subjects_body_kps[body_id][frame_idx] = subject_body_kps
            subjects_body_kps_scores[body_id][frame_idx] = subject_body_kps_scores

        del body_kps
        pbar.update(1)

    for subject_id, subject_kps in subjects_body_kps.items():
        subjects_body_kps[subject_id] = standardize_keypoints(
            subject_kps,
            src_world_unit_in_meters=0.01,  # 1 world unit = 1 cm
            src_world_basis=RH_NEG_Y_UP_BASIS,
        )

    pbar.close()

    assert len(subjects_body_kps) > 0, f"No subjects found in sequence {sequence_name}"

    return (
        start_frame_idx,
        end_frame_idx,
        subjects_body_kps,
        subjects_body_kps_scores,
    )


def _get_cameras_from_file(
    calibration_file_path: str,
) -> list[dict[str, Any]]:
    with open(calibration_file_path, "rb") as f:
        data = json_loads(f.read())

    camera_params = []

    calib_data_source = data["calibDataSource"]

    for camera_data in data["cameras"]:
        camera_name = camera_data["name"]
        camera_type = camera_data["type"]

        if camera_name not in DOWNLOADED_CAMERAS:
            continue

        K = torch.as_tensor(camera_data["K"]).to(torch.float32)
        distortion_coefficients = torch.as_tensor(camera_data["distCoef"]).to(
            torch.float32
        )
        R = torch.as_tensor(camera_data["R"]).to(torch.float32).view(3, 3)
        t = torch.as_tensor(camera_data["t"]).to(torch.float32).view(3, 1)
        t = t / 100.0  # Convert from cm to m
        Rt = torch.cat([R, t], dim=-1)

        # Convert the camera basis to OpenCV world basis
        Rt = convert_Rt_basis(Rt, RH_NEG_Y_UP_BASIS, OPENCV_WORLD_BASIS)

        camera_params.append(
            dict(
                name=camera_name,
                type=camera_type,
                resolution=camera_data["resolution"],
                panel=camera_data["panel"],
                node=camera_data["node"],
                K=K,
                distortion_coefficients=distortion_coefficients,
                Rt=Rt,
                calib_data_source=calib_data_source,
            )
        )

    return camera_params
