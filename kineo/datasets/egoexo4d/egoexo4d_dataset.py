# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from typing import Literal, Any
import orjson
import os
import torch
from tqdm import tqdm
from kineo.io.frame_sequence_loader import VideoLoader
from kineo.datasets.keypoints_sequence_dataset import (
    KeypointsSequenceDataset,
    KeypointsSequence,
)
from kineo.datasets.keypoints_metadata import KeypointsMetadata

KEYPOINTS_METADATA = KeypointsMetadata.from_mmpose_dataset("coco")

# Mapping from MMPose COCO keypoints names to EgoExo4D keypoints names
KEYPOINTS_NAME_TO_MMPOSE_IDX = {
    keypoint_name.replace("_", "-"): keypoint_idx
    for keypoint_idx, keypoint_name in enumerate(KEYPOINTS_METADATA.names)
}

KEYPOINTS_MMPOSE_IDX_TO_NAME = {
    keypoint_idx: keypoint_name.replace("_", "-")
    for keypoint_idx, keypoint_name in enumerate(KEYPOINTS_METADATA.names)
}


class EgoExo4DSequenceDataset(KeypointsSequenceDataset):
    def __init__(
        self, dataset_dirpath: str, split: Literal["train", "val"], device: torch.device
    ):
        if not os.path.exists(dataset_dirpath):
            raise FileNotFoundError(
                f"Dataset directory {dataset_dirpath} does not exist"
            )

        self.dataset_dirpath = dataset_dirpath
        self.split = split

        self.sequences = self._load_sequences(dataset_dirpath)
        self.device = device

    def _load_sequences(self, dataset_dirpath: str) -> list[dict[str, Any]]:
        splits_filepath = os.path.join(dataset_dirpath, "annotations", "splits.json")
        takes_filepath = os.path.join(dataset_dirpath, "takes.json")

        if not os.path.exists(splits_filepath):
            raise FileNotFoundError(f"Splits file {splits_filepath} does not exist")

        if not os.path.exists(takes_filepath):
            raise FileNotFoundError(f"Takes file {takes_filepath} does not exist")

        with open(splits_filepath, "r") as f:
            splits = orjson.loads(f.read())

        with open(takes_filepath, "r") as f:
            takes = orjson.loads(f.read())

        takes_uids = [
            take_uid
            for take_uid in splits["take_uid_to_split"].keys()
            if splits["take_uid_to_split"][take_uid] == self.split
            and "egobodypose" in splits["take_uid_to_benchmark"].get(take_uid, [])
        ]

        sequences = []

        for take_uid in tqdm(takes_uids, desc="Loading sequences", leave=False):
            take = next(take for take in takes if take["take_uid"] == take_uid)

            root_dir = take["root_dir"]

            n_frames = take["timesync_end_idx"] - take["timesync_start_idx"]

            sequence_annotations = self._load_sequence_annotations(
                dataset_dirpath,
                take=take,
                n_frames=n_frames,
            )

            included_camera_ids = list(sequence_annotations["cameras"].keys())

            sequence_info = {
                "sequence_name": take["take_name"],
                "annotations": sequence_annotations,
                "n_frames": n_frames,
                "duration_s": take["duration_sec"],
                "views": {
                    cam_id: {
                        "video_path": os.path.join(
                            root_dir,
                            take["frame_aligned_videos"][cam_id]["0"]["relative_path"],
                        ),
                    }
                    for cam_id in included_camera_ids
                },
            }
            sequences.append(sequence_info)

            break

        return sequences

    def _load_sequence_annotations(
        self,
        dataset_dirpath: str,
        take: dict[str, Any],
        n_frames: int,
    ) -> dict[str, Any]:
        take_uid = take["take_uid"]
        body_annotations = self._load_body_annotations(dataset_dirpath, take_uid)
        body_bboxes_annotations = self._load_body_bboxes_annotations(
            dataset_dirpath, take_uid
        )
        cameras_annotations = self._load_cameras_annotations(dataset_dirpath, take_uid)

        # Only keep the exo cameras
        included_camera_ids = [
            cam_data["cam_id"]
            for cam_data in take["capture"]["cameras"]
            if not cam_data["is_ego"]
            and not cam_data["has_walkaround"]
            and cam_data["cam_id"] in cameras_annotations
        ]

        annotations = {
            "3d": {},
            "2d": {camera_id: {} for camera_id in included_camera_ids},
            "cameras": {
                camera_id: {
                    "intrinsics": torch.as_tensor(
                        cameras_annotations[camera_id]["camera_intrinsics"],
                        dtype=torch.float32,
                    ),
                    "extrinsics": torch.as_tensor(
                        cameras_annotations[camera_id]["camera_extrinsics"],
                        dtype=torch.float32,
                    ),
                    "distortion_coefficients": torch.as_tensor(
                        cameras_annotations[camera_id]["distortion_coeffs"],
                        dtype=torch.float32,
                    ),
                }
                for camera_id in included_camera_ids
            },
        }

        for frame_idx in tqdm(
            range(n_frames), desc="Loading sequence annotations", leave=False
        ):
            frame_idx_str = str(frame_idx)

            if (
                frame_idx_str not in body_annotations
                and frame_idx_str not in body_bboxes_annotations
            ):
                continue

            if frame_idx_str in body_annotations:
                body_annotations_3d = body_annotations[frame_idx_str][0]["annotation3D"]
                body_annotations_2d = body_annotations[frame_idx_str][0]["annotation2D"]

                # 3D keypoints
                keypoints_3d = torch.zeros(
                    (KEYPOINTS_METADATA.n_keypoints, 3), dtype=torch.float32
                )
                keypoints_3d_annotation_mask = torch.zeros(
                    KEYPOINTS_METADATA.n_keypoints, dtype=torch.bool
                )
                for keypoint_name in body_annotations_3d:
                    keypoints_mmpose_idx = KEYPOINTS_NAME_TO_MMPOSE_IDX[keypoint_name]
                    keypoint_3d = keypoints_3d[keypoints_mmpose_idx]
                    keypoint_3d[0] = body_annotations_3d[keypoint_name]["x"]
                    keypoint_3d[1] = body_annotations_3d[keypoint_name]["y"]
                    keypoint_3d[2] = body_annotations_3d[keypoint_name]["z"]
                    keypoints_3d_annotation_mask[keypoints_mmpose_idx] = True

                annotations["3d"].setdefault("keypoints", {})
                annotations["3d"]["keypoints"][frame_idx] = {
                    "xyz": keypoints_3d,
                    "annotation_mask": keypoints_3d_annotation_mask,
                }

                # 2D keypoints
                for camera_id, camera_annotations in body_annotations_2d.items():
                    if camera_id not in included_camera_ids:
                        continue

                    keypoints_2d = torch.zeros(
                        (KEYPOINTS_METADATA.n_keypoints, 2), dtype=torch.float32
                    )
                    keypoints_2d_annotation_mask = torch.zeros(
                        KEYPOINTS_METADATA.n_keypoints, dtype=torch.bool
                    )

                    for (
                        keypoint_name,
                        keypoint_annotation,
                    ) in camera_annotations.items():
                        keypoints_mmpose_idx = KEYPOINTS_NAME_TO_MMPOSE_IDX[
                            keypoint_name
                        ]
                        keypoints_2d[keypoints_mmpose_idx, 0] = keypoint_annotation["x"]
                        keypoints_2d[keypoints_mmpose_idx, 1] = keypoint_annotation["y"]
                        keypoints_2d_annotation_mask[keypoints_mmpose_idx] = True

                    annotations["2d"][camera_id].setdefault("keypoints", {})
                    annotations["2d"][camera_id]["keypoints"][frame_idx] = {
                        "xy": keypoints_2d,
                        "annotation_mask": keypoints_2d_annotation_mask,
                    }

            if frame_idx_str in body_bboxes_annotations:
                bbox_annotations = body_bboxes_annotations[frame_idx_str][0]

                for camera_id, camera_annotations in bbox_annotations.items():
                    if camera_id not in included_camera_ids:
                        continue

                    annotations["2d"][camera_id].setdefault("bboxes_xyxy", {})
                    annotations["2d"][camera_id]["bboxes_xyxy"][frame_idx] = (
                        torch.as_tensor(
                            camera_annotations["bbox_xyxy"], dtype=torch.float32
                        )
                    )

        return annotations

    def _load_body_annotations(
        self, dataset_dirpath: str, take_uid: str
    ) -> dict[str, Any]:
        body_annotations_filepath = os.path.join(
            dataset_dirpath,
            "annotations/ego_pose",
            self.split,
            "body/annotation",
            f"{take_uid}.json",
        )

        if not os.path.exists(body_annotations_filepath):
            raise FileNotFoundError(
                f"Body annotations file {body_annotations_filepath} does not exist"
            )

        with open(body_annotations_filepath, "r") as f:
            body_annotations = orjson.loads(f.read())

        return body_annotations

    def _load_body_bboxes_annotations(
        self, dataset_dirpath: str, take_uid: str
    ) -> dict[str, Any]:
        body_bboxes_annotations_filepath = os.path.join(
            dataset_dirpath,
            "annotations/ego_pose",
            self.split,
            "body_bboxes",
            f"{take_uid}.json",
        )

        if not os.path.exists(body_bboxes_annotations_filepath):
            raise FileNotFoundError(
                f"Body bboxes annotations file {body_bboxes_annotations_filepath} does not exist"
            )

        with open(body_bboxes_annotations_filepath, "r") as f:
            body_bboxes_annotations = orjson.loads(f.read())

        return body_bboxes_annotations

    def _load_cameras_annotations(
        self, dataset_dirpath: str, take_uid: str
    ) -> dict[str, Any]:
        cameras_annotations_filepath = os.path.join(
            dataset_dirpath,
            "annotations/ego_pose",
            self.split,
            "camera_pose",
            f"{take_uid}.json",
        )

        if not os.path.exists(cameras_annotations_filepath):
            raise FileNotFoundError(
                f"Cameras annotations file {cameras_annotations_filepath} does not exist"
            )

        with open(cameras_annotations_filepath, "r") as f:
            cameras_annotations = orjson.loads(f.read())

        return cameras_annotations

    def __getitem__(self, index: int) -> KeypointsSequence:
        sequence = self.sequences[index]
        sequence_annotations = sequence.get("annotations", None)

        view_inputs = {
            view_name: VideoLoader(
                video_path=os.path.join(self.dataset_dirpath, view_info["video_path"]),
                device=self.device,
            )
            for view_name, view_info in sequence["views"].items()
        }

        return KeypointsSequence(
            sequence_name=sequence["sequence_name"],
            views_inputs=view_inputs,
            sequence_annotations=sequence_annotations,
        )

    def __len__(self) -> int:
        return len(self.sequences)


import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd


def undistort_exocam(image, intrinsics, distortion_coeffs, dimension=(3840, 2160)):
    DIM = dimension
    dim2 = None
    dim3 = None
    balance = 0.8
    # Load the distortion parameters
    distortion_coeffs = distortion_coeffs
    # Load the camera intrinsic parameters
    intrinsics = intrinsics

    dim1 = image.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort

    # Change the calibration dim dynamically (bouldering cam01 and cam04 are verticall for examples)
    if DIM[0] != dim1[0]:
        DIM = (DIM[1], DIM[0])

    assert dim1[0] / dim1[1] == DIM[0] / DIM[1], (
        "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    )
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = (
        intrinsics * dim1[0] / DIM[0]
    )  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_K, distortion_coeffs, dim2, np.eye(3), balance=balance
    )

    print("new_K", new_K)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_K, distortion_coeffs, np.eye(3), new_K, dim3, cv2.CV_16SC2
    )
    undistorted_image = cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return undistorted_image, new_K


def get_distortion_and_intrinsics(_raw_camera):
    intrinsics = np.array(
        [
            [_raw_camera["intrinsics_0"], 0, _raw_camera["intrinsics_2"]],
            [0, _raw_camera["intrinsics_1"], _raw_camera["intrinsics_3"]],
            [0, 0, 1],
        ]
    )
    distortion_coeffs = np.array(
        [
            _raw_camera["intrinsics_4"],
            _raw_camera["intrinsics_5"],
            _raw_camera["intrinsics_6"],
            _raw_camera["intrinsics_7"],
        ]
    )
    return distortion_coeffs, intrinsics


def load_csv_to_df(filepath: str) -> pd.DataFrame:
    with open(filepath, "r") as csv_file:
        return pd.read_csv(csv_file)


if __name__ == "__main__":
    from kornia.geometry.calibration import distort_points

    dataset = EgoExo4DSequenceDataset(
        dataset_dirpath="E:/Datasets/EgoExo4D/",
        split="val",
        device=torch.device("cuda"),
    )

    first = dataset[0]

    # for camera_id, video_loader in first.views_inputs.items():

    #     seq_name = first.sequence_name

    #     exo_traj_df = load_csv_to_df(
    #         f"E:/Datasets/EgoExo4D/takes/{seq_name}/trajectory/gopro_calibs.csv"
    #     )
    #     calib_df = exo_traj_df[exo_traj_df.cam_uid == camera_id]
    #     calib_df = calib_df.iloc[0].to_dict()
    #     D, I = get_distortion_and_intrinsics(calib_df)

    #     camera_intrinsics = first.sequence_annotations["cameras"][camera_id][
    #         "intrinsics"
    #     ]
    #     camera_distortion_coefficients = first.sequence_annotations["cameras"][
    #         camera_id
    #     ]["distortion_coefficients"]

    #     print("camera_intrinsics", camera_intrinsics)
    #     print("I", I)
    #     print("D", D)

    #     print("camera_distortion_coefficients", camera_distortion_coefficients)

    #     for frame_idx in range(len(video_loader)):
    #         frame = video_loader[frame_idx]
    #         frame = frame.permute(1, 2, 0).cpu().numpy()

    #         keypoints2d = first.sequence_annotations["2d"][camera_id]["keypoints"]

    #         if frame_idx in keypoints2d:
    #             kps_2d_xy = keypoints2d[frame_idx]["xy"]
    #             kps_2d_xy_mask = keypoints2d[frame_idx]["annotation_mask"]

    #             # undistorted_frame, new_K_latest = undistort_exocam(
    #             #     frame, I, D
    #             # )

    #             # distorted_kps_2d_xy = distort_points(
    #             #     points=kps_2d_xy,
    #             #     K=camera_intrinsics,
    #             #     dist=torch.from_numpy(D),
    #             #     new_K=torch.from_numpy(I),
    #             # )

    #             distorted_kps_2d_xy =

    #             distorted_kps_2d_xy = cv2.fisheye.distortPoints(
    #                 undistorted=kps_2d_xy.cpu().numpy().reshape(1, -1, 2),
    #                 Kundistorted=camera_intrinsics.cpu().numpy(),
    #                 K=I,
    #                 D=D
    #             ).reshape(17, 2)
    #             distorted_kps_2d_xy = torch.from_numpy(distorted_kps_2d_xy)

    #             plt.imshow(frame)
    #             plt.scatter(
    #                 distorted_kps_2d_xy[kps_2d_xy_mask][:, 0].cpu().numpy(),
    #                 distorted_kps_2d_xy[kps_2d_xy_mask][:, 1].cpu().numpy(),
    #                 color="red",
    #             )
    #             plt.show()
