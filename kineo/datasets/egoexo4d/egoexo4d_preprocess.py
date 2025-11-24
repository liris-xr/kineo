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
import warnings
from typing import Any, Literal

import pandas as pd
import torch
from tqdm import tqdm
import roma

from kineo.annotations.keypoints_format import KeypointsFormat
from kineo.geometry.camera import inverse_Rt

KEYPOINTS_METADATA = KeypointsFormat.from_mmpose_dataset("coco")


def preprocess_egoexo4d(dataset_dirpath: str, split: Literal["train", "val", "test"]):
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
        for take_uid, benchmarks in splits["take_uid_to_benchmark"].items()
        if "egobodypose" in benchmarks
        and take_uid in splits["take_uid_to_split"]
        and splits["take_uid_to_split"][take_uid] == split
    ]

    for take_uid in tqdm(
        takes_uids, desc="Generating body bboxes annotations", leave=False
    ):
        take = next(take for take in takes if take["take_uid"] == take_uid)

        if "covid" in take["take_name"] or "pcr" in take["take_name"]:
            continue

        print(take["take_name"])

        # if "upenn_0706_Dance_4_2" not in take["take_name"]:
        #     continue

        _generate_annotations(
            dataset_dirpath,
            take,
            split,
            bbox_padding_x=100,
            bbox_padding_y=120,
            bbox_clamp_to_image_size=True,
        )

    print(f"Finished preprocess {split} split.")


def _load_gopro_calibs(filepath: str) -> dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Gopro calibs file {filepath} does not exist")

    with open(filepath, "r") as csv_file:
        gopro_calibs_df = pd.read_csv(csv_file)

    gopro_calibs = {}

    for _, row in gopro_calibs_df.iterrows():
        camera_id = row["cam_uid"]
        calib = row.to_dict()

        assert calib["intrinsics_type"] == "KANNALABRANDTK3"

        extrinsics = torch.eye(4, dtype=torch.float32)[:3, :]
        world_cam_quat = torch.tensor(
            [
                calib["qx_world_cam"],
                calib["qy_world_cam"],
                calib["qz_world_cam"],
                calib["qw_world_cam"],
            ]
        )
        extrinsics[:3, :3] = roma.unitquat_to_rotmat(world_cam_quat)
        extrinsics[:3, 3] = torch.tensor(
            [
                calib["tx_world_cam"],
                calib["ty_world_cam"],
                calib["tz_world_cam"],
            ],
            dtype=torch.float32,
        )

        extrinsics = inverse_Rt(extrinsics)

        intrinsics = torch.tensor(
            [
                [calib["intrinsics_0"], 0, calib["intrinsics_2"]],
                [0, calib["intrinsics_1"], calib["intrinsics_3"]],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )
        distortion_coeffs = torch.tensor(
            [
                calib["intrinsics_4"],
                calib["intrinsics_5"],
                calib["intrinsics_6"],
                calib["intrinsics_7"],
            ],
            dtype=torch.float32,
        )

        gopro_calibs[camera_id] = {
            "distortion_coefficients": distortion_coeffs,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "image_width": calib["image_width"],
            "image_height": calib["image_height"],
        }

    # from aitviewer.viewer import Viewer
    # from aitviewer.scene.camera import OpenCVCamera

    # viewer = Viewer()

    # for gopro in gopro_calibs:
    #     camera = OpenCVCamera(
    #         Rt=gopro_calibs[gopro]["extrinsics"].cpu().numpy(),
    #         K=gopro_calibs[gopro]["intrinsics"].cpu().numpy(),
    #         cols=gopro_calibs[gopro]["image_width"],
    #         rows=gopro_calibs[gopro]["image_height"],
    #     )
    #     viewer.scene.add(camera)

    # viewer.scene.floor.enabled = False
    # viewer.run()

    return gopro_calibs


import cv2
import matplotlib.pyplot as plt
import numpy as np


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


def _generate_annotations(
    dataset_dirpath: str,
    take: dict[str, Any],
    split: str,
    bbox_padding_x: int,
    bbox_padding_y: int,
    bbox_clamp_to_image_size: bool,
) -> dict[str, Any]:
    take_uid = take["take_uid"]

    gopro_calibs_filepath = os.path.join(
        dataset_dirpath,
        "takes",
        f"{take['take_name']}/trajectory/gopro_calibs.csv",
    )

    gopro_calibs = _load_gopro_calibs(gopro_calibs_filepath)

    camera_pose_annotations_filepath = os.path.join(
        dataset_dirpath,
        "annotations/ego_pose",
        split,
        "camera_pose",
        f"{take_uid}.json",
    )

    if not os.path.exists(camera_pose_annotations_filepath):
        raise FileNotFoundError(
            f"Camera pose annotations file {camera_pose_annotations_filepath} does not exist"
        )

    with open(camera_pose_annotations_filepath, "r") as f:
        camera_pose_annotations = orjson.loads(f.read())

    for gopro in gopro_calibs:
        Rt = torch.as_tensor(
            camera_pose_annotations[gopro]["camera_extrinsics"], dtype=torch.float32
        )
        K = torch.as_tensor(
            camera_pose_annotations[gopro]["camera_intrinsics"], dtype=torch.float32
        )
        # gopro_calibs[gopro]["extrinsics"] = Rt
        gopro_calibs[gopro]["intrinsics_undistorted"] = K
        gopro_calibs[gopro]["projection_matrix_undistorted"] = K @ Rt

    body_annotations_filepath = os.path.join(
        dataset_dirpath,
        "annotations/ego_pose",
        split,
        "body/annotation",
        f"{take_uid}.json",
    )

    if not os.path.exists(body_annotations_filepath):
        warnings.warn(
            f"Take annotations file {body_annotations_filepath} does not exist. Skipping."
        )
        return

    with open(body_annotations_filepath, "r") as f:
        body_annotations = orjson.loads(f.read())

    body_bboxes_annotations = {}

    n_frames = len(body_annotations)

    n_cameras = len(gopro_calibs)

    all_keypoints_2d = torch.zeros((n_frames, n_cameras, 17, 2), dtype=torch.float32)
    # all_keypoints_3d = torch.zeros((n_frames, 17, 3), dtype=torch.float32)
    all_keypoints_3d_og = torch.zeros((n_frames, 17, 3), dtype=torch.float32)

    for frame_idx, (frame_idx_str, frame_annotations) in enumerate(
        body_annotations.items()
    ):
        annotation2D = frame_annotations[0]["annotation2D"]

        for keypoint_idx, keypoint_name in enumerate(KEYPOINTS_METADATA.names):
            # Names in annotation3D use '-' instead of '_' as separator
            keypoint_name = keypoint_name.replace("_", "-")
            keypoint_annotation = frame_annotations[0]["annotation3D"].get(
                keypoint_name, None
            )

            if keypoint_annotation is None:
                continue

            all_keypoints_3d_og[frame_idx, keypoint_idx] = torch.tensor(
                [
                    keypoint_annotation["x"],
                    keypoint_annotation["y"],
                    keypoint_annotation["z"],
                ],
                dtype=torch.float32,
            )

        for cam_idx, camera_name in enumerate(gopro_calibs):
            if camera_name not in annotation2D:
                print(f"camera {camera_name} not found")
                continue

            cam_annot_2d = annotation2D[camera_name]

            for keypoint_idx, keypoint_name in enumerate(KEYPOINTS_METADATA.names):
                # Names in annotation3D use '-' instead of '_' as separator
                keypoint_name = keypoint_name.replace("_", "-")
                keypoint_annotation = cam_annot_2d.get(keypoint_name, None)

                if keypoint_annotation is None:
                    continue

                all_keypoints_2d[frame_idx, cam_idx, keypoint_idx] = torch.tensor(
                    [
                        keypoint_annotation["x"],
                        keypoint_annotation["y"],
                    ],
                    dtype=torch.float32,
                )

    # gopro_cap = {
    #     gopro: cv2.VideoCapture(
    #         f"{dataset_dirpath}/takes/{take['take_name']}/frame_aligned_videos/{gopro}.mp4"
    #     )
    #     for gopro in gopro_calibs
    # }

    # print(n_frames)
    # for frame_idx in range(0, n_frames, 10):
    #     frame_idx_str = list(body_annotations.keys())[frame_idx]

    #     plt.figure(figsize=(10, 10))
    #     for i, gopro in enumerate(gopro_calibs):
    #         cap = gopro_cap[gopro]
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx_str))
    #         ret, frame = cap.read()
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         frame, new_K = undistort_exocam(
    #             frame,
    #             gopro_calibs[gopro]["intrinsics"].cpu().numpy(),
    #             gopro_calibs[gopro]["distortion_coefficients"].cpu().numpy(),
    #         )
    #         K = torch.from_numpy(new_K)

    #         # print("camera_intrinsics", camera_pose_annotations[gopro]["camera_intrinsics"])
    #         # print("new_K", new_K)

    #         Rt1 = gopro_calibs[gopro]["extrinsics"]
    #         Rt2 = torch.as_tensor(camera_pose_annotations[gopro]["camera_extrinsics"])

    #         rvec1 = cv2.Rodrigues(Rt1[:3, :3].cpu().numpy())[0]
    #         tvec1 = Rt1[:3, 3].cpu().numpy()
    #         rvec2 = cv2.Rodrigues(Rt2[:3, :3].cpu().numpy())[0]
    #         tvec2 = Rt2[:3, 3].cpu().numpy()

    #         kps2d_1 = cv2.projectPoints(
    #             all_keypoints_3d_og[frame_idx].cpu().numpy().reshape(1, 17, 3),
    #             rvec1,
    #             tvec1,
    #             cameraMatrix=new_K,
    #             distCoeffs=np.zeros(4),
    #         )[0][:, 0, :]
    #         kps2d_2 = cv2.projectPoints(
    #             all_keypoints_3d_og[frame_idx].cpu().numpy().reshape(1, 17, 3),
    #             rvec2,
    #             tvec2,
    #             cameraMatrix=new_K,
    #             distCoeffs=np.zeros(4),
    #         )[0][:, 0, :]

    #         plt.subplot(1, len(gopro_calibs), i + 1)
    #         plt.imshow(frame)
    #         plt.scatter(
    #             all_keypoints_2d[frame_idx, i, :, 0],
    #             all_keypoints_2d[frame_idx, i, :, 1],
    #             color="red",
    #         )
    #         plt.scatter(
    #             kps2d_1[:, 0],
    #             kps2d_1[:, 1],
    #             color="blue",
    #         )
    #         plt.scatter(
    #             kps2d_2[:, 0],
    #             kps2d_2[:, 1],
    #             color="green",
    #         )

    #     plt.show()

    from aitviewer.viewer import Viewer
    from aitviewer.renderables.point_clouds import PointClouds
    from aitviewer.scene.camera import OpenCVCamera

    viewer = Viewer()

    pcd_og = PointClouds(all_keypoints_3d_og.reshape(n_frames, 17, 3).cpu().numpy())
    pcd_og.name = "keypoints_3d_og"
    pcd_og.color = (0.0, 0.0, 1.0, 1.0)

    viewer.scene.add(pcd_og)
    for gopro in gopro_calibs:
        camera = OpenCVCamera(
            Rt=np.asarray(camera_pose_annotations[gopro]["camera_extrinsics"]),
            K=gopro_calibs[gopro]["intrinsics"].cpu().numpy(),
            cols=gopro_calibs[gopro]["image_width"],
            rows=gopro_calibs[gopro]["image_height"],
        )
        viewer.scene.add(camera)

    viewer.scene.floor.enabled = False
    viewer.run()

    # print(keypoints_2d)

    # Re-project 3D annotations to 2D with and without distortion

    # cameras_annotations = {}

    # for camera_id, camera_annotation in annotation2D.items():
    #     if cameras_types[camera_id] not in SUPPORTED_DEVICE_TYPES:
    #         continue

    #     image_size_hw = RESOLUTION_HW_BY_DEVICE_TYPE[cameras_types[camera_id]]

    #     visible_keypoints = []

    #     for keypoint_annotation in camera_annotation.values():
    #         visible_keypoints.append(
    #             [
    #                 keypoint_annotation["x"],
    #                 keypoint_annotation["y"],
    #             ]
    #         )

    #     visible_keypoints = torch.as_tensor(visible_keypoints)

    #     if visible_keypoints.numel() == 0:
    #         bbox_visible = False
    #     else:
    #         bbox_xyxy = compute_bboxes_xyxy(
    #             poses_2d=visible_keypoints,
    #             padding_x=bbox_padding_x,
    #             padding_y=bbox_padding_y,
    #             image_size_hw=image_size_hw,
    #             clamp_to_image_size=bbox_clamp_to_image_size,
    #         )
    #         bbox_visible = torch.any(bbox_xyxy > 0).item()

    #     if bbox_visible:
    #         cameras_annotations[camera_id] = {"bbox_xyxy": bbox_xyxy.tolist()}

    # if len(cameras_annotations) > 0:
    #     body_bboxes_annotations.setdefault(frame_idx, [])
    #     body_bboxes_annotations[frame_idx].append(cameras_annotations)

    body_bboxes_annotations_filepath = os.path.join(
        dataset_dirpath,
        "annotations/ego_pose",
        split,
        "body_bboxes",
        f"{take_uid}.json",
    )

    body_bboxes_annotations_dirpath = os.path.dirname(body_bboxes_annotations_filepath)
    os.makedirs(body_bboxes_annotations_dirpath, exist_ok=True)

    with open(body_bboxes_annotations_filepath, "wb") as f:
        f.write(orjson.dumps(body_bboxes_annotations))

    print(f"Body bboxes annotations saved to {body_bboxes_annotations_filepath}")
