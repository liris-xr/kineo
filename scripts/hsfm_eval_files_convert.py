from smplx import SMPL
from smplx.lbs import vertices2joints
import torch
import pickle
import orjson
import os
from tqdm import tqdm
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import glob
from typing import Any

from kineo.geometry.camera import inverse_Rt
from kineo.annotations.camera_extrinsics import (
    CameraExtrinsicsAnnotation,
    CameraExtrinsicsAnnotations,
    CameraExtrinsicsAnnotationsMetadata,
)
from kineo.annotations.camera_intrinsics import (
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotations,
    CameraIntrinsicsAnnotationsMetadata,
)
from kineo.annotations.keypoints_3d import (
    Keypoints3DAnnotation,
    Keypoints3DAnnotations,
    Keypoints3DAnnotationsMetadata,
    KeypointsFormat,
)
from kineo.annotations.camera_intrinsics import CameraDistortionModel
import cv2

# Fix to properly load smpl model without errors
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.long = np.int_
np.complex = np.complex_
np.object = np.object_
np.str = np.str_
np.unicode = np.unicode_


def load_h36m_joints_regressor(body_model_dir: str) -> torch.Tensor:
    # From https://github.com/ubc-vision/joint-regressor-refinement/tree/master
    J_regressor = (
        torch.load(os.path.join(body_model_dir, "h36m_joints_regressor.pt"))
        .float()
        .cpu()
    )
    return J_regressor


def get_joints_from_smpl(
    smplx: SMPL, J_regressor: torch.Tensor, smpl_params: dict[str, Any]
) -> torch.Tensor:
    betas = torch.as_tensor(smpl_params["betas"], dtype=torch.float32)
    body_pose = torch.as_tensor(smpl_params["body_pose"], dtype=torch.float32).reshape(
        1, -1, 3, 3
    )
    global_orient = torch.as_tensor(
        smpl_params["global_orient"], dtype=torch.float32
    ).reshape(1, 1, 3, 3)
    root_transl = torch.as_tensor(
        smpl_params["root_transl"], dtype=torch.float32
    ).reshape(1, 1, 3)

    with torch.no_grad():
        smpl_output = smplx.forward(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            pose2rot=False,
            return_verts=True,
        )
        # Similar to what's done in the original code in vis_viser_hsfm.py
        smplx_vertices = smpl_output.vertices
        smplx_j3d = smpl_output.joints  # (1, J, 3), joints in the world coordinate from the world mesh decoded by the optimizing parameters
        smplx_vertices = smplx_vertices - smplx_j3d[:, 0:1, :] + root_transl
        smplx_j3d = smplx_j3d - smplx_j3d[:, 0:1, :] + root_transl

        joints = vertices2joints(J_regressor, smplx_vertices)[0]
    return joints


def load_hsfm_predictions_in_folder(
    folder_path: str,
    views_ids: list[str],
    views_resolution_hw: list[tuple[int, int]],
    smpl_model: SMPL,
    J_regressor: torch.Tensor,
    subject_id: str,
    n_workers: int = 8,
) -> tuple[
    CameraIntrinsicsAnnotations,
    CameraExtrinsicsAnnotations,
    Keypoints3DAnnotations,
]:
    files = sorted(glob.glob(os.path.join(folder_path, "*.pkl")))

    def load_hsfm_prediction(
        file: str,
        subject_id: str,
    ) -> tuple[
        list[CameraIntrinsicsAnnotation],
        list[CameraExtrinsicsAnnotation],
        list[Keypoints3DAnnotation],
    ]:
        frame_idx = int(file.split("_")[-1].split(".")[0])

        camera_intrinsics_annotations: list[CameraIntrinsicsAnnotation] = []
        camera_extrinsics_annotations: list[CameraExtrinsicsAnnotation] = []
        keypoints_3d_annotations: list[Keypoints3DAnnotation] = []

        with open(file, "rb") as f:
            data = pickle.load(f)

            if data is None:
                return None, None, None

            h36m_smpl = data["hsfm_people(smplx_params)"][1]
            h36m_joints = get_joints_from_smpl(smpl_model, J_regressor, h36m_smpl)

            keypoints_3d_annotations.append(
                Keypoints3DAnnotation(
                    frame_idx=frame_idx,
                    subject_id=subject_id,
                    xyz=h36m_joints.reshape(-1, 3).to(torch.float32),
                    scores=torch.ones_like(h36m_joints[:, 0], dtype=torch.float32),
                    format="h36m",
                )
            )

            n_cameras = len(data["hsfm_places_cameras"])

            for cam_idx in range(n_cameras):
                Rt = inverse_Rt(
                    torch.from_numpy(data["hsfm_places_cameras"][cam_idx]["cam2world"])[
                        :3, :
                    ]
                )
                K = torch.from_numpy(data["hsfm_places_cameras"][cam_idx]["intrinsic"])

                camera_intrinsics_annotations.append(
                    CameraIntrinsicsAnnotation(
                        view_id=views_ids[cam_idx],
                        frame_idx=frame_idx,
                        K=K,
                        distortion_coefficients=torch.zeros(5),
                        distortion_model=CameraDistortionModel.BROWN_CONRADY,
                        resolution_hw=views_resolution_hw[cam_idx],
                    )
                )
                camera_extrinsics_annotations.append(
                    CameraExtrinsicsAnnotation(
                        view_id=views_ids[cam_idx],
                        frame_idx=frame_idx,
                        R=Rt[:3, :3],
                        t=Rt[:3, 3],
                    )
                )

        camera_intrinsics_annotations = CameraIntrinsicsAnnotations(
            metadata=CameraIntrinsicsAnnotationsMetadata(),
            annotations=camera_intrinsics_annotations,
        )
        camera_extrinsics_annotations = CameraExtrinsicsAnnotations(
            metadata=CameraExtrinsicsAnnotationsMetadata(),
            annotations=camera_extrinsics_annotations,
        )
        keypoints_3d_annotations = Keypoints3DAnnotations(
            metadata=Keypoints3DAnnotationsMetadata(
                formats=[KeypointsFormat.from_mmpose_dataset("h36m")],
            ),
            annotations=keypoints_3d_annotations,
        )

        return (
            camera_intrinsics_annotations,
            camera_extrinsics_annotations,
            keypoints_3d_annotations,
        )

    executor = ThreadPoolExecutor(max_workers=n_workers)
    futures = [
        executor.submit(load_hsfm_prediction, file, subject_id) for file in files
    ]

    all_camera_intrinsics_annotations: list[CameraIntrinsicsAnnotations] = []
    all_camera_extrinsics_annotations: list[CameraExtrinsicsAnnotations] = []
    all_keypoints_3d_annotations: list[Keypoints3DAnnotations] = []

    for future in tqdm(futures, desc="Loading HSFM predictions", leave=False):
        (
            camera_intrinsics_annotations,
            camera_extrinsics_annotations,
            keypoints_3d_annotations,
        ) = future.result()

        if (
            camera_intrinsics_annotations is None
            or camera_extrinsics_annotations is None
            or keypoints_3d_annotations is None
        ):
            continue

        all_camera_intrinsics_annotations.extend(camera_intrinsics_annotations)
        all_camera_extrinsics_annotations.extend(camera_extrinsics_annotations)
        all_keypoints_3d_annotations.extend(keypoints_3d_annotations)

    camera_intrinsics_annotations = CameraIntrinsicsAnnotations(
        metadata=CameraIntrinsicsAnnotationsMetadata(),
        annotations=all_camera_intrinsics_annotations,
    )
    camera_extrinsics_annotations = CameraExtrinsicsAnnotations(
        metadata=CameraExtrinsicsAnnotationsMetadata(),
        annotations=all_camera_extrinsics_annotations,
    )
    keypoints_3d_annotations = Keypoints3DAnnotations(
        metadata=Keypoints3DAnnotationsMetadata(
            formats=[KeypointsFormat.from_mmpose_dataset("h36m")]
        ),
        annotations=all_keypoints_3d_annotations,
    )

    return (
        camera_intrinsics_annotations,
        camera_extrinsics_annotations,
        keypoints_3d_annotations,
    )


def load_kineo_predictions_in_folder(
    folder_path: str,
) -> tuple[
    CameraIntrinsicsAnnotations,
    CameraExtrinsicsAnnotations,
    Keypoints3DAnnotations,
]:
    camera_intrinsics_file = os.path.join(folder_path, "camera_intrinsics.pkl")
    camera_extrinsics_file = os.path.join(folder_path, "camera_extrinsics.pkl")
    keypoints_3d_file = os.path.join(folder_path, "keypoints_3d.pkl")

    if not os.path.exists(camera_intrinsics_file):
        raise FileNotFoundError(
            f"Camera intrinsics file not found: {camera_intrinsics_file}"
        )

    if not os.path.exists(camera_extrinsics_file):
        raise FileNotFoundError(
            f"Camera extrinsics file not found: {camera_extrinsics_file}"
        )

    if not os.path.exists(keypoints_3d_file):
        raise FileNotFoundError(f"Keypoints 3D file not found: {keypoints_3d_file}")

    with open(camera_intrinsics_file, "rb") as f:
        kineo_camera_intrinsics = CameraIntrinsicsAnnotations.from_dict(pickle.load(f))

    with open(camera_extrinsics_file, "rb") as f:
        kineo_camera_extrinsics = CameraExtrinsicsAnnotations.from_dict(pickle.load(f))

    with open(keypoints_3d_file, "rb") as f:
        kineo_keypoints_3d = Keypoints3DAnnotations.from_dict(pickle.load(f))

    return kineo_camera_intrinsics, kineo_camera_extrinsics, kineo_keypoints_3d


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument(
        "hsfm_eval_data_dir", type=str, help="Path to HSFM evaluation data directory"
    )
    parser.add_argument(
        "hsfm_processed_eval_data_dir",
        type=str,
        help="Path to HSFM processed evaluation data directory",
    )
    parser.add_argument("smpl_model_dir", type=str, help="Path to SMPL model directory")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    sequences_file = os.path.join(dataset_dir, "h36m_protocol1_sequences.json")

    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())

    smpl_model = SMPL(
        model_path=args.smpl_model_dir,
        gender="neutral",
        num_betas=10,
        batch_size=1,
    )
    J_regressor = load_h36m_joints_regressor(args.smpl_model_dir)

    for sequence in sequences:
        if sequence["split"] != "val":
            continue
        sequence_name = sequence["sequence_name"]

        try:
            views_ids = sorted(list(sequence["views"].keys()))
            views_resolution_hw = []

            for view_id in views_ids:
                video_path = os.path.join(
                    args.dataset_dir,
                    sequence["views"][view_id]["video_path"],
                )
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                views_resolution_hw.append((height, width))

            sequence_dir = os.path.join(args.hsfm_eval_data_dir, sequence_name)
            (
                hsfm_camera_intrinsics,
                hsfm_camera_extrinsics,
                hsfm_keypoints_3d,
            ) = load_hsfm_predictions_in_folder(
                sequence_dir,
                views_ids=views_ids,
                views_resolution_hw=views_resolution_hw,
                smpl_model=smpl_model,
                J_regressor=J_regressor,
                subject_id=sequence["subject_name"],
                n_workers=8,
            )

            os.makedirs(
                os.path.join(args.hsfm_processed_eval_data_dir, sequence_name),
                exist_ok=True,
            )

            camera_intrinsics_file = os.path.join(
                args.hsfm_processed_eval_data_dir,
                sequence_name,
                "camera_intrinsics.pkl",
            )
            camera_extrinsics_file = os.path.join(
                args.hsfm_processed_eval_data_dir,
                sequence_name,
                "camera_extrinsics.pkl",
            )
            keypoints_3d_file = os.path.join(
                args.hsfm_processed_eval_data_dir, sequence_name, "keypoints_3d.pkl"
            )

            with open(camera_intrinsics_file, "wb") as f:
                pickle.dump(hsfm_camera_intrinsics.to_dict(), f)
            with open(camera_extrinsics_file, "wb") as f:
                pickle.dump(hsfm_camera_extrinsics.to_dict(), f)
            with open(keypoints_3d_file, "wb") as f:
                pickle.dump(hsfm_keypoints_3d.to_dict(), f)

            print(f"Saved new annotations for sequence {sequence_name}")

        except Exception as e:
            print(f"Error processing sequence {sequence_name}: {e}")
            continue
