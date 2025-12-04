from kineo.demo.online.multicam_recorder import MultiCamRecorder
from PyQt5 import QtWidgets
import sys
import torch
from kineo.pipeline.pipeline import Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.io.frame_sequence_loader import VideoLoader
from kineo.demo.online.live_video_loader import LiveVideoLoader
import cv2
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.reconstructed_scene import WorldReconstructedSceneAnnotations
import sys
import platform

from kineo.demo.online.camera_utils import get_available_cameras
import pickle
import argparse
import os

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def create_views_from_temp_videos(temp_videos, cam_names, device: torch.device):
    views = []
    for i, temp_video in enumerate(temp_videos):
        video_path = temp_video.name

        views.append(
            ViewInput(
                view_id=cam_names[i],
                frame_loader=VideoLoader(
                    video_path=video_path,
                    device=device,
                ),
                audio_loader=None,
            )
        )
    return views

def create_live_views(cam_indices: list[int], cam_names: list[str], device: torch.device, api_preference: int = cv2.CAP_ANY):
    views = []
    for cam_idx, cam_name in zip(cam_indices, cam_names):
        views.append(
            ViewInput(
                view_id=cam_name,
                frame_loader=LiveVideoLoader(
                    camera_idx=cam_idx,
                    device=device,
                    api_preference=api_preference,
                ),
            )
        )
    return views


def load_camera_calibrations(cam_ids: list[str], calibration_output_root_dir: str) -> tuple[
    CameraIntrinsicsAnnotations, CameraExtrinsicsAnnotations, WorldReconstructedSceneAnnotations]:
    cam_extrinsics_filepath = f"{calibration_output_root_dir}/annotations/calibration/camera_extrinsics.pkl"
    cam_intrinsics_filepath = f"{calibration_output_root_dir}/annotations/calibration/camera_intrinsics.pkl"
    world_reconstructed_scene_filepath = f"{calibration_output_root_dir}/annotations/calibration/world_reconstructed_scene.pkl"

    if not os.path.exists(cam_extrinsics_filepath):
        raise FileNotFoundError(f"Camera extrinsics file {cam_extrinsics_filepath} does not exist")
    if not os.path.exists(cam_intrinsics_filepath):
        raise FileNotFoundError(f"Camera intrinsics file {cam_intrinsics_filepath} does not exist")

    with open(cam_extrinsics_filepath, "rb") as f:
        cam_extrinsics = CameraExtrinsicsAnnotations.from_dict(pickle.load(f))
    with open(cam_intrinsics_filepath, "rb") as f:
        cam_intrinsics = CameraIntrinsicsAnnotations.from_dict(pickle.load(f))

    world_reconstructed_scene = None
    if os.path.exists(world_reconstructed_scene_filepath):
        with open(world_reconstructed_scene_filepath, "rb") as f:
            world_reconstructed_scene = WorldReconstructedSceneAnnotations.from_dict(pickle.load(f))

    cam_extrinsics = cam_extrinsics.filter_by_view_ids(cam_ids)
    cam_intrinsics = cam_intrinsics.filter_by_view_ids(cam_ids)

    if not len(cam_extrinsics.views_ids) == len(cam_ids) or not len(cam_intrinsics.views_ids) == len(cam_ids):
        raise Exception("All views were not found in the calibration data")

    return cam_intrinsics, cam_extrinsics, world_reconstructed_scene


def main(target_fps, target_res, live_viz_config, skip_calibration, api_preference: int = cv2.CAP_ANY):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calibration_config_file = "configs/demo/realtime/calibration.yaml"

    if not skip_calibration:
        print("Loading calibration pipeline")
        calibration_pipeline = Pipeline.build_pipeline_from_config(calibration_config_file, device)
        print("Pipelines loaded")

        app = QtWidgets.QApplication(sys.argv)
        recorder = MultiCamRecorder(
            target_fps=target_fps,
            target_res=target_res,
            api_preference=api_preference
        )
        recorder.show()
        app.exec_()

        cam_indices = recorder.camera_indices
        cam_names = recorder.camera_ids

        views = create_views_from_temp_videos(recorder.temp_videos, cam_names, device=device, api_preference=api_preference)
        calibration_pipeline.run(
            sequence_name="calibration",
            views=views,
            annotations={},
            gt_annotations={},
        )

    print("Loading live viz pipeline")
    live_viz_pipeline = Pipeline.build_pipeline_from_config(live_viz_config, device)
    print("Live viz pipeline loaded")

    camera_indices = []
    camera_ids = []

    for camera_info in get_available_cameras(api_preference=api_preference):
        camera_id = f"{camera_info.pid}_{camera_info.vid}_{camera_info.index}"
        camera_indices.append(camera_info.index)
        camera_ids.append(camera_id)

    if len(camera_indices) <= 1:
        raise Exception(f"Expected at least 2 cameras, got {len(camera_indices)}")

    views = create_live_views(camera_indices, camera_ids, device=device, api_preference=api_preference)

    calibration_output_root_dir = "./outputs/realtime_demo_calibration"
    cam_intrinsics, cam_extrinsics, world_reconstructed_scene = load_camera_calibrations(
        camera_ids, calibration_output_root_dir
    )

    live_viz_pipeline.run(
        sequence_name="realtime_viz",
        views=views,
        annotations={
            "camera_intrinsics": cam_intrinsics,
            "camera_extrinsics": cam_extrinsics,
            "world_reconstructed_scene": world_reconstructed_scene,
        },
        gt_annotations={},
    )


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--target-res", type=str, default="640x480")
    parser.add_argument("--live-viz-config", type=str, default="configs/demo/realtime/realtime_viz.yaml")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument(
        "--api-preference",
        type=str,
        choices=["any", "dshow", "v4l2", "msmf"],
        default=None
    )
    args = parser.parse_args()

    target_res = tuple(int(x) for x in args.target_res.split("x"))

    available_backends = cv2.videoio_registry.getBackends()

    if args.api_preference is None:
        if platform.system() == "Windows" and cv2.CAP_DSHOW in available_backends:
            api_preference = cv2.CAP_DSHOW
        elif platform.system() == "Linux" and cv2.CAP_V4L2 in available_backends:
            api_preference = cv2.CAP_V4L2
        else:
            api_preference = cv2.CAP_ANY
    elif args.api_preference == "any":
        api_preference = cv2.CAP_ANY
    elif args.api_preference == "dshow":
        api_preference = cv2.CAP_DSHOW
    elif args.api_preference == "v4l2":
        api_preference = cv2.CAP_V4L2
    elif args.api_preference == "msmf":
        api_preference = cv2.CAP_MSMF

    main(
        target_fps=args.target_fps,
        target_res=target_res,
        live_viz_config=args.live_viz_config,
        skip_calibration=args.skip_calibration,
        api_preference=api_preference,
    )

if __name__ == "__main__":
    cli()