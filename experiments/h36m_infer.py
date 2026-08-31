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
from kineo.pipeline.pipeline import Pipeline

from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.io.frame_sequence_loader import VideoLoader
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.bboxes_2d import BBox2DAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations

import orjson
from tqdm import tqdm
import argparse
import traceback

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def print_system_info(device: torch.device):
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB"
        )


def main(
    dataset_dir: str,
    config_file: str,
    sequences_filter: list[str] = [],
    use_subsampled_frames: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_info(device)

    pipeline = Pipeline.build_pipeline_from_config(config_file, device)

    sequences_file = os.path.join(dataset_dir, "h36m_protocol1_sequences.json")

    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())

    if sequences_filter:
        sequences = [s for s in sequences if s["sequence_name"] in sequences_filter]
    sequences = [s for s in sequences if s["split"] == "val"]

    print("The following sequences will be processed:")
    for sequence in sequences:
        print(f"- {sequence['sequence_name']}")

    pbar = tqdm(sequences, desc="Processing sequences")

    failed_sequences = []

    for sequence in pbar:
        sequence_name = sequence["sequence_name"]

        pbar.set_postfix(sequence_name=sequence_name)

        try:
            cameras = list(sequence["views"].keys())

            keypoints_3d_file = sequence["annotations"]["keypoints_3d"]
            keypoints_3d_file = os.path.join(dataset_dir, keypoints_3d_file)

            with open(keypoints_3d_file, "rb") as f:
                gt_keypoints_3d = Keypoints3DAnnotations.from_dict(
                    orjson.loads(f.read())
                )

            cam_intrinsics_file = sequence["annotations"]["cameras_intrinsics"]
            cam_intrinsics_file = os.path.join(dataset_dir, cam_intrinsics_file)

            with open(cam_intrinsics_file, "rb") as f:
                gt_cam_intrinsics = CameraIntrinsicsAnnotations.from_dict(
                    orjson.loads(f.read())
                )

            cam_extrinsics_file = sequence["annotations"]["cameras_extrinsics"]
            cam_extrinsics_file = os.path.join(dataset_dir, cam_extrinsics_file)

            with open(cam_extrinsics_file, "rb") as f:
                gt_cam_extrinsics = CameraExtrinsicsAnnotations.from_dict(
                    orjson.loads(f.read())
                )

            views = []
            for camera in cameras:
                video_path = os.path.join(
                    dataset_dir, sequence["views"][camera]["video_path"]
                )
                if use_subsampled_frames:
                    selected_frames = sequence["views"][camera]["selected_frames"]
                    selected_frames_start = int(selected_frames["start"])
                    selected_frames_stop = int(selected_frames["stop"])
                    selected_frames_step = int(selected_frames["step"])
                    selected_frames = range(
                        selected_frames_start, selected_frames_stop, selected_frames_step
                    )
                else:
                    selected_frames = None

                views.append(
                    ViewInput(
                        view_id=camera,
                        frame_loader=VideoLoader(
                            video_path=video_path,
                            selected_frames=selected_frames,
                            device=device,
                        ),
                        audio_loader=None,
                    )
                )

            if not use_subsampled_frames:
                # H3.6M Ground truth is at 10 fps, we resample to 50 fps for visualization
                gt_keypoints_3d = gt_keypoints_3d.resample(
                    original_fps=10,
                    dst_timestamps=views[0]["frame_loader"].frame_timestamps_local,
                )

            _ = pipeline.run(
                sequence_name=sequence_name,
                views=views,
                annotations={},
                gt_annotations={
                    "keypoints_3d": gt_keypoints_3d,
                    "cameras_extrinsics": gt_cam_extrinsics,
                    "cameras_intrinsics": gt_cam_intrinsics,
                },
            )
        except Exception:
            tqdm.write(
                f"Error processing sequence {sequence_name}: {traceback.format_exc()}"
            )
            failed_sequences.append(sequence_name)
            continue

    pbar.close()

    print(f"Failed sequences: {failed_sequences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/experiments/viz/kps_formats/h36m_kps_format_viz_nlf_smpl.yaml",
    )
    parser.add_argument(
        "--sequences-filter",
        nargs="+",
        default=[],
        help="List of sequences to process",
    )
    parser.add_argument(
        "--use-subsampled-frames",
        action="store_true",
        default=False,
        help="Use subsampled frames",
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    config_file = args.config_file
    main(dataset_dir, config_file, args.sequences_filter, args.use_subsampled_frames)
