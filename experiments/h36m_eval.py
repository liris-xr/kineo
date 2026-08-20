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

# For deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from omegaconf import OmegaConf
from kineo.eval.dataset_metrics import (
    aggregate_sequence_metrics_files,
    export_metrics_statistics,
    print_metrics_statistics,
)
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
from collections import defaultdict
import json

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
    use_cache: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_info(device)

    cfg = OmegaConf.load(config_file)
    if use_cache:
        cfg.use_cache = True
    pipeline = Pipeline.build_pipeline_from_config(cfg, device)

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
    processed_sequences = []

    for sequence in pbar:
        sequence_name = sequence["sequence_name"]

        pbar.set_postfix(sequence_name=sequence_name)

        try:
            cameras = list(sequence["views"].keys())

            bboxes_2d_file = sequence["annotations"]["bboxes_2d"]
            bboxes_2d_file = os.path.join(dataset_dir, bboxes_2d_file)

            with open(bboxes_2d_file, "rb") as f:
                gt_bboxes_2d = BBox2DAnnotations.from_dict(orjson.loads(f.read()))

            keypoints_3d_file = sequence["annotations"]["keypoints_3d"]
            keypoints_3d_file = os.path.join(dataset_dir, keypoints_3d_file)

            with open(keypoints_3d_file, "rb") as f:
                gt_keypoints_3d = Keypoints3DAnnotations.from_dict(
                    orjson.loads(f.read())
                )

            keypoints_2d_file = sequence["annotations"]["keypoints_2d"]
            keypoints_2d_file = os.path.join(dataset_dir, keypoints_2d_file)

            with open(keypoints_2d_file, "rb") as f:
                gt_keypoints_2d = Keypoints2DAnnotations.from_dict(
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
                selected_frames = sequence["views"][camera]["selected_frames"]
                selected_frames_start = int(selected_frames["start"])
                selected_frames_stop = int(selected_frames["stop"])
                selected_frames_step = int(selected_frames["step"])
                selected_frames = range(
                    selected_frames_start, selected_frames_stop, selected_frames_step
                )

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

            _ = pipeline.run(
                sequence_name=sequence_name,
                views=views,
                annotations={},
                gt_annotations={
                    "bboxes_2d": gt_bboxes_2d,
                    "keypoints_2d": gt_keypoints_2d,
                    "keypoints_3d": gt_keypoints_3d,
                    "camera_extrinsics": gt_cam_extrinsics,
                    "camera_intrinsics": gt_cam_intrinsics,
                },
            )
            processed_sequences.append(sequence_name)
        except Exception:
            tqdm.write(
                f"Error processing sequence {sequence_name}: {traceback.format_exc()}"
            )
            failed_sequences.append(sequence_name)
            continue

    pbar.close()

    print(f"Failed sequences: {failed_sequences}")

    metrics_export_cfg = cfg.pipeline.stages.get("metrics_export", None)
    if metrics_export_cfg is None:
        print("No metrics_export stage in the config, skipping aggregation.")
        return

    metrics_path_template = metrics_export_cfg.runtime_cfg.output_path_template
    metrics_files = [
        metrics_path_template.format(sequence_name=name)
        for name in processed_sequences
    ]
    metrics_files = [f for f in metrics_files if os.path.isfile(f)]

    if not metrics_files:
        print("No per-sequence metrics files found, skipping aggregation.")
        return

    cam_metrics_stats, human_metrics_stats = aggregate_sequence_metrics_files(
        metrics_files
    )
    print_metrics_statistics(
        cam_metrics_stats, human_metrics_stats, failed_sequences
    )
    export_metrics_statistics(
        os.path.join(cfg.output_root_dir, "metrics_summary.json"),
        cam_metrics_stats,
        human_metrics_stats,
        failed_sequences,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/pipeline/h36m_eval_rtmpose_smpl_scaling.yaml",
    )
    parser.add_argument(
        "--sequences-filter",
        nargs="+",
        default=[],
        help="List of sequences to process",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Override the config's use_cache to reuse cached stage outputs",
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    config_file = args.config_file
    main(dataset_dir, config_file, args.sequences_filter, args.use_cache)
