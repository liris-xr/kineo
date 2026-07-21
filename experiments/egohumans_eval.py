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
from kineo.pipeline.pipeline import Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.datasets.egohumans.egohumans_smpl_gt import (
    load_egohumans_smpl_keypoints_3d,
)
from kineo.io.frame_sequence_loader import ImagesLoader
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.bboxes_2d import BBox2DAnnotations

from collections import defaultdict

import glob
import orjson
from tqdm import tqdm
import argparse
import traceback
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


def print_statistics(
    cam_metrics_stats: dict[str, float],
    human_metrics_stats: dict[str, float],
    failed_sequences: list[str],
):
    print("\n=== Statistics Report ===\n")
    print("📷 Camera Metrics:")
    for metric_name, metric_stats in cam_metrics_stats.items():
        print(f"- {metric_name}:")
        for key, value in metric_stats.items():
            print(f"\t- {key:<10}: {value:.4f}")

    print("\n🧑 Human Metrics:")
    for metric_name, metric_stats in human_metrics_stats.items():
        print(f"- {metric_name}:")
        for key, value in metric_stats.items():
            print(f"\t- {key:<10}: {value:.4f}")

    if failed_sequences:
        print("\n❌ Failed Sequences:")
        for seq in failed_sequences:
            print(f"  - {seq}")

    print("\n=========================\n")


def export_statistics(
    output_path: str,
    cam_metrics_stats: dict[str, dict[str, float]],
    human_metrics_stats: dict[str, dict[str, float]],
    failed_sequences: list[str],
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "camera_metrics": cam_metrics_stats,
                "human_metrics": human_metrics_stats,
                "failed_sequences": failed_sequences,
            },
            f,
            indent=2,
        )


def main(
    dataset_dir: str,
    config_file: str,
    sequences_filter: list[str] = [],
    views_filter: list[str] = [],
    human_gt_format: str = "coco",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_info(device)

    pipeline = Pipeline.build_pipeline_from_config(config_file, device)

    sequences_file = os.path.join(dataset_dir, "egohumans_sequences.json")

    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())

    if sequences_filter:
        sequences = [s for s in sequences if s["sequence_name"] in sequences_filter]
    
    print("The following sequences will be processed:")
    for sequence in sequences:
        print(f"- {sequence['sequence_name']}")

    pbar = tqdm(sequences, desc="Processing sequences")

    failed_sequences = []

    for sequence in pbar:
        sequence_name = sequence["sequence_name"]

        pbar.set_description(f"Processing sequence: {sequence_name}")

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

            if human_gt_format == "smpl":
                # SMPL GT lives beside the annotations dir:
                # <seq>/annotations/keypoints_3d.json -> <seq>/processed_data/smpl
                sequence_rel_dir = os.path.dirname(
                    os.path.dirname(sequence["annotations"]["keypoints_3d"])
                )
                smpl_dir = os.path.join(
                    dataset_dir, sequence_rel_dir, "processed_data", "smpl"
                )
                gt_keypoints_3d = load_egohumans_smpl_keypoints_3d(
                    smpl_dir,
                    valid_subject_ids=gt_keypoints_3d.subjects_ids,
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

            if views_filter:
                gt_bboxes_2d = gt_bboxes_2d.filter_by_view_ids(views_filter)
                gt_keypoints_2d = gt_keypoints_2d.filter_by_view_ids(views_filter)
                gt_cam_extrinsics = gt_cam_extrinsics.filter_by_view_ids(views_filter)
                gt_cam_intrinsics = gt_cam_intrinsics.filter_by_view_ids(views_filter)
                cameras = [c for c in cameras if c in views_filter]

            views = []
            for camera in cameras:
                images_dir = sequence["views"][camera]["images_dir"]
                fps = sequence["views"][camera]["fps"]
                imgs_paths = sorted(
                    glob.glob(
                        os.path.join(
                            dataset_dir,
                            images_dir,
                            "*.jpg",
                        )
                    )
                )
                n_imgs = len(imgs_paths)

                frame_timestamps_local = (torch.arange(n_imgs) / fps).tolist()

                views.append(
                    ViewInput(
                        view_id=camera,
                        frame_loader=ImagesLoader(
                            img_paths=imgs_paths,
                            frame_timestamps_local=frame_timestamps_local,
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
                    "camera_intrinsics": gt_cam_intrinsics,
                    "camera_extrinsics": gt_cam_extrinsics,
                    "keypoints_3d": gt_keypoints_3d,
                    "keypoints_2d": gt_keypoints_2d,
                },
            )
        except Exception as e:
            print(
                f"Error processing sequence {sequence_name}: {e}\n{traceback.format_exc()}"
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
        default="configs/pipeline/egohumans_eval_dwpose_smpl_scaling.yaml",
    )
    parser.add_argument(
        "--sequences-filter",
        nargs="+",
        default=[],
        help="List of sequences to process",
    )
    parser.add_argument(
        "--views-filter",
        nargs="+",
        default=[],
        help="List of views to process",
    )
    parser.add_argument(
        "--human-gt-format",
        type=str,
        default="coco",
        choices=["coco", "smpl"],
        help="GT joint set for human metrics: coco (default) or smpl (SMPL-22).",
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    config_file = args.config_file

    main(
        dataset_dir,
        config_file,
        args.sequences_filter,
        args.views_filter,
        args.human_gt_format,
    )
