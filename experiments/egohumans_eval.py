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
from kineo.pipeline.pipeline import Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.datasets.egohumans.egohumans_smpl_gt import (
    load_egohumans_smpl_keypoints_3d,
)
from kineo.eval.dataset_metrics import (
    aggregate_sequence_metrics_files,
    export_metrics_statistics,
    print_metrics_statistics,
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
    views_filter: list[str] = [],
    human_gt_format: str = "coco",
    use_cache: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_info(device)

    cfg = OmegaConf.load(config_file)
    if use_cache:
        cfg.use_cache = True
    pipeline = Pipeline.build_pipeline_from_config(cfg, device)

    # Benchmark configs that reproduce a fixed-view protocol (e.g. HSfM's 2-,
    # 4- and 8-view settings) carry the resolved camera set of every sequence
    # they cover; without the key every view is used.
    camera_selection = cfg.get("camera_selection", None)
    if camera_selection is not None:
        camera_selection = OmegaConf.to_container(camera_selection)

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
    processed_sequences = []

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

            sequence_views_filter = views_filter
            if camera_selection is not None:
                if sequence_name not in camera_selection:
                    print(
                        f"Skipping sequence {sequence_name}: not covered by the "
                        f"config's camera_selection"
                    )
                    continue
                sequence_views_filter = camera_selection[sequence_name]
                missing_views = [c for c in sequence_views_filter if c not in cameras]
                if missing_views:
                    raise ValueError(
                        f"Sequence {sequence_name} lacks the cameras "
                        f"{missing_views} its camera_selection asks for"
                    )
                print(f"Cameras for {sequence_name}: {sequence_views_filter}")

            if sequence_views_filter:
                gt_bboxes_2d = gt_bboxes_2d.filter_by_view_ids(sequence_views_filter)
                gt_keypoints_2d = gt_keypoints_2d.filter_by_view_ids(
                    sequence_views_filter
                )
                gt_cam_extrinsics = gt_cam_extrinsics.filter_by_view_ids(
                    sequence_views_filter
                )
                gt_cam_intrinsics = gt_cam_intrinsics.filter_by_view_ids(
                    sequence_views_filter
                )
                cameras = [c for c in cameras if c in sequence_views_filter]

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
            processed_sequences.append(sequence_name)
        except Exception as e:
            print(
                f"Error processing sequence {sequence_name}: {e}\n{traceback.format_exc()}"
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
        help="List of views to process. Ignored when the config carries a camera_selection.",
    )
    parser.add_argument(
        "--human-gt-format",
        type=str,
        default="coco",
        choices=["coco", "smpl"],
        help="GT joint set for human metrics: coco (default) or smpl (SMPL-22).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Override the config's use_cache to reuse cached stage outputs",
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
        args.use_cache,
    )
