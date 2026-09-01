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
from kineo.datasets.egohumans.egohumans_dataset import EgoHumansSequenceDataset


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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_info(device)

    pipeline = Pipeline.build_pipeline_from_config(config_file, device)

    dataset = EgoHumansSequenceDataset(
        os.path.join(dataset_dir, "egohumans_sequences.json"), device=device
    )

    indices = [
        index
        for index, sequence_data in enumerate(dataset.sequences_data)
        if not sequences_filter
        or sequence_data["sequence_name"] in sequences_filter
    ]

    print("The following sequences will be processed:")
    for index in indices:
        print(f"- {dataset.sequences_data[index]['sequence_name']}")

    pbar = tqdm(indices, desc="Processing sequences")

    failed_sequences = []

    for index in pbar:
        sequence_name = dataset.sequences_data[index]["sequence_name"]

        pbar.set_description(f"Processing sequence: {sequence_name}")

        try:
            sequence = dataset[index]
            annotations = sequence["annotations"]

            gt_bboxes_2d = annotations["bboxes_2d"]
            gt_keypoints_2d = annotations["keypoints_2d"]
            gt_keypoints_3d = annotations["keypoints_3d"]
            gt_cam_intrinsics = annotations["cameras_intrinsics"]
            gt_cam_extrinsics = annotations["cameras_extrinsics"]

            cameras = [view["view_id"] for view in sequence["views_inputs"]]

            # Preprocess keeps moving cameras with their per-segment poses, but
            # the pipeline assumes one pose per view for the whole sequence, so
            # they are left out of the run rather than out of the dataset.
            non_static_views = [
                view_id
                for view_id in gt_cam_extrinsics.views_ids
                if not gt_cam_extrinsics.is_view_static(view_id)
            ]

            if non_static_views:
                tqdm.write(
                    f"{sequence_name}: skipping non-static views "
                    f"{', '.join(non_static_views)}"
                )

            cameras = [
                camera
                for camera in cameras
                if camera not in non_static_views
                and (not views_filter or camera in views_filter)
            ]

            gt_bboxes_2d = gt_bboxes_2d.filter_by_view_ids(cameras)
            gt_keypoints_2d = gt_keypoints_2d.filter_by_view_ids(cameras)
            gt_cam_extrinsics = gt_cam_extrinsics.filter_by_view_ids(cameras)
            gt_cam_intrinsics = gt_cam_intrinsics.filter_by_view_ids(cameras)

            views = [
                view
                for view in sequence["views_inputs"]
                if view["view_id"] in cameras
            ]

            _ = pipeline.run(
                sequence_name=sequence_name,
                views=views,
                annotations={},
                gt_annotations={
                    "bboxes_2d": gt_bboxes_2d,
                    "cameras_intrinsics": gt_cam_intrinsics,
                    "cameras_extrinsics": gt_cam_extrinsics,
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
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    config_file = args.config_file

    main(
        dataset_dir,
        config_file,
        args.sequences_filter,
        args.views_filter,
    )
