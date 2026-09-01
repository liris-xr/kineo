# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Evaluation of the pipeline on AIST++, over the unsynchronized raw videos.

Alongside the camera and human metrics the other benchmarks report, this scores
the time offsets the pipeline recovers against the ground-truth ones the
preprocessing derived from AIST's own refined cuts.
"""

import os

# For deterministic behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import traceback

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from kineo.datasets.aistpp.aistpp_dataset import AISTPPSequenceDataset
from kineo.datasets.aistpp.aistpp_download import VIDEO_VARIANTS
from kineo.pipeline.pipeline import Pipeline

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# The ground truth is the frame AIST cut its refined videos at, so half a frame
# is the tightest tolerance it can support.

GT_ANNOTATION_KEYS = (
    "keypoints_2d",
    "keypoints_3d",
    "bboxes_2d",
    "cameras_intrinsics",
    "cameras_extrinsics",
    "cameras_temporal",
    "global_time_reference",
)


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
    split: str = "pose_test",
    variant: str = "raw",
    sequences_filter: list[str] = [],
    use_cache: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_info(device)

    cfg = OmegaConf.load(config_file)
    if use_cache:
        cfg.use_cache = True
    pipeline = Pipeline.build_pipeline_from_config(cfg, device)

    sequences_file = os.path.join(
        dataset_dir, f"aistpp_{split}_{variant}_sequences.json"
    )
    dataset = AISTPPSequenceDataset(sequences_file, device)

    indices = range(len(dataset))
    if sequences_filter:
        indices = [
            index
            for index in indices
            if dataset.sequences_data[index]["sequence_name"] in sequences_filter
        ]

    print(f"The following sequences will be processed ({len(indices)}):")
    for index in indices:
        print(f"- {dataset.sequences_data[index]['sequence_name']}")

    failed_sequences = []
    processed_sequences = []

    pbar = tqdm(indices, desc="Processing sequences")

    for index in pbar:
        sequence = dataset[index]
        sequence_name = sequence["sequence_name"]
        pbar.set_postfix(sequence_name=sequence_name)

        gt_annotations = {
            key: sequence["annotations"][key]
            for key in GT_ANNOTATION_KEYS
            if key in sequence["annotations"]
        }
        try:
            pipeline.run(
                sequence_name=sequence_name,
                views=sequence["views_inputs"],
                annotations={},
                gt_annotations=gt_annotations,
            )

            processed_sequences.append(sequence_name)
        except Exception:
            tqdm.write(
                f"Error processing sequence {sequence_name}: {traceback.format_exc()}"
            )
            failed_sequences.append(sequence_name)
        finally:
            # A split holds thousands of videos, so readers cannot be left to
            # the garbage collector.
            for view in sequence["views_inputs"]:
                view["frame_loader"].close()

    pbar.close()

    print(f"Failed sequences: {failed_sequences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/experiments/benchmarks/aistpp_benchmark_nlf_estRt_estK_estDk1k2.yaml",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pose_test",
        help="AIST++ split to evaluate, as preprocessed",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=VIDEO_VARIANTS,
        default="raw",
        help="Video variant to evaluate. 'raw' videos are unsynchronized and "
        "carry ground-truth time offsets; 'refined' ones are already aligned.",
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
    main(
        args.dataset_dir,
        args.config_file,
        args.split,
        args.variant,
        args.sequences_filter,
        args.use_cache,
    )
