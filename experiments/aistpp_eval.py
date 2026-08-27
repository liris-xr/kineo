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
import statistics
import traceback

import orjson
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from kineo.annotations.camera_temporal import CameraTemporalAnnotations
from kineo.annotations.global_time_reference import (
    GlobalTimeReferenceAnnotation,
    GlobalTimeReferenceAnnotations,
    GlobalTimeReferenceAnnotationsMetadata,
)
from kineo.datasets.aistpp.aistpp_dataset import VIDEO_FPS, AISTPPSequenceDataset
from kineo.datasets.aistpp.aistpp_download import VIDEO_VARIANTS
from kineo.eval.dataset_metrics import (
    aggregate_sequence_metrics_files,
    export_metrics_statistics,
    print_metrics_statistics,
)
from kineo.pipeline.pipeline import Pipeline

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# The ground truth is the frame AIST cut its refined videos at, so it is exact
# on the frame grid and says nothing finer. Half a frame is therefore the
# tightest tolerance it can support: an estimate inside it selects the frame
# the annotation sits on, and a perfect estimate cannot fall outside it.
GT_TOLERANCE_S = 0.5 / VIDEO_FPS

GT_ANNOTATION_KEYS = (
    "keypoints_2d",
    "keypoints_3d",
    "bboxes_2d",
    "cameras_intrinsics",
    "cameras_extrinsics",
    "cameras_temporal",
)


def print_system_info(device: torch.device):
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB"
        )


def compute_time_offset_errors(
    gt: CameraTemporalAnnotations,
    pred: CameraTemporalAnnotations,
) -> dict[str, float]:
    """Scores recovered inter-camera time offsets against the ground truth.

    Both sides are rebased on a shared reference view. Only the differences
    between views are observable -- shifting every camera by the same amount
    describes the same synchronization -- so a common offset is not an error,
    and whichever view each side happened to pin to zero must not count.

    Args:
        gt: Ground-truth time offsets, one annotation per view.
        pred: Time offsets recovered by the pipeline.

    Returns:
        Maps each view scored to its signed offset error, in seconds, positive
        when the estimate runs late. Views missing from either side are left
        out.

    Raises:
        ValueError: If the two share no view to score.
    """
    gt_offsets = {a.view_id: a.time_offset for a in gt.annotations}
    pred_offsets = {a.view_id: a.time_offset for a in pred.annotations}

    shared_views = sorted(set(gt_offsets) & set(pred_offsets))
    if not shared_views:
        raise ValueError(
            "Ground-truth and predicted time offsets share no view: "
            f"{sorted(gt_offsets)} vs {sorted(pred_offsets)}."
        )

    reference_view = shared_views[0]

    return {
        view_id: (pred_offsets[view_id] - pred_offsets[reference_view])
        - (gt_offsets[view_id] - gt_offsets[reference_view])
        for view_id in shared_views
    }


def summarize_time_offset_errors(
    errors_by_sequence: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Aggregates per-view offset errors over every scored sequence.

    Reported in frames as well as milliseconds: the ground truth is quantized
    to a frame boundary, so a frame is the finest distinction it can support,
    and picking the same frame as the annotation is what actually matters
    downstream. Milliseconds are kept because they survive that quantization,
    carry the sub-frame bias frames round away, and compare across datasets
    running at other rates.

    Per view, no difference below the tolerance is meaningful. Over a whole
    split the quantization averages out, so `mean_signed_error_ms` separates
    two methods well below a frame where `mean_error_ms`, inflated by that same
    noise, cannot.
    """
    signed = [
        error
        for view_errors in errors_by_sequence.values()
        for error in view_errors.values()
    ]

    if not signed:
        return {}

    errors = sorted(abs(error) for error in signed)
    signed_frames = [error * VIDEO_FPS for error in signed]
    frames = sorted(abs(error) for error in signed_frames)
    n = len(errors)

    return {
        "n_sequences": len(errors_by_sequence),
        "n_views": n,
        "mean_error_ms": statistics.fmean(errors) * 1e3,
        "median_error_ms": statistics.median(errors) * 1e3,
        "p95_error_ms": errors[int(0.95 * (n - 1))] * 1e3,
        "max_error_ms": errors[-1] * 1e3,
        # Signed, so a systematic lead or lag shows up instead of averaging
        # away against itself.
        "mean_signed_error_ms": statistics.fmean(signed) * 1e3,
        "mean_error_frames": statistics.fmean(frames),
        "median_error_frames": statistics.median(frames),
        "max_error_frames": frames[-1],
        "mean_signed_error_frames": statistics.fmean(signed_frames),
        # Within half a frame is both the tolerance the ground truth supports
        # and the condition for resampling to pick the frame AIST picked.
        "ratio_within_tolerance": sum(e <= GT_TOLERANCE_S for e in errors) / n,
        "ratio_within_1_frame": sum(f <= 1.0 for f in frames) / n,
        "ratio_within_2_frames": sum(f <= 2.0 for f in frames) / n,
    }


def print_time_offset_statistics(statistics_summary: dict[str, float]):
    if not statistics_summary:
        print(
            "\nNo time offsets were recovered. Does the config run a temporal "
            "calibration stage?"
        )
        return

    print("\nTime offset metrics")
    print(
        f"  sequences scored     {statistics_summary['n_sequences']} "
        f"({statistics_summary['n_views']} views)"
    )
    print(
        f"  within tolerance     "
        f"{statistics_summary['ratio_within_tolerance']:.1%} "
        f"(+/-{GT_TOLERANCE_S * 1e3:.2f} ms, half a frame)"
    )
    print(f"  within 1 frame       {statistics_summary['ratio_within_1_frame']:.1%}")
    print(f"  within 2 frames      {statistics_summary['ratio_within_2_frames']:.1%}")
    print(
        f"  error (frames)       mean {statistics_summary['mean_error_frames']:.3f}"
        f"  median {statistics_summary['median_error_frames']:.3f}"
        f"  max {statistics_summary['max_error_frames']:.3f}"
    )
    print(
        f"  error (ms)           mean {statistics_summary['mean_error_ms']:.2f}"
        f"  median {statistics_summary['median_error_ms']:.2f}"
        f"  p95 {statistics_summary['p95_error_ms']:.2f}"
        f"  max {statistics_summary['max_error_ms']:.2f}"
    )
    print(
        f"  signed bias          "
        f"{statistics_summary['mean_signed_error_frames']:+.3f} frames "
        f"({statistics_summary['mean_signed_error_ms']:+.2f} ms)"
    )


def build_gt_time_reference(
    n_frames: int, fps: float
) -> GlobalTimeReferenceAnnotations:
    """Timeline the ground-truth frame indices sit on.

    Lets the metrics align a prediction resampled onto its own grid with the
    annotation by timestamp instead of by frame index. Only the timestamps are
    read, so the per-view frame mapping is left empty rather than restated
    here, where the offsets that define it are not known.
    """
    return GlobalTimeReferenceAnnotations(
        metadata=GlobalTimeReferenceAnnotationsMetadata(),
        annotations=[
            GlobalTimeReferenceAnnotation(
                timestamps=torch.arange(n_frames, dtype=torch.float64) / fps,
                closest_local_frame_idx={},
            )
        ],
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
    time_offset_errors = {}

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
        sequence_data = dataset.sequences_data[index]
        gt_annotations["global_time_reference"] = build_gt_time_reference(
            sequence_data["n_frames"], sequence_data["fps"]
        )

        try:
            predictions = pipeline.run(
                sequence_name=sequence_name,
                views=sequence["views_inputs"],
                annotations={},
                gt_annotations=gt_annotations,
            )

            predicted_temporal = predictions.get("cameras_temporal")
            if predicted_temporal is not None and "cameras_temporal" in gt_annotations:
                time_offset_errors[sequence_name] = compute_time_offset_errors(
                    gt_annotations["cameras_temporal"], predicted_temporal
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

    time_offset_statistics = summarize_time_offset_errors(time_offset_errors)
    print_time_offset_statistics(time_offset_statistics)

    os.makedirs(cfg.output_root_dir, exist_ok=True)
    time_offsets_path = os.path.join(
        cfg.output_root_dir, f"time_offsets_summary_{variant}.json"
    )
    with open(time_offsets_path, "wb") as f:
        f.write(
            orjson.dumps(
                {
                    "statistics": time_offset_statistics,
                    "errors_by_sequence": time_offset_errors,
                    "failed_sequences": failed_sequences,
                },
                option=orjson.OPT_INDENT_2,
            )
        )
    print(f'Saved time offset metrics to "{time_offsets_path}"')

    metrics_export_cfg = cfg.pipeline.stages.get("metrics_export", None)
    if metrics_export_cfg is None:
        print("No metrics_export stage in the config, skipping aggregation.")
        return

    metrics_path_template = metrics_export_cfg.runtime_cfg.output_path_template
    metrics_files = [
        metrics_path_template.format(sequence_name=name) for name in processed_sequences
    ]
    metrics_files = [f for f in metrics_files if os.path.isfile(f)]

    if not metrics_files:
        print("No per-sequence metrics files found, skipping aggregation.")
        return

    cam_metrics_stats, human_metrics_stats = aggregate_sequence_metrics_files(
        metrics_files
    )
    print_metrics_statistics(cam_metrics_stats, human_metrics_stats, failed_sequences)
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
