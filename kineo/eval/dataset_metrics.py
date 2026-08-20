# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import json
import os

import numpy as np
import orjson
from typing import Any
from collections import defaultdict
from tqdm import tqdm

from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.stage_timing import StageTimingsAnnotations

from kineo.eval.sequence_metrics import (
    compute_camera_metrics_over_sequence,
    compute_human_metrics_over_sequence,
)

def _summarize(values_by_metric: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": np.nanmean(values),
            "median": np.nanmedian(values),
            "std": np.nanstd(values),
            "min": np.nanmin(values),
            "max": np.nanmax(values),
        }
        for name, values in values_by_metric.items()
    }


def aggregate_sequence_metrics_files(
    metrics_files: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Aggregates per-sequence metrics JSON files into overall statistics.

    Args:
        metrics_files: Paths to the per-sequence metrics JSON files written
            by the pipeline's metrics export stage.

    Returns:
        A (camera_metrics, human_metrics) pair mapping each metric name to
        its mean/median/std/min/max computed over the per-sequence means.
    """
    cam_values = defaultdict(list)
    human_values = defaultdict(list)

    for metrics_file in metrics_files:
        with open(metrics_file, "rb") as f:
            metrics = orjson.loads(f.read())
        for name, stats in metrics["cam_stats"].items():
            cam_values[name].append(stats["mean"])
        for name, stats in metrics["human_stats"].items():
            human_values[name].append(stats["mean"])

    return _summarize(cam_values), _summarize(human_values)


def print_metrics_statistics(
    cam_metrics_stats: dict[str, dict[str, float]],
    human_metrics_stats: dict[str, dict[str, float]],
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


def export_metrics_statistics(
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


def compute_predictions_metrics(
    gt_keypoints_3d_annotations: dict[str, Keypoints3DAnnotations],
    gt_cam_extrinsics_annotations: dict[str, CameraExtrinsicsAnnotations],
    gt_cam_intrinsics_annotations: dict[str, CameraIntrinsicsAnnotations],
    pred_keypoints_3d_annotations: dict[str, Keypoints3DAnnotations],
    pred_cam_extrinsics_annotations: dict[str, CameraExtrinsicsAnnotations],
    pred_cam_intrinsics_annotations: dict[str, CameraIntrinsicsAnnotations],
    stage_timings_annotations: dict[str, StageTimingsAnnotations],
) -> dict[str, Any]:
    """
    Compute the metrics for a set of predictions against a set of ground truth annotations.

    Args:
        gt_keypoints_3d_annotations: Dictionary of ground truth keypoints 3D annotations for each sequence.
        gt_cam_extrinsics_annotations: Dictionary of ground truth camera extrinsics annotations for each sequence.
        gt_cam_intrinsics_annotations: Dictionary of ground truth camera intrinsics annotations for each sequence.
        pred_keypoints_3d_annotations: Dictionary of predicted keypoints 3D annotations for each sequence.
        pred_cam_extrinsics_annotations: Dictionary of predicted camera extrinsics annotations for each sequence.
        pred_cam_intrinsics_annotations: Dictionary of predicted camera intrinsics annotations for each sequence.
        stage_timings_annotations: Dictionary of stage timings annotations for each sequence.

    Note: the keys of the dictionaries are the sequence names and should be the same for all the dictionaries.
    """
    sequences_names = set(gt_keypoints_3d_annotations.keys())
    missing_sequences_predictions = []

    all_human_metrics = defaultdict(list)
    all_camera_metrics = defaultdict(list)
    all_stage_timings = defaultdict(list)  # timings in seconds per frame
    n_total_frames = 0

    for sequence_name in tqdm(sequences_names, desc="Computing sequence metrics", leave=False):
        if (
            sequence_name not in pred_keypoints_3d_annotations
            or sequence_name not in pred_cam_extrinsics_annotations
            or sequence_name not in pred_cam_intrinsics_annotations
        ):
            missing_sequences_predictions.append(sequence_name)
            continue

        seq_gt_keypoints_3d = gt_keypoints_3d_annotations[sequence_name]
        seq_gt_cam_extrinsics = gt_cam_extrinsics_annotations[sequence_name]
        seq_gt_cam_intrinsics = gt_cam_intrinsics_annotations[sequence_name]
        seq_pred_keypoints_3d = pred_keypoints_3d_annotations[sequence_name]
        seq_pred_cam_extrinsics = pred_cam_extrinsics_annotations[sequence_name]
        seq_pred_cam_intrinsics = pred_cam_intrinsics_annotations[sequence_name]

        seq_human_metrics = compute_human_metrics_over_sequence(
            gt_keypoints_3d_annotations=seq_gt_keypoints_3d,
            gt_cam_extrinsics_annotations=seq_gt_cam_extrinsics,
            pred_keypoints_3d_annotations=seq_pred_keypoints_3d,
            pred_cam_extrinsics_annotations=seq_pred_cam_extrinsics,
        )
        seq_camera_metrics = compute_camera_metrics_over_sequence(
            gt_cam_intrinsics_annotations=seq_gt_cam_intrinsics,
            gt_cam_extrinsics_annotations=seq_gt_cam_extrinsics,
            pred_cam_intrinsics_annotations=seq_pred_cam_intrinsics,
            pred_cam_extrinsics_annotations=seq_pred_cam_extrinsics,
        )
        seq_human_metrics_avg = {
            key: seq_human_metrics[key]["mean"] for key in seq_human_metrics.keys()
        }
        seq_camera_metrics_avg = {
            key: seq_camera_metrics[key]["mean"] for key in seq_camera_metrics.keys()
        }

        for key, value in seq_human_metrics_avg.items():
            all_human_metrics[key].append(value)
        for key, value in seq_camera_metrics_avg.items():
            all_camera_metrics[key].append(value)

        if sequence_name in stage_timings_annotations:
            seq_stage_timings = stage_timings_annotations[sequence_name]
            for stage_timing in seq_stage_timings.annotations:
                n_frames = seq_gt_keypoints_3d.n_frames
                n_views = len(seq_gt_cam_extrinsics.views_ids)
                all_stage_timings[stage_timing.stage_name].append(
                    stage_timing.duration_seconds / (n_frames * n_views)
                )
                n_total_frames += n_frames * n_views

    all_human_metrics_stats = {
        human_metric_name: {
            "mean": np.nanmean(value),
            "median": np.nanmedian(value),
            "std": np.nanstd(value),
            "min": np.nanmin(value),
            "max": np.nanmax(value),
        }
        for human_metric_name, value in all_human_metrics.items()
    }

    all_camera_metrics_stats = {
        camera_metric_name: {
            "mean": np.nanmean(value),
            "median": np.nanmedian(value),
            "std": np.nanstd(value),
            "min": np.nanmin(value),
            "max": np.nanmax(value),
        }
        for camera_metric_name, value in all_camera_metrics.items()
    }

    all_stage_timings_stats = {
        stage_name: {
            "mean": np.nanmean(value),
            "median": np.nanmedian(value),
            "std": np.nanstd(value),
            "min": np.nanmin(value),
            "max": np.nanmax(value),
        }
        for stage_name, value in all_stage_timings.items()
    }

    return {
        "human_metrics": all_human_metrics_stats,
        "camera_metrics": all_camera_metrics_stats,
        "stage_timings": all_stage_timings_stats,
        "n_total_frames": n_total_frames,
        "missing_sequences_predictions": missing_sequences_predictions,
    }
