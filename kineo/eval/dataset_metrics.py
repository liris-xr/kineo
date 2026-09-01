# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import numpy as np
import orjson
from typing import Any, Sequence
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


def _summarize(values_by_name: dict[str, list]) -> dict[str, dict[str, Any]]:
    """Mean, median, std, min and max of each named series, ignoring NaNs."""
    return {
        name: {
            "mean": np.nanmean(values),
            "median": np.nanmedian(values),
            "std": np.nanstd(values),
            "min": np.nanmin(values),
            "max": np.nanmax(values),
        }
        for name, values in values_by_name.items()
    }


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

    for sequence_name in tqdm(
        sequences_names, desc="Computing sequence metrics", leave=False
    ):
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

    all_human_metrics_stats = _summarize(all_human_metrics)

    all_camera_metrics_stats = _summarize(all_camera_metrics)

    all_stage_timings_stats = _summarize(all_stage_timings)

    return {
        "human_metrics": all_human_metrics_stats,
        "camera_metrics": all_camera_metrics_stats,
        "stage_timings": all_stage_timings_stats,
        "n_total_frames": n_total_frames,
        "missing_sequences_predictions": missing_sequences_predictions,
    }


def aggregate_sequence_metrics_files(
    metrics_files: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Aggregates per-sequence metrics files over a dataset.

    Args:
        metrics_files: Paths to the JSON files a metrics export stage wrote,
            one per sequence.

    Returns:
        The camera and human statistics, each mapping a metric name to
        statistics taken over the sequences' mean values.
    """
    camera: dict[str, list] = defaultdict(list)
    human: dict[str, list] = defaultdict(list)

    for filepath in metrics_files:
        with open(filepath, "rb") as f:
            sequence_metrics = orjson.loads(f.read())

        for key, collected in (("cam_stats", camera), ("human_stats", human)):
            for name, stats in sequence_metrics.get(key, {}).items():
                collected[name].append(stats["mean"])

    return _summarize(camera), _summarize(human)


def print_metrics_statistics(
    camera_stats: dict[str, dict[str, Any]],
    human_stats: dict[str, dict[str, Any]],
    failed_sequences: Sequence[str] = (),
):
    """Prints dataset-level statistics, and whatever failed to produce them."""
    for title, stats in (("Camera", camera_stats), ("Human", human_stats)):
        print()
        print(f"{title} metrics")
        for name, values in stats.items():
            print(
                f"  {name:<14} mean {values['mean']:<10.4f} "
                f"median {values['median']:<10.4f} std {values['std']:<10.4f} "
                f"[{values['min']:.4f}, {values['max']:.4f}]"
            )

    if failed_sequences:
        print()
        print(f"Failed sequences ({len(failed_sequences)}):")
        for sequence_name in failed_sequences:
            print(f"  {sequence_name}")


def export_metrics_statistics(
    filepath: str,
    camera_stats: dict[str, dict[str, Any]],
    human_stats: dict[str, dict[str, Any]],
    failed_sequences: Sequence[str] = (),
):
    """Writes dataset-level statistics as JSON."""
    with open(filepath, "wb") as f:
        f.write(
            orjson.dumps(
                {
                    "cam_stats": camera_stats,
                    "human_stats": human_stats,
                    "failed_sequences": list(failed_sequences),
                },
                default=float,
                option=orjson.OPT_INDENT_2,
            )
        )
