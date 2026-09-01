# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Scoring of recovered inter-camera time offsets."""

import statistics

from kineo.annotations.camera_temporal import CameraTemporalAnnotations


def compute_time_offset_errors(
    gt: CameraTemporalAnnotations,
    pred: CameraTemporalAnnotations,
) -> dict[str, float]:
    """Scores recovered inter-camera time offsets against the ground truth.

    Both sides are rebased on a shared reference view: only inter-view
    differences are observable, so a common offset is not an error and whichever
    view each side pinned to zero must not count.

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
    errors_by_sequence: dict[str, dict[str, float]], fps: float
) -> dict[str, float]:
    """Aggregates per-view offset errors over every scored sequence.

    Reported in frames as well as milliseconds: the ground truth is quantized to
    a frame boundary, so picking the frame the annotation sits on is what
    matters downstream, while milliseconds keep the sub-frame bias frames round
    away and compare across datasets running at other rates.

    Args:
        errors_by_sequence: Signed per-view errors, in seconds, by sequence.
        fps: Rate the recordings run at, to express errors in frames.

    Returns:
        The aggregated statistics, or an empty mapping if nothing was scored.
    """
    signed = [
        error
        for view_errors in errors_by_sequence.values()
        for error in view_errors.values()
    ]

    if not signed:
        return {}

    tolerance_s = 0.5 / fps
    errors = sorted(abs(error) for error in signed)
    frames = [error * fps for error in errors]
    n = len(errors)

    return {
        "n_sequences": len(errors_by_sequence),
        "n_views": n,
        "mean_error_ms": statistics.fmean(errors) * 1e3,
        "median_error_ms": statistics.median(errors) * 1e3,
        "p95_error_ms": errors[int(0.95 * (n - 1))] * 1e3,
        "max_error_ms": errors[-1] * 1e3,
        # Signed, so a systematic lead or lag shows up instead of averaging away.
        "mean_signed_error_ms": statistics.fmean(signed) * 1e3,
        "mean_error_frames": statistics.fmean(frames),
        "median_error_frames": statistics.median(frames),
        "max_error_frames": frames[-1],
        "mean_signed_error_frames": statistics.fmean(signed) * fps,
        "ratio_within_tolerance": sum(e <= tolerance_s for e in errors) / n,
        "ratio_within_1_frame": sum(f <= 1.0 for f in frames) / n,
        "ratio_within_2_frames": sum(f <= 2.0 for f in frames) / n,
    }


def print_time_offset_statistics(summary: dict[str, float], fps: float):
    """Prints the aggregated offset statistics."""
    if not summary:
        print(
            "\nNo time offsets were recovered. Does the config run a temporal "
            "calibration stage?"
        )
        return

    print("\nTime offset metrics")
    print(
        f"  sequences scored     {summary['n_sequences']} "
        f"({summary['n_views']} views)"
    )
    print(
        f"  within tolerance     {summary['ratio_within_tolerance']:.1%} "
        f"(+/-{0.5 / fps * 1e3:.2f} ms, half a frame)"
    )
    print(f"  within 1 frame       {summary['ratio_within_1_frame']:.1%}")
    print(f"  within 2 frames      {summary['ratio_within_2_frames']:.1%}")
    print(
        f"  error (frames)       mean {summary['mean_error_frames']:.3f}"
        f"  median {summary['median_error_frames']:.3f}"
        f"  max {summary['max_error_frames']:.3f}"
    )
    print(
        f"  error (ms)           mean {summary['mean_error_ms']:.2f}"
        f"  median {summary['median_error_ms']:.2f}"
        f"  p95 {summary['p95_error_ms']:.2f}"
        f"  max {summary['max_error_ms']:.2f}"
    )
    print(
        f"  signed bias          {summary['mean_signed_error_frames']:+.3f} frames "
        f"({summary['mean_signed_error_ms']:+.2f} ms)"
    )
