# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Summarizes a benchmark run from what it wrote to disk.

Kept out of the evaluation scripts so that a run split over several processes
summarizes once, over every sequence, instead of each process writing a summary
of the sequences it happened to be given.
"""

import argparse
import glob
import os
import pickle

import orjson

from kineo.annotations.camera_temporal import CameraTemporalAnnotations
from kineo.eval.dataset_metrics import (
    aggregate_sequence_metrics_files,
    export_metrics_statistics,
    print_metrics_statistics,
)
from kineo.eval.temporal_metrics import (
    compute_time_offset_errors,
    print_time_offset_statistics,
    summarize_time_offset_errors,
)


def score_time_offsets(
    benchmark_dir: str, dataset_file: str
) -> dict[str, dict[str, float]]:
    """Scores the offsets a run recovered against the dataset's own.

    Args:
        benchmark_dir: Directory the run wrote its annotations to.
        dataset_file: The dataset's `sequences.json`, read for ground truth.

    Returns:
        Signed per-view errors, in seconds, by sequence.
    """
    dataset_dir = os.path.dirname(dataset_file)

    with open(dataset_file, "rb") as f:
        sequences = orjson.loads(f.read())

    errors = {}

    for sequence in sequences:
        name = sequence["sequence_name"]
        predicted_path = os.path.join(
            benchmark_dir, "annotations", name, "cameras_temporal.pkl"
        )
        gt_relpath = sequence.get("annotations", {}).get("cameras_temporal")

        if gt_relpath is None or not os.path.isfile(predicted_path):
            continue

        with open(os.path.join(dataset_dir, gt_relpath), "rb") as f:
            gt = CameraTemporalAnnotations.from_dict(orjson.loads(f.read()))

        with open(predicted_path, "rb") as f:
            predicted = CameraTemporalAnnotations.from_dict(pickle.load(f))

        errors[name] = compute_time_offset_errors(gt, predicted)

    return errors


def main(benchmark_dir: str, dataset_file: str | None, fps: float):
    metrics_files = sorted(glob.glob(os.path.join(benchmark_dir, "metrics", "*.json")))

    if not metrics_files:
        raise FileNotFoundError(f"No per-sequence metrics under {benchmark_dir}")

    print(f"Summarizing {len(metrics_files)} sequences from {benchmark_dir}")

    camera_stats, human_stats = aggregate_sequence_metrics_files(metrics_files)
    print_metrics_statistics(camera_stats, human_stats)
    export_metrics_statistics(
        os.path.join(benchmark_dir, "metrics_summary.json"),
        camera_stats,
        human_stats,
    )

    if dataset_file is None:
        return

    errors = score_time_offsets(benchmark_dir, dataset_file)
    summary = summarize_time_offset_errors(errors, fps)
    print_time_offset_statistics(summary, fps)

    with open(os.path.join(benchmark_dir, "time_offsets_summary.json"), "wb") as f:
        f.write(
            orjson.dumps(
                {"statistics": summary, "errors_by_sequence": errors},
                option=orjson.OPT_INDENT_2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark_dir", type=str, help="Directory holding metrics/ and annotations/"
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="A dataset's sequences.json, to also score recovered time offsets",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60000 / 1001,
        help="Rate the recordings run at, to express offset errors in frames",
    )
    args = parser.parse_args()
    main(args.benchmark_dir, args.dataset_file, args.fps)
