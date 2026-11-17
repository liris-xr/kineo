# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Summarize a benchmark run over every sequence it wrote to disk."""

import argparse
import glob
import os

import orjson

from kineo.eval.dataset_metrics import (
    aggregate_sequence_metrics_files,
    export_metrics_statistics,
    print_metrics_statistics,
)
from kineo.eval.run_scoring import (
    load_sequence_list,
    load_sequence_names,
    score_time_offsets,
)
from kineo.eval.temporal_metrics import (
    print_time_offset_statistics,
    summarize_time_offset_errors,
)


def scored_sequences(benchmark_dir: str) -> dict[str, str]:
    metrics_files = glob.glob(os.path.join(benchmark_dir, "metrics", "*.json"))
    return {os.path.splitext(os.path.basename(f))[0]: f for f in metrics_files}


def main(
    benchmark_dir: str,
    dataset_file: str | None,
    fps: float,
    exclude_list: str | None = None,
    common_with: list[str] = (),
):
    excluded = set()
    if exclude_list is not None:
        excluded = load_sequence_list(exclude_list)
    scored = scored_sequences(benchmark_dir)

    not_common = set()
    for other_dir in common_with:
        not_common |= set(scored) - set(scored_sequences(other_dir))
    excluded |= not_common

    excluded_scored = sorted(excluded & set(scored))
    metrics_files = [
        f for name, f in sorted(scored.items()) if name not in excluded
    ]

    if not metrics_files:
        raise FileNotFoundError(f"No per-sequence metrics under {benchmark_dir}")

    failed = []
    if dataset_file is not None:
        listed = set(load_sequence_names(dataset_file)) - excluded
        failed = sorted(listed - set(scored))

    print(
        f"Summarizing {len(metrics_files)} sequences from {benchmark_dir}, "
        f"{len(excluded_scored)} excluded ({len(not_common)} not scored by "
        f"every run compared with), {len(failed)} failed"
    )

    camera_stats, human_stats = aggregate_sequence_metrics_files(metrics_files)
    print_metrics_statistics(camera_stats, human_stats)
    export_metrics_statistics(
        os.path.join(benchmark_dir, "metrics_summary.json"),
        camera_stats,
        human_stats,
        failed_sequences=failed,
        excluded_sequences=excluded_scored,
    )

    if dataset_file is None:
        return

    errors = score_time_offsets(benchmark_dir, dataset_file, excluded)
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
    parser.add_argument(
        "--exclude-list",
        type=str,
        default=None,
        help="Sequence names to leave out, one per line, '#' comments",
    )
    parser.add_argument(
        "--common-with",
        nargs="+",
        default=(),
        metavar="BENCHMARK_DIR",
    )
    args = parser.parse_args()
    main(
        args.benchmark_dir,
        args.dataset_file,
        args.fps,
        args.exclude_list,
        args.common_with,
    )
