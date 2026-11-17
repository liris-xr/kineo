# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Generate the EgoHumans sampling strategy tables."""

import argparse
import glob
import os
import statistics

import orjson

from experiments.figures.generate_online_rows import format_metric

_TEMPLATE = os.path.join(
    os.path.dirname(__file__), "sampling_rows_template.tex"
)

COLUMNS = (
    ("TE", "cam_stats"),
    ("s-TE", "cam_stats"),
    ("AE", "cam_stats"),
    ("vfov_error", "cam_stats"),
    ("w-mpjpe", "human_stats"),
    ("ga-mpjpe", "human_stats"),
    ("pa-mpjpe", "human_stats"),
    ("sampling_seconds", None),
)
SAMPLING_STAGES = ("Keypoints Pairs Sampling", "Bundle Adjustment Sampling")
CATEGORY_LABELS = {"legoassemble": "Lego Assembly"}
ROW_ORDER = ("FPS", "Top-P", "Random")
ARM_LABELS = {"Random": "Random (ours)"}


def scored_sequences(run_dir: str) -> dict[str, str]:
    pattern = os.path.join(run_dir, "metrics", "*.json")
    return {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob.glob(pattern)
    }


def load_sequence(path: str) -> dict[str, float]:
    with open(path, "rb") as f:
        metrics = orjson.loads(f.read())
    durations = {
        timing["stage_name"]: timing["duration_seconds"]
        for timing in metrics["stage_timings"]
    }
    row = {
        name: metrics[section][name]["mean"]
        for name, section in COLUMNS
        if section is not None
    }
    row["sampling_seconds"] = sum(durations[s] for s in SAMPLING_STAGES)
    return row


def category_of(sequence_name: str) -> str:
    return sequence_name.rsplit("_", 1)[0]


def summarize(
    rows: dict[str, dict[str, float]], sequences: list[str]
) -> dict[str, float]:
    return {
        name: statistics.median(
            rows[s][name] for s in sequences if rows[s][name] is not None
        )
        for name, _ in COLUMNS
    }


def format_cells(stats: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    cells = {arm: [] for arm in stats}
    for name, _ in COLUMNS:
        printed = {
            arm: format_metric(name, arm_stats[name])
            for arm, arm_stats in stats.items()
        }
        values = sorted({float(text) for text in printed.values()})
        holders = {v: [a for a, t in printed.items() if float(t) == v] for v in values}
        best = holders[values[0]]
        second = holders[values[1]] if len(values) > 1 and len(best) == 1 else []
        for arm, text in printed.items():
            if len(best) == 1 and best[0] == arm:
                text = rf"\textbf{{{text}}}"
            elif len(stats) >= 3 and len(second) == 1 and second[0] == arm:
                text = rf"\underline{{{text}}}"
            cells[arm].append(text)
    return cells


def table_rows(
    cells: dict[str, list[str]], category: str | None = None
) -> list[str]:
    lines = []
    ordered = sorted(cells, key=ROW_ORDER.index)
    for i, arm in enumerate(ordered):
        lead = []
        if category is not None:
            label = CATEGORY_LABELS.get(category, category.capitalize())
            lead = [rf"\multirow{{{len(cells)}}}{{*}}{{{label}}}" if i == 0 else ""]
        lines.append("        " + " & ".join(lead + [ARM_LABELS.get(arm, arm)] + cells[arm]) + r" \\")
    return lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "random_run_dir",
        type=str,
        help="Run that samples keypoints uniformly at random",
    )
    parser.add_argument(
        "fps_run_dir",
        type=str,
        help="Run that samples keypoints by farthest-point sampling",
    )
    parser.add_argument(
        "--topp-run-dir",
        type=str,
        help="Run that keeps the most confident keypoints (top-P)",
    )
    parser.add_argument(
        "--runtime-json",
        type=str,
        help="Sampling runtimes from experiments/benchmark_sampling_runtime.py",
    )
    args = parser.parse_args()

    scored = {
        "Random": scored_sequences(args.random_run_dir),
        "FPS": scored_sequences(args.fps_run_dir),
    }
    if args.topp_run_dir:
        scored["Top-P"] = scored_sequences(args.topp_run_dir)
    common = sorted(set.intersection(*(set(files) for files in scored.values())))
    if not common:
        raise FileNotFoundError("No sequence was scored by every run")
    rows = {
        arm: {name: load_sequence(files[name]) for name in common}
        for arm, files in scored.items()
    }
    if args.runtime_json:
        with open(args.runtime_json, "rb") as f:
            timed = orjson.loads(f.read())["per_sequence"]
        for arm, arm_rows in rows.items():
            for name, row in arm_rows.items():
                timing = timed[arm].get(name)
                row["sampling_seconds"] = (
                    timing["pairs"] + timing["ba"] if timing else None
                )

    global_rows = table_rows(
        format_cells({arm: summarize(r, common) for arm, r in rows.items()})
    )
    category_rows = []
    categories = sorted({category_of(name) for name in common})
    for category in categories:
        members = [name for name in common if category_of(name) == category]
        if category_rows:
            category_rows.append(r"        \midrule")
        category_rows += table_rows(
            format_cells(
                {arm: summarize(r, members) for arm, r in rows.items()}
            ),
            category,
        )

    with open(_TEMPLATE, "r") as f:
        template = f.read()
    print(
        template.replace("{n_sequences}", str(len(common)))
        .replace("{global_rows}", "\n".join(global_rows))
        .replace("{category_rows}", "\n".join(category_rows))
    )
