# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Generate the rows of the audio sync ablation table on AIST++."""

import argparse
import json
import os
import statistics
import sys

_TEMPLATE = os.path.join(
    os.path.dirname(__file__), "sync_ablation_rows_template.tex"
)

ARMS = (
    "Beep",
    "NoBeep",
    "Park20dB",
    "Park10dB",
    "Park0dB",
    "Cafeteria20dB",
    "Cafeteria10dB",
    "Cafeteria0dB",
    "NoBeepSynthBeep",
)

MARKDOWN_COLUMNS = (
    ("median_error_ms", "median (ms)"),
    ("mean_error_ms", "mean (ms)"),
    ("p95_error_ms", "p95 (ms)"),
    ("mean_signed_error_ms", "bias (ms)"),
    ("ratio_within_1_frame", "<= 17 ms (%)"),
    ("ratio_within_2_frames", "<= 33 ms (%)"),
    ("ratio_within_4_frames", "<= 67 ms (%)"),
    ("confidence_median", "conf median"),
    ("confidence_min", "conf min"),
)


def confidence_statistics(summary: dict) -> dict[str, float]:
    confidences = [
        c
        for views in summary["confidences_by_sequence"].values()
        for c in views.values()
    ]
    return {
        "confidence_median": statistics.median(confidences),
        "confidence_min": min(confidences),
    }


def load_run(run_dir: str) -> dict[str, float]:
    with open(os.path.join(run_dir, "time_offsets_summary.json"), "rb") as f:
        summary = json.load(f)
    return {**summary["statistics"], **confidence_statistics(summary)}


def format_cell(name: str, value: float, n_decimals: int) -> str:
    if name.startswith("ratio_within"):
        return f"{100 * value:.0f}"
    if name == "mean_signed_error_ms":
        return f"${value:+.1f}$"
    if name.endswith("_ms"):
        return f"{value:.1f}"
    return f"{value:.{n_decimals}f}"


def fill_rows(template: str, values: dict[str, dict], n_decimals: int) -> str:
    rows = template
    for arm, stats in values.items():
        for name, value in stats.items():
            rows = rows.replace(
                f"{{kineo_offset_{name}_aistpp_{arm}}}",
                format_cell(name, value, n_decimals),
            )
    return rows


def markdown_table(values: dict[str, dict], n_decimals: int) -> str:
    header = "| arm | " + " | ".join(label for _, label in MARKDOWN_COLUMNS) + " |"
    sep = "|---" * (len(MARKDOWN_COLUMNS) + 1) + "|"
    lines = [header, sep]
    for arm, stats in values.items():
        cells = [format_cell(key, stats[key], n_decimals) for key, _ in MARKDOWN_COLUMNS]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"{arm}_run_dir", type=str)
    parser.add_argument("--n-decimals", type=int, default=2)
    args = parser.parse_args()

    values = {arm: load_run(getattr(args, f"{arm}_run_dir")) for arm in ARMS}

    with open(_TEMPLATE, "r") as f:
        template = f.read()
    print(fill_rows(template, values, args.n_decimals))
    print(markdown_table(values, args.n_decimals), file=sys.stderr)
