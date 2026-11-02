# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Generate the AIST++ time offset table rows."""

import argparse
import json
import os

_TEMPLATE = os.path.join(
    os.path.dirname(__file__), "aistpp_sync_rows_template.tex"
)


def format_offset(name: str, value: float) -> str:
    if name.startswith("ratio_within"):
        return f"{100 * value:.0f}"
    if name == "mean_signed_error_ms":
        return f"${value:+.1f}$"
    return f"{value:.1f}"


def fill_rows(
    template: str, summaries: dict[str, tuple[dict, dict]], n_decimals: int
) -> str:
    rows = template
    for arm, (metrics_summary, offsets_summary) in summaries.items():
        for section in ("human_stats", "cam_stats"):
            for name, stats in metrics_summary[section].items():
                rows = rows.replace(
                    f"{{kineo_{name}_aistpp_{arm}}}",
                    f"{stats['mean']:.{n_decimals}f}",
                )
        for name, value in offsets_summary["statistics"].items():
            rows = rows.replace(
                f"{{kineo_offset_{name}_aistpp_{arm}}}", format_offset(name, value)
            )
    return rows


def load_run_summaries(run_dir: str) -> tuple[dict, dict]:
    with open(os.path.join(run_dir, "metrics_summary.json"), "rb") as f:
        metrics_summary = json.load(f)
    with open(os.path.join(run_dir, "time_offsets_summary.json"), "rb") as f:
        offsets_summary = json.load(f)
    return metrics_summary, offsets_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "est_sync_run_dir",
        type=str,
        help="Run of aistpp_benchmark_nlf_estRt_estK_estD (MFCC offsets)",
    )
    parser.add_argument(
        "gt_sync_run_dir",
        type=str,
        help="Run of aistpp_benchmark_nlf_gtSync_estRt_estK_estD (GT offsets)",
    )
    parser.add_argument(
        "--n-decimals", type=int, default=3, help="Decimals of PA-MPJPE"
    )
    args = parser.parse_args()

    with open(_TEMPLATE, "r") as f:
        template = f.read()

    summaries = {
        "EstSync": load_run_summaries(args.est_sync_run_dir),
        "GtSync": load_run_summaries(args.gt_sync_run_dir),
    }
    print(fill_rows(template, summaries, args.n_decimals))
