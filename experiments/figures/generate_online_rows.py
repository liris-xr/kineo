# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Generate the H3.6M online vs offline table rows."""

import argparse
import glob
import os

import orjson

from experiments import summarize_online_latency
from experiments.report_online_benchmark import count_views
from kineo.eval.dataset_metrics import aggregate_sequence_metrics_files

_TEMPLATE = os.path.join(
    os.path.dirname(__file__), "online_rows_template.tex"
)

_IMPLAUSIBLE_ANGULAR_ERROR_DEG = 10.0

METRES = ("TE", "s-TE", "w-mpjpe", "ga-mpjpe", "pa-mpjpe")


def scored_sequences(run_dir: str, arm_name: str) -> dict[str, str]:
    pattern = os.path.join(
        run_dir, f"stream_{arm_name}", "metrics", "*.json"
    )
    return {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in glob.glob(pattern)
    }


def format_metric(name: str, value: float) -> str:
    if name in METRES:
        return f"{value:.3f}"
    return f"{value:.2f}"


def diverged_sequences(files: dict[str, str]) -> dict[str, float]:
    diverged = {}
    for name, path in files.items():
        with open(path, "rb") as f:
            angular_error = orjson.loads(f.read())["cam_stats"]["AE"]["mean"]
        if angular_error > _IMPLAUSIBLE_ANGULAR_ERROR_DEG:
            diverged[name] = angular_error
    return diverged


def read_gpu(run_dir: str) -> str:
    path = os.path.join(run_dir, "environment.json")
    if not os.path.exists(path):
        return "an unrecorded device"
    with open(path, "rb") as f:
        return orjson.loads(f.read()).get("gpu", "an unrecorded device")


def latency_figures(run_dir: str, arm_name: str) -> dict[str, str]:
    sequences = summarize_online_latency.load_arm_timings(
        os.path.join(run_dir, f"stream_{arm_name}")
    )
    timings = [timing for sequence in sequences for timing in sequence]
    p50, p95, _ = summarize_online_latency.summarize(timings)[
        "total_seconds"
    ]
    rate = summarize_online_latency.throughput(timings, "total_seconds")
    n_views = count_views(run_dir)

    return {
        "latency_p50": f"{p50:.1f}",
        "latency_p95": f"{p95:.1f}",
        "fps": f"{rate:.1f}",
        "views": str(n_views),
    }


def fill_rows(
    template: str,
    arms: dict[str, tuple[dict, dict]],
    latency: dict[str, str],
    statistic: str,
) -> str:
    rows = template
    for arm, (camera_stats, human_stats) in arms.items():
        for stats in (camera_stats, human_stats):
            for name, values in stats.items():
                rows = rows.replace(
                    f"{{kineo_{name}_h36m_{arm}}}",
                    format_metric(name, values[statistic]),
                )
    for name, value in latency.items():
        rows = rows.replace(f"{{kineo_{name}_h36m}}", value)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "offline_run_dir",
        type=str,
        help="Run of the arm whose calibration read the whole sequence",
    )
    parser.add_argument(
        "online_run_dir",
        type=str,
        help="Run of the arm whose calibration read the first seconds only",
    )
    parser.add_argument("--arm-name", type=str, default="h36m_trt_fp16")
    parser.add_argument(
        "--statistic",
        type=str,
        default="mean",
        choices=("mean", "median"),
        help="Statistic over sequences",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=[],
        metavar="SEQUENCE",
    )
    args = parser.parse_args()

    runs = {
        "Offline": args.offline_run_dir,
        "Online": args.online_run_dir,
    }
    scored = {
        arm: scored_sequences(run_dir, args.arm_name)
        for arm, run_dir in runs.items()
    }
    common = sorted(
        set(scored["Offline"]) & set(scored["Online"]) - set(args.exclude)
    )
    if not common:
        raise FileNotFoundError(
            f"No sequence was scored by both arms under {runs}"
        )

    # Run summaries hold only the last summarized arm
    arms = {
        arm: aggregate_sequence_metrics_files([files[name] for name in common])
        for arm, files in scored.items()
    }

    with open(_TEMPLATE, "r") as f:
        template = f.read()

    print(
        fill_rows(
            template,
            arms,
            latency_figures(args.online_run_dir, args.arm_name)
            | {"gpu": read_gpu(args.online_run_dir)},
            args.statistic,
        )
    )
