# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Report the accuracy and latency of an online benchmark run."""

import argparse
import contextlib
import glob
import io
import os
import pickle
import subprocess

import orjson

from experiments import summarize_benchmark, summarize_online_latency


def count_views(benchmark_dir: str) -> int | None:
    pattern = os.path.join(
        benchmark_dir, "calibration", "annotations", "*",
        "cameras_extrinsics.pkl",
    )
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            annotations = pickle.load(f)["annotations"]
        return len({a["view_id"] for a in annotations})
    return None


def describe_environment() -> dict:
    import torch

    environment = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    if torch.cuda.is_available():
        environment["gpu"] = torch.cuda.get_device_name(0)

    try:
        import onnxruntime

        environment["onnxruntime"] = onnxruntime.__version__
        environment["providers"] = onnxruntime.get_available_providers()
    except ImportError:
        pass

    environment["driver"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()

    return environment


def main(benchmark_dir: str, arms: list[str], warmup_frames: int):
    n_views = count_views(benchmark_dir)

    for arm in arms:
        stream_dir = os.path.join(benchmark_dir, f"stream_{arm}")
        if not os.path.isdir(os.path.join(stream_dir, "metrics")):
            print(f"{arm}: nothing scored under {stream_dir}")
            continue
        print(f"===== {arm}: accuracy =====")
        summarize_benchmark.main(stream_dir, dataset_file=None, fps=0.0)
        print()

    print("===== latency =====")
    table = io.StringIO()
    with contextlib.redirect_stdout(table):
        summarize_online_latency.main(
            output_root_dir=benchmark_dir,
            arms=arms,
            warmup_frames=warmup_frames,
            n_views=n_views,
        )
    print(table.getvalue())

    latency_path = os.path.join(benchmark_dir, "latency_summary.txt")
    with open(latency_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(table.getvalue())

    environment = describe_environment()
    environment_path = os.path.join(benchmark_dir, "environment.json")
    with open(environment_path, "wb") as f:
        f.write(orjson.dumps(environment, option=orjson.OPT_INDENT_2))

    print(f"views per frame: {n_views}")
    print(f"latency table written to {latency_path}")
    print(f"environment written to {environment_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark_dir",
        type=str,
        help="The run's h36m_online directory, holding calibration/ and "
        "stream_<arm>/",
    )
    parser.add_argument(
        "--arms", type=str, nargs="*", default=["h36m_trt_fp16"]
    )
    parser.add_argument("--warmup-frames", type=int, default=3)
    args = parser.parse_args()
    main(
        benchmark_dir=args.benchmark_dir,
        arms=args.arms,
        warmup_frames=args.warmup_frames,
    )
