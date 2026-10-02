# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import argparse
import glob
import os
import pickle
import statistics

from kineo.annotations.frame_timing import FrameTiming, FrameTimingsAnnotations

PHASES = (
    "read_seconds",
    "detection_seconds",
    "keypoints_seconds",
    "triangulation_seconds",
    "total_seconds",
)


def load_arm_timings(
    arm_dir: str, warmup_frames: int = 3
) -> list[list[FrameTiming]]:
    sequences: list[list[FrameTiming]] = []
    pattern = os.path.join(arm_dir, "annotations", "*", "frame_timings.pkl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            annotations = FrameTimingsAnnotations.from_dict(pickle.load(f))
            sequences.append(annotations.annotations[warmup_frames:])
    return sequences


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(fraction * len(ordered)), len(ordered) - 1)]


def milliseconds(timings: list[FrameTiming], phase: str) -> list[float]:
    return [getattr(timing, phase) * 1000 for timing in timings]


def summarize(
    timings: list[FrameTiming],
) -> dict[str, tuple[float, float, float]]:
    summary = {}
    for phase in PHASES:
        values = milliseconds(timings, phase)
        summary[phase] = (
            statistics.median(values),
            percentile(values, 0.95),
            percentile(values, 0.99),
        )

    pipeline = [
        total - read
        for total, read in zip(
            milliseconds(timings, "total_seconds"),
            milliseconds(timings, "read_seconds"),
        )
    ]
    summary["pipeline_seconds"] = (
        statistics.median(pipeline),
        percentile(pipeline, 0.95),
        percentile(pipeline, 0.99),
    )
    return summary


def throughput(timings: list[FrameTiming], phase: str) -> float:
    return 1000.0 * len(timings) / sum(milliseconds(timings, phase))


def jitter(sequences: list[list[FrameTiming]]) -> tuple[float, float]:
    deltas = []
    for timings in sequences:
        values = milliseconds(timings, "total_seconds")
        deltas.extend(
            abs(later - earlier)
            for earlier, later in zip(values, values[1:])
        )
    return statistics.median(deltas), percentile(deltas, 0.95)


def main(
    output_root_dir: str,
    arms: list[str],
    warmup_frames: int,
    n_views: int | None = None,
):
    rows = {}
    for arm in arms:
        sequences = load_arm_timings(
            os.path.join(output_root_dir, f"stream_{arm}"), warmup_frames
        )
        timings = [t for sequence in sequences for t in sequence]
        if not timings:
            print(f"{arm}: no timings found")
            continue
        rows[arm] = (sequences, timings, summarize(timings))

    if not rows:
        return

    width = 26
    header = f"{'phase (p50/p95/p99 ms)':<26}" + "".join(
        f"{arm:>{width}}" for arm in rows
    )
    print(header)
    print("-" * len(header))
    for phase in PHASES + ("pipeline_seconds",):
        line = f"{phase:<26}"
        for _, _, summary in rows.values():
            p50, p95, p99 = summary[phase]
            line += f"{p50:>8.2f}/{p95:<8.2f}/{p99:<8.2f}"
        print(line)

    print("-" * len(header))
    for label, phase in (
        ("frames/s sustained", "total_seconds"),
        ("frames/s without the read", "pipeline_seconds"),
    ):
        line = f"{label:<26}"
        for _, timings, summary in rows.values():
            if phase == "pipeline_seconds":
                rate = 1000.0 * len(timings) / (
                    sum(milliseconds(timings, "total_seconds"))
                    - sum(milliseconds(timings, "read_seconds"))
                )
            else:
                rate = throughput(timings, phase)
            line += f"{rate:>8.2f}{'':<18}"
        print(line)

    line = f"{'frames/s at the median':<26}"
    for _, _, summary in rows.values():
        line += f"{1000.0 / summary['total_seconds'][0]:>8.2f}{'':<18}"
    print(line)

    line = f"{'jitter p50/p95':<26}"
    for sequences, _, _ in rows.values():
        p50, p95 = jitter(sequences)
        line += f"{p50:>8.2f}/{p95:<17.2f}"
    print(line)

    if n_views is not None:
        line = f"{'view-frames/s sustained':<26}"
        for _, timings, _ in rows.values():
            rate = n_views * throughput(timings, "total_seconds")
            line += f"{rate:>8.2f}{'':<18}"
        print(line)

        line = f"{'ms per view (p50)':<26}"
        for _, _, summary in rows.values():
            line += f"{summary['total_seconds'][0] / n_views:>8.2f}{'':<18}"
        print(line)

    print()
    for arm, (sequences, timings, _) in rows.items():
        print(
            f"{arm}: {len(timings)} frames over {len(sequences)} sequences, "
            f"first {warmup_frames} of each dropped"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root-dir", type=str, default="./outputs/h36m_online"
    )
    parser.add_argument(
        "--arms",
        type=str,
        nargs="*",
        default=["h36m_trt_fp16"],
    )
    parser.add_argument(
        "--views",
        type=int,
        default=None,
        help="Views a frame covers, to also report the per view rate",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=3,
        help="Frames to drop from the start of each sequence (video warm-up)",
    )
    args = parser.parse_args()
    main(
        output_root_dir=args.output_root_dir,
        arms=args.arms,
        warmup_frames=args.warmup_frames,
        n_views=args.views,
    )
