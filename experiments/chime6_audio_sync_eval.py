# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""
Measure the accuracy of MFCC temporal calibration on CHiME-6.

CHiME-6 audio is already synchronized, so a known de-sync is injected by
reading the target device at a shifted position and the estimate is compared
against it. Errors are reported against window length, acoustic content, and
the distance between the two microphone arrays.

Outputs:
  - chime6_sync_eval.csv       one row per (window, pair, hop, offset)
  - chime6_sync_summary.json   aggregates by distance, class and length

Usage:
    pixi run python experiments/chime6_audio_sync_eval.py \
        <chime6_dir> <output_dir> [--sessions S02] \
        [--lengths 30] [--max-windows-per-cell 2]
"""

import argparse
import csv
import itertools
import json
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from kineo.datasets.chime6.chime6_dataset import AudioView, CHiME6AudioDataset
from kineo.datasets.chime6.chime6_preprocess import (
    AUDIO_START_GUARD_S,
    WINDOWS_FILENAME,
)
from kineo.io.audio_file import get_waveform_info
from kineo.io.audio_loader import WaveformAudioLoader
from kineo.pipeline.stages.mfcc_temporal_calibration import estimate_time_offsets

C_SOUND = 343.0

# Half a frame at 30 fps: the budget a pose pipeline can absorb.
HALF_FRAME_MS = 1000.0 / 30.0 / 2.0

INJECTED_OFFSETS_S = [0.0, 0.05, -0.05, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0]

# The pipeline default of 5 ms quantizes every estimate onto a 5 ms grid,
# which is coarser than the effect the distance axis is looking for.
HOP_DURATIONS_S = [0.005, 0.001]

CSV_FIELDS = [
    "window_id", "session_id", "content_class", "duration_s",
    "n_distinct_speakers", "max_concurrent_speakers",
    "silence_frac", "single_frac", "overlap_frac",
    "ref_device", "target_device", "pair_distance_m", "same_room",
    "tof_bound_ms", "hop_duration_s", "injected_offset_s", "estimated_offset_s",
    "signed_error_ms", "abs_error_ms", "within_half_frame", "runtime_s",
]


def load_shifted(
    view: AudioView, start_s: float, duration_s: float
) -> tuple[torch.Tensor, int] | None:
    """Read a view at an arbitrary position on the session clock.

    Returns:
        The waveform and its sample rate, or None if the read would run past
        the end of the file or reach into the opening sync beep. Windows are
        selected clear of both, so only an injected offset can trip this.
    """
    info = get_waveform_info(view["audio_path"])
    start_frame = round(start_s * info.sample_rate)
    n_frames = round(duration_s * info.sample_rate)
    guard_frame = round(AUDIO_START_GUARD_S * info.sample_rate)
    if start_frame < guard_frame or start_frame + n_frames > info.n_frames:
        return None
    return WaveformAudioLoader(
        view["audio_path"],
        view["audio_loader"].device,
        start_frame,
        n_frames,
    ).load_audio()


def evaluate_window(
    window: dict,
    offsets: list[float],
    hops: list[float],
    device: torch.device,
) -> tuple[list[dict], int]:
    """Run every (pair, hop, injected offset) combination of one window.

    The reference is read at the window position and the target at a position
    shifted by the injected offset, so the pair is de-synchronised by a known
    amount. See tests/pipeline/test_mfcc_temporal_calibration.py for the sign.

    Returns:
        Tuple of (rows, n_skipped), with one row per combination that could be
        read and a count of those refused by the file bounds or the beep guard.
    """
    t0 = window["start_time_s"]
    length = window["duration_s"]

    rows = []
    skipped = 0
    for ref_view, target_view in itertools.combinations(window["views"], 2):
        ref_waveform, ref_sample_rate = ref_view["audio_loader"].load_audio()
        distance_m = math.dist(ref_view["position_m"], target_view["position_m"])

        for offset in offsets:
            shifted = load_shifted(target_view, t0 - offset, length)
            if shifted is None:
                skipped += len(hops)
                continue
            target_waveform, target_sample_rate = shifted

            for hop in hops:
                started = time.perf_counter()
                estimated = estimate_time_offsets(
                    audio_waveforms=[ref_waveform, target_waveform],
                    audio_sample_rates=[ref_sample_rate, target_sample_rate],
                    ref_idx=0,
                    hop_duration=hop,
                    compute_device=device,
                )[1].item()
                runtime = time.perf_counter() - started

                # The target's content sits `offset` later on the reference
                # clock, so t_ref - t_target is -offset.
                signed_error_ms = (estimated + offset) * 1000.0
                rows.append(
                    {
                        "window_id": window["window_id"],
                        "session_id": window["session_id"],
                        "content_class": window["content_class"],
                        "duration_s": length,
                        "n_distinct_speakers": window["n_distinct_speakers"],
                        "max_concurrent_speakers": window["max_concurrent_speakers"],
                        "silence_frac": window["composition"]["silence"],
                        "single_frac": window["composition"]["single"],
                        "overlap_frac": window["composition"]["overlap"],
                        "ref_device": ref_view["view_id"],
                        "target_device": target_view["view_id"],
                        "pair_distance_m": round(distance_m, 3),
                        "same_room": ref_view["room"] == target_view["room"],
                        "tof_bound_ms": round(distance_m / C_SOUND * 1000.0, 3),
                        "hop_duration_s": hop,
                        "injected_offset_s": offset,
                        "estimated_offset_s": round(estimated, 6),
                        "signed_error_ms": round(signed_error_ms, 3),
                        "abs_error_ms": round(abs(signed_error_ms), 3),
                        "within_half_frame": abs(signed_error_ms) <= HALF_FRAME_MS,
                        "runtime_s": round(runtime, 3),
                    }
                )
    return rows, skipped


DISTANCE_BINS = [(0.0, 0.5), (0.5, 2.0), (2.0, 3.5), (3.5, 5.0), (5.0, 99.0)]


def distance_bin(distance_m: float) -> str:
    """Label the distance bin a pair falls into."""
    for low, high in DISTANCE_BINS:
        if low <= distance_m < high:
            return f"{low:.1f}-{high:.1f}m"
    return "unbinned"


def summarize(rows: list[dict]) -> dict:
    """Aggregate absolute error by hop, distance, class and window length."""
    for row in rows:
        row["distance_bin"] = distance_bin(row["pair_distance_m"])

    def stats(subset: list[dict]) -> dict:
        errors = np.array([r["abs_error_ms"] for r in subset])
        return {
            "n": len(subset),
            "median_abs_error_ms": round(float(np.median(errors)), 3),
            "p95_abs_error_ms": round(float(np.percentile(errors, 95)), 3),
            "success_rate": round(
                float(np.mean([r["within_half_frame"] for r in subset])), 4
            ),
        }

    summary: dict = {"overall": stats(rows)}
    for hop in sorted({r["hop_duration_s"] for r in rows}):
        at_hop = [r for r in rows if r["hop_duration_s"] == hop]
        key = f"hop_{hop * 1000:.0f}ms"
        summary[key] = {"overall": stats(at_hop)}
        for name, group in (
            ("by_distance_bin", "distance_bin"),
            ("by_content_class", "content_class"),
            ("by_duration_s", "duration_s"),
            ("by_session", "session_id"),
        ):
            summary[key][name] = {
                str(value): stats([r for r in at_hop if r[group] == value])
                for value in sorted({r[group] for r in at_hop}, key=str)
            }
    return summary


def print_summary(summary: dict):
    for key, block in summary.items():
        if key == "overall":
            continue
        print(f"\n===== {key} =====")
        for name, group in block.items():
            if name == "overall":
                s = group
                print(
                    f"  overall: n={s['n']} median={s['median_abs_error_ms']:.2f} ms "
                    f"p95={s['p95_abs_error_ms']:.2f} ms "
                    f"success={s['success_rate'] * 100:.1f}%"
                )
                continue
            print(f"  {name}:")
            for value, s in group.items():
                print(
                    f"    {value:<14} n={s['n']:<6} "
                    f"median={s['median_abs_error_ms']:8.2f} ms "
                    f"p95={s['p95_abs_error_ms']:8.2f} ms "
                    f"success={s['success_rate'] * 100:5.1f}%"
                )


def main(
    dataset_dir: str,
    output_dir: str,
    windows_json: str = "",
    sessions: list[str] = [],
    lengths: list[float] = [],
    classes: list[str] = [],
    max_windows_per_cell: int = 0,
    offsets: list[float] = INJECTED_OFFSETS_S,
    hops: list[float] = HOP_DURATIONS_S,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {device}")

    dataset = CHiME6AudioDataset(
        windows_json or os.path.join(dataset_dir, WINDOWS_FILENAME),
        device=device,
    )

    windows = [
        window
        for window in dataset
        if (not sessions or window["session_id"] in sessions)
        and (not lengths or window["duration_s"] in lengths)
        and (not classes or window["content_class"] in classes)
        and (
            not max_windows_per_cell
            or window["selection"]["rank"] < max_windows_per_cell
        )
    ]

    n_views = {len(w["views"]) for w in windows}
    print(f"Windows: {len(windows)}")
    print(f"Views per window: {sorted(n_views)}")
    print(f"Offsets: {offsets}")
    print(f"Hops: {hops}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "chime6_sync_eval.csv")

    rows: list[dict] = []
    skipped = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for window in tqdm(windows, desc="Windows"):
            window_rows, window_skipped = evaluate_window(
                window, offsets, hops, device
            )
            writer.writerows(window_rows)
            # Flushed per window so an interrupted run keeps its results.
            f.flush()
            rows.extend(window_rows)
            skipped += window_skipped

    print(f"\nWrote {csv_path} ({len(rows)} rows)")
    if skipped:
        print(
            f"Skipped {skipped} combinations that ran past a file end or into "
            f"the first {AUDIO_START_GUARD_S:.0f} s sync beep"
        )

    summary = summarize(rows)
    print_summary(summary)
    summary_path = os.path.join(output_dir, "chime6_sync_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--windows-json", type=str, default="")
    parser.add_argument("--sessions", type=str, nargs="+", default=[])
    parser.add_argument("--lengths", type=float, nargs="+", default=[])
    parser.add_argument("--classes", type=str, nargs="+", default=[])
    parser.add_argument("--max-windows-per-cell", type=int, default=0)
    parser.add_argument(
        "--offsets", type=float, nargs="+", default=INJECTED_OFFSETS_S
    )
    parser.add_argument("--hops", type=float, nargs="+", default=HOP_DURATIONS_S)
    args = parser.parse_args()

    main(
        args.dataset_dir,
        args.output_dir,
        args.windows_json,
        args.sessions,
        args.lengths,
        args.classes,
        args.max_windows_per_cell,
        args.offsets,
        args.hops,
    )
