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
        <chime6_dev_dir> <windows_json> <output_dir> [--sessions S02] \
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

from kineo.io.audio_file import get_waveform_info, load_waveform
from kineo.pipeline.stages.mfcc_temporal_calibration import estimate_time_offsets

C_SOUND = 343.0

# Half a frame at 30 fps: the budget a pose pipeline can absorb.
HALF_FRAME_MS = 1000.0 / 30.0 / 2.0

# The 4 microphones of one Kinect are sample-synchronised and span 22.6 cm,
# so an intra-array pair measures the estimator's own noise floor.
INTRA_ARRAY_DISTANCE_M = 0.226

INJECTED_OFFSETS_S = [0.0, 0.05, -0.05, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0]

# Sessions open with a sync beep shared by every device, which aligns clips far
# more easily than speech does. Windows already start after it, but injecting a
# positive offset reads the target earlier and can pull it back in, so the
# guard is enforced on every read rather than on the window position.
AUDIO_START_GUARD_S = 5.0

# The pipeline default of 5 ms quantizes every estimate onto a 5 ms grid,
# which is coarser than the effect the distance axis is looking for.
HOP_DURATIONS_S = [0.005, 0.001]

CSV_FIELDS = [
    "window_id", "session_id", "class", "duration_s",
    "n_distinct_speakers", "max_concurrent_speakers",
    "silence_frac", "single_frac", "overlap_frac",
    "ref_device", "target_device", "pair_kind", "pair_distance_m", "same_room",
    "tof_bound_ms", "hop_duration_s", "injected_offset_s", "estimated_offset_s",
    "signed_error_ms", "abs_error_ms", "within_half_frame", "runtime_s",
]


def build_pairs(positions: dict, session: str) -> list[dict]:
    """List the device pairs of a session, closest first.

    Intra-array pairs (two channels of one Kinect) come first, then every
    combination of two arrays that have audio.

    Args:
        positions: Parsed device position file.
        session: Session identifier, e.g. "S02".

    Returns:
        List of pair descriptions with distance and room agreement.
    """
    units = positions["sessions"][session]["units"]
    usable = sorted(u for u, spec in units.items() if spec["has_audio"])

    pairs = [
        {
            "ref": f"{unit}.CH1",
            "target": f"{unit}.CH4",
            "kind": "intra_array",
            "distance_m": INTRA_ARRAY_DISTANCE_M,
            "same_room": True,
        }
        for unit in usable
    ]
    for a, b in itertools.combinations(usable, 2):
        pairs.append(
            {
                "ref": f"{a}.CH1",
                "target": f"{b}.CH1",
                "kind": "inter_array",
                "distance_m": math.hypot(
                    units[a]["x"] - units[b]["x"], units[a]["y"] - units[b]["y"]
                ),
                "same_room": units[a]["room"] == units[b]["room"],
            }
        )
    return pairs


def load_window(path: str, start_s: float, duration_s: float) -> torch.Tensor | None:
    """Read one window of a wav file.

    Returns:
        The waveform, or None if the read would run past the end of the file
        or reach into the opening sync beep.
    """
    info = get_waveform_info(path)
    start_frame = int(round(start_s * info.sample_rate))
    num_frames = int(round(duration_s * info.sample_rate))
    guard_frame = int(round(AUDIO_START_GUARD_S * info.sample_rate))
    if start_frame < guard_frame or start_frame + num_frames > info.n_frames:
        return None
    waveform, _ = load_waveform(path, start_frame=start_frame, n_frames=num_frames)
    return waveform


def evaluate_window(
    window: dict,
    audio_dir: str,
    pairs: list[dict],
    offsets: list[float],
    hops: list[float],
    device: torch.device,
) -> list[dict]:
    """Run every (pair, hop, injected offset) combination of one window.

    The reference is read at the window position and the target at a position
    shifted by the injected offset, so the pair is de-synchronised by a known
    amount. See tests/pipeline/test_mfcc_temporal_calibration.py for the sign.

    Returns:
        Tuple of (rows, n_skipped), with one row per combination that could be
        read and a count of those refused by the file bounds or the beep guard.
    """
    session = window["session_id"]
    t0 = window["start_time_s"]
    length = window["duration_s"]
    sample_rate = 16000

    rows = []
    skipped = 0
    for pair in pairs:
        ref_path = os.path.join(audio_dir, f"{session}_{pair['ref']}.wav")
        target_path = os.path.join(audio_dir, f"{session}_{pair['target']}.wav")
        ref = load_window(ref_path, t0, length)
        if ref is None:
            skipped += len(offsets) * len(hops)
            continue

        for offset in offsets:
            target = load_window(target_path, t0 - offset, length)
            if target is None:
                skipped += len(hops)
                continue

            for hop in hops:
                started = time.perf_counter()
                estimated = estimate_time_offsets(
                    audio_waveforms=[ref, target],
                    audio_sample_rates=[sample_rate, sample_rate],
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
                        "session_id": session,
                        "class": window["class"],
                        "duration_s": length,
                        "n_distinct_speakers": window["n_distinct_speakers"],
                        "max_concurrent_speakers": window["max_concurrent_speakers"],
                        "silence_frac": window["composition"]["silence"],
                        "single_frac": window["composition"]["single"],
                        "overlap_frac": window["composition"]["overlap"],
                        "ref_device": pair["ref"],
                        "target_device": pair["target"],
                        "pair_kind": pair["kind"],
                        "pair_distance_m": round(pair["distance_m"], 3),
                        "same_room": pair["same_room"],
                        "tof_bound_ms": round(
                            pair["distance_m"] / C_SOUND * 1000.0, 3
                        ),
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
            ("by_pair_kind", "pair_kind"),
            ("by_distance_bin", "distance_bin"),
            ("by_class", "class"),
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
    windows_json: str,
    output_dir: str,
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

    audio_dir = os.path.join(dataset_dir, "CHiME6", "audio", "dev")
    with open(windows_json) as f:
        windows = json.load(f)["windows"]
    with open(
        os.path.join(os.path.dirname(__file__), "chime6_device_positions.json")
    ) as f:
        positions = json.load(f)

    if sessions:
        windows = [w for w in windows if w["session_id"] in sessions]
    if lengths:
        windows = [w for w in windows if w["duration_s"] in lengths]
    if classes:
        windows = [w for w in windows if w["class"] in classes]
    if max_windows_per_cell:
        windows = [
            w for w in windows if w["selection"]["rank"] < max_windows_per_cell
        ]

    pairs = {s: build_pairs(positions, s) for s in {w["session_id"] for w in windows}}
    print(f"Windows: {len(windows)}")
    for session, session_pairs in sorted(pairs.items()):
        print(f"  {session}: {len(session_pairs)} pairs")
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
                window,
                audio_dir,
                pairs[window["session_id"]],
                offsets,
                hops,
                device,
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
    parser.add_argument("windows_json", type=str)
    parser.add_argument("output_dir", type=str)
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
        args.windows_json,
        args.output_dir,
        args.sessions,
        args.lengths,
        args.classes,
        args.max_windows_per_cell,
        args.offsets,
        args.hops,
    )
