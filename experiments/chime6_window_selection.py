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
Select CHiME-6 evaluation windows by acoustic content.

Reads the CHiME-6 transcripts and emits time ranges on the synchronized audio
clock, grouped by how many people are talking: silence, a single speaker,
overlapping speakers, or a mix when the corpus offers nothing purer.

No audio is decoded; only the per-session duration is read from the wav
headers so windows stay inside the shortest device.

Outputs:
  - chime6_windows.json    windows plus a per-cell generation report

Usage:
    pixi run python experiments/chime6_window_selection.py \
        <chime6_dev_dir> <output_json> [--sessions S02 S09] [--lengths 30 60]
"""

import argparse
import glob
import json
import os

import numpy as np
import orjson
import soundfile as sf

WINDOW_LENGTHS_S = [30.0, 60.0, 120.0, 300.0, 600.0]

# Sessions open with a sync beep shared by every device. It is a far stronger
# alignment cue than speech, so a window containing it measures the beep
# rather than the method.
AUDIO_START_GUARD_S = 5.0
TARGET_CLASSES = ["silence", "single", "overlap"]
WINDOWS_PER_CELL = 5

# A window qualifies for a class when its composition clears the threshold.
# Overlap sits lowest because sustained multi-speaker talk is rare: a dinner
# party spends most of its time with one person speaking.
THRESHOLDS = {"silence": 0.80, "single": 0.70, "overlap": 0.30}
SINGLE_MAX_OVERLAP = 0.10

# Two selected windows of the same cell may not share more than this fraction
# of their span, otherwise the "5 windows" are five views of one moment.
MAX_MUTUAL_OVERLAP = 0.50


def parse_time(value: str) -> float:
    """Convert a CHiME-6 "HH:MM:SS.ss" timestamp to seconds."""
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def load_utterances(path: str) -> tuple[np.ndarray, np.ndarray, list, list]:
    """Load one session transcript.

    Returns:
        Tuple of (starts, ends, speakers, locations), each ordered by start
        time. Times are seconds on the synchronized clock.
    """
    with open(path, "rb") as f:
        entries = orjson.loads(f.read())

    entries.sort(key=lambda e: parse_time(e["start_time"]))
    starts = np.array([parse_time(e["start_time"]) for e in entries])
    ends = np.array([parse_time(e["end_time"]) for e in entries])
    speakers = [e["speaker"] for e in entries]
    locations = [e.get("location") for e in entries]
    return starts, ends, speakers, locations


def concurrency_segments(
    starts: np.ndarray, ends: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the piecewise-constant count of simultaneously active speakers.

    Returns:
        Tuple of (seg_starts, seg_ends, seg_counts) covering the whole session,
        including the gaps where nobody speaks.
    """
    edges = np.unique(np.concatenate([[0.0], starts, ends]))
    seg_starts, seg_ends = edges[:-1], edges[1:]
    mids = (seg_starts + seg_ends) / 2.0
    counts = ((starts[None, :] <= mids[:, None]) & (mids[:, None] < ends[None, :])).sum(
        axis=1
    )
    return seg_starts, seg_ends, counts


def composition(
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    counts: np.ndarray,
    window_starts: np.ndarray,
    length: float,
) -> np.ndarray:
    """Fraction of each window spent with 0, exactly 1, and 2+ speakers.

    Args:
        seg_starts: Start of every constant-concurrency segment.
        seg_ends: End of every constant-concurrency segment.
        counts: Speaker count on each segment.
        window_starts: Start time of every candidate window.
        length: Window length in seconds.

    Returns:
        Array of shape (n_windows, 3) whose rows sum to one.
    """
    lo = np.maximum(seg_starts[None, :], window_starts[:, None])
    hi = np.minimum(seg_ends[None, :], window_starts[:, None] + length)
    covered = np.clip(hi - lo, 0.0, None)

    out = np.empty((len(window_starts), 3))
    out[:, 0] = covered[:, counts == 0].sum(axis=1)
    out[:, 1] = covered[:, counts == 1].sum(axis=1)
    out[:, 2] = covered[:, counts >= 2].sum(axis=1)
    return out / length


def session_duration(audio_dir: str, session: str) -> float:
    """Duration of the shortest Kinect channel of a session, in seconds."""
    paths = sorted(glob.glob(os.path.join(audio_dir, f"{session}_U0*.CH1.wav")))
    if not paths:
        raise FileNotFoundError(f"No Kinect audio for {session} in {audio_dir}")
    return min(sf.info(p).duration for p in paths)


def select_cell(
    scores: np.ndarray,
    qualifies: np.ndarray,
    stride: float,
    length: float,
    count: int,
) -> list[int]:
    """Pick the best mutually distinct candidate windows of one cell.

    Candidates are taken by descending score, skipping any that overlaps an
    already selected window by more than ``MAX_MUTUAL_OVERLAP``.

    Args:
        scores: Purity score of every candidate.
        qualifies: Whether each candidate clears its class threshold.
        stride: Spacing between consecutive candidates, in seconds.
        length: Window length in seconds.
        count: How many windows to select.

    Returns:
        Indices of the selected candidates, best first.
    """
    # Qualifying candidates always outrank relaxed ones.
    order = np.lexsort((-scores, ~qualifies))
    min_gap = length * (1.0 - MAX_MUTUAL_OVERLAP)
    selected: list[int] = []
    for idx in order:
        if all(abs(idx - other) * stride >= min_gap for other in selected):
            selected.append(int(idx))
        if len(selected) == count:
            break
    return selected


def build_windows(
    session: str,
    transcript_path: str,
    duration: float,
    lengths: list[float],
    count: int,
) -> tuple[list[dict], dict]:
    """Select windows of every length and class for one session.

    Returns:
        Tuple of (windows, report). The report records, per cell, how many
        candidates qualified and why a cell had to fall back to "mixed".
    """
    starts, ends, speakers, locations = load_utterances(transcript_path)
    seg_starts, seg_ends, counts = concurrency_segments(starts, ends)

    spans = seg_ends - seg_starts
    # Summing utterances would double-count overlapping speech, so coverage is
    # measured on the concurrency segments instead.
    speaker_seconds = float(np.clip(ends - starts, 0.0, None).sum())
    silent = float(spans[counts == 0].sum())
    overlapped = float(spans[counts >= 2].sum())
    gaps = spans[counts == 0]
    print(f"\n===== {session} =====")
    print(f"  duration:        {duration:.1f} s")
    print(f"  utterances:      {len(starts)}")
    print(f"  speakers:        {sorted(set(speakers))}")
    print(f"  locations:       {sorted({v for v in locations if v})}")
    print(f"  speaker-seconds: {speaker_seconds:.1f} s summed over speakers")
    print(f"  silent:          {silent / duration * 100:.1f}% of session")
    print(f"  2+ speakers:     {overlapped / duration * 100:.1f}% of session")
    print(f"  longest silence: {gaps.max() if len(gaps) else 0.0:.1f} s")

    windows: list[dict] = []
    report: dict = {}

    for length in lengths:
        stride = length / 4.0
        window_starts = np.arange(AUDIO_START_GUARD_S, duration - length, stride)
        if len(window_starts) == 0:
            print(f"\n  [L={length:.0f}s] session too short, skipped")
            report[f"L{length:.0f}"] = {"skipped": "session shorter than window"}
            continue

        comp = composition(seg_starts, seg_ends, counts, window_starts, length)
        print(f"\n  [L={length:.0f}s] {len(window_starts)} candidates, stride {stride:.1f}s")

        for target in TARGET_CLASSES:
            if target == "silence":
                scores = comp[:, 0]
                qualifies = scores >= THRESHOLDS["silence"]
            elif target == "single":
                scores = comp[:, 1]
                qualifies = (scores >= THRESHOLDS["single"]) & (
                    comp[:, 2] <= SINGLE_MAX_OVERLAP
                )
            else:
                scores = comp[:, 2]
                qualifies = scores >= THRESHOLDS["overlap"]

            chosen = select_cell(scores, qualifies, stride, length, count)
            n_qualified = int(qualifies.sum())
            relaxed = n_qualified < count

            key = f"L{length:.0f}_{target}"
            report[key] = {
                "requested": count,
                "candidates": len(window_starts),
                "qualified": n_qualified,
                "selected": len(chosen),
                "threshold": THRESHOLDS[target],
                "best_score": round(float(scores.max()), 3),
                "median_score": round(float(np.median(scores)), 3),
                "relaxed": relaxed,
            }
            if relaxed:
                report[key]["reason"] = (
                    f"only {n_qualified} of {len(window_starts)} candidates reach "
                    f"{target} >= {THRESHOLDS[target]:.2f}; best candidate reaches "
                    f"{scores.max():.3f}"
                )

            print(
                f"    {target:<8} qualified {n_qualified:4d}/{len(window_starts):<5d} "
                f"selected {len(chosen)}  best={scores.max():.3f} "
                f"median={np.median(scores):.3f}"
                + ("  RELAXED -> mixed" if relaxed else "")
            )

            for rank, idx in enumerate(chosen):
                t0 = float(window_starts[idx])
                qualified = bool(qualifies[idx])
                active = (starts < t0 + length) & (ends > t0)
                window_speakers = sorted({speakers[i] for i in np.nonzero(active)[0]})
                in_window = np.nonzero(
                    (seg_starts < t0 + length) & (seg_ends > t0)
                )[0]
                windows.append(
                    {
                        "window_id": f"{session}_L{length:.0f}_{target}_{rank:02d}",
                        "session_id": session,
                        "start_time_s": round(t0, 3),
                        "duration_s": length,
                        "class": target if qualified else "mixed",
                        "composition": {
                            "silence": round(float(comp[idx, 0]), 3),
                            "single": round(float(comp[idx, 1]), 3),
                            "overlap": round(float(comp[idx, 2]), 3),
                        },
                        "n_distinct_speakers": len(window_speakers),
                        "max_concurrent_speakers": int(counts[in_window].max()),
                        "speakers": window_speakers,
                        "selection": {
                            "target_class": target,
                            "relaxed": not qualified,
                            "rank": rank,
                            "purity": round(float(scores[idx]), 3),
                        },
                    }
                )

    return windows, report


def main(
    dataset_dir: str,
    output_json: str,
    sessions: list[str] = ["S02", "S09"],
    lengths: list[float] = WINDOW_LENGTHS_S,
    count: int = WINDOWS_PER_CELL,
):
    audio_dir = os.path.join(dataset_dir, "CHiME6", "audio", "dev")
    transcript_dir = os.path.join(
        dataset_dir, "CHiME6_transcriptions", "transcriptions", "transcriptions", "dev"
    )

    windows: list[dict] = []
    report: dict = {}
    for session in sessions:
        duration = session_duration(audio_dir, session)
        session_windows, session_report = build_windows(
            session,
            os.path.join(transcript_dir, f"{session}.json"),
            duration,
            lengths,
            count,
        )
        windows.extend(session_windows)
        report[session] = {"duration_s": round(duration, 3), "cells": session_report}

    classes: dict = {}
    for window in windows:
        classes[window["class"]] = classes.get(window["class"], 0) + 1
    print("\n===== total =====")
    print(f"  windows: {len(windows)}")
    for name, n in sorted(classes.items()):
        print(f"    {name:<8} {n}")

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(
            {
                "meta": {
                    "dataset_dir": dataset_dir,
                    "sessions": sessions,
                    "lengths_s": lengths,
                    "windows_per_cell": count,
                    "audio_start_guard_s": AUDIO_START_GUARD_S,
                    "thresholds": THRESHOLDS,
                    "single_max_overlap": SINGLE_MAX_OVERLAP,
                    "report": report,
                },
                "windows": windows,
            },
            f,
            indent=2,
        )
        f.write("\n")
    print(f"\nWrote {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("output_json", type=str)
    parser.add_argument("--sessions", type=str, nargs="+", default=["S02", "S09"])
    parser.add_argument(
        "--lengths", type=float, nargs="+", default=WINDOW_LENGTHS_S
    )
    parser.add_argument("--windows-per-cell", type=int, default=WINDOWS_PER_CELL)
    args = parser.parse_args()

    main(
        args.dataset_dir,
        args.output_json,
        args.sessions,
        args.lengths,
        args.windows_per_cell,
    )
