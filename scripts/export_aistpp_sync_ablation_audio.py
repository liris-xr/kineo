# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Export one camera's audio for each arm of the audio sync ablation."""

import argparse
import json
import os

import numpy as np
import soundfile
import torch

from kineo.audio.degradations import add_tone_bursts, mix_at_snr
from kineo.datasets.aistpp.aistpp_dataset import build_global_time_reference
from kineo.io.ffmpeg import extract_audio
from kineo.pipeline.stages.mfcc_temporal_calibration import (
    audio_cut_starts,
    load_noise_segment,
)

BACKGROUNDS = {"park": "NPARK", "cafeteria": "PCAFETER"}


def save(path: str, waveform: torch.Tensor, sample_rate: int):
    samples = waveform.numpy().astype(np.float64)
    samples = samples / max(1e-9, np.abs(samples).max()) * 0.9
    soundfile.write(path, samples, sample_rate, subtype="PCM_16")


def main(
    dataset_dir: str,
    demand_dir: str,
    sequence_name: str,
    view_id: str,
    output_dir: str,
    levels_db: list[float],
):
    with open(os.path.join(dataset_dir, "aistpp_pose_test_raw_sequences.json")) as f:
        sequence = next(s for s in json.load(f) if s["sequence_name"] == sequence_name)
    views = [{"view_id": v} for v in sequence["views"]]
    view_idx = [v["view_id"] for v in views].index(view_id)
    reference_id = views[0]["view_id"]

    audio, sample_rate = extract_audio(
        os.path.join(dataset_dir, sequence["views"][view_id]["video_path"]),
        show_progress=False,
    )
    audio = audio.float().mean(0) if audio.ndim > 1 else audio.float()

    timeline = build_global_time_reference(
        {v: info["annotated_start_frame"] for v, info in sequence["views"].items()},
        sequence["n_frames"],
        reference_id,
    )
    cut_s = audio_cut_starts(views, timeline, jitter_s=1.0)[view_idx]
    cut = audio[int(round(cut_s * sample_rate)) :]

    with open(os.path.join(dataset_dir, sequence["annotations"]["cameras_temporal"])) as f:
        gt = {a["view_id"]: a["time_offset"] for a in json.load(f)["annotations"]}
    shift_s = gt[view_id] - gt[reference_id] + cut_s
    period_s = 0.5
    n_beeps = int((cut.shape[-1] / sample_rate + abs(shift_s)) / period_s) + 2
    beeped = add_tone_bursts(
        cut,
        sample_rate,
        [k * period_s - shift_s for k in range(-n_beeps, n_beeps)],
        frequency_hz=1000.0,
        duration_s=0.05,
        level_db=0.0,
    )

    exports = {
        "1_raw_with_sync_beeps": audio,
        "2_sync_beeps_removed": cut,
        "4_synthetic_periodic_beep_0dB": beeped,
    }
    for label, environment in BACKGROUNDS.items():
        noise = load_noise_segment(
            os.path.join(demand_dir, environment),
            channel=view_idx % 16 + 1,
            start_s=view_idx * 33.0,
            n_samples=cut.shape[-1],
            sample_rate=sample_rate,
        )
        for level_db in levels_db:
            exports[f"3_{label}_{level_db:g}dB"] = mix_at_snr(cut, noise, -level_db)

    os.makedirs(output_dir, exist_ok=True)
    for name, waveform in exports.items():
        path = os.path.join(output_dir, f"{sequence_name}_{view_id}_{name}.wav")
        save(path, waveform, sample_rate)
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=str, help="Preprocessed AIST++ directory")
    parser.add_argument(
        "demand_dir", type=str, help="DEMAND directory, see demand-download"
    )
    parser.add_argument("--sequence", type=str, default="gBR_sBM_cAll_d06_mBR2_ch01")
    parser.add_argument("--view", type=str, default="c04")
    parser.add_argument("--output-dir", type=str, default="outputs/listen")
    parser.add_argument(
        "--levels-db",
        type=float,
        nargs="+",
        default=[-20, -10, 0],
        help="Background level relative to the music",
    )
    args = parser.parse_args()
    main(
        args.dataset_dir,
        args.demand_dir,
        args.sequence,
        args.view,
        args.output_dir,
        args.levels_db,
    )
