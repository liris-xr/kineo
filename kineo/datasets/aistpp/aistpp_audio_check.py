# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import os

import orjson
import torch
from tqdm import tqdm

from kineo.datasets.aistpp.aistpp_dataset import VIDEO_FPS
from kineo.io.ffmpeg import extract_audio
from kineo.pipeline.stages.mfcc_temporal_calibration import (
    estimate_time_offsets,
)

DEFAULT_TOLERANCE_S = 0.25


def compare_time_offsets(
    frame_offsets: dict[str, int],
    audio_offsets: dict[str, float],
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    fps: float = VIDEO_FPS,
) -> dict[str, dict]:
    reference = sorted(frame_offsets)[0]

    comparison = {}
    for camera in sorted(frame_offsets):
        frame_match_s = (frame_offsets[reference] - frame_offsets[camera]) / fps
        audio_s = audio_offsets[camera] - audio_offsets[reference]
        disagreement_s = audio_s - frame_match_s
        comparison[camera] = {
            "frame_match_s": frame_match_s,
            "audio_s": audio_s,
            "disagreement_s": disagreement_s,
            "flagged": abs(disagreement_s) > tolerance_s,
        }

    return comparison


def flagged_views(comparisons: dict[str, dict[str, dict]]) -> dict[str, dict]:
    flagged = {}
    for sequence_name, comparison in comparisons.items():
        views = {
            camera: stats["disagreement_s"]
            for camera, stats in comparison.items()
            if stats["flagged"]
        }
        if views:
            flagged[sequence_name] = views
    return flagged


def audio_check_report_path(dataset_dir: str, split: str) -> str:
    return os.path.join(
        dataset_dir, f"aistpp_{split}_time_offsets_audio_check.json"
    )


def check_time_offsets_against_audio(
    dataset_dir: str,
    split: str = "pose_test",
    tolerance_s: float = DEFAULT_TOLERANCE_S,
    sequence_names: list[str] | None = None,
    device: torch.device | None = None,
) -> dict[str, dict]:
    """Write the audio check report and return the flagged views."""
    offsets_path = os.path.join(
        dataset_dir, f"aistpp_{split}_time_offsets.json"
    )
    with open(offsets_path, "rb") as f:
        time_offsets = orjson.loads(f.read())

    if sequence_names is None:
        sequence_names = sorted(time_offsets)

    comparisons = {}
    n_missing_videos = 0

    pbar = tqdm(sequence_names, desc=f"Checking AIST++ offsets ({split})")
    for sequence_name in pbar:
        pbar.set_postfix_str(sequence_name)
        offsets = time_offsets[sequence_name]
        cameras = sorted(offsets)

        video_paths = [
            os.path.join(
                dataset_dir,
                "videos",
                "raw",
                f"{sequence_name.replace('cAll', camera)}.mp4",
            )
            for camera in cameras
        ]
        if not all(os.path.exists(path) for path in video_paths):
            n_missing_videos += 1
            continue

        waveforms = []
        sample_rates = []
        for path in video_paths:
            waveform, sample_rate = extract_audio(path)
            waveforms.append(waveform)
            sample_rates.append(sample_rate)

        kwargs = {} if device is None else {"compute_device": device}
        audio_offsets = estimate_time_offsets(waveforms, sample_rates, **kwargs)

        comparisons[sequence_name] = compare_time_offsets(
            {camera: offsets[camera]["offset_frames"] for camera in cameras},
            {
                camera: audio_offsets[i].item()
                for i, camera in enumerate(cameras)
            },
            tolerance_s,
        )
    pbar.close()

    with open(audio_check_report_path(dataset_dir, split), "wb") as f:
        f.write(orjson.dumps(comparisons, option=orjson.OPT_INDENT_2))

    flagged = flagged_views(comparisons)
    print(
        f"Checked {len(comparisons)} sequences against their audio, skipped "
        f"{n_missing_videos} with a raw video missing: {len(flagged)} flagged."
    )
    for sequence_name, views in flagged.items():
        views_str = ", ".join(f"{c}: {d:+.3f} s" for c, d in views.items())
        print(f"  {sequence_name}: {views_str}")
    if flagged:
        print(
            "A flagged view's refined clip was not cut where the other views "
            "were; add the sequence to aistpp_ignore_list.txt."
        )

    return flagged
