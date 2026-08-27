# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import subprocess
from typing import Dict, Any, Tuple
import torch
from tqdm import tqdm
import tempfile
import os

from kineo.io.audio_file import load_waveform

def get_audio_stream_info(video_path: str, stream_idx: int = 0) -> Dict[str, Any]:
    ffprobe_cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        f"a:{stream_idx}",
        "-show_entries",
        "stream=duration,sample_rate,nb_frames,sample_fmt,channels,codec_name,channel_layout",
        "-of",
        "default=noprint_wrappers=1",
        video_path,
    ]
    try:
        ffprobe_cmd_stdout = subprocess.run(
            ffprobe_cmd, capture_output=True, text=True
        ).stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to run command: {e}")

    if ffprobe_cmd_stdout.strip() == "":
        # No audio stream found
        return None

    output_dict = {
        k: v
        for k, v in [line.split("=") for line in ffprobe_cmd_stdout.strip().split("\n")]
    }

    return dict(
        sample_rate=int(output_dict["sample_rate"]),
        nb_frames=int(output_dict["nb_frames"]),
        duration=float(output_dict["duration"]),
        sample_fmt=output_dict["sample_fmt"],
        channels=int(output_dict["channels"]),
        codec_name=output_dict["codec_name"],
        channel_layout=output_dict["channel_layout"],
    )

def extract_audio(
    video_path: str, show_progress: bool = False
) -> tuple[torch.Tensor | None, int | None]:
    audio_metadata = get_audio_stream_info(video_path)

    # No audio stream found
    if audio_metadata is None:
        return None, None

    # Extract audio to temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        output_path = tmp_file.name
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    extract_audio_cmd = [
        "ffmpeg",
        "-y",
        "-progress",
        "pipe:1",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-vsync",
        "0",
        output_path,
    ]

    p = subprocess.Popen(
        extract_audio_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    pbar = tqdm(
        desc="Extracting audio",
        total=audio_metadata["duration"] * 1000,
        unit="ms",
        leave=False,
        disable=not show_progress,
    )

    while True:
        line = p.stdout.readline()
        if not line:
            break
        line = line.strip()

        if line.startswith("out_time_us="):
            time_ms = int(line.split("out_time_us=")[1]) // 1000
            pbar.n = time_ms
            pbar.refresh()
        if line == "progress=end":
            pbar.close()
            break
    pbar.n = pbar.total
    pbar.close()
    p.wait()

    if p.returncode != 0:
        raise Exception("ffmpeg command failed.")

    audio, sample_rate = load_waveform(output_path)

    # Remove temporary file
    os.remove(output_path)
    return audio, sample_rate

def get_frames_timing_info(
    video_path: str,
    stream_idx: int = 0,
    device: torch.device = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    p = subprocess.Popen(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-select_streams",
            f"v:{stream_idx}",
            "-show_entries",
            "packet=pts_time,duration_time",
            "-of",
            "default=noprint_wrappers=1",
            video_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    frames_pts_ns = []
    frames_duration_ns = []

    for line in tqdm(p.stdout, desc="Getting frames timing info", leave=False):
        key, value = line.split("=")
        if key == "pts_time":
            frames_pts_ns.append(int(float(value) * 1e9))
        elif key == "duration_time":
            frames_duration_ns.append(int(float(value) * 1e9))
        else:
            raise ValueError(f"Unexpected key: {key}")

    frames_pts_ns = torch.tensor(frames_pts_ns, dtype=torch.long, device=device)
    frames_duration_ns = torch.tensor(
        frames_duration_ns, dtype=torch.long, device=device
    )

    frames_pts_ns, indices = torch.sort(frames_pts_ns)
    frames_duration_ns = frames_duration_ns[indices]
    return frames_pts_ns, frames_duration_ns

def get_video_duration(video_path: str) -> float:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")

    ffprobe_cmd = [
        "ffprobe",
        "-hide_banner",
        "-i",
        video_path,
        "-show_entries",
        "format=duration",
        "-v",
        "quiet",
        "-of",
        "csv=p=0",
    ]

    try:
        ffprobe_cmd_stdout = subprocess.run(
            ffprobe_cmd, capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Failed to run command: {e}")

    output = ffprobe_cmd_stdout.strip()
    return float(output)