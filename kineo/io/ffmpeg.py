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

def decode_video_to_grayscale(video_path: str, width: int, height: int) -> torch.Tensor:
    """Decodes a whole video to downscaled grayscale frames.

    Downscaling happens inside ffmpeg, so a full-resolution frame is never
    materialized and a several-minute 1080p video costs a few megabytes.

    Args:
        video_path: Path to the video to decode.
        width: Width the frames are scaled down to.
        height: Height the frames are scaled down to.

    Returns:
        Frames flattened per frame, of shape (n_frames, height * width) and
        dtype uint8.

    Raises:
        FileNotFoundError: If `video_path` does not exist.
        ValueError: If ffmpeg fails, or emits a truncated final frame.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")

    decode_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        video_path,
        "-vf",
        f"scale={width}:{height}",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "pipe:1",
    ]

    result = subprocess.run(decode_cmd, capture_output=True)

    if result.returncode != 0:
        raise ValueError(
            f"Failed to decode {video_path}: {result.stderr.decode().strip()}"
        )

    frame_size = width * height

    if len(result.stdout) % frame_size != 0:
        raise ValueError(
            f"Decoding {video_path} yielded {len(result.stdout)} bytes, "
            f"not a multiple of the {frame_size} B frame size."
        )

    return torch.frombuffer(bytearray(result.stdout), dtype=torch.uint8).reshape(
        -1, frame_size
    )


def scale_filter_args(scale: float) -> list[str]:
    """Builds the ffmpeg arguments resizing frames by `scale`.

    Sides are rounded to even numbers, which the 4:2:0 chroma subsampling the
    encoders use requires.

    Args:
        scale: Factor the frames are resized by.

    Returns:
        The arguments, empty when the frames are left alone.
    """
    if scale == 1.0:
        return []

    return ["-vf", f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2"]


def encode_images_to_video(
    image_paths: list[str],
    output_path: str,
    fps: float,
    scale: float = 1.0,
):
    """Encodes an image sequence into a video.

    Args:
        image_paths: Images to encode, in the order they are shown.
        output_path: Video file to write.
        fps: Rate the images are played back at.
        scale: Factor the images are resized by. Anything expressed in their
            pixels has to be resized with them.

    Raises:
        ValueError: If ffmpeg fails.
    """
    # A list file takes the images whatever they are named, which a numbered
    # input pattern does not.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False
    ) as list_file:
        for path in image_paths:
            list_file.write(f"file '{path.replace(os.sep, '/')}'\n")

    encode_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-r",
        str(fps),
        "-i",
        list_file.name,
        *scale_filter_args(scale),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        # Visually lossless enough for a preview at a fraction of the size.
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]

    try:
        result = subprocess.run(encode_cmd, capture_output=True)
    finally:
        os.remove(list_file.name)

    if result.returncode != 0:
        raise ValueError(
            f"Failed to encode {output_path}: {result.stderr.decode().strip()}"
        )


def get_video_codec(video_path: str) -> str:
    """Reads the codec a video's first video stream is encoded with.

    Args:
        video_path: Path to the video to probe.

    Returns:
        The codec name, as ffprobe spells it.

    Raises:
        ValueError: If ffprobe fails, or the file holds no video stream.
    """
    ffprobe_cmd = [
        "ffprobe",
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise ValueError(
            f"Failed to probe {video_path}: {result.stderr.strip()}"
        )

    codec_name = result.stdout.strip()

    if not codec_name:
        raise ValueError(f"{video_path} holds no video stream.")

    return codec_name


def transcode_video_to_h264(
    video_path: str, output_path: str, scale: float = 1.0
):
    """Re-encodes a video to H.264, frame for frame.

    Frames are passed through rather than resampled, so an index into the
    source video is still an index into the result.

    Args:
        video_path: Video to re-encode.
        output_path: Video file to write.
        scale: Factor the frames are resized by. Anything expressed in their
            pixels has to be resized with them.

    Raises:
        ValueError: If ffmpeg fails.
    """
    transcode_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-y",
        "-i",
        video_path,
        *scale_filter_args(scale),
        "-fps_mode",
        "passthrough",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-an",
        output_path,
    ]

    result = subprocess.run(transcode_cmd, capture_output=True)

    if result.returncode != 0:
        raise ValueError(
            f"Failed to transcode {video_path}: {result.stderr.decode().strip()}"
        )
