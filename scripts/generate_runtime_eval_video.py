import argparse
import os
import orjson
from math import ceil
from kineo.io.ffmpeg import get_video_duration
import subprocess

def repeat_video(video_path: str, n_repetitions: int, output_path: str):
    ffmpeg_cmd = [
        "ffmpeg",
        "-stream_loop",
        str(n_repetitions),
        "-i",
        video_path,
        "-c",
        "copy",
        output_path,
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Repeated video {video_path} {n_repetitions} times and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h36m_dataset_dir", type=str, help="Path to Human3.6M dataset directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    parser.add_argument("sequence_name", type=str, help="Name of the sequence to generate the long video for", default="S11_WalkTogether 1")
    parser.add_argument("--min-duration", type=float, help="Minimum duration of the long video in seconds (default: 20min)", default=1200)
    args = parser.parse_args()

    dataset_dir = args.h36m_dataset_dir
    output_dir = args.output_dir
    sequence_name = args.sequence_name

    sequences_file = os.path.join(dataset_dir, "h36m_protocol1_sequences.json")
    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())
    sequence = next(s for s in sequences if s["sequence_name"] == sequence_name)

    assert sequence is not None

    os.makedirs(output_dir, exist_ok=True)

    for view_id, view in sequence["views"].items():
        video_path = view["video_path"]
        # Original video duration
        video_duration = get_video_duration(os.path.join(dataset_dir, video_path))
        n_repetitions = ceil(args.min_duration / video_duration)

        print(video_path, video_duration, n_repetitions)
        repeat_video(os.path.join(dataset_dir, video_path), n_repetitions, os.path.join(output_dir, f"{view_id}.mp4"))