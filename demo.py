# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import torch
from kineo.pipeline.pipeline import Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.io.frame_sequence_loader import VideoLoader
from kineo.io.audio_loader import VideoAudioLoader
import argparse

import rerun as rr

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def print_system_info(device: torch.device):
    print(f"Torch version: {torch.__version__}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB"
        )

def create_views(video_paths: list[str], device: torch.device):
    views = []
    for i, video_path in enumerate(video_paths):
        views.append(
            ViewInput(
                view_id=f"view_{i}",
                frame_loader=VideoLoader(video_path=video_path, device=device),
                audio_loader=VideoAudioLoader(video_path=video_path, device=device),
            )
        )
    return views

def main(video_paths: list[str]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_info(device)

    config_file = "configs/demo.yaml"
    pipeline = Pipeline.build_pipeline_from_config(config_file, device)

    views = create_views(video_paths, device)

    _ = pipeline.run(
        sequence_name="demo_sequence",
        views=views,
        annotations={},
        gt_annotations={}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_paths", type=str, nargs="+")
    args = parser.parse_args()
    video_paths = args.video_paths
    main(video_paths)

    rr.init("demo", spawn=True)
    rr.log_file_from_path("./outputs/rerun/demo_sequence.rrd")