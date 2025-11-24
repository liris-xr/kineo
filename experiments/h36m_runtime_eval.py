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
import torch
from kineo.pipeline.pipeline import Pipeline

from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.io.frame_sequence_loader import VideoLoader

import argparse

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def main(
    long_video_dir: str,
    config_file: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.build_pipeline_from_config(config_file, device)

    camera_names = ["54138969", "55011271", "58860488", "60457274"]

    views = []
    for camera_name in camera_names:
        video_path = os.path.join(long_video_dir, f"{camera_name}.mp4")
        views.append(
            ViewInput(
                view_id=camera_name,
                frame_loader=VideoLoader(
                    video_path=video_path,
                    device=device,
                ),
                audio_loader=None,
            )
        )

    _ = pipeline.run(
        sequence_name="runtime_eval",
        views=views,
        annotations={},
        gt_annotations={},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=str)
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/experiments/runtime/h36m_long_video_rtmpose_runtime.yaml",
    )
    args = parser.parse_args()
    video_dir = args.video_dir
    config_file = args.config_file
    main(video_dir, config_file)
