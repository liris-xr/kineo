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
import warnings
warnings.filterwarnings("ignore")

from kineo.pipeline.pipeline import Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.io.frame_sequence_loader import VideoLoader
from kineo.io.audio_loader import VideoAudioLoader
import argparse
from omegaconf import OmegaConf

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

def main(config_file: str, sequence_name: str, video_paths: list[str], batch_size: int | None = None, target_fps: int | None = 50, shared_intrinsics: bool = False, use_cache: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_system_info(device)

    cfg = OmegaConf.load(config_file)

    if batch_size is not None:
        cfg.batch_size = batch_size

    if target_fps is not None:
        cfg["pipeline"]["stages"]["global_time_resampling"]["runtime_cfg"]["target_fps"] = target_fps

    cfg.shared_intrinsics = shared_intrinsics
    cfg.use_cache = use_cache

    pipeline = Pipeline.build_pipeline_from_config(cfg, device)

    views = create_views(video_paths, device)

    _ = pipeline.run(
        sequence_name=sequence_name,
        views=views,
        annotations={},
        gt_annotations={}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="configs/demo/offline/nlf_single_person_sam2.yaml")
    parser.add_argument("--sequence-name", type=str, default="offline_demo")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-fps", type=int, default=None)
    parser.add_argument("--shared-intrinsics", action="store_true", default=False)
    parser.add_argument("--use-cache", action="store_true", default=False)
    parser.add_argument("video_paths", type=str, nargs="+")
    args = parser.parse_args()
    config_file = args.config_file
    sequence_name = args.sequence_name
    video_paths = args.video_paths
    batch_size = args.batch_size
    target_fps = args.target_fps
    shared_intrinsics = args.shared_intrinsics
    use_cache = args.use_cache
    main(config_file, sequence_name, video_paths, batch_size, target_fps, shared_intrinsics, use_cache)