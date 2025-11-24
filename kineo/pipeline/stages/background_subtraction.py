# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional

from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.pipeline import ViewInput
from kineo.pipeline.pipeline import Annotations
from kineo.annotations.bboxes_2d import BBox2DAnnotations
from kineo.annotations.background_image import (
    BackgroundImageAnnotation,
    BackgroundImageAnnotations,
)
import torch
from tqdm import tqdm
import onnxruntime
from torchvision.transforms.functional import resize, normalize
import numpy as np


@dataclass(frozen=True)
class BackgroundSubtractionRuntimeConfig:
    batch_size: int = 4
    bbox_padding: int = 5
    n_frames_to_accumulate: int = 100
    run_skyseg: bool = True


class BackgroundSubtractionStage(PipelineStage[BackgroundSubtractionRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: BackgroundSubtractionRuntimeConfig,
        dynamic_runtime_cfg: Optional[
            dict[str, BackgroundSubtractionRuntimeConfig]
        ] = None,
        skyseg_model_path: str = "./checkpoints/skyseg.onnx",
        skyseg_input_size: tuple[int, int] = (320, 320),
    ):
        super().__init__(name, order, runtime_cfg, dynamic_runtime_cfg)
        # https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx
        self.skyseg_model_path = skyseg_model_path
        self.skyseg_input_size = skyseg_input_size

    def run_skyseg(
        self,
        skyseg_session: onnxruntime.InferenceSession,
        skyseg_input_size_hw: tuple[int, int],
        img: torch.Tensor,
    ):
        device = img.device
        orig_h, orig_w = img.shape[-2], img.shape[-1]

        if img.dtype == torch.uint8:
            img = img / 255.0

        img = resize(img, size=skyseg_input_size_hw, antialias=True)
        img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = (
            img.reshape(-1, 3, skyseg_input_size_hw[0], skyseg_input_size_hw[1])
            .cpu()
            .numpy()
        )
        img = img.astype(np.float32)

        input_name = skyseg_session.get_inputs()[0].name
        output_name = skyseg_session.get_outputs()[0].name
        sky_map = skyseg_session.run([output_name], {input_name: img})

        sky_map = torch.from_numpy(np.array(sky_map)).to(device=device)
        min_value = torch.min(sky_map)
        max_value = torch.max(sky_map)
        sky_map = (sky_map - min_value) / (max_value - min_value)

        sky_map = sky_map.reshape((1, skyseg_input_size_hw[0], skyseg_input_size_hw[1]))
        sky_map = resize(sky_map, size=(orig_h, orig_w), antialias=True)
        sky_map = (sky_map * 255).to(torch.uint8)

        # The model outputs low values for sky, high values for non-sky
        output_mask = (sky_map >= 32).squeeze(0)
        return output_mask

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: BackgroundSubtractionRuntimeConfig,
    ):
        device = pipeline.device

        if runtime_cfg.run_skyseg:
            session_options = onnxruntime.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1

            skyseg_session = onnxruntime.InferenceSession(
                path_or_bytes=self.skyseg_model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                sess_options=session_options,
            )

        bboxes_2d: BBox2DAnnotations = annotations["bboxes_2d"]
        batch_size = runtime_cfg.batch_size

        background_imgs_annotations: list[BackgroundImageAnnotation] = []

        n_frames_to_accumulate = runtime_cfg.n_frames_to_accumulate

        for view in tqdm(views, desc="Extracting background", leave=False, unit="view"):
            frame_loader = view["frame_loader"]
            view_n_frames = frame_loader.n_frames
            view_id = view["view_id"]
            view_bboxes_2d = bboxes_2d.filter_by_view_id(view_id)
            resolution_hw = frame_loader.resolution_hw

            img = torch.zeros(
                (3, resolution_hw[0], resolution_hw[1]),
                dtype=torch.float32,
                device=device,
            )
            pixel_counter = torch.zeros(
                (resolution_hw[0], resolution_hw[1]),
                dtype=torch.long,
                device=device,
            )

            first_frame_rgb = frame_loader.load_frame_at(0).to(device)
            first_frame_rgb = first_frame_rgb / 255.0

            pbar = tqdm(
                total=n_frames_to_accumulate,
                desc="Accumulating frames",
                leave=False,
                unit="frame",
            )

            # Randomly select n_frames_to_accumulate frames from the view
            frame_indices = torch.randperm(view_n_frames)[:n_frames_to_accumulate]
            frame_indices = frame_indices.sort().values

            n_batches = n_frames_to_accumulate // batch_size

            for batch_idx in range(n_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, n_frames_to_accumulate)
                batch_frames = frame_indices[batch_start:batch_end]
                actual_batch_size = len(batch_frames)

                if actual_batch_size == 0:
                    continue

                # Load batch of frames (B, C, H, W) in RGB (0-255)
                frames_rgb = frame_loader.load_frames_at(frame_indices=batch_frames).to(
                    device
                )
                frames_rgb = frames_rgb / 255.0

                frames_bboxes_xyxy = []

                for frame_idx in batch_frames:
                    frame_bboxes_xyxy = view_bboxes_2d.filter_by_frame_idx(frame_idx)
                    if len(frame_bboxes_xyxy) == 0:
                        frames_bboxes_xyxy.append(torch.zeros((0, 4), device=device))
                    else:
                        frames_bboxes_xyxy.append(torch.stack([bbox_ann.xyxy for bbox_ann in frame_bboxes_xyxy]))

                frames_mask = torch.ones(
                    (actual_batch_size, resolution_hw[0], resolution_hw[1]),
                    device=device,
                    dtype=torch.bool,
                )

                for frame_idx, frame_bboxes_xyxy in enumerate(frames_bboxes_xyxy):
                    for bbox_xyxy in frame_bboxes_xyxy:
                        x1, y1, x2, y2 = bbox_xyxy
                        x1 = torch.floor(x1).int() - runtime_cfg.bbox_padding
                        y1 = torch.floor(y1).int() - runtime_cfg.bbox_padding
                        x2 = torch.ceil(x2).int() + runtime_cfg.bbox_padding
                        y2 = torch.ceil(y2).int() + runtime_cfg.bbox_padding
                        x1 = torch.clamp(x1, 0, resolution_hw[1] - 1)
                        y1 = torch.clamp(y1, 0, resolution_hw[0] - 1)
                        x2 = torch.clamp(x2, 0, resolution_hw[1] - 1)
                        y2 = torch.clamp(y2, 0, resolution_hw[0] - 1)
                        frames_mask[frame_idx, y1:y2, x1:x2] = False

                pixel_counter += frames_mask.sum(dim=0).int()
                img += (frames_rgb * frames_mask.unsqueeze(1)).sum(dim=0)

                pbar.update(actual_batch_size)

            pbar.close()

            background_mask = pixel_counter > 0
            background_img = torch.where(
                pixel_counter != 0, img / pixel_counter, torch.zeros_like(img)
            )
            background_img = background_img.clamp(0, 1)
            background_img[:, ~background_mask] = first_frame_rgb[:, ~background_mask]
            background_img = (background_img * 255).to(torch.uint8)

            if runtime_cfg.run_skyseg:
                sky_mask = self.run_skyseg(
                    skyseg_session, self.skyseg_input_size, background_img
                )
            else:
                sky_mask = torch.zeros_like(background_mask, dtype=torch.bool)

            background_imgs_annotations.append(
                BackgroundImageAnnotation(
                    view_id=view_id,
                    image=background_img.cpu(),
                    mask=background_mask.cpu(),
                    sky_mask=sky_mask.cpu(),
                )
            )

        annotations["background_images"] = BackgroundImageAnnotations(
            annotations=background_imgs_annotations
        )
