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

from . import torch_processings
from ..base import BaseTool


class YOLOXPerson(BaseTool):
    """YOLOX person detector for a person-only export without NMS."""

    def __init__(
        self,
        onnx_model: str,
        model_input_size: tuple = (416, 416),
        score_thr: float = 0.3,
        nms_thr: float = 0.65,
        backend: str = "onnxruntime",
        device: str = "cpu",
        use_tensorrt: bool = False,
        fp16: bool = False,
        engine_cache_dir: str = None,
        strict_providers: bool = False,
    ):
        super().__init__(
            onnx_model,
            model_input_size,
            backend=backend,
            device=device,
            use_tensorrt=use_tensorrt,
            fp16=fp16,
            engine_cache_dir=engine_cache_dir,
            strict_providers=strict_providers,
        )
        self.score_thr = score_thr
        self.nms_thr = nms_thr

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "YOLOXPerson detects on the device in batches; use forward_gpu."
        )

    def forward_gpu(
        self, frames: list[torch.Tensor]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        prepared = [
            torch_processings.resize_and_pad(
                frame, tuple(self.model_input_size))
            for frame in frames
        ]
        batch = torch.cat([image for image, _ in prepared], dim=0).contiguous()

        boxes, scores = self.inference_gpu(batch)

        return torch_processings.decode_batched_detections(
            boxes,
            scores,
            [ratio for _, ratio in prepared],
            score_threshold=self.score_thr,
            nms_iou_threshold=self.nms_thr,
        )
