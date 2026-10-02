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
import torch.nn.functional as F
from torchvision.ops import batched_nms

PADDING_VALUE = 114.0


def resize_and_pad(
    image: torch.Tensor, input_size_hw: tuple[int, int]
) -> tuple[torch.Tensor, float]:
    height, width = input_size_hw
    source = image.to(torch.float32)

    if tuple(source.shape[1:]) == (height, width):
        return source.unsqueeze(0).contiguous(), 1.0

    ratio = min(height / source.shape[1], width / source.shape[2])
    resized = F.interpolate(
        source.unsqueeze(0),
        size=(int(source.shape[1] * ratio), int(source.shape[2] * ratio)),
        mode="bilinear",
        align_corners=False,
    )

    padded = torch.full(
        (1, 3, height, width),
        PADDING_VALUE,
        dtype=torch.float32,
        device=image.device,
    )
    padded[:, :, : resized.shape[2], : resized.shape[3]] = resized

    return padded.contiguous(), ratio


def decode_batched_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    ratios: list[float],
    score_threshold: float = 0.3,
    nms_iou_threshold: float = 0.65,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    n_views, n_anchors = scores.shape[0], scores.shape[1]
    device = boxes.device

    best_scores = scores.amax(dim=-1)

    view_of = torch.arange(n_views, device=device).repeat_interleave(n_anchors)
    flat_boxes = boxes.reshape(-1, 4)
    flat_scores = best_scores.reshape(-1)

    kept = flat_scores > score_threshold
    flat_boxes = flat_boxes[kept]
    flat_scores = flat_scores[kept]
    view_of = view_of[kept]

    scale = torch.tensor(ratios, device=device, dtype=flat_boxes.dtype)
    flat_boxes = flat_boxes / scale[view_of].unsqueeze(-1)

    order = batched_nms(flat_boxes, flat_scores, view_of, nms_iou_threshold)
    surviving_view = view_of[order]

    return [
        (
            flat_boxes[order][surviving_view == view],
            flat_scores[order][surviving_view == view],
        )
        for view in range(n_views)
    ]
