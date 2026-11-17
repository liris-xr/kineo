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
from torchvision.ops import roi_align


def bboxes_to_center_scale(
    bboxes_xyxy: torch.Tensor,
    input_size_wh: tuple[int, int],
    padding: float = 1.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = input_size_wh
    aspect_ratio = width / height

    centers = (bboxes_xyxy[:, :2] + bboxes_xyxy[:, 2:]) * 0.5
    scales = (bboxes_xyxy[:, 2:] - bboxes_xyxy[:, :2]) * padding

    box_widths = scales[:, 0]
    box_heights = scales[:, 1]
    widened = box_widths > box_heights * aspect_ratio
    scales = torch.stack(
        (
            torch.where(widened, box_widths, box_heights * aspect_ratio),
            torch.where(widened, box_widths / aspect_ratio, box_heights),
        ),
        dim=-1,
    )

    return centers, scales


def crop_and_normalize(
    image: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    input_size_wh: tuple[int, int],
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
) -> torch.Tensor:
    width, height = input_size_wh
    half = scales / 2

    image_index = torch.zeros(
        (centers.shape[0], 1), dtype=torch.float32, device=centers.device
    )
    regions = torch.cat((image_index, centers - half, centers + half), dim=-1)
    crops = roi_align(
        image.to(torch.float32).unsqueeze(0),
        regions,
        output_size=(height, width),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=True,
    )

    if mean is not None:
        crops = (crops - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

    return crops.contiguous()


def decode_simcc(
    simcc_x: torch.Tensor,
    simcc_y: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    input_size_wh: tuple[int, int],
    simcc_split_ratio: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_x, locs_x = simcc_x.max(dim=-1)
    max_y, locs_y = simcc_y.max(dim=-1)

    locations = torch.stack((locs_x, locs_y), dim=-1).to(torch.float32)
    scores = 0.5 * (max_x + max_y)
    locations[scores <= 0.0] = -1.0

    input_size = torch.tensor(
        input_size_wh, dtype=torch.float32, device=simcc_x.device
    )
    keypoints = locations / simcc_split_ratio / input_size * scales.unsqueeze(1)
    keypoints = keypoints + centers.unsqueeze(1) - scales.unsqueeze(1) / 2

    return keypoints, scores
