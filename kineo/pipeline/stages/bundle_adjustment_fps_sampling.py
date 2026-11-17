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
from kineo.pipeline.stages.bundle_adjustment_sampling import (
    BundleAdjustmentSamplingRuntimeConfig,
    BundleAdjustmentSamplingStage,
)
from kineo.sampling import farthest_point_sampling, normalized_uv

from dataclasses import dataclass


@dataclass(frozen=True)
class BundleAdjustmentFPSSamplingRuntimeConfig(
    BundleAdjustmentSamplingRuntimeConfig
):
    image_plane_weight: float = 1.0
    time_weight: float = 1.0


class BundleAdjustmentFPSSamplingStage(BundleAdjustmentSamplingStage):
    """BundleAdjustmentSamplingStage with farthest-point instead of random draws."""

    def pick_samples(
        self,
        pipeline: Pipeline,
        runtime_cfg: BundleAdjustmentFPSSamplingRuntimeConfig,
        kps_xy: torch.Tensor,
        kps_scores: torch.Tensor,
        candidates_indices: torch.Tensor,
        cameras_resolutions_hw: list[tuple[int, int]],
        n_frames: int,
        n_points_per_frame: int,
    ) -> torch.Tensor:
        device = pipeline.device
        n_views = kps_xy.shape[0]
        generator = torch.Generator(device=device)
        generator.manual_seed(pipeline.seed)

        resolutions_hw = torch.as_tensor(
            cameras_resolutions_hw, dtype=kps_xy.dtype, device=device
        )
        uvs = normalized_uv(kps_xy[:, candidates_indices], resolutions_hw)
        candidates_frame_indices = candidates_indices // n_points_per_frame
        ts = candidates_frame_indices.to(kps_xy.dtype) / max(n_frames - 1, 1)

        features = torch.stack(
            [
                runtime_cfg.image_plane_weight * uvs[..., 0],
                runtime_cfg.image_plane_weight * uvs[..., 1],
                runtime_cfg.time_weight * ts.unsqueeze(0).expand(n_views, -1),
            ],
            dim=-1,
        )
        selected, selected_valid = farthest_point_sampling(
            features,
            runtime_cfg.n_kp_samples,
            generator,
            kps_scores[:, candidates_indices] > runtime_cfg.min_kp_score,
        )
        selected = selected.tolist()
        selected_valid = selected_valid.tolist()

        # Interleave views rank by rank before truncating
        merged = dict.fromkeys(
            selected[view_idx][rank]
            for rank in range(len(selected[0]))
            for view_idx in range(n_views)
            if selected_valid[view_idx][rank]
        )
        return candidates_indices[
            torch.as_tensor(
                list(merged)[: runtime_cfg.n_kp_samples],
                dtype=torch.long,
                device=device,
            )
        ]
