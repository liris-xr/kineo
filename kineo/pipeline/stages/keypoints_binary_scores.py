# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import dataclasses
from dataclasses import dataclass

import torch

from kineo.annotations import Annotations
from kineo.annotations import Keypoints2DAnnotations
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.pipeline import PipelineStage


def with_binary_scores(keypoints: Keypoints2DAnnotations) -> Keypoints2DAnnotations:
    return Keypoints2DAnnotations(
        metadata=keypoints.metadata,
        annotations=[
            dataclasses.replace(a, scores=(a.scores > 0).to(a.scores.dtype))
            for a in keypoints.annotations
        ],
    )


@dataclass(frozen=True)
class KeypointsBinaryScoresRuntimeConfig:
    pass


class KeypointsBinaryScoresStage(PipelineStage[KeypointsBinaryScoresRuntimeConfig]):
    """Stage for binarizing the 2D keypoint scores after sampling."""

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: KeypointsBinaryScoresRuntimeConfig,
        dynamic_runtime_cfg: dict[str, KeypointsBinaryScoresRuntimeConfig]
        | None = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: KeypointsBinaryScoresRuntimeConfig,
    ):
        annotations["keypoints_2d"] = with_binary_scores(annotations["keypoints_2d"])
