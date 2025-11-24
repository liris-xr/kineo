# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.pipeline import ViewInput
from kineo.pipeline.pipeline import Annotations

from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.keypoints_format import KeypointsFormat

from dataclasses import dataclass


@dataclass(frozen=True)
class KeypointsConversionRuntimeConfig:
    dst_keypoints_format_name: str


class MetricsExportStage(PipelineStage[KeypointsConversionRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: KeypointsConversionRuntimeConfig,
        dynamic_runtime_cfg: dict[str, KeypointsConversionRuntimeConfig] | None = None,
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
        runtime_cfg: KeypointsConversionRuntimeConfig,
    ):
        pred_keypoints_3d: Keypoints3DAnnotations = annotations["keypoints_3d"]
        pred_keypoints_2d: Keypoints2DAnnotations = annotations["keypoints_2d"]

        dst_keypoints_format = KeypointsFormat.from_mmpose_dataset(runtime_cfg.dst_keypoints_format_name)

        pred_keypoints_3d = pred_keypoints_3d.convert_to_format(dst_keypoints_format)
        pred_keypoints_2d = pred_keypoints_2d.convert_to_format(dst_keypoints_format)

        annotations["keypoints_3d"] = pred_keypoints_3d
        annotations["keypoints_2d"] = pred_keypoints_2d
