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

from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.global_scale import (
    GlobalScaleAnnotation,
    GlobalScaleAnnotations,
    GlobalScaleAnnotationsMetadata,
)


class GTScaleComputationStage(PipelineStage[None]):
    def __init__(
        self,
        name: str,
        order: int,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=None,
            dynamic_runtime_cfg=None,
        )

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: None = None,
    ):
        pred_camera_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "cameras_extrinsics"
        ]
        gt_camera_extrinsics: CameraExtrinsicsAnnotations = gt_annotations.get(
            "cameras_extrinsics"
        )

        _, _, s = pred_camera_extrinsics.compute_similarity_transform(
            gt_camera_extrinsics, estimate_scale=True
        )

        global_scale_annotations = GlobalScaleAnnotations(
            metadata=GlobalScaleAnnotationsMetadata(),
            annotations=[GlobalScaleAnnotation(frame_idx=0, scale=s.squeeze().item())],
        )
        annotations["global_scale"] = global_scale_annotations
