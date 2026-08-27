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
from kineo.pipeline.pipeline import Annotations
from kineo.visualization.viz_2d import show_bboxes_and_keypoints
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.bboxes_2d import BBox2DAnnotations
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from dataclasses import dataclass

from kineo.pipeline.pipeline import ViewInput


@dataclass(frozen=True)
class Visualization2DRuntimeConfig:
    min_keypoint_score: float = 0.2
    keypoints_indices: list[int] = None


class Visualization2DStage(PipelineStage[Visualization2DRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: Visualization2DRuntimeConfig,
        dynamic_runtime_cfg: dict[str, Visualization2DRuntimeConfig] | None = None,
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
        runtime_cfg: Visualization2DRuntimeConfig,
    ):
        global_time_reference: GlobalTimeReferenceAnnotation = annotations["global_time_reference"]
        pred_keypoints_2d: Keypoints2DAnnotations = annotations["keypoints_2d"]
        pred_bboxes_2d: BBox2DAnnotations = annotations["bboxes_2d"]

        show_bboxes_and_keypoints(
            views=[v for v in views if v["view_id"]],
            bboxes_2d=pred_bboxes_2d,
            keypoints_2d=pred_keypoints_2d,
            score_threshold=runtime_cfg.min_keypoint_score,
            keypoints_indices=runtime_cfg.keypoints_indices,
            global_time_reference=global_time_reference,
        )
