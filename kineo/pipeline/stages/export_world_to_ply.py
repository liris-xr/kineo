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
from kineo.visualization.viz_3d import export_world_to_ply
from kineo.annotations.reconstructed_scene import WorldReconstructedSceneAnnotations
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ExportWorldToPlyRuntimeConfig:
    output_path_template: str = "./outputs/ply/{sequence_name}.ply"
    max_world_points_to_show: int = -1
    world_z_clipping_m: float = None
    min_world_point_confidence: float = 0.2


class ExportWorldToPlyStage(PipelineStage[ExportWorldToPlyRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: ExportWorldToPlyRuntimeConfig,
        dynamic_runtime_cfg: dict[str, ExportWorldToPlyRuntimeConfig] | None = None,
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
        runtime_cfg: ExportWorldToPlyRuntimeConfig,
    ):
        pred_world_reconstruction: WorldReconstructedSceneAnnotations = annotations.get(
            "world_reconstructed_scene"
        )

        formatted_output_path = runtime_cfg.output_path_template.format(
            sequence_name=sequence_name
        )
        formatted_output_path = os.path.abspath(formatted_output_path)
        os.makedirs(os.path.dirname(formatted_output_path), exist_ok=True)

        export_world_to_ply(
            path=formatted_output_path,
            world_reconstruction=pred_world_reconstruction,
            max_world_points_to_show=runtime_cfg.max_world_points_to_show,
            world_z_clipping_m=runtime_cfg.world_z_clipping_m,
            min_world_point_confidence=runtime_cfg.min_world_point_confidence,
            convert_coordinates_to_opengl=True,
        )
        print(f"World reconstruction exported to {formatted_output_path}")
