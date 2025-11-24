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
from kineo.visualization.viz_3d import export_usd
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
import torch
from dataclasses import dataclass
import warnings
import os


@dataclass(frozen=True)
class ExportToUsdRuntimeConfig:
    output_path_template: str = "./outputs/usd/{sequence_name}.usd"
    export_as_directory: bool = False
    keypoints_score_threshold: float = 0.2
    camera_scale: float = 1
    camera_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    joints_radius: float = 0.01
    skeleton_color_override: tuple[float, float, float] = None
    keypoints_to_show: list[int] | None = None
    keypoints_to_hide: list[int] | None = None
    start_frame_idx: int | None = None
    end_frame_idx: int | None = None


class ExportToUsdStage(PipelineStage[ExportToUsdRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: ExportToUsdRuntimeConfig,
        dynamic_runtime_cfg: dict[str, ExportToUsdRuntimeConfig] | None = None,
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
        runtime_cfg: ExportToUsdRuntimeConfig,
    ):
        global_time_reference: GlobalTimeReferenceAnnotations = annotations.get(
            "global_time_reference"
        )

        pred_keypoints_3d: Keypoints3DAnnotations = annotations["keypoints_3d"]
        pred_camera_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "camera_extrinsics"
        ]
        pred_camera_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "camera_intrinsics"
        ]

        if global_time_reference is not None:
            dt = (
                torch.diff(global_time_reference.first_or_default().timestamps)
                .mean()
                .item()
            )

            if dt > 0:
                target_fps = 1 / dt
            else:
                target_fps = 25
                warnings.warn(
                    f"Sequence {sequence_name} has a dt of {dt} seconds, which is less than 0.01 seconds. Setting target fps to 25."
                )
        else:
            target_fps = 25

        formatted_output_path = runtime_cfg.output_path_template.format(
            sequence_name=sequence_name
        )
        formatted_output_path = os.path.abspath(formatted_output_path)
        os.makedirs(os.path.dirname(formatted_output_path), exist_ok=True)

        start_frame_idx = runtime_cfg.start_frame_idx if runtime_cfg.start_frame_idx is not None else 0
        end_frame_idx = runtime_cfg.end_frame_idx if runtime_cfg.end_frame_idx is not None else pred_keypoints_3d.n_frames - 1

        frames = list(pred_keypoints_3d.frames)
        frames = [f for f in frames if f >= start_frame_idx and f <= end_frame_idx]
        pred_keypoints_3d = pred_keypoints_3d.filter_by_frames_idxs(frames)

        # Shift all the frames by the start_frame_idx
        pred_keypoints_3d = pred_keypoints_3d.apply_frame_idx_shift(-start_frame_idx)

        export_usd(
            path=formatted_output_path,
            keypoints_3d=pred_keypoints_3d,
            camera_extrinsics=pred_camera_extrinsics,
            camera_intrinsics=pred_camera_intrinsics,
            score_threshold=runtime_cfg.keypoints_score_threshold,
            fps=target_fps,
            camera_scale=runtime_cfg.camera_scale,
            camera_color=runtime_cfg.camera_color,
            joints_radius=runtime_cfg.joints_radius,
            keypoints_to_show=runtime_cfg.keypoints_to_show,
            keypoints_to_hide=runtime_cfg.keypoints_to_hide,
            export_as_directory=runtime_cfg.export_as_directory,
            convert_coordinates_to_opengl=True,
        )
