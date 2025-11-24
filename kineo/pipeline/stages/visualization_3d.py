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
from kineo.visualization.viz_3d import show_keypoints_and_cameras
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.reconstructed_scene import WorldReconstructedSceneAnnotations
from kineo.annotations.smpl_params import SMPLShapeAnnotations, SMPLPoseAnnotations
import torch
from dataclasses import dataclass
import warnings


@dataclass(frozen=True)
class Visualization3DRuntimeConfig:
    keypoints_score_threshold: float = 0.2
    show_world_reconstruction: bool = True
    world_points_size: float = 0.7
    max_world_points_to_show: int = -1
    world_z_clipping_threshold: float = None
    camera_scale: float = 1
    camera_color: tuple[float, float, float] = (1.0, 0.0, 0.0)
    world_points_confidence_threshold: float = 0.2
    estimate_scale: bool = True
    skeleton_radius: float = 0.01
    skeleton_color_override: tuple[float, float, float] = None
    skeleton_confidence_colormap: str = "viridis"
    show_skeleton_confidence: bool = False
    normalize_skeleton_confidence_per_frame: bool = False
    keypoints_to_show: list[int] | None = None
    keypoints_to_hide: list[int] | None = None
    show_floor: bool = True


class Visualization3DStage(PipelineStage[Visualization3DRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: Visualization3DRuntimeConfig,
        dynamic_runtime_cfg: dict[str, Visualization3DRuntimeConfig] | None = None,
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
        runtime_cfg: Visualization3DRuntimeConfig,
    ):
        global_time_reference: GlobalTimeReferenceAnnotations = annotations.get(
            "global_time_reference"
        )

        pred_keypoints_3d: Keypoints3DAnnotations = annotations.get("keypoints_3d")
        pred_camera_extrinsics: CameraExtrinsicsAnnotations = annotations.get(
            "camera_extrinsics"
        )
        pred_camera_intrinsics: CameraIntrinsicsAnnotations = annotations.get(
            "camera_intrinsics"
        )
        pred_world_reconstruction: WorldReconstructedSceneAnnotations = annotations.get(
            "world_reconstructed_scene"
        )

        pred_smpl_shape: SMPLShapeAnnotations = annotations.get("smpl_shape")
        pred_smpl_pose: SMPLPoseAnnotations = annotations.get("smpl_pose")

        gt_keypoints_3d: Keypoints3DAnnotations = gt_annotations.get("keypoints_3d")
        gt_camera_extrinsics: CameraExtrinsicsAnnotations = gt_annotations.get(
            "camera_extrinsics"
        )
        gt_camera_intrinsics: CameraIntrinsicsAnnotations = gt_annotations.get(
            "camera_intrinsics"
        )

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

        show_keypoints_and_cameras(
            window_name=sequence_name,
            keypoints_3d=pred_keypoints_3d,
            camera_extrinsics=pred_camera_extrinsics,
            camera_intrinsics=pred_camera_intrinsics,
            score_threshold=runtime_cfg.keypoints_score_threshold,
            fps=target_fps,
            world_reconstruction=(
                pred_world_reconstruction
                if runtime_cfg.show_world_reconstruction
                else None
            ),
            world_points_size=runtime_cfg.world_points_size,
            world_z_clipping_threshold=runtime_cfg.world_z_clipping_threshold,
            max_world_points_to_show=runtime_cfg.max_world_points_to_show,
            world_points_confidence_threshold=runtime_cfg.world_points_confidence_threshold,
            camera_scale=runtime_cfg.camera_scale,
            gt_keypoints_3d=gt_keypoints_3d,
            gt_camera_extrinsics=gt_camera_extrinsics,
            gt_camera_intrinsics=gt_camera_intrinsics,
            estimate_scale=runtime_cfg.estimate_scale,
            camera_color=runtime_cfg.camera_color,
            skeleton_radius=runtime_cfg.skeleton_radius,
            skeleton_color_override=runtime_cfg.skeleton_color_override,
            skeleton_confidence_colormap=runtime_cfg.skeleton_confidence_colormap,
            show_skeleton_confidence=runtime_cfg.show_skeleton_confidence,
            normalize_skeleton_confidence_per_frame=runtime_cfg.normalize_skeleton_confidence_per_frame,
            keypoints_to_show=runtime_cfg.keypoints_to_show,
            keypoints_to_hide=runtime_cfg.keypoints_to_hide,
            smpl_shape=pred_smpl_shape,
            smpl_pose=pred_smpl_pose,
            show_floor=runtime_cfg.show_floor,
        )
