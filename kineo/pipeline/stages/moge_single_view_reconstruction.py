# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional
from torchvision.io.image import decode_image, ImageReadMode
from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.pipeline import ViewInput
from kineo.pipeline.pipeline import Annotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.reconstructed_scene import (
    WorldReconstructedSceneAnnotation,
    WorldReconstructedSceneAnnotations,
)
import torch
from moge.model.v2 import MoGeModel
from kineo.geometry.transformations import (
    undistort_image,
    compute_optimal_K,
)
from kineo.geometry.camera import transform_points_from_camera_to_world


@dataclass(frozen=True)
class MoGeSingleViewReconstructionRuntimeConfig:
    view_id: str
    frame_idx: int | None = None
    img_path: str | None = None

    def __post_init__(self):
        if self.frame_idx is not None and self.img_path is not None:
            raise ValueError("Only frame_idx or img_path can be provided at once.")
        if self.frame_idx is None and self.img_path is None:
            raise ValueError("At least frame_idx or img_path needs to be provided.")


class MoGeSingleViewReconstructionStage(
    PipelineStage[MoGeSingleViewReconstructionRuntimeConfig]
):
    def __init__(
        self,
        name: str,
        order: int,
        model_name_or_path: str,
        runtime_cfg: MoGeSingleViewReconstructionRuntimeConfig,
        dynamic_runtime_cfg: Optional[
            dict[str, MoGeSingleViewReconstructionRuntimeConfig]
        ] = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moge_model = MoGeModel.from_pretrained(model_name_or_path).cpu()
        self.moge_model.eval()

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: MoGeSingleViewReconstructionRuntimeConfig,
    ):
        device = pipeline.device

        self.moge_model = self.moge_model.to(device)

        camera_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "camera_intrinsics"
        ]
        camera_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "camera_extrinsics"
        ]

        views_ids = [v["view_id"] for v in views]

        points_xyz_world: list[torch.Tensor] = []
        points_colors: list[torch.Tensor] = []
        points_confidences: list[torch.Tensor] = []

        intrinsics_annotation = camera_intrinsics.filter_by_view_id(
            runtime_cfg.view_id
        ).first_or_default()
        extrinsics_annotation = camera_extrinsics.filter_by_view_id(
            runtime_cfg.view_id
        ).first_or_default()

        if intrinsics_annotation is None or extrinsics_annotation is None:
            raise ValueError(
                f"View {runtime_cfg.view_id} is missing intrinsics, extrinsics"
            )

        resolution_hw = intrinsics_annotation.resolution_hw

        view_input = next(v for v in views if v["view_id"] == runtime_cfg.view_id)

        frame_loader = view_input["frame_loader"]

        if runtime_cfg.frame_idx is not None:
            frame_rgb = frame_loader.load_frame_at(runtime_cfg.frame_idx).to(device)
            frame_rgb = frame_rgb / 255.0
        elif runtime_cfg.img_path is not None:
            frame_rgb = decode_image(
                runtime_cfg.img_path,
                mode=ImageReadMode.RGB,
                apply_exif_orientation=True,
            )
            frame_rgb = frame_rgb / 255.0

            _, img_h, img_w = frame_rgb.shape

            if resolution_hw != (img_h, img_w):
                raise ValueError(f"Image should have the same resolution as view input. Expected {resolution_hw}, got ({img_h}, {img_w})")

        # Undistort both background image and calibration frame
        frame_rgb = undistort_image(
            frame_rgb,
            K=intrinsics_annotation.K,
            D=intrinsics_annotation.distortion_coefficients,
            distortion_model=intrinsics_annotation.distortion_model.value,
        )

        new_K, roi_xyxy = compute_optimal_K(
            resolution_hw,
            K=intrinsics_annotation.K,
            D=intrinsics_annotation.distortion_coefficients,
            distortion_model=intrinsics_annotation.distortion_model.value,
        )
        fov_x = (2 * torch.atan(resolution_hw[1] / (2 * new_K[0, 0]))).rad2deg()

        frame_rgb = frame_rgb[..., roi_xyxy[1] : roi_xyxy[3], roi_xyxy[0] : roi_xyxy[2]]

        # Use MoGe to infer depth on both images
        moge_result = self.moge_model.infer(
            frame_rgb,
            fov_x=fov_x,
        )

        # Rescale the calibration frame depth to match the scene scale
        bg_frame_points = moge_result["points"].reshape(-1, 3)  # (H, W, 3)
        bg_frame_valid_mask = moge_result["mask"].reshape(-1)  # (H, W)
        bg_frame_points_colors = frame_rgb.permute(1, 2, 0).reshape(-1, 3)
        bg_frame_points_conf = bg_frame_valid_mask.float()

        bg_frame_points_world = transform_points_from_camera_to_world(
            bg_frame_points,
            extrinsics_annotation.Rt.to(device),
        )

        points_xyz_world = bg_frame_points_world.cpu()
        points_colors = bg_frame_points_colors.cpu()
        points_confidences = bg_frame_points_conf.cpu()

        world_reconstructed_scene_annotations = WorldReconstructedSceneAnnotations(
            annotations=[
                WorldReconstructedSceneAnnotation(
                    points_xyz=points_xyz_world,
                    points_colors=points_colors,
                    points_confidences=points_confidences,
                )
            ],
        )

        annotations["world_reconstructed_scene"] = world_reconstructed_scene_annotations
        self.moge_model = self.moge_model.cpu()
