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

from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.pipeline import ViewInput
from kineo.pipeline.pipeline import Annotations
from kineo.annotations.background_image import (
    BackgroundImageAnnotations,
    BackgroundImageAnnotation,
)
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

from tqdm import tqdm


@dataclass(frozen=True)
class MoGeSceneReconstructionRuntimeConfig:
    mask_grow_px: int = 3
    use_background_image: bool = False
    use_half_precision: bool = False
    view_idx: int | None = None


class MoGeSceneReconstructionStage(PipelineStage[MoGeSceneReconstructionRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        model_name_or_path: str,
        runtime_cfg: MoGeSceneReconstructionRuntimeConfig,
        dynamic_runtime_cfg: Optional[
            dict[str, MoGeSceneReconstructionRuntimeConfig]
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
        runtime_cfg: MoGeSceneReconstructionRuntimeConfig,
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

        if runtime_cfg.view_idx is not None:
            views_ids = [views_ids[runtime_cfg.view_idx]]

        for view_idx, view_id in tqdm(
            enumerate(views_ids),
            desc="Reconstructing camera views",
            total=len(views_ids),
            leave=False,
        ):
            intrinsics_annotation = camera_intrinsics.filter_by_view_id(
                view_id
            ).first_or_default()
            extrinsics_annotation = camera_extrinsics.filter_by_view_id(
                view_id
            ).first_or_default()

            if runtime_cfg.use_background_image:
                if "background_images" not in annotations:
                    raise ValueError(
                        "Background images are not available. Use runtime_cfg.use_background_image=False to use the frame loader instead."
                    )

                background_images: BackgroundImageAnnotations = annotations[
                    "background_images"
                ]
                background_image_annotation: BackgroundImageAnnotation = (
                    background_images.filter_by_view_id(view_id).first_or_default()
                )
                background_image_rgb = background_image_annotation.image.to(device)
                background_image_rgb = background_image_rgb / 255.0
                background_mask = background_image_annotation.mask.to(device)
                background_sky_mask = background_image_annotation.sky_mask.to(device)
            else:
                background_image_rgb = (
                    views[view_idx]["frame_loader"].load_frame_at(0).to(device)
                )
                background_image_rgb = background_image_rgb / 255.0
                background_mask = torch.ones_like(background_image_rgb[0], dtype=torch.bool)
                background_sky_mask = torch.zeros_like(background_image_rgb[0], dtype=torch.bool)

            if intrinsics_annotation is None:
                raise ValueError(f"View {view_id} is missing intrinsics")

            resolution_hw = intrinsics_annotation.resolution_hw

            background_image_rgb = undistort_image(
                background_image_rgb,
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

            background_image_rgb = background_image_rgb[
                ..., roi_xyxy[1] : roi_xyxy[3], roi_xyxy[0] : roi_xyxy[2]
            ]
            background_mask = background_mask[
                ..., roi_xyxy[1] : roi_xyxy[3], roi_xyxy[0] : roi_xyxy[2]
            ]
            background_sky_mask = background_sky_mask[
                ..., roi_xyxy[1] : roi_xyxy[3], roi_xyxy[0] : roi_xyxy[2]
            ]

            # Use MoGe to infer depth on both images
            moge_result = self.moge_model.infer(
                background_image_rgb,
                fov_x=fov_x,
                use_fp16=runtime_cfg.use_half_precision,
            )

            # Rescale the calibration frame depth to match the scene scale
            bg_frame_points = moge_result["points"].reshape(-1, 3)  # (H, W, 3)
            bg_frame_valid_mask = moge_result["mask"].reshape(-1)  # (H, W)
            bg_frame_points_colors = background_image_rgb.permute(1, 2, 0).reshape(
                -1, 3
            )
            bg_frame_points_conf = torch.ones_like(bg_frame_points[..., 0])

            bg_frame_points_conf *= background_mask.reshape(-1)
            bg_frame_points_conf *= bg_frame_valid_mask.reshape(-1)
            bg_frame_points_conf *= ~background_sky_mask.reshape(-1)

            bg_frame_points_world = transform_points_from_camera_to_world(
                bg_frame_points,
                extrinsics_annotation.Rt.to(device),
            )

            points_xyz_world.append(bg_frame_points_world.cpu())
            points_colors.append(bg_frame_points_colors.cpu())
            points_confidences.append(bg_frame_points_conf.cpu())

        points_xyz_world = torch.cat(points_xyz_world, dim=0)
        points_colors = torch.cat(points_colors, dim=0)
        points_confidences = torch.cat(points_confidences, dim=0)

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
