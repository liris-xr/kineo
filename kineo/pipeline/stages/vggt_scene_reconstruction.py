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
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from kineo.geometry.transformations import (
    undistort_image,
    compute_optimal_K,
    inverse_Rt,
)
from kineo.geometry.transformations import (
    apply_similarity_transform_to_points,
    compute_similarity_transform,
)

from tqdm import tqdm


@dataclass(frozen=True)
class VGGTSceneReconstructionRuntimeConfig:
    target_image_size: int = 518
    use_background_image: bool = False

class VGGTSceneReconstructionStage(PipelineStage[VGGTSceneReconstructionRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        model_name_or_path: str,
        runtime_cfg: VGGTSceneReconstructionRuntimeConfig,
        dynamic_runtime_cfg: Optional[
            dict[str, VGGTSceneReconstructionRuntimeConfig]
        ] = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name_or_path.endswith(".pt"):
            self.vggt_model = VGGT()
            self.vggt_model.load_state_dict(
                torch.load(model_name_or_path, weights_only=True, map_location="cpu")
            )
        else:
            self.vggt_model = VGGT.from_pretrained(model_name_or_path)

        self.vggt_model = self.vggt_model.cpu().eval()

        # Convert to mixed precision to save VRAM
        for module in self.vggt_model.modules():
            if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                module.float()  # keep these in FP32
            else:
                for param in module.parameters():
                    param.data = param.data.half()

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: VGGTSceneReconstructionRuntimeConfig,
    ):
        device = pipeline.device

        self.vggt_model = self.vggt_model.to(device)

        camera_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "camera_intrinsics"
        ]
        camera_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "camera_extrinsics"
        ]

        views_ids = [v["view_id"] for v in views]

        all_background_images_rgb = []
        cam_poses = torch.empty((len(views_ids), 3), device=device, dtype=torch.float32)

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

            if intrinsics_annotation is None or extrinsics_annotation is None:
                raise ValueError(
                    f"View {view_id} is missing intrinsics or extrinsics"
                )
            
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


            resolution_hw = intrinsics_annotation.resolution_hw

            background_image_rgb = background_image_annotation.image.to(device)
            background_image_rgb = background_image_rgb / 255.0
            background_mask = background_image_annotation.mask.to(device)

            # Undistort both background image and calibration frame
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

            background_image_rgb = background_image_rgb[
                ..., roi_xyxy[1] : roi_xyxy[3], roi_xyxy[0] : roi_xyxy[2]
            ]
            background_mask = background_mask[
                ..., roi_xyxy[1] : roi_xyxy[3], roi_xyxy[0] : roi_xyxy[2]
            ]
            background_sky_mask = background_sky_mask[
                ..., roi_xyxy[1] : roi_xyxy[3], roi_xyxy[0] : roi_xyxy[2]
            ]

            all_background_images_rgb.append(background_image_rgb)

            cam_poses[view_idx] = inverse_Rt(extrinsics_annotation.Rt.to(device))[
                ..., :3, 3
            ]

        all_background_images_rgb = preprocess_images(
            all_background_images_rgb,
            mode="crop",
            target_size=runtime_cfg.target_image_size,
        )

        with (
            torch.no_grad(),
            torch.amp.autocast(device_type="cuda", dtype=torch.float16),
            torch.amp.autocast(device_type="cpu", dtype=torch.float16),
        ):
            inputs = all_background_images_rgb.unsqueeze(0)
            aggregated_tokens_list, ps_idx = self.vggt_model.aggregator(inputs)
            pose_enc = self.vggt_model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                pose_enc, inputs.shape[-2:]
            )
            extrinsic = extrinsic.reshape(-1, 3, 4).float()
            intrinsic = intrinsic.reshape(-1, 3, 3).float()
            depth_map, depth_conf = self.vggt_model.depth_head(
                aggregated_tokens_list, inputs, ps_idx, frames_chunk_size=4
            )
            point_map = torch.from_numpy(
                unproject_depth_map_to_point_map(
                    depth_map.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0)
                )
            ).float()

            colors = all_background_images_rgb.permute(0, 2, 3, 1).reshape(-1, 3)
            points = point_map.reshape(-1, 3).float().to(device)
            points_conf = depth_conf.reshape(-1).float()

        vggt_cam_poses = inverse_Rt(extrinsic)[..., :3, 3]

        # Align using the previously estimated camera extrinsics
        R, t, s = compute_similarity_transform(
            vggt_cam_poses, cam_poses, estimate_scale=True
        )
        points = apply_similarity_transform_to_points(points, R, t, s)

        world_reconstructed_scene_annotations = WorldReconstructedSceneAnnotations(
            annotations=[
                WorldReconstructedSceneAnnotation(
                    points_xyz=points.cpu(),
                    points_colors=colors.cpu(),
                    points_confidences=points_conf.cpu(),
                )
            ],
        ).cpu()

        annotations["world_reconstructed_scene"] = world_reconstructed_scene_annotations
        self.vggt_model = self.vggt_model.cpu()


def paste(base: torch.Tensor, patch: torch.Tensor, left: int, top: int) -> torch.Tensor:
    """
    Paste `patch` into `base` at (x, y) safely.
    Both tensors are [C, H, W] with values in [0,1].
    """
    Cb, Hb, Wb = base.shape
    Cp, Hp, Wp = patch.shape
    assert Cb == Cp, "Channel mismatch"

    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(Wb, left + Wp), min(Hb, top + Hp)

    if x1 >= x2 or y1 >= y2:
        return base

    patch_x1 = x1 - left
    patch_y1 = y1 - top
    patch_x2 = patch_x1 + (x2 - x1)
    patch_y2 = patch_y1 + (y2 - y1)

    base[:, y1:y2, x1:x2] = patch[:, patch_y1:patch_y2, patch_x1:patch_x2]
    return base


def preprocess_images(
    src_images: list[torch.Tensor], mode: str = "crop", target_size: int = 518
) -> torch.Tensor:
    """
    A quick start function to preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(src_images) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()

    # First process all images and collect their shapes
    for img in src_images:
        width, height = img.shape[-2], img.shape[-1]

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = (
                    round(height * (new_width / width) / 14) * 14
                )  # Make divisible by 14
            else:
                new_height = target_size
                new_width = (
                    round(width * (new_height / height) / 14) * 14
                )  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = torch.nn.functional.interpolate(
            img.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(src_images) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images