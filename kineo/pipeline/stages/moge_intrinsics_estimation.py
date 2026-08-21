# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import torch
import os
import pickle
from dataclasses import dataclass

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    Annotations,
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotations,
    CameraIntrinsicsAnnotationsMetadata,
    CameraDistortionModel,
)
from kineo.pipeline.pipeline import Pipeline


from moge.model.v2 import MoGeModel


@dataclass(frozen=True)
class MoGeIntrinsicsEstimationRuntimeConfig:
    use_cache: bool = True
    cache_output_path_template: str = "cache/{sequence_name}/{annotation_key}.pkl"
    use_half_precision: bool = True
    shared_focal_init: bool = False


def share_focal_lengths(
    intrinsics: CameraIntrinsicsAnnotations,
) -> CameraIntrinsicsAnnotations:
    """Gives every view the median focal length of the rig.

    Per-view monocular estimates disagree by far more than the cameras really
    differ, and the two-view relative pose is sensitive enough to that spread to
    settle on a mirrored geometry. The median is robust to a single bad view;
    bundle adjustment is free to pull the focals apart again afterwards.

    Args:
        intrinsics: Per-view estimated intrinsics.

    Returns:
        The same intrinsics with fx and fy replaced by their median over views,
        principal points untouched.
    """
    annotations = list(intrinsics.annotations)

    if len(annotations) < 2:
        return intrinsics

    fx = torch.stack([a.K[0, 0] for a in annotations]).median()
    fy = torch.stack([a.K[1, 1] for a in annotations]).median()

    shared_annotations = []
    for annotation in annotations:
        K = annotation.K.clone()
        K[0, 0] = fx
        K[1, 1] = fy
        shared_annotations.append(
            CameraIntrinsicsAnnotation(
                view_id=annotation.view_id,
                frame_idx=annotation.frame_idx,
                K=K,
                distortion_coefficients=annotation.distortion_coefficients,
                distortion_model=annotation.distortion_model,
                resolution_hw=annotation.resolution_hw,
            )
        )

    return CameraIntrinsicsAnnotations(
        metadata=intrinsics.metadata,
        annotations=shared_annotations,
    )


class MoGeIntrinsicsEstimationStage(PipelineStage):
    """
    Stage for estimating the camera intrinsics parameters.

    Produces :class:`CameraIntrinsicsAnnotations` with the estimated camera intrinsics parameters.
    """

    def __init__(
        self,
        name: str,
        order: int,
        model_name_or_path: str,
        runtime_cfg: MoGeIntrinsicsEstimationRuntimeConfig,
        dynamic_runtime_cfg: (
            dict[str, MoGeIntrinsicsEstimationRuntimeConfig] | None
        ) = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )
        self.moge_model = MoGeModel.from_pretrained(model_name_or_path).cpu()
        self.moge_model.eval()

    def _estimate_intrinsics(
        self,
        frame_rgb: torch.Tensor,
        resolution_hw: tuple[int, int],
        use_half_precision: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        img_h, img_w = resolution_hw
        output = self.moge_model.infer(frame_rgb, use_fp16=use_half_precision)
        K_norm = output["intrinsics"]
        fx = K_norm[0, 0] * img_w
        fy = K_norm[1, 1] * img_h
        cx = K_norm[0, 2] * img_w
        cy = K_norm[1, 2] * img_h

        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=torch.float32,
            device=frame_rgb.device,
        )
        return K

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: MoGeIntrinsicsEstimationRuntimeConfig,
    ):
        device = pipeline.device

        device = pipeline.device
        self.moge_model = self.moge_model.to(device)

        if runtime_cfg.use_cache:
            intrinsics_cache_filepath = runtime_cfg.cache_output_path_template.format(
                sequence_name=sequence_name, annotation_key="camera_intrinsics"
            )

            if os.path.exists(intrinsics_cache_filepath):
                with open(intrinsics_cache_filepath, "rb") as f:
                    intrinsics_annotations = CameraIntrinsicsAnnotations.from_dict(
                        pickle.load(f)
                    )
                    print(
                        f"Loaded camera intrinsics annotations from cache: {intrinsics_cache_filepath}"
                    )
                    if runtime_cfg.shared_focal_init:
                        intrinsics_annotations = share_focal_lengths(
                            intrinsics_annotations
                        )
                    annotations["camera_intrinsics"] = intrinsics_annotations
                    return

        intrinsics_annotations: list[CameraIntrinsicsAnnotation] = []

        for view in views:
            frame_loader = view["frame_loader"]
            view_name = view["view_id"]

            frame_rgb = frame_loader.load_frame_at(0)

            if frame_rgb.dtype == torch.uint8:
                frame_rgb = frame_rgb.float() / 255.0

            img_h, img_w = frame_rgb.shape[-2:]

            K = self._estimate_intrinsics(
                frame_rgb, (img_h, img_w), runtime_cfg.use_half_precision
            )
            distortion_coefficients = torch.zeros(5, device=device)
            distortion_model = CameraDistortionModel.BROWN_CONRADY

            view_annotation = CameraIntrinsicsAnnotation(
                view_id=view_name,
                frame_idx=0,
                K=K,
                distortion_coefficients=distortion_coefficients,
                distortion_model=distortion_model,
                resolution_hw=(img_h, img_w),
            )
            intrinsics_annotations.append(view_annotation)

        camera_intrinsics_annotations = CameraIntrinsicsAnnotations(
            metadata=CameraIntrinsicsAnnotationsMetadata(),
            annotations=intrinsics_annotations,
        ).cpu()

        annotations["camera_intrinsics"] = camera_intrinsics_annotations

        self.moge_model = self.moge_model.cpu()

        # Cache what the model estimated, so the cache stays valid whatever the
        # sharing setting; sharing is applied on read as well.
        if runtime_cfg.use_cache:
            os.makedirs(os.path.dirname(intrinsics_cache_filepath), exist_ok=True)

            with open(intrinsics_cache_filepath, "wb") as f:
                pickle.dump(camera_intrinsics_annotations.to_dict(), f)
                print(
                    f"Saved camera intrinsics annotations to cache: {intrinsics_cache_filepath}"
                )

        if runtime_cfg.shared_focal_init:
            annotations["camera_intrinsics"] = share_focal_lengths(
                camera_intrinsics_annotations
            )
