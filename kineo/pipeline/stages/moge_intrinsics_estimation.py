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
from kineo.pipeline import per_view_cache


from moge.model.v2 import MoGeModel


@dataclass(frozen=True)
class MoGeIntrinsicsEstimationRuntimeConfig:
    use_cache: bool = True
    cache_output_path_template: str = (
        "cache/{sequence_name}/{annotation_key}/{view_id}.pkl"
    )
    use_half_precision: bool = True


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

        def infer_missing(missing_views: list[ViewInput]) -> dict[str, Annotations]:
            self.moge_model = self.moge_model.to(device)
            try:
                return {
                    "cameras_intrinsics": self._estimate_views_intrinsics(
                        missing_views, device, runtime_cfg.use_half_precision
                    )
                }
            finally:
                self.moge_model = self.moge_model.cpu()

        cached = per_view_cache.load_or_infer_per_view(
            views=views,
            specs={
                "cameras_intrinsics": per_view_cache.PerViewCacheSpec(
                    annotations_cls=CameraIntrinsicsAnnotations,
                    metadata=CameraIntrinsicsAnnotationsMetadata(),
                )
            },
            infer_missing=infer_missing,
            sequence_name=sequence_name,
            cache_output_path_template=runtime_cfg.cache_output_path_template,
            use_cache=runtime_cfg.use_cache,
        )

        annotations["cameras_intrinsics"] = cached["cameras_intrinsics"]

    def _estimate_views_intrinsics(
        self,
        views: list[ViewInput],
        device: torch.device,
        use_half_precision: bool,
    ) -> CameraIntrinsicsAnnotations:
        """Estimates intrinsics from the first frame of each view."""
        intrinsics_annotations: list[CameraIntrinsicsAnnotation] = []

        for view in views:
            frame_loader = view["frame_loader"]
            view_name = view["view_id"]

            frame_rgb = frame_loader.load_frame_at(0)

            if frame_rgb.dtype == torch.uint8:
                frame_rgb = frame_rgb.float() / 255.0

            img_h, img_w = frame_rgb.shape[-2:]

            K = self._estimate_intrinsics(
                frame_rgb, (img_h, img_w), use_half_precision
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

        return CameraIntrinsicsAnnotations(
            metadata=CameraIntrinsicsAnnotationsMetadata(),
            annotations=intrinsics_annotations,
        ).cpu()
