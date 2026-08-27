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
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.global_scale import (
    GlobalScaleAnnotations,
    GlobalScaleAnnotationsMetadata,
    GlobalScaleAnnotation,
)
from kineo.geometry.camera import (
    transform_points_from_world_to_camera,
    project_points_from_camera_to_image,
)
from kineo.geometry.metrics import compute_reprojection_residuals
import torch
from math import atan2, degrees
from moge.model.v2 import MoGeModel
from tqdm import tqdm
import cv2
import numpy as np
from kineo.maths import weighted_mean
from dataclasses import dataclass


@dataclass(frozen=True)
class MoGeGlobalScaleEstimationRuntimeConfig:
    show: bool = False
    min_keypoint_score: float = 0.3
    # Used when selecting the best frame for each view
    keypoints_indices: list[int] | None = None
    use_half_precision: bool = False
    exponential_decay_factor: float = 1.0


class MoGeGlobalScaleEstimationStage(
    PipelineStage[MoGeGlobalScaleEstimationRuntimeConfig]
):
    def __init__(
        self,
        name: str,
        order: int,
        model_name_or_path: str,
        runtime_cfg: MoGeGlobalScaleEstimationRuntimeConfig,
        dynamic_runtime_cfg: (
            dict[str, MoGeGlobalScaleEstimationRuntimeConfig] | None
        ) = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MoGeModel.from_pretrained(model_name_or_path).cpu()
        self.model.eval()

    def _sample_best_frame_for_each_view(
        self,
        views_ids: list[str],
        keypoints_2d: Keypoints2DAnnotations,
        keypoints_3d: Keypoints3DAnnotations,
        camera_intrinsics: CameraIntrinsicsAnnotations,
        camera_extrinsics: CameraExtrinsicsAnnotations,
        device: torch.device,
        exponential_decay_factor: float = 1.0,
        keypoints_indices: list[int] | None = None,
    ) -> dict[str, tuple[int, torch.Tensor, torch.Tensor]]:
        result: dict[str, dict[str, torch.Tensor]] = {}

        # Assume all keypoints have the same format
        kps_format_name = keypoints_3d.first_or_default().format
        kps_format = next(
            format
            for format in keypoints_3d.metadata.formats
            if format.name == kps_format_name
        )
        subjects_ids = keypoints_3d.subjects_ids

        n_frames = keypoints_3d.n_frames
        n_keypoints = kps_format.n_keypoints
        n_subjects = len(subjects_ids)

        if keypoints_indices is None:
            keypoints_indices = torch.arange(n_keypoints, dtype=torch.long)
        else:
            keypoints_indices = torch.tensor(keypoints_indices, dtype=torch.long)

        n_selected_keypoints = len(keypoints_indices)

        subject_id_to_idx = {subject_id: i for i, subject_id in enumerate(subjects_ids)}

        kps_3d_xyz = torch.zeros(
            (n_frames, n_subjects, n_selected_keypoints, 3),
            dtype=torch.float32,
            device=device,
        )

        for ann in tqdm(
            keypoints_3d.annotations, desc="Collecting 3D keypoints", leave=False
        ):
            kps_3d_xyz[ann.frame_idx, subject_id_to_idx[ann.subject_id], :] = ann.xyz[
                keypoints_indices
            ]

        for view_idx, view_id in enumerate(views_ids):
            view_kps_annotations = keypoints_2d.filter_by_view_id(view_id)
            view_camera_intrinsics = camera_intrinsics.filter_by_view_id(
                view_id
            ).first_or_default()
            view_camera_extrinsics = camera_extrinsics.filter_by_view_id(
                view_id
            ).first_or_default()

            K = view_camera_intrinsics.K.to(device)
            D = view_camera_intrinsics.distortion_coefficients.to(device)
            Rt = view_camera_extrinsics.Rt.to(device)
            distortion_model = view_camera_intrinsics.distortion_model.value

            view_kps_xy = torch.zeros(
                (n_frames, n_subjects, n_selected_keypoints, 2), device=device
            )
            view_kps_scores = torch.zeros(
                (n_frames, n_subjects, n_selected_keypoints), device=device
            )

            for ann in tqdm(
                view_kps_annotations.annotations,
                desc="Collecting 2D keypoints",
                leave=False,
            ):
                view_kps_xy[ann.frame_idx, subject_id_to_idx[ann.subject_id], :] = (
                    ann.xy[keypoints_indices]
                )
                view_kps_scores[ann.frame_idx, subject_id_to_idx[ann.subject_id], :] = (
                    ann.scores[keypoints_indices]
                )

            view_kps_xy = view_kps_xy.reshape(
                -1, 1, n_subjects * n_selected_keypoints, 2
            )
            view_kps_scores = view_kps_scores.reshape(
                -1, 1, n_subjects * n_selected_keypoints
            )

            e, _ = compute_reprojection_residuals(
                kps_3d=kps_3d_xyz.view(-1, n_subjects * n_selected_keypoints, 3),
                kps_2d=view_kps_xy,
                Ks=K.reshape(1, 3, 3),
                Rts=Rt.reshape(1, 3, 4),
                Ds=D.reshape(1, -1),
                distortion_model=distortion_model,
            )

            h = torch.exp(-exponential_decay_factor * e).sum(dim=-1) / (
                n_subjects * n_selected_keypoints
            )
            h = h.reshape(n_frames)

            best_frame_idx = h.argmax().item()

            kps_cam = transform_points_from_world_to_camera(
                kps_3d_xyz[best_frame_idx], Rt
            )
            _, kps_proj_depth = project_points_from_camera_to_image(
                kps_cam, K, D, distortion_model
            )

            result[view_id] = {
                "frame_idx": best_frame_idx,
                "kps_scores": view_kps_scores[best_frame_idx].reshape(
                    n_subjects * n_selected_keypoints
                ),
                "kps_xy": view_kps_xy[best_frame_idx].reshape(
                    n_subjects * n_selected_keypoints, 2
                ),
                "kps_depth": kps_proj_depth.reshape(n_subjects * n_selected_keypoints),
            }

        return result

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: MoGeGlobalScaleEstimationRuntimeConfig,
    ):
        device = pipeline.device
        n_views = len(views)

        self.model = self.model.to(device)

        camera_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "cameras_intrinsics"
        ]
        camera_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "cameras_extrinsics"
        ]

        keypoints_3d: Keypoints3DAnnotations = annotations["keypoints_3d"]
        keypoints_2d: Keypoints2DAnnotations = annotations["keypoints_2d"]

        global_time_reference: GlobalTimeReferenceAnnotation = annotations[
            "global_time_reference"
        ].first_or_default()

        if global_time_reference is None:
            raise ValueError("Global time reference is not available")

        views_ids = [view["view_id"] for view in views]

        result = self._sample_best_frame_for_each_view(
            views_ids=views_ids,
            keypoints_2d=keypoints_2d,
            keypoints_3d=keypoints_3d,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            device=device,
            exponential_decay_factor=runtime_cfg.exponential_decay_factor,
            keypoints_indices=runtime_cfg.keypoints_indices,
        )

        best_frame_closest_view_frame_idx = {
            view_id: global_time_reference.closest_local_frame_idx[view_id][
                result[view_id]["frame_idx"]
            ]
            for view_id in views_ids
        }

        scale_factors = torch.empty((0,), device=device)
        scale_factors_weights = torch.empty((0,), device=device)
        for view_idx in tqdm(
            range(n_views), desc="Estimating global scale", leave=False
        ):
            view_id = views[view_idx]["view_id"]

            view_camera_intrinsics = camera_intrinsics.filter_by_view_id(
                view_id
            ).first_or_default()

            closest_frame_idx = best_frame_closest_view_frame_idx[view_id]

            img_h, img_w = views[view_idx]["frame_loader"].resolution_hw

            K = view_camera_intrinsics.K
            fx = K[0, 0].item()
            fov_x = degrees(2 * atan2(img_w, 2 * fx))

            frame_rgb = views[view_idx]["frame_loader"].load_frame_at(closest_frame_idx)
            frame_rgb = frame_rgb.float() / 255.0

            moge_result = self.model.infer(
                frame_rgb, fov_x=fov_x, use_fp16=runtime_cfg.use_half_precision
            )
            frame_depth = moge_result["depth"]

            kps_img_xy = result[view_id]["kps_xy"]
            kps_arbitrary_scale_depth = result[view_id]["kps_depth"]
            kps_2d_scores = result[view_id]["kps_scores"]

            frame_depth_x = kps_img_xy[..., 0].round().long()
            frame_depth_y = kps_img_xy[..., 1].round().long()

            kps_img_inside_frame = torch.where(
                (frame_depth_x >= 0)
                & (frame_depth_x < img_w)
                & (frame_depth_y >= 0)
                & (frame_depth_y < img_h),
                torch.ones_like(frame_depth_x, dtype=torch.bool, device=device),
                torch.zeros_like(frame_depth_x, dtype=torch.bool, device=device),
            )

            # Some keypoints might be outside the image, so we need to mask them
            # to prevent out of bounds errors.
            valid_kps_mask = (kps_img_inside_frame) & (
                kps_2d_scores > runtime_cfg.min_keypoint_score
            )

            frame_depth_x = frame_depth_x[valid_kps_mask]
            frame_depth_y = frame_depth_y[valid_kps_mask]
            kps_arbitrary_scale_depth = kps_arbitrary_scale_depth[valid_kps_mask]
            kps_2d_scores = kps_2d_scores[valid_kps_mask]

            if len(frame_depth_x) == 0 or len(frame_depth_y) == 0:
                continue

            # Sample the value of the depth in the depth map
            kps_metric_scale_depth = frame_depth[frame_depth_y, frame_depth_x]

            # Remove invalid depths
            valid_metric_scale_depth_mask = kps_metric_scale_depth.isfinite()
            kps_metric_scale_depth = kps_metric_scale_depth[valid_metric_scale_depth_mask]
            kps_arbitrary_scale_depth = kps_arbitrary_scale_depth[valid_metric_scale_depth_mask]
            kps_2d_scores = kps_2d_scores[valid_metric_scale_depth_mask]

            if kps_metric_scale_depth.numel() == 0:
                continue

            # NB: there might be a slight difference between the depth of the keypoints
            # and the depth of the image because depth is sampled on the surface, and keypoints in often located inside the object (joints, etc.)
            scale_factors = torch.cat(
                [
                    scale_factors,
                    (kps_metric_scale_depth / kps_arbitrary_scale_depth.squeeze()),
                ],
                dim=0,
            )
            scale_factors_weights = torch.cat(
                [
                    scale_factors_weights,
                    kps_2d_scores,
                ],
                dim=0,
            )

            if runtime_cfg.show:

                depth_min = frame_depth[frame_depth.isfinite()].min()
                depth_max = frame_depth[frame_depth.isfinite()].max()
                frame_depth = frame_depth.clamp(depth_min, depth_max)
                normalized_depth = (frame_depth - depth_min) / (depth_max - depth_min)
                depth_np = normalized_depth.cpu().numpy()
                depth_np = (depth_np * 255.0).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_np, cv2.COLORMAP_INFERNO)

                frame_bgr_np = (
                    frame_rgb.permute(1, 2, 0).flip(-1).cpu().contiguous().numpy()
                )
                frame_bgr_np = frame_bgr_np * 255.0
                frame_bgr_np = frame_bgr_np.astype(np.uint8)

                valid_kps_img_xy = kps_img_xy[valid_kps_mask].cpu().numpy()

                for kp, kp_arbitrary_scale_depth, kp_metric_scale_depth, score in zip(
                    valid_kps_img_xy,
                    kps_arbitrary_scale_depth,
                    kps_metric_scale_depth,
                    kps_2d_scores,
                ):
                    cv2.circle(
                        depth_vis, (int(kp[0]), int(kp[1])), 3, (255, 255, 255), -1
                    )
                    cv2.circle(
                        frame_bgr_np, (int(kp[0]), int(kp[1])), 3, (255, 255, 255), -1
                    )
                    cv2.putText(
                        depth_vis,
                        f"s={score:.2f}, d_m={kp_metric_scale_depth.item():.2f}m, d_a={kp_arbitrary_scale_depth.item():.2f}u",
                        (int(kp[0]), int(kp[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame_bgr_np,
                        f"s={score:.2f}, d_m={kp_metric_scale_depth.item():.2f}m, d_a={kp_arbitrary_scale_depth.item():.2f}u",
                        (int(kp[0]), int(kp[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                vis = np.hstack((frame_bgr_np, depth_vis))

                window_name = f"RGB and Depth - View {view_idx}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1000, 500)
                cv2.imshow(window_name, vis)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if scale_factors.numel() == 0:
            raise ValueError("No valid scale factors found")

        global_scale_factor = weighted_mean(
            values=scale_factors, weights=scale_factors_weights
        ).item()

        global_scale_annotations = GlobalScaleAnnotations(
            metadata=GlobalScaleAnnotationsMetadata(),
            annotations=[GlobalScaleAnnotation(frame_idx=0, scale=global_scale_factor)],
        )

        annotations["global_scale"] = global_scale_annotations

        self.model = self.model.cpu()


def create_coordinate_grid(H, W, device="cpu", dtype=torch.float32):
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # Shape: (H, W)
    grid = torch.stack([xx, yy], dim=-1)  # Shape: (H, W, 2)
    return grid
