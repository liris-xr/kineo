import os
import torch
import rerun as rr
from uuid import uuid4
from dataclasses import dataclass
from typing import Dict, List, Tuple

from kineo.pipeline.pipeline import PipelineStage, Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import Annotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.bundle_adjustment_history import (
    BundleAdjustmentHistoryAnnotations,
    BundleAdjustmentHistoryAnnotation,
)
from kineo.annotations.bundle_adjustment_keypoints import BundleAdjustmentKeypointsAnnotations
from kineo.geometry.transformations import inverse_Rt, undistort_points
from kineo.geometry.triangulation import triangulate_points
from kineo.geometry.camera import transform_points_from_world_to_camera, project_points_from_camera_to_image


@dataclass(frozen=True)
class BundleAdjustmentHistoryRerunExportRuntimeConfig:
    output_path_template: str = "./outputs/rerun/{sequence_name}_ba_history.rrd"
    image_plane_distance: float = 0.2


class BundleAdjustmentHistoryRerunExportStage(
    PipelineStage[BundleAdjustmentHistoryRerunExportRuntimeConfig]
):
    def __init__(
            self,
            name: str,
            order: int,
            runtime_cfg: BundleAdjustmentHistoryRerunExportRuntimeConfig,
            dynamic_runtime_cfg: Dict[str, BundleAdjustmentHistoryRerunExportRuntimeConfig] | None = None,
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
            views: List[ViewInput],
            annotations: Dict[str, Annotations],
            gt_annotations: Dict[str, Annotations],
            runtime_cfg: BundleAdjustmentHistoryRerunExportRuntimeConfig,
    ):
        history: BundleAdjustmentHistoryAnnotations | None = annotations.get(
            "bundle_adjustment_history"
        )
        ba_kps_annots: BundleAdjustmentKeypointsAnnotations | None = annotations.get(
            "bundle_adjustment_keypoints"
        )

        if history is None or len(history) == 0:
            print("No bundle adjustment history to export.")
            return

        if ba_kps_annots is None:
            print("No bundle adjustment keypoints found. Skipping.")
            return

        ba_kps = ba_kps_annots.first_or_default()
        device = pipeline.device
        cameras_intrinsics: CameraIntrinsicsAnnotations = annotations["cameras_intrinsics"]
        distortion_model = cameras_intrinsics.first_or_default().distortion_model.value

        # Build resolution lookup
        resolution_by_view_id = {
            v_id: cameras_intrinsics.filter_by_view_id(v_id).first_or_default().resolution_hw
            for v_id in ba_kps.view_ids
        }

        # Undistort BA sample 2D points for triangulation
        ba_kps_2d = ba_kps.kps_2d_xy.to(device)  # (n_views, n_kps, 2)
        ba_kps_scores = ba_kps.kps_2d_scores.to(device)  # (n_views, n_kps)

        ba_kps_2d_undistorted = ba_kps_2d.clone()
        for v_idx, v_id in enumerate(ba_kps.view_ids):
            cam = cameras_intrinsics.filter_by_view_id(v_id).first_or_default()
            ba_kps_2d_undistorted[v_idx] = undistort_points(
                points=ba_kps_2d[v_idx],
                K=cam.K.to(device),
                D=cam.distortion_coefficients.to(device),
                distortion_model=cam.distortion_model.value,
            )

        # Export Loop
        formatted_output_path = runtime_cfg.output_path_template.format(sequence_name=sequence_name)
        os.makedirs(os.path.dirname(formatted_output_path), exist_ok=True)
        rr.init(sequence_name, recording_id=uuid4())
        rr.save(formatted_output_path)

        sorted_entries = sorted(history, key=lambda e: (e.stage_order, e.iteration))
        iteration_offset = 0
        prev_stage_order = None
        prev_iter = 0

        for entry in sorted_entries:
            if prev_stage_order is not None and entry.stage_order != prev_stage_order:
                iteration_offset = prev_iter

            normalized_iteration = iteration_offset + entry.iteration
            prev_iter = normalized_iteration
            prev_stage_order = entry.stage_order

            _log_history_entry(
                entry=entry,
                normalized_iteration=normalized_iteration,
                resolution_by_view_id=resolution_by_view_id,
                ba_kps_2d_undistorted=ba_kps_2d_undistorted,
                ba_kps_scores=ba_kps_scores,
                distortion_model=distortion_model,
                image_plane_distance=runtime_cfg.image_plane_distance,
            )

        print(f"Exported bundle adjustment history to {formatted_output_path}")


def _log_history_entry(
        entry: BundleAdjustmentHistoryAnnotation,
        normalized_iteration: int,
        resolution_by_view_id: Dict[str, Tuple[int, int]],
        ba_kps_2d_undistorted: torch.Tensor,
        ba_kps_scores: torch.Tensor,
        distortion_model: str,
        image_plane_distance: float = 0.2,
):
    rr.set_time("iteration", sequence=normalized_iteration)
    n_views = entry.Ks.shape[0]

    Ps = []
    for view_idx in range(n_views):
        view_id = entry.view_ids[view_idx]
        K_view = entry.Ks[view_idx].numpy()
        Rt_view = entry.Rts[view_idx]

        # Projection matrix for triangulation
        Ps.append(entry.Ks[view_idx] @ Rt_view)

        # Logging Camera
        Rt_inv = inverse_Rt(Rt_view)
        translation = Rt_inv[:3, 3].numpy()
        rotation = Rt_inv[:3, :3].numpy()
        height, width = resolution_by_view_id.get(view_id, (480, 640))

        rr.log(f"ba/cameras/{view_id}", rr.Pinhole(
            image_from_camera=K_view, width=width, height=height,
            image_plane_distance=image_plane_distance
        ))
        rr.log(f"ba/cameras/{view_id}", rr.Transform3D(translation=translation, mat3x3=rotation))

    # Triangulate BA sample points for this iteration
    Ps_tensor = torch.stack(Ps)

    points_3d = triangulate_points(
        Ps=Ps_tensor.to(ba_kps_2d_undistorted.device),
        points=ba_kps_2d_undistorted,
        points_weights=ba_kps_scores,
    )

    # Log 3D Triangulation Result
    rr.log("ba/triangulation", rr.Points3D(
        positions=points_3d.reshape(-1, 3).cpu().numpy(),
        radii=0.02,
        colors=[255, 255, 255]
    ))

    # Log 2D Reprojection in each camera
    device = ba_kps_2d_undistorted.device
    points_3d_cam = transform_points_from_world_to_camera(points_3d, entry.Rts.to(device))
    proj_2d, _ = project_points_from_camera_to_image(
        points_3d_cam, entry.Ks.to(device), entry.dist_coeffs.to(device), distortion_model
    )

    for view_idx in range(n_views):
        view_id = entry.view_ids[view_idx]
        rr.log(f"ba/cameras/{view_id}/reprojection", rr.Points2D(
            positions=proj_2d[view_idx].detach().cpu().numpy(),
            radii=3.0,
            colors=[0, 255, 0]
        ))

    if entry.loss is not None:
        rr.log("ba/loss", rr.Scalars(entry.loss))