import os
import torch
import rerun as rr
from uuid import uuid4
from dataclasses import dataclass
from typing import Dict, List, Tuple

from kineo.pipeline.pipeline import PipelineStage, Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import Annotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations, CameraDistortionModel
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.annotations.bundle_adjustment_history import (
    BundleAdjustmentHistoryAnnotations,
    BundleAdjustmentHistoryAnnotation,
)
from kineo.geometry.transformations import inverse_Rt, undistort_points
from kineo.geometry.triangulation import triangulate_points


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
        kps_2d: Keypoints2DAnnotations | None = annotations.get("keypoints_2d")

        if history is None or len(history) == 0:
            print("No bundle adjustment history to export.")
            return

        if kps_2d is None:
            print("No 2D keypoints found. Skipping triangulation export.")
            return

        device = pipeline.device
        cameras_intrinsics: CameraIntrinsicsAnnotations = annotations["camera_intrinsics"]
        global_time_ref: GlobalTimeReferenceAnnotation = annotations["global_time_reference"].first_or_default()

        n_frames = global_time_ref.timestamps.numel()
        view_ids = cameras_intrinsics.views_ids
        subject_ids = kps_2d.subjects_ids
        n_kpts = kps_2d.metadata.formats[0].n_keypoints

        # 1. Identify the Best Frame based on average 2D score
        # Shape: (frames, views, subjects, kpts)
        all_scores = torch.zeros((n_frames, len(view_ids), len(subject_ids), n_kpts), device=device)
        all_coords = torch.zeros((n_frames, len(view_ids), len(subject_ids), n_kpts, 2), device=device)

        view_to_idx = {v_id: i for i, v_id in enumerate(view_ids)}
        subj_to_idx = {s_id: i for i, s_id in enumerate(subject_ids)}

        for annot in kps_2d.annotations:
            v_idx = view_to_idx[annot.view_id]
            s_idx = subj_to_idx[annot.subject_id]
            all_scores[annot.frame_idx, v_idx, s_idx] = annot.scores
            all_coords[annot.frame_idx, v_idx, s_idx] = annot.xy

        # Avg score per frame across all views, subjects, and keypoints
        frame_avg_scores = all_scores.mean(dim=(1, 2, 3))
        best_frame_idx = torch.argmax(frame_avg_scores).item()
        print(
            f"Selected best frame for triangulation visualization: {best_frame_idx} (score: {frame_avg_scores[best_frame_idx]:.4f})")

        # 2. Prepare undistorted points for the best frame
        best_kps_2d = all_coords[best_frame_idx]  # (views, subjects, kpts, 2)
        best_kps_weights = all_scores[best_frame_idx]  # (views, subjects, kpts)

        undistorted_best_kps = best_kps_2d.clone()
        for v_idx, v_id in enumerate(view_ids):
            cam = cameras_intrinsics.filter_by_view_id(v_id).first_or_default()
            undistorted_best_kps[v_idx] = undistort_points(
                points=best_kps_2d[v_idx].reshape(-1, 2),
                K=cam.K.to(device),
                D=cam.distortion_coefficients.to(device),
                distortion_model=cam.distortion_model.value
            ).reshape(len(subject_ids), n_kpts, 2)

        # Build resolution lookup
        resolution_by_view_id = {
            v_id: cameras_intrinsics.filter_by_view_id(v_id).first_or_default().resolution_hw
            for v_id in view_ids
        }

        # 3. Export Loop
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
                best_kps_undistorted=undistorted_best_kps,
                best_kps_weights=best_kps_weights,
                image_plane_distance=runtime_cfg.image_plane_distance,
            )

        print(f"Exported bundle adjustment history to {formatted_output_path}")


def _log_history_entry(
        entry: BundleAdjustmentHistoryAnnotation,
        normalized_iteration: int,
        resolution_by_view_id: Dict[str, Tuple[int, int]],
        best_kps_undistorted: torch.Tensor,
        best_kps_weights: torch.Tensor,
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

    # Triangulate Best Frame for this iteration
    Ps_tensor = torch.stack(Ps)
    # Flatten subjects/kpts for triangulation: (views, N_points, 2)
    pts_2d = best_kps_undistorted.transpose(0, 1).reshape(n_views, -1, 2)
    weights = best_kps_weights.transpose(0, 1).reshape(n_views, -1)

    points_3d = triangulate_points(
        Ps=Ps_tensor.to(pts_2d.device),
        points=pts_2d,  # triangulate_points expects (views, ..., 2)
        points_weights=weights
    )

    # Log Triangulation Result
    rr.log("ba/triangulation", rr.Points3D(
        positions=points_3d.reshape(-1, 3).cpu().numpy(),
        radii=0.02,
        colors=[255, 255, 255]
    ))

    if entry.loss is not None:
        rr.log(f"ba/loss", rr.Scalars(entry.loss))