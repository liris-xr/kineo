# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from __future__ import annotations

import cv2
import numpy as np
import os
from dataclasses import dataclass
from tqdm import tqdm

from kineo.pipeline.pipeline import PipelineStage, Pipeline
from kineo.pipeline.pipeline import ViewInput
from kineo.pipeline.pipeline import Annotations
from kineo.annotations.calibration_points_pairs import CalibrationPointsAnnotations
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.visualization.viz_2d import draw_keypoints


@dataclass(frozen=True)
class CalibrationPointsVideoExportRuntimeConfig:
    view1_id: str = "view_0"
    view2_id: str = "view_1"
    output_path_template: str = "./outputs/{sequence_name}_calib_points_{view1_id}_{view2_id}.mp4"
    point_radius: int = 5
    line_thickness: int = 1
    fps: float | None = None
    skeleton_bone_width: int = 2
    skeleton_keypoint_radius: int = 2
    skeleton_color_bgr: tuple[int, int, int] = (200, 200, 200)
    keypoints_indices: tuple[int, ...] | None = None
    show: bool = False


class CalibrationPointsVideoExportStage(
    PipelineStage[CalibrationPointsVideoExportRuntimeConfig]
):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: CalibrationPointsVideoExportRuntimeConfig,
        dynamic_runtime_cfg: dict[str, CalibrationPointsVideoExportRuntimeConfig] | None = None,
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
        runtime_cfg: CalibrationPointsVideoExportRuntimeConfig,
    ):
        calib_points_annots: CalibrationPointsAnnotations = annotations["calibration_points"]
        global_time_ref_annots: GlobalTimeReferenceAnnotations = annotations["global_time_reference"]
        global_time_ref = global_time_ref_annots.first_or_default()

        view1_id = runtime_cfg.view1_id
        view2_id = runtime_cfg.view2_id

        calib_ann = calib_points_annots.get_annotation((view1_id, view2_id))
        if calib_ann is None:
            raise ValueError(
                f"No calibration points annotation found for view pair ({view1_id}, {view2_id})"
            )

        # Find view objects
        view1 = next((v for v in views if v["view_id"] == view1_id), None)
        view2 = next((v for v in views if v["view_id"] == view2_id), None)
        if view1 is None or view2 is None:
            raise ValueError(
                f"Could not find views for IDs: {view1_id}, {view2_id}"
            )

        # Get point data on CPU
        points1 = calib_ann.points1.cpu().numpy()  # (n_points, 2)
        points2 = calib_ann.points2.cpu().numpy()  # (n_points, 2)
        confidence_scores = calib_ann.confidence_scores.cpu().numpy()  # (n_points,)
        frame_indices = calib_ann.frame_indices.cpu().numpy()  # (n_points,)

        # Sort by frame index for efficient accumulation
        sort_order = np.argsort(frame_indices)
        points1 = points1[sort_order]
        points2 = points2[sort_order]
        confidence_scores = confidence_scores[sort_order]
        frame_indices = frame_indices[sort_order]

        n_global_frames = global_time_ref.timestamps.shape[0]
        fps = runtime_cfg.fps if runtime_cfg.fps is not None else global_time_ref.fps

        # Format output path
        output_path = runtime_cfg.output_path_template.format(
            sequence_name=sequence_name,
            view1_id=view1_id,
            view2_id=view2_id,
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load first frames to determine output resolution
        local_idx1 = int(global_time_ref.closest_local_frame_idx[view1_id][0])
        local_idx2 = int(global_time_ref.closest_local_frame_idx[view2_id][0])
        frame1 = view1["frame_loader"].load_frame_at(local_idx1)
        frame2 = view2["frame_loader"].load_frame_at(local_idx2)

        h1, w1 = frame1.shape[1], frame1.shape[2]
        h2, w2 = frame2.shape[1], frame2.shape[2]
        min_height = min(h1, h2)
        out_width = w1 + w2
        out_height = min_height

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (out_width, out_height),
        )

        # Track how many points have been accumulated
        acc_idx = 0

        # Build per-(view_id, local_frame_idx) lookup for skeleton overlay
        kps_2d: Keypoints2DAnnotations | None = annotations.get("keypoints_2d")
        kps_2d_lookup: dict[tuple[str, int], list] = {}
        skeleton_connectivity = None
        skeleton_kp_indices = None
        if kps_2d is not None:
            kps_format = kps_2d.metadata.formats[0]
            for annot in kps_2d.annotations:
                kps_2d_lookup.setdefault((annot.view_id, annot.frame_idx), []).append(annot)

            if runtime_cfg.keypoints_indices is not None:
                skeleton_kp_indices = list(runtime_cfg.keypoints_indices)
                old_to_new = {old: new for new, old in enumerate(skeleton_kp_indices)}
                skeleton_connectivity = [
                    (old_to_new[a], old_to_new[b])
                    for a, b in kps_format.keypoints_connectivity
                    if a in old_to_new and b in old_to_new
                ]
            else:
                skeleton_connectivity = kps_format.keypoints_connectivity

        for global_frame_idx in tqdm(
            range(n_global_frames),
            desc=f"Exporting calibration points video ({view1_id}, {view2_id})",
            leave=False,
        ):
            # Load frames for this global frame
            local_idx1 = int(global_time_ref.closest_local_frame_idx[view1_id][global_frame_idx])
            local_idx2 = int(global_time_ref.closest_local_frame_idx[view2_id][global_frame_idx])

            frame1 = view1["frame_loader"].load_frame_at(local_idx1)
            frame2 = view2["frame_loader"].load_frame_at(local_idx2)

            frame1_rgb = frame1.permute(1, 2, 0).cpu().numpy()
            frame1_bgr = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2BGR)
            frame2_rgb = frame2.permute(1, 2, 0).cpu().numpy()
            frame2_bgr = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2BGR)

            # Equalize heights by cropping the taller frame from the bottom
            dy1 = frame1_bgr.shape[0] - min_height
            dy2 = frame2_bgr.shape[0] - min_height
            if dy1 > 0:
                frame1_bgr = frame1_bgr[:-dy1, :, :]
            if dy2 > 0:
                frame2_bgr = frame2_bgr[:-dy2, :, :]

            combined = np.concatenate([frame1_bgr, frame2_bgr], axis=1)

            # Draw skeleton overlays (on top of video, below matching points)
            if kps_2d is not None:
                for annot in kps_2d_lookup.get((view1_id, local_idx1), []):
                    xy = annot.xy.cpu().numpy()
                    visibility = annot.annotated.cpu().numpy()
                    if skeleton_kp_indices is not None:
                        xy = xy[skeleton_kp_indices]
                        visibility = visibility[skeleton_kp_indices]
                    draw_keypoints(
                        combined, xy,
                        colors_bgr=runtime_cfg.skeleton_color_bgr,
                        connectivity=skeleton_connectivity,
                        connectivity_colors_bgr=runtime_cfg.skeleton_color_bgr,
                        keypoints_radius=runtime_cfg.skeleton_keypoint_radius,
                        bone_width=runtime_cfg.skeleton_bone_width,
                        visibility=visibility,
                    )
                for annot in kps_2d_lookup.get((view2_id, local_idx2), []):
                    xy = annot.xy.cpu().numpy()
                    visibility = annot.annotated.cpu().numpy()
                    if skeleton_kp_indices is not None:
                        xy = xy[skeleton_kp_indices]
                        visibility = visibility[skeleton_kp_indices]
                    xy = xy.copy()
                    xy[:, 0] += w1  # offset x for view2
                    draw_keypoints(
                        combined, xy,
                        colors_bgr=runtime_cfg.skeleton_color_bgr,
                        connectivity=skeleton_connectivity,
                        connectivity_colors_bgr=runtime_cfg.skeleton_color_bgr,
                        keypoints_radius=runtime_cfg.skeleton_keypoint_radius,
                        bone_width=runtime_cfg.skeleton_bone_width,
                        visibility=visibility,
                    )

            # Advance accumulation index to include all points up to current frame
            while acc_idx < len(frame_indices) and frame_indices[acc_idx] <= global_frame_idx:
                acc_idx += 1

            # Draw all accumulated points
            for i in range(acc_idx):
                score = confidence_scores[i]
                color = tuple(
                    int(c)
                    for c in cv2.applyColorMap(
                        np.array([int(score * 255)], dtype=np.uint8),
                        cv2.COLORMAP_VIRIDIS,
                    ).reshape(3)
                )

                pt1 = (int(points1[i, 0]), int(points1[i, 1]))
                pt2 = (int(points2[i, 0]) + w1, int(points2[i, 1]))

                cv2.circle(combined, pt1, runtime_cfg.point_radius, color, -1, cv2.LINE_AA)
                cv2.circle(combined, pt2, runtime_cfg.point_radius, color, -1, cv2.LINE_AA)
                cv2.line(combined, pt1, pt2, color, runtime_cfg.line_thickness, cv2.LINE_AA)

            if runtime_cfg.show:
                cv2.imshow("combined", combined)
                cv2.waitKey(0)

            writer.write(combined)

        if runtime_cfg.show:
            cv2.destroyAllWindows()

        writer.release()
        print(f"Calibration points video saved to {output_path}")
