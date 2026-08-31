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

from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import Annotations
from kineo.annotations.camera_temporal import CameraTemporalAnnotations
from kineo.annotations.keypoints_2d import (
    Keypoints2DAnnotations,
    Keypoints2DAnnotation,
)
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.datasets.annotations_io import build_synchronized_camera_temporal
from kineo.pipeline.stages.global_time_reference import (
    build_global_time_reference,
    build_view_timelines,
    is_pass_through,
)
from kineo.annotations.bboxes_2d import BBox2DAnnotations, BBox2DAnnotation

from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class GlobalTimeResamplingRuntimeConfig:
    target_fps: int = 50
    interpolation_dt_thr_s: float = 1 / 50 * 3  # 3 frames at 50 Hz
    show: bool = False

def pad_to_size(img, target_h, target_w):
    """Pad image (H, W, C) with black pixels to target size."""
    h, w = img.shape[:2]
    pad_h = target_h - h
    pad_w = target_w - w
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))

class GlobalTimeResamplingStage(PipelineStage[GlobalTimeResamplingRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: GlobalTimeResamplingRuntimeConfig,
        dynamic_runtime_cfg: dict[str, GlobalTimeResamplingRuntimeConfig] | None = None,
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
        runtime_cfg: GlobalTimeResamplingRuntimeConfig,
    ):
        device = pipeline.device
        n_views = len(views)

        camera_temporal: CameraTemporalAnnotations | None = annotations.get(
            "cameras_temporal", None
        )

        if camera_temporal is None:
            print("No camera temporal annotations provided, assuming synchronized cameras.")
            camera_temporal = build_synchronized_camera_temporal(
                [view["view_id"] for view in views]
            )
            annotations["cameras_temporal"] = camera_temporal

        # A stage may have settled the timeline before the detectors ran.
        if "global_time_reference" not in annotations:
            annotations["global_time_reference"] = build_global_time_reference(
                views=views,
                camera_temporal=camera_temporal,
                target_fps=runtime_cfg.target_fps,
                device=device,
            )

        global_time_reference: GlobalTimeReferenceAnnotation = annotations[
            "global_time_reference"
        ].first_or_default()

        if is_pass_through(global_time_reference):
            return

        kps_2d: Keypoints2DAnnotations = annotations["keypoints_2d"]
        bboxes_2d: BBox2DAnnotations = annotations["bboxes_2d"]

        views_frame_timestamps = build_view_timelines(views, camera_temporal, device)

        resampled_timestamps = global_time_reference.timestamps.to(device)
        n_frames = resampled_timestamps.numel()
        closest_local_frame_idx = global_time_reference.closest_local_frame_idx

        resampled_kps_2d_annotations: list[Keypoints2DAnnotation] = []
        resampled_bboxes_2d_annotations: list[BBox2DAnnotation] = []

        for subject_id in kps_2d.subjects_ids:
            kps_2d_subject = kps_2d.filter_by_subject_id(subject_id)
            bboxes_2d_subject = bboxes_2d.filter_by_subject_id(subject_id)

            if len(bboxes_2d_subject.annotations) == 0:
                raise ValueError("No bboxes for subject " + subject_id)

            # Here we assume that the keypoints format doesn't for a given subject.
            # In reality, we might have different formats (e.g. body, hand, face),
            # but this isn't implemented yet.

            kps_format_name = kps_2d_subject.first_or_default().format
            subject_category_id = bboxes_2d_subject.first_or_default().category_id

            n_kps = next(
                (
                    format.n_keypoints
                    for format in kps_2d_subject.metadata.formats
                    if format.name == kps_format_name
                ),
                None,
            )

            resampled_kps_2d_xy = torch.zeros(
                (n_frames, n_views, n_kps, 2), device=device
            )
            resampled_kps_2d_scores = torch.zeros(
                (n_frames, n_views, n_kps), device=device
            )

            resampled_bboxes_2d_xyxy = torch.zeros(
                (n_frames, n_views, 4), device=device
            )
            resampled_bboxes_2d_scores = torch.zeros((n_frames, n_views), device=device)

            for view_idx in range(n_views):
                view_id = views[view_idx]["view_id"]
                view_frame_timestamps = views_frame_timestamps[view_idx]
                view_kps_2d = kps_2d_subject.filter_by_view_id(view_id)
                view_bboxes_2d = bboxes_2d_subject.filter_by_view_id(view_id)

                if view_kps_2d is None:
                    raise ValueError(
                        f"No keypoints 2d annotations found for subject {subject_id} and view {views[view_idx]['view_id']}"
                    )

                if view_bboxes_2d is None:
                    raise ValueError(
                        f"No bboxes 2d annotations found for subject {subject_id} and view {views[view_idx]['view_id']}"
                    )

                # One view might not have any keypoints for a given subject (not visible at all)
                if len(view_kps_2d) == 0:
                    continue

                if len(view_bboxes_2d) == 0:
                    continue

                view_kps_2d_xy = torch.stack([kps.xy for kps in view_kps_2d], dim=0).to(
                    device
                )
                view_kps_2d_scores = torch.stack(
                    [kps.scores for kps in view_kps_2d], dim=0
                ).to(device)
                view_kps_timestamps = view_frame_timestamps[
                    [kps.frame_idx for kps in view_kps_2d]
                ].to(device)

                view_bboxes_2d_xyxy = torch.stack(
                    [bboxes.xyxy for bboxes in view_bboxes_2d], dim=0
                ).to(device)
                view_bboxes_2d_scores = torch.tensor(
                    [bboxes.score for bboxes in view_bboxes_2d],
                    dtype=torch.float32,
                    device=device,
                )
                view_bboxes_2d_timestamps = view_frame_timestamps[
                    [bboxes.frame_idx for bboxes in view_bboxes_2d]
                ].to(device)

                resampled_view_kps_2d_xy, resampled_view_kps_2d_scores = (
                    _resample_keypoints(
                        src_keypoints=view_kps_2d_xy,
                        src_scores=view_kps_2d_scores,
                        src_timestamps=view_kps_timestamps,
                        dst_timestamps=resampled_timestamps,
                        dt_threshold_s=runtime_cfg.interpolation_dt_thr_s,
                    )
                )

                resampled_view_bboxes_2d_xy, resampled_view_bboxes_2d_scores = (
                    _resample_bboxes(
                        src_bboxes=view_bboxes_2d_xyxy,
                        src_scores=view_bboxes_2d_scores,
                        src_timestamps=view_bboxes_2d_timestamps,
                        dst_timestamps=resampled_timestamps,
                        dt_threshold_s=runtime_cfg.interpolation_dt_thr_s,
                    )
                )

                resampled_kps_2d_xy[:, view_idx] = resampled_view_kps_2d_xy
                resampled_kps_2d_scores[:, view_idx] = resampled_view_kps_2d_scores

                resampled_bboxes_2d_xyxy[:, view_idx] = resampled_view_bboxes_2d_xy
                resampled_bboxes_2d_scores[:, view_idx] = (
                    resampled_view_bboxes_2d_scores
                )

                for f in range(n_frames):
                    # NB: the frame_idx is now expressed in global time
                    resampled_kps_2d_annotations.append(
                        Keypoints2DAnnotation(
                            view_id=view_id,
                            frame_idx=f,
                            subject_id=subject_id,
                            xy=resampled_kps_2d_xy[f, view_idx],
                            scores=resampled_kps_2d_scores[f, view_idx],
                            format=kps_format_name,
                        )
                    )

                    resampled_bboxes_2d_annotations.append(
                        BBox2DAnnotation(
                            view_id=view_id,
                            frame_idx=f,
                            subject_id=subject_id,
                            category_id=subject_category_id,
                            xyxy=resampled_bboxes_2d_xyxy[f, view_idx],
                            score=resampled_bboxes_2d_scores[f, view_idx].item(),
                        )
                    )

        if runtime_cfg.show:

            cv2.namedWindow("Synchronized Views", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Synchronized Views", 1280, 720)

            for frame_idx in range(n_frames):
                frames_bgr = []
        
                # --- Load all views for this frame ---
                max_h, max_w = 0, 0
                for view_idx in range(n_views):
                    view_id = views[view_idx]["view_id"]
                    frame_loader = views[view_idx]["frame_loader"]
                    local_frame_idx = closest_local_frame_idx[view_id][frame_idx]

                    frame = frame_loader.load_frame_at(local_frame_idx)
                    frame_rgb = frame.permute(1, 2, 0).cpu().numpy()

                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    h, w = frame_bgr.shape[:2]
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)

                    frames_bgr.append(frame_bgr)

                padded_frames = [pad_to_size(f, max_h, max_w) for f in frames_bgr]

                concatenated_frame = np.concatenate(padded_frames, axis=1)

                cv2.imshow("Synchronized Views", concatenated_frame)
                cv2.waitKey(1)
        
            cv2.destroyAllWindows()

        annotations["keypoints_2d"] = Keypoints2DAnnotations(
            metadata=kps_2d.metadata, annotations=resampled_kps_2d_annotations
        ).cpu()

        annotations["bboxes_2d"] = BBox2DAnnotations(
            metadata=bboxes_2d.metadata, annotations=resampled_bboxes_2d_annotations
        ).cpu()



def _resample_keypoints(
    src_keypoints: torch.Tensor,
    src_scores: torch.Tensor,
    src_timestamps: torch.Tensor,
    dst_timestamps: torch.Tensor,
    dt_threshold_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resample keypoints to a target timestamp grid.
    """
    idx = torch.searchsorted(src_timestamps, dst_timestamps)
    idx = torch.clamp(idx, 0, src_timestamps.numel() - 1)

    closest_src_left_idx = torch.clamp(idx - 1, 0, src_timestamps.numel() - 1)
    closest_src_right_idx = torch.clamp(idx, 0, src_timestamps.numel() - 1)

    src_keypoints_left = src_keypoints[closest_src_left_idx, :]
    src_keypoints_right = src_keypoints[closest_src_right_idx, :]

    src_scores_left = src_scores[closest_src_left_idx, :]
    src_scores_right = src_scores[closest_src_right_idx, :]

    dt_left = torch.abs(dst_timestamps - src_timestamps[closest_src_left_idx])
    dt_right = torch.abs(dst_timestamps - src_timestamps[closest_src_right_idx])

    dt_left_ok = dt_left < dt_threshold_s
    dt_right_ok = dt_right < dt_threshold_s

    # We do a simple linear interpolation between the two closest timestamps
    w_l = torch.where(
        closest_src_left_idx == closest_src_right_idx,
        1.0,
        dt_right / (dt_left + dt_right),
    )
    w_r = torch.where(
        closest_src_left_idx == closest_src_right_idx,
        0.0,
        dt_left / (dt_left + dt_right),
    )

    resampled_kps_xy = (
        w_l.view(-1, 1, 1) * src_keypoints_left
        + w_r.view(-1, 1, 1) * src_keypoints_right
    )

    # Do not trust the interpolated keypoints if the dt is above the threshold
    resampled_kps_scores = torch.where(
        dt_left_ok.view(-1, 1) & dt_right_ok.view(-1, 1),
        w_l.view(-1, 1) * src_scores_left + w_r.view(-1, 1) * src_scores_right,
        torch.zeros_like(w_l.view(-1, 1)),
    )

    return resampled_kps_xy, resampled_kps_scores


def _resample_bboxes(
    src_bboxes: torch.Tensor,
    src_scores: torch.Tensor,
    src_timestamps: torch.Tensor,
    dst_timestamps: torch.Tensor,
    dt_threshold_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resample bounding boxes to a target timestamp grid.
    Args:
        src_bboxes: Source bounding boxes in xyxy format
        src_scores: Source bounding box scores
        src_timestamps: Source timestamps
        dst_timestamps: Target timestamps to resample to
        dt_threshold_s: Time threshold for interpolation
    Returns:
        Tuple of resampled bboxes (in xyxy format) and scores
    """
    # Convert xyxy to xywh format
    src_bboxes_xywh = torch.zeros_like(src_bboxes)
    src_bboxes_xywh[..., 0] = src_bboxes[..., 0]  # x
    src_bboxes_xywh[..., 1] = src_bboxes[..., 1]  # y
    src_bboxes_xywh[..., 2] = src_bboxes[..., 2] - src_bboxes[..., 0]  # width
    src_bboxes_xywh[..., 3] = src_bboxes[..., 3] - src_bboxes[..., 1]  # height

    idx = torch.searchsorted(src_timestamps, dst_timestamps)
    idx = torch.clamp(idx, 0, src_timestamps.numel() - 1)

    closest_src_left_idx = torch.clamp(idx - 1, 0, src_timestamps.numel() - 1)
    closest_src_right_idx = torch.clamp(idx, 0, src_timestamps.numel() - 1)

    src_bboxes_left = src_bboxes_xywh[closest_src_left_idx, :]
    src_bboxes_right = src_bboxes_xywh[closest_src_right_idx, :]

    src_scores_left = src_scores[closest_src_left_idx]
    src_scores_right = src_scores[closest_src_right_idx]

    dt_left = torch.abs(dst_timestamps - src_timestamps[closest_src_left_idx])
    dt_right = torch.abs(dst_timestamps - src_timestamps[closest_src_right_idx])

    dt_left_ok = dt_left < dt_threshold_s
    dt_right_ok = dt_right < dt_threshold_s

    # We do a simple linear interpolation between the two closest timestamps
    w_l = torch.where(
        closest_src_left_idx == closest_src_right_idx,
        1.0,
        dt_right / (dt_left + dt_right),
    )
    w_r = torch.where(
        closest_src_left_idx == closest_src_right_idx,
        0.0,
        dt_left / (dt_left + dt_right),
    )

    resampled_bboxes_xywh = (
        w_l.view(-1, 1) * src_bboxes_left + w_r.view(-1, 1) * src_bboxes_right
    )

    # Do not trust the interpolated bboxes if the dt is above the threshold
    resampled_scores = torch.where(
        dt_left_ok & dt_right_ok,
        w_l * src_scores_left + w_r * src_scores_right,
        torch.zeros_like(w_l),
    )

    # Convert back to xyxy format
    resampled_bboxes = torch.zeros_like(resampled_bboxes_xywh)
    resampled_bboxes[..., 0] = resampled_bboxes_xywh[..., 0]  # x1
    resampled_bboxes[..., 1] = resampled_bboxes_xywh[..., 1]  # y1
    resampled_bboxes[..., 2] = (
        resampled_bboxes_xywh[..., 0] + resampled_bboxes_xywh[..., 2]
    )  # x2
    resampled_bboxes[..., 3] = (
        resampled_bboxes_xywh[..., 1] + resampled_bboxes_xywh[..., 3]
    )  # y2

    return resampled_bboxes, resampled_scores
