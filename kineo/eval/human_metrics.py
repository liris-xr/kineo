# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import math

import torch
from kineo.geometry.transformations import (
    compute_similarity_transform,
    apply_similarity_transform_to_points,
    inverse_Rt,
)
from typing import Any
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations

def get_min_median_max_frames(
    human_metrics: dict[int, dict[str, float]]
) -> dict[int, dict[str, float]]:
    aggregated_human_metrics = aggregate_human_metrics_per_frame(human_metrics)
    frames = sorted(list(aggregated_human_metrics.keys()))

    w_mpjpe = torch.tensor([aggregated_human_metrics[frame_idx]["w-mpjpe"] for frame_idx in frames])
    pa_mpjpe = torch.tensor([aggregated_human_metrics[frame_idx]["pa-mpjpe"] for frame_idx in frames])

    if len(w_mpjpe) == 0 or len(pa_mpjpe) == 0:
        return None

    min_w_mpjpe_frame = w_mpjpe.argmin().item()
    median_w_mpjpe = w_mpjpe.nanmedian(dim=-1, keepdim=False)
    median_w_mpjpe_frame = median_w_mpjpe[1].item()
    median_w_mpjpe_value = median_w_mpjpe[0].item()
    max_w_mpjpe_frame = w_mpjpe.argmax().item()
    min_pa_mpjpe_frame = pa_mpjpe.argmin().item()
    median_pa_mpjpe = pa_mpjpe.nanmedian(dim=-1, keepdim=False)
    median_pa_mpjpe_frame = median_pa_mpjpe[1].item()
    median_pa_mpjpe_value = median_pa_mpjpe[0].item()
    max_pa_mpjpe_frame = pa_mpjpe.argmax().item()
    min_w_mpjpe_value = w_mpjpe[min_w_mpjpe_frame]
    max_w_mpjpe_value = w_mpjpe[max_w_mpjpe_frame]
    min_pa_mpjpe_value = pa_mpjpe[min_pa_mpjpe_frame]
    max_pa_mpjpe_value = pa_mpjpe[max_pa_mpjpe_frame]

    mean_w_mpjpe_value = w_mpjpe.nanmean().item()
    mean_pa_mpjpe_value = pa_mpjpe.nanmean().item()
    std_w_mpjpe_value = w_mpjpe[w_mpjpe.isfinite()].std().item()
    std_pa_mpjpe_value = pa_mpjpe[pa_mpjpe.isfinite()].std().item()

    return {
        "min_w_mpjpe_frame": min_w_mpjpe_frame,
        "min_w_mpjpe_value": min_w_mpjpe_value,
        "median_w_mpjpe_frame": median_w_mpjpe_frame,
        "median_w_mpjpe_value": median_w_mpjpe_value,
        "max_w_mpjpe_frame": max_w_mpjpe_frame,
        "max_w_mpjpe_value": max_w_mpjpe_value,
        "min_pa_mpjpe_frame": min_pa_mpjpe_frame,
        "min_pa_mpjpe_value": min_pa_mpjpe_value,
        "median_pa_mpjpe_frame": median_pa_mpjpe_frame,
        "median_pa_mpjpe_value": median_pa_mpjpe_value,
        "max_pa_mpjpe_frame": max_pa_mpjpe_frame,
        "max_pa_mpjpe_value": max_pa_mpjpe_value,
        "std_w_mpjpe_value": std_w_mpjpe_value,
        "std_pa_mpjpe_value": std_pa_mpjpe_value,
        "mean_w_mpjpe_value": mean_w_mpjpe_value,
        "mean_pa_mpjpe_value": mean_pa_mpjpe_value,
    }

def aggregate_human_metrics_per_frame(
    human_metrics: dict[str, Any]
) -> dict[int, dict[str, float]]:
    human_metrics_per_frame: dict[int, dict[str, float]] = {}

    for frame_idx, frame_metrics in human_metrics.items():
        
        frame_w_mpjpe: list[float] = []
        frame_pa_mpjpe: list[float] = []

        for subject_metrics in frame_metrics:
            for keypoint_metrics in subject_metrics["joints"]:
                frame_w_mpjpe.append(keypoint_metrics["w-mpjpe"])
                frame_pa_mpjpe.append(keypoint_metrics["pa-mpjpe"])

        frame_w_mpjpe = torch.as_tensor(frame_w_mpjpe).nanmean().item()
        frame_pa_mpjpe = torch.as_tensor(frame_pa_mpjpe).nanmean().item()

        human_metrics_per_frame[frame_idx] = {
            "w-mpjpe": frame_w_mpjpe,
            "pa-mpjpe": frame_pa_mpjpe,
        }

    return human_metrics_per_frame

def compute_human_metrics(
    gt_keypoints_3d_annotations: Keypoints3DAnnotations,
    gt_cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
    pred_keypoints_3d_annotations: Keypoints3DAnnotations,
    pred_cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames = gt_keypoints_3d_annotations.frames
    gt_subjects_ids = gt_keypoints_3d_annotations.subjects_ids
    pred_subjects_ids = pred_keypoints_3d_annotations.subjects_ids
    views_ids = gt_cam_extrinsics_annotations.views_ids
    n_frames = len(frames)
    n_views = len(views_ids)
    n_subjects = len(gt_subjects_ids)

    # Assuming the same format for all subjects
    gt_kps_format = gt_keypoints_3d_annotations.metadata.formats[0]
    pred_kps_format = pred_keypoints_3d_annotations.metadata.formats[0]

    if gt_kps_format.name != pred_kps_format.name:
        pred_keypoints_3d_annotations = pred_keypoints_3d_annotations.convert_to_format(gt_kps_format)
        pred_kps_format = gt_kps_format

    n_keypoints = gt_kps_format.n_keypoints

    gt_world2cam = torch.zeros((n_views, 3, 4), device=device)
    gt_kps3d = torch.zeros((n_frames, n_subjects, n_keypoints, 3), device=device)
    pred_kps3d = torch.zeros((n_frames, n_subjects, n_keypoints, 3), device=device)
    # A GT frame the predictions never fill, or a keypoint the pipeline zeroed
    # because triangulation was not finite, is a missing prediction rather than
    # one at the world origin. Track them so they do not count as errors.
    pred_valid = torch.zeros(
        (n_frames, n_subjects, n_keypoints), dtype=torch.bool, device=device
    )

    view_id_to_idx = {view_id: i for i, view_id in enumerate(views_ids)}
    subject_id_to_idx = {subject_id: i for i, subject_id in enumerate(gt_subjects_ids)}
    frame_to_idx = {frame_idx: i for i, frame_idx in enumerate(frames)}

    if set(pred_subjects_ids) != set(gt_subjects_ids):
        if len(pred_subjects_ids) == len(gt_subjects_ids) == 1:
            subject_id_to_idx.update({subject_id: i for i, subject_id in enumerate(pred_subjects_ids)})
        else:
            raise NotImplementedError(
                f"Predicted subjects ids do not match ground truth subjects ids and multi-subject auto associating in metric computation is not implemented yet"
            )
    
    # One pose per view for the camera-center alignment; non-static (moving) views
    # keep multiple annotations, their inter-segment translation is negligible here.
    for view_id in views_ids:
        ann = gt_cam_extrinsics_annotations.filter_by_view_id(view_id).first_or_default()
        gt_world2cam[view_id_to_idx[view_id]] = ann.Rt

    gt_world2cam = gt_world2cam.unsqueeze(0).expand(n_frames, -1, -1, -1)

    for ann in gt_keypoints_3d_annotations.annotations:
        subject_idx = subject_id_to_idx[ann.subject_id]
        frame_idx = frame_to_idx[ann.frame_idx]
        gt_kps3d[frame_idx, subject_idx, :] = ann.xyz

    for ann in pred_keypoints_3d_annotations.annotations:
        subject_idx = subject_id_to_idx[ann.subject_id]
        frame_idx = frame_to_idx.get(ann.frame_idx, -1)
        if frame_idx == -1:
            continue
        pred_kps3d[frame_idx, subject_idx] = ann.xyz
        pred_valid[frame_idx, subject_idx] = (
            torch.as_tensor(ann.scores, device=device) > 0
        )

    if len(pred_cam_extrinsics_annotations.annotations) == n_views:
        pred_world2cam = torch.empty((n_views, 3, 4), device=device)

        for ann in pred_cam_extrinsics_annotations.annotations:
            view_idx = view_id_to_idx[ann.view_id]
            pred_world2cam[view_idx] = ann.Rt

        pred_world2cam = pred_world2cam.unsqueeze(0).expand(n_frames, -1, -1, -1)
    else:
        pred_world2cam = torch.empty((n_frames, n_views, 3, 4), device=device)

        for ann in pred_cam_extrinsics_annotations.annotations:
            view_idx = view_id_to_idx[ann.view_id]
            frame_idx = frame_to_idx.get(ann.frame_idx, -1)
            if frame_idx == -1:
                continue
            pred_world2cam[frame_idx, view_idx] = ann.Rt

    gt_cam2world = inverse_Rt(gt_world2cam)
    pred_cam2world = inverse_Rt(pred_world2cam)
    gt_cam_pos = gt_cam2world[..., :3, 3]
    pred_cam_pos = pred_cam2world[..., :3, 3]

    R, t, s = compute_similarity_transform(
        X=pred_cam_pos.reshape(n_frames, n_views, 3),
        Y=gt_cam_pos.reshape(n_frames, n_views, 3),
        estimate_scale=False,
    )

    pred_kps3d_aligned = apply_similarity_transform_to_points(
        pred_kps3d.reshape(n_frames, -1, 3),
        R.reshape(n_frames, 3, 3),
        t.reshape(n_frames, 3),
        s.reshape(n_frames),
    ).reshape(n_frames, n_subjects, n_keypoints, 3)

    w_mpjpe = torch.norm(
        gt_kps3d - pred_kps3d_aligned,
        dim=-1,
    )
    w_mpjpe = torch.where(pred_valid, w_mpjpe, torch.nan).cpu()

    pa_mpjpe = torch.zeros((n_frames, n_subjects, n_keypoints))

    for subject_idx in range(n_subjects):
        # Estimates one similarity transform for each frame
        R, t, s = compute_similarity_transform(
            X=pred_kps3d[:, subject_idx].reshape(n_frames, n_keypoints, 3),
            Y=gt_kps3d[:, subject_idx].reshape(n_frames, n_keypoints, 3),
            estimate_scale=True,
        )
        subject_pred_kps3d_aligned = apply_similarity_transform_to_points(
            pred_kps3d[:, subject_idx].reshape(n_frames, n_keypoints, 3),
            R.reshape(n_frames, 3, 3),
            t.reshape(n_frames, 3),
            s.reshape(n_frames),
        ).reshape(n_frames, n_keypoints, 3)

        subject_pose_valid = pred_valid[:, subject_idx].all(dim=-1)
        pa_mpjpe[:, subject_idx] = torch.where(
            subject_pose_valid.unsqueeze(-1),
            torch.norm(
                gt_kps3d[:, subject_idx] - subject_pred_kps3d_aligned,
                dim=-1,
            ),
            torch.nan,
        ).cpu()

    return {
        frames[frame_idx]: [
            {
                "subject_id": subject_id,
                "joints": [
                    {
                        "joint_name": gt_kps_format.keypoints_names[kp_idx],
                        "w-mpjpe": w_mpjpe[frame_idx, subject_idx, kp_idx].item(),
                        "pa-mpjpe": pa_mpjpe[frame_idx, subject_idx, kp_idx].item(),
                    }
                    for kp_idx in range(n_keypoints)
                ],
            }
            for subject_idx, subject_id in enumerate(gt_subjects_ids)
        ]
        for frame_idx in range(n_frames)
    }


def flatten_human_metrics(human_metrics: dict[str, Any]) -> dict[str, Any]:
    # Aggregate the W-MPJPE and PA-MPJPE metrics for each joint and each subject in each frame
    # We collect them as a list so that the caller can compute the statistics (mean, median, std, min, max).
    all_w_mpjpe = []
    all_pa_mpjpe = []

    for frame_metrics in human_metrics.values():
        for subject_metrics in frame_metrics:
            for keypoint_metrics in subject_metrics["joints"]:
                all_w_mpjpe.append(keypoint_metrics["w-mpjpe"])
                all_pa_mpjpe.append(keypoint_metrics["pa-mpjpe"])

    # w-mpjpe drops unpredicted keypoints as NaN; its mean is only interpretable
    # alongside the percentage that were recovered at all.
    all_reconstruction_rate = [
        0.0 if math.isnan(value) else 100.0 for value in all_w_mpjpe
    ]

    return {
        "w-mpjpe": all_w_mpjpe,
        "pa-mpjpe": all_pa_mpjpe,
        "reconstruction-rate": all_reconstruction_rate,
    }
