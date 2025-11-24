# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import numpy as np
from kineo.eval.camera_metrics import compute_camera_metrics, flatten_camera_metrics
from kineo.eval.human_metrics import compute_human_metrics, flatten_human_metrics
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations


def compute_human_metrics_over_sequence(
    gt_keypoints_3d_annotations: Keypoints3DAnnotations,
    gt_cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
    pred_keypoints_3d_annotations: Keypoints3DAnnotations,
    pred_cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
) -> dict[str, dict[str, float]]:
    human_metrics = compute_human_metrics(
        gt_keypoints_3d_annotations=gt_keypoints_3d_annotations,
        gt_cam_extrinsics_annotations=gt_cam_extrinsics_annotations,
        pred_keypoints_3d_annotations=pred_keypoints_3d_annotations,
        pred_cam_extrinsics_annotations=pred_cam_extrinsics_annotations,
    )
    flattened_human_metrics = flatten_human_metrics(human_metrics)
    return {
        key: {
            "mean": np.nanmean(value),
            "median": np.nanmedian(value),
            "std": np.nanstd(value),
            "min": np.nanmin(value),
            "max": np.nanmax(value),
        }
        for key, value in flattened_human_metrics.items()
    }


def compute_camera_metrics_over_sequence(
    gt_cam_intrinsics_annotations: CameraIntrinsicsAnnotations,
    gt_cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
    pred_cam_intrinsics_annotations: CameraIntrinsicsAnnotations,
    pred_cam_extrinsics_annotations: CameraExtrinsicsAnnotations,
) -> dict[str, dict[str, float]]:
    cam_metrics = compute_camera_metrics(
        gt_cam_intrinsics_annotations=gt_cam_intrinsics_annotations,
        gt_cam_extrinsics_annotations=gt_cam_extrinsics_annotations,
        pred_cam_intrinsics_annotations=pred_cam_intrinsics_annotations,
        pred_cam_extrinsics_annotations=pred_cam_extrinsics_annotations,
    )
    flattened_cam_metrics = flatten_camera_metrics(cam_metrics)
    return {
        key: {
            "mean": np.nanmean(value),
            "median": np.nanmedian(value),
            "std": np.nanstd(value),
            "min": np.nanmin(value),
            "max": np.nanmax(value),
        }
        for key, value in flattened_cam_metrics.items()
    }
