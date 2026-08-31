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

from kineo.eval import time_alignment
from kineo.eval.sequence_metrics import compute_camera_metrics_over_sequence, compute_human_metrics_over_sequence
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.stage_timing import StageTimingsAnnotations
import numpy as np
import torch
import orjson
from dataclasses import dataclass
import os


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass(frozen=True)
class MetricsExportRuntimeConfig:
    output_path_template: str = "./outputs/metrics/{sequence_name}.json"


def _frame_timestamps(time_reference) -> torch.Tensor | None:
    """Frame timestamps of a global time reference, if there is one."""
    if time_reference is None:
        return None
    annotation = time_reference.first_or_default()
    return None if annotation is None else annotation.timestamps


def _pinned_view(camera_temporal) -> str | None:
    """View a pipeline put its predictions' clock on: the one it left at zero."""
    if camera_temporal is None:
        return None
    return next(
        (a.view_id for a in camera_temporal.annotations if a.time_offset == 0.0),
        None,
    )


def _gt_frame_timestamps(gt_time_reference, pinned_view: str | None):
    """Ground-truth instants, on the clock the predictions ended up on.

    Ground truth is annotated on the clock of whichever view it was built
    against; the predictions are on the clock of whichever view the pipeline
    pinned. Restating the annotations settles the two on one clock, whatever
    reference the pipeline was configured with.
    """
    timestamps = _frame_timestamps(gt_time_reference)

    if timestamps is None or pinned_view is None:
        return timestamps

    annotation = gt_time_reference.first_or_default()

    if pinned_view not in annotation.closest_local_frame_idx:
        return timestamps

    return time_alignment.timestamps_on_view(annotation, pinned_view)


class MetricsExportStage(PipelineStage[MetricsExportRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: MetricsExportRuntimeConfig,
        dynamic_runtime_cfg: dict[str, MetricsExportRuntimeConfig] | None = None,
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
        runtime_cfg: MetricsExportRuntimeConfig,
    ):
        pred_keypoints_3d: Keypoints3DAnnotations = annotations["keypoints_3d"]
        pred_camera_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "cameras_extrinsics"
        ]
        pred_camera_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "cameras_intrinsics"
        ]

        gt_keypoints_3d: Keypoints3DAnnotations = gt_annotations.get("keypoints_3d")
        gt_camera_extrinsics: CameraExtrinsicsAnnotations = gt_annotations.get(
            "cameras_extrinsics"
        )
        gt_camera_intrinsics: CameraIntrinsicsAnnotations = gt_annotations.get(
            "cameras_intrinsics"
        )

        stage_timings: StageTimingsAnnotations = annotations.get("stage_timings")

        n_total_frames = sum(view["frame_loader"].n_frames for view in views)
        avg_fps = None

        if stage_timings is not None:
            total_time_s = sum(
                stage_timing.duration_seconds for stage_timing in stage_timings
            )
            avg_fps = n_total_frames / total_time_s
            stage_timings = [
                {
                    "stage_name": stage_timing.stage_name,
                    "stage_idx": stage_timing.stage_idx,
                    "duration_seconds": stage_timing.duration_seconds,
                }
                for stage_timing in stage_timings.annotations
            ]
        else:
            stage_timings = []
            avg_fps = None

        cam_metrics = compute_camera_metrics_over_sequence(
            gt_cam_intrinsics_annotations=gt_camera_intrinsics,
            gt_cam_extrinsics_annotations=gt_camera_extrinsics,
            pred_cam_intrinsics_annotations=pred_camera_intrinsics,
            pred_cam_extrinsics_annotations=pred_camera_extrinsics,
        )
        human_metrics = compute_human_metrics_over_sequence(
            gt_keypoints_3d_annotations=gt_keypoints_3d,
            gt_cam_extrinsics_annotations=gt_camera_extrinsics,
            pred_keypoints_3d_annotations=pred_keypoints_3d,
            pred_cam_extrinsics_annotations=pred_camera_extrinsics,
            gt_frame_timestamps=_gt_frame_timestamps(
                gt_annotations.get("global_time_reference"),
                _pinned_view(annotations.get("cameras_temporal")),
            ),
            pred_frame_timestamps=_frame_timestamps(
                annotations.get("global_time_reference")
            ),
        )

        non_static_views = [
            view_id
            for view_id in gt_camera_extrinsics.views_ids
            if not gt_camera_extrinsics.is_view_static(view_id)
        ]

        avg_metrics = {
            "cam_stats": cam_metrics,
            "human_stats": human_metrics,
            "has_non_static_camera": len(non_static_views) > 0,
            "non_static_views": non_static_views,
            "stage_timings": stage_timings,
            "n_total_frames": n_total_frames,
            "avg_fps": avg_fps,
        }

        formatted_output_path = runtime_cfg.output_path_template.format(
            sequence_name=sequence_name
        )

        os.makedirs(os.path.dirname(formatted_output_path), exist_ok=True)

        with open(formatted_output_path, "wb") as f:
            f.write(
                orjson.dumps(avg_metrics, default=default, option=orjson.OPT_INDENT_2)
            )
        print(f"Metrics saved to {formatted_output_path}")