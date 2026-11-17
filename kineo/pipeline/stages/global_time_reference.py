# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from dataclasses import dataclass

import torch

from kineo.annotations import Annotations
from kineo.annotations.camera_temporal import CameraTemporalAnnotations
from kineo.annotations.global_time_reference import (
    GlobalTimeReferenceAnnotation,
    GlobalTimeReferenceAnnotations,
    GlobalTimeReferenceAnnotationsMetadata,
)
from kineo.datasets.annotations_io import build_synchronized_camera_temporal
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.pipeline.pipeline import Pipeline, PipelineStage


@dataclass(frozen=True)
class GlobalTimeReferenceRuntimeConfig:
    target_fps: int = 50


def build_view_timelines(
    views: list[ViewInput],
    camera_temporal: CameraTemporalAnnotations,
    device: torch.device,
) -> list[torch.Tensor]:
    timelines = []

    for view in views:
        view_temporal = camera_temporal.filter_by_view_id(
            view["view_id"]
        ).first_or_default()

        if view_temporal is None:
            raise ValueError(
                f"No camera temporal annotations found for view {view['view_id']}"
            )

        # Assumes a fixed offset per view (no clock drift).
        timelines.append(
            view["frame_loader"].frame_timestamps_local.to(device)
            + view_temporal.time_offset
        )

    return timelines


def build_global_time_reference(
    views: list[ViewInput],
    camera_temporal: CameraTemporalAnnotations,
    target_fps: float,
    device: torch.device,
) -> GlobalTimeReferenceAnnotations:
    views_frame_timestamps = build_view_timelines(views, camera_temporal, device)

    if _shares_one_timeline(views, camera_temporal):
        n_frames = views[0]["frame_loader"].n_frames
        timestamps = views[0]["frame_loader"].frame_timestamps_local.clone()
        closest_local_frame_idx = {
            view["view_id"]: torch.arange(n_frames) for view in views
        }
    else:
        timestamps = _create_uniform_timestamp_grid(
            views_frame_timestamps, target_fps, device
        )
        closest_local_frame_idx = {
            view["view_id"]: _get_closest_view_local_frame_for_global_frames(
                view_frame_timestamps=view_frame_timestamps,
                global_frame_timestamps=timestamps,
            )
            for view, view_frame_timestamps in zip(views, views_frame_timestamps)
        }

    return GlobalTimeReferenceAnnotations(
        metadata=GlobalTimeReferenceAnnotationsMetadata(),
        annotations=[
            GlobalTimeReferenceAnnotation(
                timestamps=timestamps,
                closest_local_frame_idx=closest_local_frame_idx,
            )
        ],
    ).cpu()


def is_pass_through(reference: GlobalTimeReferenceAnnotation) -> bool:
    """Check whether the timeline is each view's own frames, unchanged."""
    n_frames = reference.timestamps.numel()

    return all(
        len(local_frame_idx) == n_frames
        and bool(
            torch.equal(
                local_frame_idx,
                torch.arange(n_frames, device=local_frame_idx.device),
            )
        )
        for local_frame_idx in reference.closest_local_frame_idx.values()
    )


def build_inference_frames(
    annotations: dict[str, Annotations], views: list[ViewInput]
) -> dict[str, list[int]] | None:
    reference = annotations.get("global_time_reference", None)

    if reference is None:
        return None

    closest_local_frame_idx = reference.first_or_default().closest_local_frame_idx

    return {
        view["view_id"]: torch.unique(
            closest_local_frame_idx[view["view_id"]]
        ).tolist()
        for view in views
    }


class GlobalTimeReferenceStage(PipelineStage[GlobalTimeReferenceRuntimeConfig]):
    """Stage for building the "global_time_reference" before frame decoding."""

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: GlobalTimeReferenceRuntimeConfig,
        dynamic_runtime_cfg: dict[str, GlobalTimeReferenceRuntimeConfig]
        | None = None,
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
        runtime_cfg: GlobalTimeReferenceRuntimeConfig,
    ):
        camera_temporal = annotations.get("cameras_temporal", None)

        if camera_temporal is None:
            print(
                "No camera temporal annotations provided, assuming synchronized cameras."
            )
            camera_temporal = build_synchronized_camera_temporal(
                [view["view_id"] for view in views]
            )
            annotations["cameras_temporal"] = camera_temporal

        annotations["global_time_reference"] = build_global_time_reference(
            views=views,
            camera_temporal=camera_temporal,
            target_fps=runtime_cfg.target_fps,
            device=pipeline.device,
        )


def _shares_one_timeline(
    views: list[ViewInput], camera_temporal: CameraTemporalAnnotations
) -> bool:
    return all(a.time_offset == 0.0 for a in camera_temporal.annotations) and all(
        view["frame_loader"].n_frames == views[0]["frame_loader"].n_frames
        for view in views
    )


def _create_uniform_timestamp_grid(
    view_frame_timestamps: list[torch.Tensor],
    fps: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a uniform timestamp grid from a given timestamp tensor.
    """
    min_timestamp = max([timestamps.min() for timestamps in view_frame_timestamps])
    max_timestamp = min([timestamps.max() for timestamps in view_frame_timestamps])
    duration = max_timestamp - min_timestamp
    n_frames = int(duration * fps)
    return torch.arange(
        min_timestamp, max_timestamp, duration / n_frames, device=device
    )


def _get_closest_view_local_frame_for_global_frames(
    view_frame_timestamps: torch.Tensor,
    global_frame_timestamps: torch.Tensor,
) -> torch.Tensor:
    """
    Get the closest local frame index for each global frame index.
    """
    right_indices = torch.searchsorted(view_frame_timestamps, global_frame_timestamps)

    right_indices = torch.clamp(right_indices, 1, len(view_frame_timestamps))
    left_indices = right_indices - 1

    left_dist = torch.abs(view_frame_timestamps[left_indices] - global_frame_timestamps)
    right_dist = torch.where(
        right_indices < len(view_frame_timestamps),
        torch.abs(view_frame_timestamps[right_indices] - global_frame_timestamps),
        float("inf"),
    )

    return torch.where(left_dist <= right_dist, left_indices, right_indices)
