# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import dataclasses
import os
from typing import Any, TypeVar

import rerun as rr
import rerun.blueprint as rrb
import torch

from kineo.annotations.annotations import Annotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.datasets.keypoints_sequence_dataset import KeypointsSequence, ViewInput
from kineo.io.ffmpeg import (
    encode_images_to_video,
    get_video_codec,
    transcode_video_to_h264,
)
from kineo.io.frame_sequence_loader import ImagesLoader, VideoLoader
from kineo.visualization import viz_rerun

GROUND_TRUTH_PREFIX = "ground_truth"

UP_AXIS_COORDINATES = {
    "x": rr.ViewCoordinates.RIGHT_HAND_X_UP,
    "y": rr.ViewCoordinates.RIGHT_HAND_Y_UP,
    "z": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
}
DEFAULT_UP_AXIS = "z"
RECORDINGS_PREFIX = "recordings"

ROLLING_STATE = "rolling"
ANNOTATED_STATE = "annotated"
STATE_COLORS = [(120, 120, 130), (60, 180, 110)]

DEFAULT_DOWNSCALE_FACTOR = 4

# Codecs the rerun viewer can decode
VIEWABLE_CODECS = frozenset({"h264", "hevc", "av1", "vp9"})

# Fraction of the image height
KEYPOINTS_RADIUS_RATIO = 0.004
MIN_KEYPOINTS_RADIUS = 2.0

AnnotationsT = TypeVar("AnnotationsT", bound=Annotations)


@dataclasses.dataclass(frozen=True)
class PreviewTimeline:
    """Shared timeline of all views, -1 where a view is not recording."""

    n_steps: int
    lead_in: int
    window_length: int
    local_by_step: dict[str, torch.Tensor]


def add_dataset_dir_argument(parser, dataset_name: str):
    root = os.environ.get("KINEO_DATASETS_DIR")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=os.path.join(root, dataset_name) if root else None,
        help="Path to the directory the dataset was preprocessed into, "
        f"$KINEO_DATASETS_DIR/{dataset_name} by default",
    )


def find_sequence(names: list[str], query: str | None) -> int:
    if not query:
        return 0

    if query in names:
        return names.index(query)

    matches = [
        index
        for index, name in enumerate(names)
        if query.lower() in name.lower()
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise LookupError(
            f"no sequence matches '{query}'. The listing holds "
            f"{len(names)}, such as {', '.join(names[:3])}"
        )

    raise LookupError(
        f"'{query}' matches {len(matches)} sequences: "
        f"{', '.join(names[index] for index in matches[:4])}"
        + (", ..." if len(matches) > 4 else "")
    )


def sequence_fps(views_inputs: list[ViewInput]) -> float:
    timestamps = views_inputs[0]["frame_loader"].frame_timestamps_local
    return 1 / float(torch.diff(timestamps).median())


def build_timeline(
    views_inputs: list[ViewInput],
    time_reference: GlobalTimeReferenceAnnotation | None,
) -> PreviewTimeline:
    n_frames = {
        view_input["view_id"]: len(view_input["frame_loader"])
        for view_input in views_inputs
    }

    if time_reference is None:
        cut_points = {view_id: 0 for view_id in n_frames}
        window_length = max(n_frames.values())
    else:
        cut_points = {
            view_id: int(local_indices[0])
            for view_id, local_indices in (
                time_reference.closest_local_frame_idx.items()
            )
        }
        window_length = len(time_reference.timestamps)

    lead_in = max(cut_points.values())
    starts = {
        view_id: lead_in - cut_point for view_id, cut_point in cut_points.items()
    }
    n_steps = max(start + n_frames[view_id] for view_id, start in starts.items())

    local_by_step = {}
    for view_id, start in starts.items():
        local = torch.arange(n_steps) - start
        local[local < 0] = -1
        local[local >= n_frames[view_id]] = -1
        local_by_step[view_id] = local

    return PreviewTimeline(
        n_steps=n_steps,
        lead_in=lead_in,
        window_length=window_length,
        local_by_step=local_by_step,
    )


def log_recording_states(timeline: PreviewTimeline, fps: float):
    window_end = timeline.lead_in + timeline.window_length

    for view_id, local in timeline.local_by_step.items():
        entity_path = f"{RECORDINGS_PREFIX}/{view_id}"
        rr.log(
            entity_path,
            rr.StateConfiguration(
                values=[ROLLING_STATE, ANNOTATED_STATE], colors=STATE_COLORS
            ),
            static=True,
        )

        rolling = (local >= 0).nonzero().flatten()
        states = {
            int(rolling[0]): ROLLING_STATE,
            timeline.lead_in: ANNOTATED_STATE,
            window_end: ROLLING_STATE,
        }

        for step, state in sorted(states.items()):
            rr.set_time("frame_idx", sequence=step)
            rr.set_time("time", timestamp=step / fps)
            rr.log(entity_path, rr.StateChange(state=state))

        last_step = int(rolling[-1]) + 1
        rr.set_time("frame_idx", sequence=last_step)
        rr.set_time("time", timestamp=last_step / fps)
        rr.log(entity_path, rr.Clear(recursive=False))


def sequence_blueprint(view_ids: list[str]) -> rrb.Blueprint:
    scene = rrb.Spatial3DView(
        name="Scene",
        origin=GROUND_TRUTH_PREFIX,
        contents=["+ $origin/**", "- $origin/cameras/*/rgb"],
    )

    cameras = rrb.Grid(
        contents=[
            rrb.Spatial2DView(
                name=view_id,
                origin=f"{GROUND_TRUTH_PREFIX}/cameras/{view_id}",
            )
            for view_id in view_ids
        ]
    )

    recordings = rrb.StateTimelineView(
        name="Recordings", origin=RECORDINGS_PREFIX
    )

    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(scene, cameras, column_shares=[2, 3]),
            recordings,
            row_shares=[4, 1],
        ),
        collapse_panels=True,
    )


def scale_pixel_space(
    annotations: dict[str, Any], scale: float
) -> dict[str, Any]:
    if scale == 1.0:
        return annotations

    resized = dict(annotations)

    for kind, fields in (
        ("keypoints_2d", {"xy": lambda xy: xy * scale}),
        ("bboxes_2d", {"xyxy": lambda xyxy: xyxy * scale}),
        (
            "cameras_intrinsics",
            {
                "K": lambda K: K * torch.tensor([[scale], [scale], [1.0]]),
                "resolution_hw": lambda hw: (
                    round(hw[0] * scale),
                    round(hw[1] * scale),
                ),
            },
        ),
    ):
        if kind in resized:
            resized[kind] = _replace_fields(resized[kind], fields)

    return resized


def _replace_fields(annotations: AnnotationsT, fields: dict) -> AnnotationsT:
    return type(annotations)(
        metadata=annotations.metadata,
        annotations=[
            dataclasses.replace(
                annotation,
                **{
                    name: transform(getattr(annotation, name))
                    for name, transform in fields.items()
                },
            )
            for annotation in annotations
        ],
    )


def keypoints_radius(cameras_intrinsics: CameraIntrinsicsAnnotations) -> float:
    heights = [
        annotation.resolution_hw[0] for annotation in cameras_intrinsics
    ]
    return max(MIN_KEYPOINTS_RADIUS, KEYPOINTS_RADIUS_RATIO * max(heights))


def rebase_on_global_frames(
    annotations: AnnotationsT,
    timeline: PreviewTimeline,
) -> AnnotationsT:
    global_frames = {
        view_id: {
            local_frame_idx: step
            for step, local_frame_idx in enumerate(local.tolist())
            if local_frame_idx >= 0
        }
        for view_id, local in timeline.local_by_step.items()
    }

    rebased = [
        dataclasses.replace(
            annotation,
            frame_idx=global_frames[annotation.view_id][annotation.frame_idx],
        )
        for annotation in annotations
        if annotation.frame_idx in global_frames[annotation.view_id]
    ]

    return type(annotations)(metadata=annotations.metadata, annotations=rebased)


def preview_sequence(
    sequence: KeypointsSequence,
    fps: float | None = None,
    downscale_factor: int = DEFAULT_DOWNSCALE_FACTOR,
    up_axis: str = DEFAULT_UP_AXIS,
):
    """Show a sequence and its ground truth in the rerun viewer."""
    if downscale_factor < 1:
        raise ValueError(
            f"A preview cannot be larger than the dataset: downscale_factor "
            f"is {downscale_factor}."
        )

    if up_axis not in UP_AXIS_COORDINATES:
        raise ValueError(
            f"Unknown up axis '{up_axis}', expected any of "
            f"{list(UP_AXIS_COORDINATES)}."
        )

    if fps is None:
        fps = sequence_fps(sequence["views_inputs"])

    scale = 1 / downscale_factor
    annotations = scale_pixel_space(sequence["annotations"] or {}, scale)

    time_reference = annotations.get("global_time_reference")
    if time_reference is not None:
        time_reference = time_reference.first_or_default()

    timeline = build_timeline(sequence["views_inputs"], time_reference)

    rr.init(sequence["sequence_name"], spawn=True)

    rr.send_blueprint(
        sequence_blueprint(
            [view_input["view_id"] for view_input in sequence["views_inputs"]]
        )
    )

    rr.log(GROUND_TRUTH_PREFIX, UP_AXIS_COORDINATES[up_axis], static=True)

    cameras_intrinsics = annotations.get("cameras_intrinsics")
    cameras_extrinsics = annotations.get("cameras_extrinsics")
    if cameras_intrinsics is not None and cameras_extrinsics is not None:
        viz_rerun.log_cameras(
            cameras_extrinsics=cameras_extrinsics,
            cameras_intrinsics=cameras_intrinsics,
            prefix=GROUND_TRUTH_PREFIX,
        )

    keypoints_3d = annotations.get("keypoints_3d")
    if keypoints_3d is not None:
        # 3D ground truth starts at the annotated window
        keypoints_3d = _replace_fields(
            keypoints_3d, {"frame_idx": lambda idx: idx + timeline.lead_in}
        )

        viz_rerun.log_skeletons_3d(
            keypoints_3d=keypoints_3d, prefix=GROUND_TRUTH_PREFIX, fps=fps
        )

    keypoints_2d = annotations.get("keypoints_2d")
    if keypoints_2d is not None:
        keypoints_2d = rebase_on_global_frames(keypoints_2d, timeline)

        radius = (
            keypoints_radius(cameras_intrinsics)
            if cameras_intrinsics is not None
            else MIN_KEYPOINTS_RADIUS
        )
        viz_rerun.log_skeletons_2d(
            keypoints_2d=keypoints_2d,
            prefix=GROUND_TRUTH_PREFIX,
            fps=fps,
            joint_radius=radius,
            bones_thickness=radius / 2,
        )

    bboxes_2d = annotations.get("bboxes_2d")
    if bboxes_2d is not None:
        bboxes_2d = rebase_on_global_frames(bboxes_2d, timeline)

        viz_rerun.log_bboxes_2d(
            bboxes_2d=bboxes_2d, prefix=GROUND_TRUTH_PREFIX, fps=fps
        )

    log_recording_states(timeline, fps)

    for view_input in sequence["views_inputs"]:
        _log_view_frames(view_input, timeline, fps, downscale_factor)


def _log_view_frames(
    view_input: ViewInput,
    timeline: PreviewTimeline,
    fps: float,
    downscale_factor: int,
):
    view_id = view_input["view_id"]
    frame_loader = view_input["frame_loader"]
    frame_indices = timeline.local_by_step[view_id]

    if isinstance(frame_loader, VideoLoader):
        if frame_loader.selected_frames is not None:
            rolling = frame_indices >= 0
            frame_indices = torch.where(
                rolling,
                frame_loader.selected_frames[frame_indices.clamp(min=0)],
                -1,
            )

        video_path = _viewable_video(frame_loader.video_path, downscale_factor)
    elif isinstance(frame_loader, ImagesLoader):
        video_path = _encoded_images(frame_loader, fps, downscale_factor)
    else:
        raise TypeError(
            f"View {view_id} is backed by {type(frame_loader).__name__}, which "
            "holds no file the viewer could read."
        )

    viz_rerun.log_video_asset(
        video_path=video_path,
        view_id=view_id,
        local_frame_indices=frame_indices.tolist(),
        prefix=GROUND_TRUTH_PREFIX,
        fps=fps,
    )


def _viewable_video(video_path: str, downscale_factor: int) -> str:
    if downscale_factor == 1 and get_video_codec(video_path) in VIEWABLE_CODECS:
        return video_path

    transcoded_path = (
        f"{os.path.splitext(video_path)[0]}_{_preview_name(downscale_factor)}"
    )

    if not os.path.exists(transcoded_path):
        transcode_video_to_h264(
            video_path, transcoded_path, 1 / downscale_factor
        )

    return transcoded_path


def _preview_name(downscale_factor: int) -> str:
    return f"preview_downscale_{downscale_factor}.mp4"


def _encoded_images(
    frame_loader: ImagesLoader, fps: float, downscale_factor: int
) -> str:
    video_path = os.path.join(
        os.path.dirname(frame_loader.img_paths[0]),
        _preview_name(downscale_factor),
    )

    if not os.path.exists(video_path):
        encode_images_to_video(
            frame_loader.img_paths, video_path, fps, 1 / downscale_factor
        )

    return video_path
