# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Rerun preview of a preprocessed sequence and the ground truth annotating it.

Views are shown on one timeline, so a sequence whose recordings each start at
their own moment needs its `global_time_reference` to say which frame of which
recording belongs to a step. Without one the views are read as frame-aligned,
which is what a dataset with no offsets between its cameras means.
"""

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
RECORDINGS_PREFIX = "recordings"

# States a view's lane on the timeline is in, and the colours they read as.
ROLLING_STATE = "rolling"
ANNOTATED_STATE = "annotated"
STATE_COLORS = [(120, 120, 130), (60, 180, 110)]

# How much a preview's footage is shrunk by default. Full-resolution frames
# cost two orders of magnitude more than a preview needs to answer whether the
# annotations sit on the subject.
DEFAULT_DOWNSCALE_FACTOR = 4

# What rerun's viewer can decode. Anything else is transcoded first.
VIEWABLE_CODECS = frozenset({"h264", "hevc", "av1", "vp9"})

# Keypoint dot radius, as a share of the image height. Radii are in pixels in
# a 2D view, so a fixed one is a speck on a 4K frame and a blob on a small one.
KEYPOINTS_RADIUS_RATIO = 0.004
MIN_KEYPOINTS_RADIUS = 2.0

AnnotationsT = TypeVar("AnnotationsT", bound=Annotations)


@dataclasses.dataclass(frozen=True)
class PreviewTimeline:
    """Placement of every view's recording on the timeline they share.

    Attributes:
        n_steps: Length of the timeline, covering every recording end to end.
        lead_in: Step the annotated window starts at, which is how much of the
            earliest recording runs before any annotation applies.
        window_length: Length of the annotated window, in steps.
        local_by_step: Frame of each view's own recording shown at each step,
            -1 where that view was not recording yet, or no longer.
    """

    n_steps: int
    lead_in: int
    window_length: int
    local_by_step: dict[str, torch.Tensor]


def build_timeline(
    views_inputs: list[ViewInput],
    time_reference: GlobalTimeReferenceAnnotation | None,
) -> PreviewTimeline:
    """Places whole recordings on one timeline, pre-roll included.

    The annotated window is the part of a sequence the ground truth covers,
    but a raw recording also holds what the camera filmed before and after it.
    Showing only the window hides the very thing the offsets describe, so the
    timeline is stretched to hold every recording, each shifted by its own cut
    point. A view that was not rolling yet simply shows nothing.

    Args:
        views_inputs: Views of the sequence, read for their lengths.
        time_reference: Sequence's time reference, or None if its views are
            frame-aligned and no window is singled out.

    Returns:
        Where every view sits on the shared timeline.
    """
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

    # The camera that started earliest is the one with the most frames before
    # the window, and it sets where the window falls for everyone.
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
    """Draws one lane per view saying when it was rolling and when annotated.

    Args:
        timeline: Placement of the recordings on the shared timeline.
        fps: Rate the steps are turned into timestamps with.
    """
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

        # The lane ends where the recording does, rather than running on to
        # the end of the longest one.
        last_step = int(rolling[-1]) + 1
        rr.set_time("frame_idx", sequence=last_step)
        rr.set_time("time", timestamp=last_step / fps)
        rr.log(entity_path, rr.Clear(recursive=False))


def sequence_blueprint(view_ids: list[str]) -> rrb.Blueprint:
    """Lays a preview out: the scene beside the views that recorded it.

    Rerun's own layout gives every entity its own view, which for a
    nine-camera sequence buries the footage under a view per skeleton. Here
    the 3D scene keeps the skeletons and the camera frustums, each camera gets
    one 2D view holding its footage and the annotations drawn over it, and a
    lane per camera underneath says when it was rolling and which part of it
    the ground truth covers.

    Args:
        view_ids: Cameras of the sequence, in the order they are shown.

    Returns:
        The blueprint to send alongside the recording.
    """
    scene = rrb.Spatial3DView(
        name="Scene",
        origin=GROUND_TRUTH_PREFIX,
        # The footage belongs in the 2D views; drawing every camera's frames
        # onto its image plane as well only costs.
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
    """Moves everything measured in image pixels onto a resized image.

    A preview shows smaller footage than the dataset holds, and the keypoints,
    the boxes and the intrinsics all count pixels of the full-size image. They
    only keep marking what they marked if they are resized with it.

    Args:
        annotations: Annotations of a sequence, keyed by kind.
        scale: Factor the footage is resized by.

    Returns:
        The same mapping, with the kinds living in image pixels resized.
    """
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
    """Rebuilds annotations with each named field passed through its map."""
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
    """Radius that draws a keypoint the same size whatever the view's size.

    Args:
        cameras_intrinsics: Intrinsics of the views being logged, read for
            their resolutions.

    Returns:
        A radius in pixels, of the tallest view logged.
    """
    heights = [
        annotation.resolution_hw[0] for annotation in cameras_intrinsics
    ]

    return max(MIN_KEYPOINTS_RADIUS, KEYPOINTS_RADIUS_RATIO * max(heights))


def rebase_on_global_frames(
    annotations: AnnotationsT,
    timeline: PreviewTimeline,
) -> AnnotationsT:
    """Re-indexes per-view annotations onto the timeline shared by the views.

    A view's annotations are numbered from the start of its own recording,
    which for a sequence whose cameras started at different moments means the
    same instant carries a different index in every view. Logging them as they
    are would scatter one instant across the timeline.

    Args:
        annotations: Annotations carrying a `view_id` and a local `frame_idx`.
        timeline: Placement of the recordings on the shared timeline.

    Returns:
        The same annotations, indexed by timeline step. Ones on a frame the
        timeline does not reach are dropped.
    """
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


def take_first_frames(
    annotations: AnnotationsT, n_frames: int
) -> AnnotationsT:
    """Keeps the annotations of the first `n_frames` timeline steps."""
    return type(annotations)(
        metadata=annotations.metadata,
        annotations=[
            annotation
            for annotation in annotations
            if annotation.frame_idx < n_frames
        ],
    )


def preview_sequence(
    sequence: KeypointsSequence,
    fps: float,
    output_path: str | None = None,
    max_frames: int | None = None,
    downscale_factor: int = DEFAULT_DOWNSCALE_FACTOR,
):
    """Logs a sequence and its ground truth to rerun.

    Each view's footage is carried into the recording whole, so a recording
    costs what the videos behind it cost, whatever `max_frames` says.

    Args:
        sequence: Sequence to preview, as a dataset yields it.
        fps: Rate the timeline steps are turned into timestamps with.
        output_path: `.rrd` file to write the recording to. The viewer is
            spawned instead when None.
        max_frames: Number of timeline steps to log, all of them when None.
            This shortens the timeline and the annotations on it, not the
            footage.
        downscale_factor: How much smaller than the dataset's the footage is
            shown, 1 for its own size. The annotations are resized with it, so
            they keep marking what they marked, in the preview's pixels rather
            than the dataset's.

    Raises:
        TypeError: If a view is backed by an unsupported frame loader.
        ValueError: If `downscale_factor` is below 1.
    """
    if downscale_factor < 1:
        raise ValueError(
            f"A preview cannot be larger than the dataset: downscale_factor "
            f"is {downscale_factor}."
        )

    scale = 1 / downscale_factor
    annotations = scale_pixel_space(sequence["annotations"] or {}, scale)

    time_reference = annotations.get("global_time_reference")
    if time_reference is not None:
        time_reference = time_reference.first_or_default()

    timeline = build_timeline(sequence["views_inputs"], time_reference)

    rr.init(sequence["sequence_name"], spawn=output_path is None)
    if output_path is not None:
        rr.save(output_path)

    rr.send_blueprint(
        sequence_blueprint(
            [view_input["view_id"] for view_input in sequence["views_inputs"]]
        )
    )

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
        # The 3D ground truth is numbered from the start of the annotated
        # window, which the pre-roll pushed down the timeline.
        keypoints_3d = _replace_fields(
            keypoints_3d, {"frame_idx": lambda idx: idx + timeline.lead_in}
        )

        if max_frames is not None:
            keypoints_3d = take_first_frames(keypoints_3d, max_frames)

        viz_rerun.log_skeletons_3d(
            keypoints_3d=keypoints_3d, prefix=GROUND_TRUTH_PREFIX, fps=fps
        )

    keypoints_2d = annotations.get("keypoints_2d")
    if keypoints_2d is not None:
        keypoints_2d = rebase_on_global_frames(keypoints_2d, timeline)

        if max_frames is not None:
            keypoints_2d = take_first_frames(keypoints_2d, max_frames)

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

        if max_frames is not None:
            bboxes_2d = take_first_frames(bboxes_2d, max_frames)

        viz_rerun.log_bboxes_2d(
            bboxes_2d=bboxes_2d, prefix=GROUND_TRUTH_PREFIX, fps=fps
        )

    log_recording_states(timeline, fps)

    for view_input in sequence["views_inputs"]:
        _log_view_frames(
            view_input, timeline, fps, max_frames, downscale_factor
        )


def _log_view_frames(
    view_input: ViewInput,
    timeline: PreviewTimeline,
    fps: float,
    max_frames: int | None,
    downscale_factor: int,
):
    """Logs a view's footage, embedding the files rather than decoding them."""
    view_id = view_input["view_id"]
    frame_loader = view_input["frame_loader"]
    frame_indices = timeline.local_by_step[view_id][:max_frames]

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
    """Prepares a video for the viewer, once, beside the original.

    A video is left alone when the viewer can decode it as it is and the
    preview shows it at its own size. Otherwise it is re-encoded, frames
    passed through so the frame a step points at is unchanged.
    """
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
    """Names a preview file after the size it was made for, so that a preview
    at another size cannot pick it up."""
    return f"preview_downscale_{downscale_factor}.mp4"


def _encoded_images(
    frame_loader: ImagesLoader, fps: float, downscale_factor: int
) -> str:
    """Encodes a view's images into a video beside them, once.

    Rerun carries the frames it is shown into the recording, and a sequence of
    full-resolution JPEGs costs two orders of magnitude more there than the
    same frames encoded.
    """
    video_path = os.path.join(
        os.path.dirname(frame_loader.img_paths[0]),
        _preview_name(downscale_factor),
    )

    if not os.path.exists(video_path):
        encode_images_to_video(
            frame_loader.img_paths, video_path, fps, 1 / downscale_factor
        )

    return video_path
