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
from typing import TypeVar

import rerun as rr
import torch

from kineo.annotations.annotations import Annotations
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

PREVIEW_VIDEO_NAME = "preview.mp4"

# What rerun's viewer can decode. Anything else is transcoded first.
VIEWABLE_CODECS = frozenset({"h264", "hevc", "av1", "vp9"})

AnnotationsT = TypeVar("AnnotationsT", bound=Annotations)


def local_frame_indices(
    view_id: str,
    n_frames: int,
    time_reference: GlobalTimeReferenceAnnotation | None,
) -> torch.Tensor:
    """Frame of a view's own recording each timeline step shows.

    Args:
        view_id: View the steps are resolved for.
        n_frames: Length of the view's recording, in frames.
        time_reference: Sequence's time reference, or None if its views are
            frame-aligned.

    Returns:
        One local frame index per timeline step.
    """
    if time_reference is None:
        return torch.arange(n_frames)

    return time_reference.closest_local_frame_idx[view_id]


def rebase_on_global_frames(
    annotations: AnnotationsT,
    time_reference: GlobalTimeReferenceAnnotation,
) -> AnnotationsT:
    """Re-indexes per-view annotations onto the timeline shared by the views.

    A view's annotations are numbered from the start of its own recording,
    which for a sequence whose cameras started at different moments means the
    same instant carries a different index in every view. Logging them as they
    are would scatter one instant across the timeline.

    Args:
        annotations: Annotations carrying a `view_id` and a local `frame_idx`.
        time_reference: Sequence's time reference.

    Returns:
        The same annotations, indexed by timeline step. Ones falling outside
        the window the time reference covers are dropped.
    """
    global_frames = {
        view_id: {
            local_frame_idx: global_frame_idx
            for global_frame_idx, local_frame_idx in enumerate(
                local_indices.tolist()
            )
        }
        for view_id, local_indices in (
            time_reference.closest_local_frame_idx.items()
        )
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

    Raises:
        TypeError: If a view is backed by an unsupported frame loader.
    """
    annotations = sequence["annotations"] or {}

    time_reference = annotations.get("global_time_reference")
    if time_reference is not None:
        time_reference = time_reference.first_or_default()

    rr.init(sequence["sequence_name"], spawn=output_path is None)
    if output_path is not None:
        rr.save(output_path)

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
        if max_frames is not None:
            keypoints_3d = take_first_frames(keypoints_3d, max_frames)

        viz_rerun.log_skeletons_3d(
            keypoints_3d=keypoints_3d, prefix=GROUND_TRUTH_PREFIX, fps=fps
        )

    keypoints_2d = annotations.get("keypoints_2d")
    if keypoints_2d is not None:
        if time_reference is not None:
            keypoints_2d = rebase_on_global_frames(keypoints_2d, time_reference)

        if max_frames is not None:
            keypoints_2d = take_first_frames(keypoints_2d, max_frames)

        viz_rerun.log_keypoints_2d(
            keypoints_2d=keypoints_2d, prefix=GROUND_TRUTH_PREFIX, fps=fps
        )

    bboxes_2d = annotations.get("bboxes_2d")
    if bboxes_2d is not None:
        if time_reference is not None:
            bboxes_2d = rebase_on_global_frames(bboxes_2d, time_reference)

        if max_frames is not None:
            bboxes_2d = take_first_frames(bboxes_2d, max_frames)

        viz_rerun.log_bboxes_2d(
            bboxes_2d=bboxes_2d, prefix=GROUND_TRUTH_PREFIX, fps=fps
        )

    for view_input in sequence["views_inputs"]:
        _log_view_frames(view_input, time_reference, fps, max_frames)


def _log_view_frames(
    view_input: ViewInput,
    time_reference: GlobalTimeReferenceAnnotation | None,
    fps: float,
    max_frames: int | None,
):
    """Logs a view's footage, embedding the files rather than decoding them."""
    view_id = view_input["view_id"]
    frame_loader = view_input["frame_loader"]
    frame_indices = local_frame_indices(
        view_id, len(frame_loader), time_reference
    )[:max_frames]

    if isinstance(frame_loader, VideoLoader):
        if frame_loader.selected_frames is not None:
            frame_indices = frame_loader.selected_frames[frame_indices]

        video_path = _viewable_video(frame_loader.video_path)
    elif isinstance(frame_loader, ImagesLoader):
        video_path = _encoded_images(frame_loader, fps)
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


def _viewable_video(video_path: str) -> str:
    """Transcodes a video the viewer cannot decode, once, beside the original.

    Frames are passed through, so the frame a step points at is unchanged.
    """
    if get_video_codec(video_path) in VIEWABLE_CODECS:
        return video_path

    transcoded_path = f"{os.path.splitext(video_path)[0]}_{PREVIEW_VIDEO_NAME}"

    if not os.path.exists(transcoded_path):
        transcode_video_to_h264(video_path, transcoded_path)

    return transcoded_path


def _encoded_images(frame_loader: ImagesLoader, fps: float) -> str:
    """Encodes a view's images into a video beside them, once.

    Rerun carries the frames it is shown into the recording, and a sequence of
    full-resolution JPEGs costs two orders of magnitude more there than the
    same frames encoded. The video keeps the images' resolution, so the
    annotations still land on the pixels they were measured on.
    """
    video_path = os.path.join(
        os.path.dirname(frame_loader.img_paths[0]), PREVIEW_VIDEO_NAME
    )

    if not os.path.exists(video_path):
        encode_images_to_video(frame_loader.img_paths, video_path, fps)

    return video_path
