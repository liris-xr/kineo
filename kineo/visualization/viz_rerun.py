# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Rerun logging of a sequence's cameras, keypoints, skeletons and footage.

Each function logs one kind of structured data under the entity path its kind
owns, so a recording is composed by calling the ones a caller has data for.
"""

from collections.abc import Iterable, Sequence

import numpy as np
import rerun as rr
import torch
from tqdm import tqdm

from kineo.annotations.bboxes_2d import BBox2DAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.geometry.transformations import inverse_Rt
from kineo.visualization.utils import get_subject_color_rgba

try:
    import av
except ImportError:
    av = None


def clear_after(entity_paths: Iterable[str], step: int, fps: float):
    """Ends entities at a step, so nothing outlives the data it came from.

    Rerun holds the last value logged to an entity until something replaces
    it, which for annotations covering part of a sequence means the final
    pose hanging over every later frame.

    Args:
        entity_paths: Entities to end.
        step: Timeline step they stop at.
        fps: Rate the steps are turned into timestamps with.
    """
    rr.set_time("frame_idx", sequence=step)
    rr.set_time("time", timestamp=step / fps)

    for entity_path in entity_paths:
        rr.log(entity_path, rr.Clear(recursive=False))


def log_cameras(
        cameras_extrinsics: CameraExtrinsicsAnnotations,
        cameras_intrinsics: CameraIntrinsicsAnnotations,
        prefix: str = "kineo",
        image_plane_distance: float = 0.2,
):
    views_ids = cameras_extrinsics.views_ids

    for view_id in views_ids:
        pred_camera_extrinsics = cameras_extrinsics.filter_by_view_id(
            view_id
        ).first_or_default()
        pred_camera_intrinsics = cameras_intrinsics.filter_by_view_id(
            view_id
        ).first_or_default()

        pred_Rt_inv = inverse_Rt(pred_camera_extrinsics.Rt).cpu().numpy()
        translation = pred_Rt_inv[:3, 3]
        rotation = pred_Rt_inv[:3, :3]
        height, width = pred_camera_intrinsics.resolution_hw

        rr.set_time("frame_idx", sequence=0)
        rr.set_time("time", timestamp=0)
        rr.log(
            f"{prefix}/cameras/{view_id}",
            rr.Pinhole(
                image_from_camera=pred_camera_intrinsics.K.cpu().numpy(),
                width=width,
                height=height,
                image_plane_distance=image_plane_distance,
            ),
        )
        rr.log(
            f"{prefix}/cameras/{view_id}",
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation,
            ),
        )


def log_keypoints_2d(
        keypoints_2d: Keypoints2DAnnotations,
        prefix: str = "kineo",
        radius: float = 0.01,
        color_override: tuple[float, float, float] | None = None,
        min_kps_score_2d: float = 0.3,
        fps: float = 25,
        start_frame_idx: int = 0,
        end_frame_idx: int = -1,
):
    frames = keypoints_2d.frames
    formats = keypoints_2d.metadata.formats

    views_ids = keypoints_2d.view_ids
    n_views = len(views_ids)
    n_subjects = len(keypoints_2d.subjects_ids)
    n_keypoints = formats[0].n_keypoints
    subject_id_to_idx = {subject_id: i for i, subject_id in enumerate(keypoints_2d.subjects_ids)}
    view_id_to_idx = {view_id: i for i, view_id in enumerate(views_ids)}

    if color_override is not None:
        subject_colors = {subject_id: color_override + (1,) for subject_id in keypoints_2d.subjects_ids}
    else:
        subject_colors = {subject_id: get_subject_color_rgba(subject_id) for subject_id in keypoints_2d.subjects_ids}

    for og_frame_idx in tqdm(frames, desc="Logging 2D keypoints", leave=False):
        if og_frame_idx < start_frame_idx or (end_frame_idx != -1 and og_frame_idx > end_frame_idx):
            continue

        frame_idx = og_frame_idx - start_frame_idx
        frame_keypoints_2d = keypoints_2d.filter_by_frame_idx(og_frame_idx)
        frame_timestamp = frame_idx / fps

        kps_xy = torch.zeros((n_views, n_subjects, n_keypoints, 2))
        kps_2d_scores = torch.zeros((n_views, n_subjects, n_keypoints))

        for annotation in frame_keypoints_2d:
            subject_idx = subject_id_to_idx[annotation.subject_id]
            view_idx = view_id_to_idx[annotation.view_id]
            kps_xy[view_idx, subject_idx] = annotation.xy
            kps_2d_scores[view_idx, subject_idx] = annotation.scores

        rr.set_time("frame_idx", sequence=frame_idx)
        rr.set_time("time", timestamp=frame_timestamp)

        for subject_idx in range(n_subjects):
            subject_id = keypoints_2d.subjects_ids[subject_idx]

            for view_idx in range(n_views):
                view_id = views_ids[view_idx]
                subject_kps_2d_valid_indices = (kps_2d_scores[view_idx, subject_idx] > min_kps_score_2d).nonzero()
                subject_valid_kps_xy = kps_xy[view_idx, subject_idx][subject_kps_2d_valid_indices]
                rr.log(
                    f"{prefix}/cameras/{view_id}/keypoints_2d_{subject_id}",
                    rr.Points2D(
                        positions=subject_valid_kps_xy.cpu().numpy(),
                        colors=[int(c * 255) for c in subject_colors[subject_id]],
                        radii=radius,
                    ),
                )


def log_keypoints_3d(
        keypoints_3d: Keypoints3DAnnotations,
        prefix: str = "kineo",
        radius: float = 0.01,
        color_override: tuple[float, float, float] | None = None,
        fps: float = 25,
        min_kps_score_3d: float = 0.5,
        start_frame_idx: int = 0,
        end_frame_idx: int = -1,
):
    frames = keypoints_3d.frames
    formats = keypoints_3d.metadata.formats

    n_subjects = len(keypoints_3d.subjects_ids)
    n_keypoints = formats[0].n_keypoints
    subject_id_to_idx = {subject_id: i for i, subject_id in enumerate(keypoints_3d.subjects_ids)}

    if color_override is not None:
        subject_colors = {subject_id: color_override + (1,) for subject_id in keypoints_3d.subjects_ids}
    else:
        subject_colors = {subject_id: get_subject_color_rgba(subject_id) for subject_id in keypoints_3d.subjects_ids}

    for og_frame_idx in tqdm(frames, desc="Logging keypoints 3D", leave=False):
        if og_frame_idx < start_frame_idx or (end_frame_idx != -1 and og_frame_idx > end_frame_idx):
            continue

        frame_idx = og_frame_idx - start_frame_idx
        frame_keypoints_3d = keypoints_3d.filter_by_frame_idx(og_frame_idx)

        frame_timestamp = frame_idx / fps

        kps_xyz = torch.zeros((n_subjects, n_keypoints, 3))
        kps_3d_scores = torch.zeros((n_subjects, n_keypoints))

        for annotation in frame_keypoints_3d:
            subject_idx = subject_id_to_idx[annotation.subject_id]
            kps_xyz[subject_idx] = annotation.xyz
            kps_3d_scores[subject_idx] = annotation.scores

        rr.set_time("frame_idx", sequence=frame_idx)
        rr.set_time("time", timestamp=frame_timestamp)

        for subject_idx in range(n_subjects):
            subject_id = keypoints_3d.subjects_ids[subject_idx]

            subject_kps_3d_valid_indices = (kps_3d_scores[subject_idx] > min_kps_score_3d).nonzero()
            subject_valid_kps_xyz = kps_xyz[subject_idx][subject_kps_3d_valid_indices]

            rr.log(
                f"{prefix}/keypoints_3d_{subject_id}",
                rr.Points3D(
                    positions=subject_valid_kps_xyz.cpu().numpy(),
                    colors=[int(c * 255) for c in subject_colors[subject_id]],
                    radii=radius,
                ),
            )


def log_skeletons_2d(
        keypoints_2d: Keypoints2DAnnotations,
        prefix: str = "kineo",
        joint_radius: float = 0.01,
        bones_thickness: float = 0.01,
        color_override: tuple[float, float, float] | None = None,
        min_kps_score_2d: float = 0.3,
        fps: float = 25,
        log_disconnected_joints: bool = False,
        start_frame_idx: int = 0,
        end_frame_idx: int = -1,
):
    frames = keypoints_2d.frames
    formats = keypoints_2d.metadata.formats

    views_ids = keypoints_2d.view_ids
    n_views = len(views_ids)
    n_subjects = len(keypoints_2d.subjects_ids)
    n_keypoints = formats[0].n_keypoints
    subject_id_to_idx = {subject_id: i for i, subject_id in enumerate(keypoints_2d.subjects_ids)}
    view_id_to_idx = {view_id: i for i, view_id in enumerate(views_ids)}

    if color_override is not None:
        subject_colors = {subject_id: color_override + (1,) for subject_id in keypoints_2d.subjects_ids}
    else:
        subject_colors = {subject_id: get_subject_color_rgba(subject_id) for subject_id in keypoints_2d.subjects_ids}

    kps_connectivity = formats[0].keypoints_connectivity
    connected_joints_indices = set(i for i, j in kps_connectivity) | set(j for i, j in kps_connectivity)
    connected_joints_indices = torch.tensor(list(connected_joints_indices))

    last_step = 0

    for og_frame_idx in tqdm(frames, desc="Logging skeleton 2D", leave=False):
        if og_frame_idx < start_frame_idx or (end_frame_idx != -1 and og_frame_idx > end_frame_idx):
            continue

        frame_idx = og_frame_idx - start_frame_idx
        frame_keypoints_2d = keypoints_2d.filter_by_frame_idx(og_frame_idx)
        frame_timestamp = frame_idx / fps

        kps_xy = torch.zeros((n_views, n_subjects, n_keypoints, 2))
        kps_2d_scores = torch.zeros((n_views, n_subjects, n_keypoints))

        for annotation in frame_keypoints_2d:
            subject_idx = subject_id_to_idx[annotation.subject_id]
            view_idx = view_id_to_idx[annotation.view_id]
            kps_xy[view_idx, subject_idx] = annotation.xy
            kps_2d_scores[view_idx, subject_idx] = annotation.scores

        last_step = frame_idx
        rr.set_time("frame_idx", sequence=frame_idx)
        rr.set_time("time", timestamp=frame_timestamp)

        for subject_idx in range(n_subjects):
            subject_id = keypoints_2d.subjects_ids[subject_idx]

            for view_idx in range(n_views):
                view_id = views_ids[view_idx]
                subject_kps_2d_valid_indices = (kps_2d_scores[view_idx, subject_idx] > min_kps_score_2d).nonzero()

                # Keep only indices present in the connectivity list
                if not log_disconnected_joints:
                    mask = torch.isin(subject_kps_2d_valid_indices, connected_joints_indices)
                    subject_kps_2d_valid_indices = subject_kps_2d_valid_indices[mask]

                subject_valid_kps_xy = kps_xy[view_idx, subject_idx][subject_kps_2d_valid_indices]
                rr.log(
                    f"{prefix}/cameras/{view_id}/skeletons_2d_joints_{subject_id}",
                    rr.Points2D(
                        positions=subject_valid_kps_xy.cpu().numpy(),
                        colors=[int(c * 255) for c in subject_colors[subject_id]],
                        radii=joint_radius,
                    ),
                )
                line_strips_2d = []
                for connection in kps_connectivity:
                    i, j = connection
                    if i in subject_kps_2d_valid_indices and j in subject_kps_2d_valid_indices:
                        line_strips_2d.append([kps_xy[view_idx, subject_idx][i].cpu().numpy(),
                                               kps_xy[view_idx, subject_idx][j].cpu().numpy()])

                line_strips_2d = np.array(line_strips_2d).reshape(-1, 2, 2)  # (N, 2, 2)

                rr.log(
                    f"{prefix}/cameras/{view_id}/skeletons_2d_bones_{subject_id}",
                    rr.LineStrips2D(
                        strips=line_strips_2d,
                        colors=[int(c * 255) for c in subject_colors[subject_id]],
                        radii=bones_thickness,
                    ),
                )

    if frames:
        clear_after(
            (
                f"{prefix}/cameras/{view_id}/skeletons_2d_{part}_{subject_id}"
                for view_id in views_ids
                for subject_id in keypoints_2d.subjects_ids
                for part in ("joints", "bones")
            ),
            last_step + 1,
            fps,
        )


def log_skeletons_3d(
        keypoints_3d: Keypoints3DAnnotations,
        prefix: str = "kineo",
        joint_radius: float = 0.01,
        bones_thickness: float = 0.01,
        color_override: tuple[float, float, float] | None = None,
        fps: float = 25,
        min_kps_score_3d: float = 0.5,
        log_disconnected_joints: bool = False,
        start_frame_idx: int = 0,
        end_frame_idx: int = -1,
):
    frames = keypoints_3d.frames
    formats = keypoints_3d.metadata.formats

    n_subjects = len(keypoints_3d.subjects_ids)
    n_keypoints = formats[0].n_keypoints
    subject_id_to_idx = {subject_id: i for i, subject_id in enumerate(keypoints_3d.subjects_ids)}

    if color_override is not None:
        subject_colors = {subject_id: color_override + (1,) for subject_id in keypoints_3d.subjects_ids}
    else:
        subject_colors = {subject_id: get_subject_color_rgba(subject_id) for subject_id in keypoints_3d.subjects_ids}

    kps_connectivity = formats[0].keypoints_connectivity
    connected_joints_indices = set(i for i, j in kps_connectivity) | set(j for i, j in kps_connectivity)
    connected_joints_indices = torch.tensor(list(connected_joints_indices))

    last_step = 0

    for og_frame_idx in tqdm(frames, desc="Logging skeleton 3D", leave=False):
        if og_frame_idx < start_frame_idx or (end_frame_idx != -1 and og_frame_idx > end_frame_idx):
            continue

        frame_idx = og_frame_idx - start_frame_idx

        frame_keypoints_3d = keypoints_3d.filter_by_frame_idx(og_frame_idx)

        frame_timestamp = frame_idx / fps

        kps_xyz = torch.zeros((n_subjects, n_keypoints, 3))
        kps_3d_scores = torch.zeros((n_subjects, n_keypoints))

        for annotation in frame_keypoints_3d:
            subject_idx = subject_id_to_idx[annotation.subject_id]
            kps_xyz[subject_idx] = annotation.xyz
            kps_3d_scores[subject_idx] = annotation.scores

        last_step = frame_idx
        rr.set_time("frame_idx", sequence=frame_idx)
        rr.set_time("time", timestamp=frame_timestamp)

        for subject_idx in range(n_subjects):
            subject_id = keypoints_3d.subjects_ids[subject_idx]

            subject_kps_3d_valid_indices = (kps_3d_scores[subject_idx] > min_kps_score_3d).nonzero()
            subject_valid_kps_xyz = kps_xyz[subject_idx][subject_kps_3d_valid_indices]

            if not log_disconnected_joints:
                mask = torch.isin(subject_kps_3d_valid_indices, connected_joints_indices)
                subject_kps_3d_valid_indices = subject_kps_3d_valid_indices[mask]
                subject_valid_kps_xyz = subject_valid_kps_xyz[mask]

            rr.log(
                f"{prefix}/skeletons_3d_joints_{subject_id}",
                rr.Points3D(
                    positions=subject_valid_kps_xyz.cpu().numpy(),
                    colors=[int(c * 255) for c in subject_colors[subject_id]],
                    radii=joint_radius,
                ),
            )

            line_strips_3d = []
            for connection in kps_connectivity:
                i, j = connection
                if i in subject_kps_3d_valid_indices and j in subject_kps_3d_valid_indices:
                    line_strips_3d.append(
                        [kps_xyz[subject_idx][i].cpu().numpy(), kps_xyz[subject_idx][j].cpu().numpy()])

            line_strips_3d = np.array(line_strips_3d).reshape(-1, 2, 3)  # (N, 2, 3)

            rr.log(
                f"{prefix}/skeletons_3d_bones_{subject_id}",
                rr.LineStrips3D(
                    strips=line_strips_3d,
                    colors=[int(c * 255) for c in subject_colors[subject_id]],
                    radii=bones_thickness,
                ),
            )

    if frames:
        clear_after(
            (
                f"{prefix}/skeletons_3d_{part}_{subject_id}"
                for subject_id in keypoints_3d.subjects_ids
                for part in ("joints", "bones")
            ),
            last_step + 1,
            fps,
        )


def quality_to_crf(quality: int, min_crf: int = 18, max_crf: int = 30) -> int:
    quality = max(0, min(100, quality))
    return round(max_crf - (max_crf - min_crf) * (quality / 100))


def log_videos(
        views: list[ViewInput],
        prefix: str,
        global_time_reference: GlobalTimeReferenceAnnotation,
        video_quality: int = 75,
        start_frame_idx: int = 0,
        end_frame_idx: int = -1,
):
    if av is None:
        raise ImportError("av is not installed. Videos cannot be logged.")

    # Some versions of rerun don't have support for video yet.
    # Don't log the videos in that case.
    try:
        codec = rr.VideoCodec.H264
    except (AttributeError, NameError):
        raise Exception("Can't find rerun H264 codec. Video will not be logged.")

    format = "h264"
    encoder = "libx264"

    n_frames = len(global_time_reference.timestamps)
    fps = int(global_time_reference.fps)

    for view in views:
        view_id = view["view_id"]
        view_frame_loader = view["frame_loader"]
        view_resolution_hw = view_frame_loader.resolution_hw

        av.logging.set_level(av.logging.VERBOSE)
        container = av.open("/dev/null", "w", format=format)  # Use AnnexB H.265 stream.
        stream = container.add_stream(encoder, rate=fps)
        stream.width = view_resolution_hw[1]
        stream.height = view_resolution_hw[0]
        # TODO(#10090): Rerun Video Streams don't support b-frames yet.
        # Note that b-frames are generally not recommended for low-latency streaming and may make logging more complex.
        stream.max_b_frames = 0
        stream.options = {
            "crf": str(quality_to_crf(video_quality)),
        }

        entity_path = f"{prefix}/cameras/{view_id}/rgb"
        rr.log(entity_path, rr.VideoStream(codec=codec), static=True)

        for og_frame_idx in tqdm(range(n_frames), desc="Logging videos", leave=False):
            if og_frame_idx < start_frame_idx or (end_frame_idx != -1 and og_frame_idx > end_frame_idx):
                continue

            frame_idx = og_frame_idx - start_frame_idx
            frame_timestamp = frame_idx / fps

            rr.set_time("frame_idx", sequence=frame_idx)
            rr.set_time("time", timestamp=frame_timestamp)

            view_frame_idx = global_time_reference.closest_local_frame_idx[
                view_id
            ][og_frame_idx].item()

            view_frame_rgb = view_frame_loader.load_frame_at(view_frame_idx)
            view_frame_rgb = view_frame_rgb.permute(1, 2, 0).cpu().numpy()

            frame = av.VideoFrame.from_ndarray(view_frame_rgb, format="rgb24")
            for packet in stream.encode(frame):
                if packet.pts is None:
                    continue
                rr.set_time("time", duration=float(packet.pts * packet.time_base))
                rr.log(entity_path, rr.VideoStream.from_fields(sample=bytes(packet)))


def log_bboxes_2d(
    bboxes_2d: BBox2DAnnotations,
    prefix: str = "kineo",
    fps: float = 25.0,
):
    """Logs 2D bounding boxes under the camera they were annotated in.

    Args:
        bboxes_2d: Boxes to log, over any number of views and subjects.
        prefix: Root entity path the boxes are logged under.
        fps: Rate the frame indices are turned into timestamps with.
    """
    entity_paths = set()

    for frame_idx in bboxes_2d.frames:
        rr.set_time("frame_idx", sequence=frame_idx)
        rr.set_time("time", timestamp=frame_idx / fps)

        for annotation in bboxes_2d.filter_by_frame_idx(frame_idx):
            color = get_subject_color_rgba(annotation.subject_id)
            entity_path = (
                f"{prefix}/cameras/{annotation.view_id}"
                f"/bboxes_2d_{annotation.subject_id}"
            )
            entity_paths.add(entity_path)
            rr.log(
                entity_path,
                rr.Boxes2D(
                    array=annotation.xyxy.cpu().numpy(),
                    array_format=rr.Box2DFormat.XYXY,
                    colors=[int(c * 255) for c in color],
                ),
            )

    if entity_paths:
        clear_after(entity_paths, max(bboxes_2d.frames) + 1, fps)
def log_video_asset(
    video_path: str,
    view_id: str,
    local_frame_indices: Sequence[int],
    prefix: str = "kineo",
    fps: float = 25.0,
):
    """Shows a video file in a camera, without decoding or re-encoding it.

    The encoded file is carried into the recording as it is and every timeline
    step points at one of its frames, so the footage keeps the quality it was
    stored at and the recording grows by the size of the file.

    Args:
        video_path: Video file, decoded by the viewer rather than here.
        view_id: View the video belongs to.
        local_frame_indices: Frame of the video each timeline step shows, -1
            for a step the recording does not cover.
        prefix: Root entity path the video is logged under.
        fps: Rate the timeline steps are turned into timestamps with.
    """
    entity_path = f"{prefix}/cameras/{view_id}/rgb"
    video = rr.AssetVideo(path=video_path)
    rr.log(entity_path, video, static=True)

    frame_timestamps_ns = video.read_frame_timestamps_nanos()

    last_step = 0

    for frame_idx, local_frame_idx in enumerate(local_frame_indices):
        if local_frame_idx < 0:
            continue

        last_step = frame_idx
        rr.set_time("frame_idx", sequence=frame_idx)
        rr.set_time("time", timestamp=frame_idx / fps)
        rr.log(
            entity_path,
            rr.VideoFrameReference(
                nanoseconds=frame_timestamps_ns[local_frame_idx]
            ),
        )

    clear_after([entity_path], last_step + 1, fps)
