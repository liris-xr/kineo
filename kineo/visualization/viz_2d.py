# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import cv2
import torch
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    BBox2DAnnotations,
    Keypoints2DAnnotations,
    Keypoints3DAnnotations,
    CameraExtrinsicsAnnotations,
    CameraIntrinsicsAnnotations,
    GlobalTimeReferenceAnnotations,
    GlobalTimeReferenceAnnotation,
)
import numpy as np
from tqdm import tqdm
from kineo.geometry.camera import (
    transform_points_from_world_to_camera,
    project_points_from_camera_to_image,
)
from kineo.visualization.utils import get_subject_color_rgba
import warnings


def show_reproj_keypoints(
    views: list[ViewInput],
    keypoints_3d: Keypoints3DAnnotations,
    camera_extrinsics: CameraExtrinsicsAnnotations,
    camera_intrinsics: CameraIntrinsicsAnnotations,
    global_time_reference: GlobalTimeReferenceAnnotations,
    fps: float,
    score_threshold: float = 0.2,
    keypoints_radius: int = 3,
    keypoints_border_width: int = 2,
    bone_width: int = 2,
):
    time_annot = global_time_reference.first_or_default()
    world_timestamps = time_annot.timestamps

    for view in views:
        view_id = view["view_id"]
        cv2.namedWindow(f"{view_id}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{view_id}", 1920, 1080)

    for frame_idx, t in tqdm(
        enumerate(world_timestamps),
        total=len(world_timestamps),
        desc="Showing 2D bboxes and reprojected 3D keypoints",
    ):
        kps_3d_annots = keypoints_3d.filter_by_frame_idx(frame_idx)

        for view in views:
            view_id = view["view_id"]
            view_frame_idx = time_annot.closest_local_frame_idx[view_id][frame_idx]
            view_camera_intrinsics = camera_intrinsics.filter_by_view_id(view_id)
            view_camera_extrinsics = camera_extrinsics.filter_by_view_id(view_id)

            frame = view["frame_loader"].load_frame_at(view_frame_idx)
            frame_rgb = frame.permute(1, 2, 0).cpu().numpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            view_camera_intrinsics = view_camera_intrinsics.first_or_default()
            view_camera_extrinsics = view_camera_extrinsics.first_or_default()

            K = view_camera_intrinsics.K
            Rt = view_camera_extrinsics.Rt
            dist_coeffs = view_camera_intrinsics.distortion_coefficients
            distortion_model = view_camera_intrinsics.distortion_model

            for subject in kps_3d_annots.subjects_ids:
                subject_color_rgba = get_subject_color_rgba(subject)
                subject_color_rgb = tuple(int(x * 255) for x in subject_color_rgba[:3])
                subject_color_bgr = tuple(reversed(subject_color_rgb))

                subject_kps_3d = kps_3d_annots.filter_by_subject_id(
                    subject
                ).first_or_default()

                kps_format = next(
                    (
                        kps_format
                        for kps_format in kps_3d_annots.metadata.formats
                        if kps_format.name == subject_kps_3d.format
                    ),
                    None,
                )

                if subject_kps_3d is not None:
                    points_3d_scores = subject_kps_3d.scores.cpu().numpy()

                    points_3d_camera = transform_points_from_world_to_camera(
                        points_3d_world=subject_kps_3d.xyz, Rt=Rt
                    )

                    points_2d_image, _ = project_points_from_camera_to_image(
                        points_3d_cam=points_3d_camera,
                        K=K,
                        D=dist_coeffs,
                        distortion_model=distortion_model.value,
                    )
                    points_2d_image = points_2d_image.cpu().numpy()

                    visibility = points_3d_scores > score_threshold

                    frame_bgr = draw_keypoints(
                        frame_bgr,
                        points_2d_image,
                        colors_bgr=subject_color_bgr,
                        visibility=visibility,
                        connectivity=kps_format.keypoints_connectivity,
                        connectivity_colors_bgr=subject_color_bgr,
                        keypoints_radius=keypoints_radius,
                        keypoints_border_width=keypoints_border_width,
                        bone_width=bone_width,
                    )

            cv2.imshow(f"{view_id}", frame_bgr)

        cv2.waitKey(int(1000 / fps))

    cv2.destroyAllWindows()


def show_bboxes_and_reproj_keypoints(
    views: list[ViewInput],
    bboxes_2d: BBox2DAnnotations,
    keypoints_3d: Keypoints3DAnnotations,
    camera_extrinsics: CameraExtrinsicsAnnotations,
    camera_intrinsics: CameraIntrinsicsAnnotations,
    fps: float,
    global_time_reference: GlobalTimeReferenceAnnotations,
    score_threshold: float = 0.2,
    keypoints_radius: int = 3,
    keypoints_border_width: int = 2,
    bboxes_border_width: int = 2,
    bone_width: int = 2,
):
    time_annot = global_time_reference.first_or_default()
    world_timestamps = time_annot.timestamps

    for view in views:
        view_id = view["view_id"]
        cv2.namedWindow(f"{view_id}", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{view_id}", 1920, 1080)

    for frame_idx, t in tqdm(
        enumerate(world_timestamps),
        total=len(world_timestamps),
        desc="Showing 2D bboxes and reprojected 3D keypoints",
    ):
        kps_3d_annots = keypoints_3d.filter_by_frame_idx(frame_idx)

        for view in views:
            view_id = view["view_id"]
            view_frame_idx = time_annot.closest_local_frame_idx[view_id][frame_idx]
            view_camera_intrinsics = camera_intrinsics.filter_by_view_id(view_id)
            view_camera_extrinsics = camera_extrinsics.filter_by_view_id(view_id)
            view_bboxes = bboxes_2d.filter_by_view_id(view_id).filter_by_frame_idx(
                frame_idx
            )

            frame = view["frame_loader"].load_frame_at(view_frame_idx)
            frame_rgb = frame.permute(1, 2, 0).cpu().numpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            view_camera_intrinsics = view_camera_intrinsics.first_or_default()
            view_camera_extrinsics = view_camera_extrinsics.first_or_default()

            K = view_camera_intrinsics.K
            Rt = view_camera_extrinsics.Rt
            dist_coeffs = view_camera_intrinsics.distortion_coefficients
            distortion_model = view_camera_intrinsics.distortion_model

            for subject in kps_3d_annots.subjects_ids:
                subject_color_rgba = get_subject_color_rgba(subject)
                subject_color_rgb = tuple(int(x * 255) for x in subject_color_rgba[:3])
                subject_color_bgr = tuple(reversed(subject_color_rgb))

                subject_kps_3d = kps_3d_annots.filter_by_subject_id(
                    subject
                ).first_or_default()

                subject_bboxes = view_bboxes.filter_by_subject_id(
                    subject
                ).first_or_default()

                kps_format = next(
                    (
                        kps_format
                        for kps_format in kps_3d_annots.metadata.formats
                        if kps_format.name == subject_kps_3d.format
                    ),
                    None,
                )

                if subject_bboxes is not None:
                    frame_bgr = draw_bboxes(
                        frame_bgr,
                        subject_bboxes.xyxy.cpu().numpy(),
                        colors_bgr=subject_color_bgr,
                        border_width=bboxes_border_width,
                    )

                if subject_kps_3d is not None:
                    points_3d_scores = subject_kps_3d.scores.cpu().numpy()

                    points_3d_camera = transform_points_from_world_to_camera(
                        points_3d_world=subject_kps_3d.xyz, Rt=Rt
                    )

                    points_2d_image, _ = project_points_from_camera_to_image(
                        points_3d_cam=points_3d_camera,
                        K=K,
                        D=dist_coeffs,
                        distortion_model=distortion_model.value,
                    )
                    points_2d_image = points_2d_image.cpu().numpy()

                    visibility = points_3d_scores > score_threshold

                    frame_bgr = draw_keypoints(
                        frame_bgr,
                        points_2d_image,
                        colors_bgr=subject_color_bgr,
                        visibility=visibility,
                        connectivity=kps_format.keypoints_connectivity,
                        connectivity_colors_bgr=subject_color_bgr,
                        keypoints_radius=keypoints_radius,
                        keypoints_border_width=keypoints_border_width,
                        bone_width=bone_width,
                    )

            cv2.imshow(f"{view_id}", frame_bgr)

        cv2.waitKey(int(1000 / fps))

    cv2.destroyAllWindows()


def show_bboxes_and_keypoints(
    views: list[ViewInput],
    bboxes_2d: BBox2DAnnotations,
    keypoints_2d: Keypoints2DAnnotations,
    global_time_reference: GlobalTimeReferenceAnnotations,
    score_threshold: float = 0.2,
    keypoints_radius: int = 3,
    keypoints_border_width: int = 2,
    bboxes_border_width: int = 2,
    bone_width: int = 2,
    window_size: tuple[int, int] = (1920 // 2, 1080 // 2),
    keypoints_indices: list[int] = None,
):
    global_time_reference: GlobalTimeReferenceAnnotation = (
        global_time_reference.first_or_default()
    )

    dt = torch.diff(global_time_reference.timestamps).mean().item()

    if dt > 0:
        target_fps = 1 / dt
    else:
        target_fps = 25
        warnings.warn(
            f"dt of {dt} seconds, which is less than 0.01 seconds. Setting target fps to 25."
        )

    kps_format = keypoints_2d.metadata.formats[0]

    if keypoints_indices is not None:
        keypoints_indices = torch.as_tensor(keypoints_indices).reshape(-1)
    else:
        keypoints_indices = torch.arange(kps_format.n_keypoints).reshape(-1)

    kp_indices_map = {
        old_idx.item(): new_idx for new_idx, old_idx in enumerate(keypoints_indices)
    }
    old_connectivity = kps_format.keypoints_connectivity
    new_connectivity = [
        (kp_indices_map[kp1], kp_indices_map[kp2])
        for kp1, kp2 in old_connectivity
        if kp1 in kp_indices_map and kp2 in kp_indices_map
    ]

    n_frames = len(global_time_reference.timestamps)

    for frame_idx in tqdm(
        range(n_frames),
        total=n_frames,
        desc="Showing 2D bboxes and 2D infered keypoints",
    ):
        for view in views:
            view_id = view["view_id"]

            # Load closest frame in the video (global frame indices are not the same as video local frame indices)
            view_frame_idx = global_time_reference.closest_local_frame_idx[view_id][
                frame_idx
            ]
            frame = view["frame_loader"].load_frame_at(view_frame_idx)
            frame_rgb = frame.permute(1, 2, 0).cpu().numpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            view_bboxes = bboxes_2d.filter_by_view_id(view_id).filter_by_frame_idx(
                frame_idx
            )
            view_keypoints = keypoints_2d.filter_by_view_id(
                view_id
            ).filter_by_frame_idx(frame_idx)

            for subject in view_keypoints.subjects_ids:
                subject_color_rgba = get_subject_color_rgba(subject)
                subject_color_rgb = tuple(
                    int(x * 255) for x in subject_color_rgba[:3]
                )
                subject_color_bgr = tuple(reversed(subject_color_rgb))

                subject_keypoints = view_keypoints.filter_by_subject_id(
                    subject
                ).first_or_default()
                subject_bboxes = view_bboxes.filter_by_subject_id(
                    subject
                ).first_or_default()

                kps_format = next(
                    (
                        kps_format
                        for kps_format in keypoints_2d.metadata.formats
                        if kps_format.name == subject_keypoints.format
                    ),
                    None,
                )

                if subject_bboxes is not None:
                    frame_bgr = draw_bboxes(
                        frame_bgr,
                        subject_bboxes.xyxy.cpu().numpy(),
                        colors_bgr=subject_color_bgr,
                        border_width=bboxes_border_width,
                    )

                if subject_keypoints is not None:
                    keypoints_xy = subject_keypoints.xy.cpu().numpy()
                    visibility = (
                        subject_keypoints.scores.cpu().numpy() > score_threshold
                    )

                    keypoints_xy = keypoints_xy[keypoints_indices].reshape(-1, 2)
                    visibility = visibility[keypoints_indices].reshape(-1)

                    frame_bgr = draw_keypoints(
                        frame_bgr,
                        keypoints_xy,
                        visibility=visibility,
                        connectivity=new_connectivity,
                        connectivity_colors_bgr=subject_color_bgr,
                        colors_bgr=subject_color_bgr,
                        keypoints_radius=keypoints_radius,
                        keypoints_border_width=keypoints_border_width,
                        bone_width=bone_width,
                    )

            cv2.namedWindow(f"{view_id}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{view_id}", window_size[0], window_size[1])
            cv2.imshow(f"{view_id}", frame_bgr)

        cv2.waitKey(int(1000 / target_fps))

    cv2.destroyAllWindows()


def draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    colors_bgr: list[tuple[int, int, int]] | tuple[int, int, int] = (0, 0, 255),
    border_colors_bgr: list[tuple[int, int, int]] | tuple[int, int, int] = (
        255,
        255,
        255,
    ),
    connectivity: list[tuple[int, int]] = None,
    connectivity_colors_bgr: list[tuple[int, int, int]] | tuple[int, int, int] = (
        0,
        0,
        255,
    ),
    keypoints_radius: int = 5,
    keypoints_border_width: int = 2,
    bone_width: int = 2,
    visibility: np.ndarray | None = None,
):
    if isinstance(colors_bgr, tuple):
        colors_bgr = [colors_bgr] * len(keypoints)

    if isinstance(connectivity_colors_bgr, tuple) and connectivity is not None:
        connectivity_colors_bgr = [connectivity_colors_bgr] * len(connectivity)

    if isinstance(border_colors_bgr, tuple):
        border_colors_bgr = [border_colors_bgr] * len(keypoints)

    if visibility is None:
        visibility = np.ones(len(keypoints), dtype=bool)

    if keypoints.ndim == 1:
        keypoints = keypoints[None, :]

    if keypoints.ndim != 2:
        raise ValueError(f"Keypoints must be of shape (N, 2), got {keypoints.shape}")

    if connectivity is not None:
        for (kp1, kp2), color_bgr in zip(connectivity, connectivity_colors_bgr):
            # Only draw the line if both keypoints are visible
            if (
                visibility[kp1]
                and visibility[kp2]
                and np.isfinite(keypoints[kp1]).all()
                and np.isfinite(keypoints[kp2]).all()
            ):
                cv2.line(
                    frame,
                    (int(keypoints[kp1][0]), int(keypoints[kp1][1])),
                    (int(keypoints[kp2][0]), int(keypoints[kp2][1])),
                    color_bgr,
                    bone_width,
                    lineType=cv2.LINE_AA,
                )

    for kp, color_bgr, border_color_bgr, vis in zip(
        keypoints, colors_bgr, border_colors_bgr, visibility
    ):
        if vis and np.isfinite(kp).all():
            cv2.circle(
                frame,
                (int(kp[0]), int(kp[1])),
                keypoints_radius + keypoints_border_width,
                border_color_bgr,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                (int(kp[0]), int(kp[1])),
                keypoints_radius,
                color_bgr,
                -1,
                lineType=cv2.LINE_AA,
            )
    return frame


def draw_bboxes(
    frame: np.ndarray,
    bboxes: np.ndarray,
    colors_bgr: list[tuple[int, int, int]] | tuple[int, int, int] = (0, 0, 255),
    border_width: int = 2,
):
    if isinstance(colors_bgr, tuple):
        colors_bgr = [colors_bgr] * len(bboxes)

    if bboxes.ndim == 1:
        bboxes = bboxes[None, :]

    if bboxes.ndim != 2:
        raise ValueError(f"Bboxes must be of shape (N, 4), got {bboxes.shape}")

    for bbox, color in zip(bboxes, colors_bgr):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            border_width,
            lineType=cv2.LINE_AA,
        )
    return frame
