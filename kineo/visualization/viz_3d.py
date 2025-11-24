# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from kineo.annotations import (
    CameraExtrinsicsAnnotations,
    CameraIntrinsicsAnnotations,
    Keypoints3DAnnotations,
)
from kineo.annotations.reconstructed_scene import (
    WorldReconstructedSceneAnnotation,
    WorldReconstructedSceneAnnotations,
)
import torch
from kineo.visualization.aitviewer.skeleton import (
    SkeletonWithVisibility,
    SkeletonWithConfidence,
)
from kineo.geometry.conversions import (
    convert_world_points_from_opencv_to_opengl,
    convert_Rt_from_opencv_to_opengl,
)
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.viewer import Viewer
from aitviewer.headless import HeadlessRenderer
from aitviewer.configuration import CONFIG as C
from aitviewer.scene.node import Node
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import SMPLLayer, SMPLSequence
from aitviewer.renderables.spheres import Spheres
import os
import trimesh

import roma
from kineo.geometry.conversions import OPENCV_CAMERA_UP_VECTOR
from kineo.geometry.transformations import apply_similarity_transform_to_points

from kineo.visualization.utils import get_subject_color_rgba
from kineo.annotations.smpl_params import (
    SMPLShapeAnnotations,
    SMPLPoseAnnotations,
    SMPLPoseFormat,
)

C.window_type = "pyglet"
C.smplx_models = "./body_models/"


def render_keypoints_and_cameras(
    keypoints_3d: Keypoints3DAnnotations,
    camera_extrinsics: CameraExtrinsicsAnnotations,
    camera_intrinsics: CameraIntrinsicsAnnotations,
    fps: float,
    video_path: str,
    skeleton_radius: float = 0.01,
    score_threshold: float = 0.2,
):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    frames = set(keypoints_3d.frames)
    max_frame = max(frames)
    n_frames = max_frame + 1

    subjects = keypoints_3d.subjects_ids

    skeletons = []
    cameras = []

    for subject in subjects:
        subject_color_rgba = get_subject_color_rgba(subject)

        subject_kps_3d_annots = keypoints_3d.filter_by_subject_id(subject)
        kps_3d_format_name = subject_kps_3d_annots.first_or_default().format

        kps_3d_format = next(
            (
                kps_3d_format
                for kps_3d_format in subject_kps_3d_annots.metadata.formats
                if kps_3d_format.name == kps_3d_format_name
            ),
            None,
        )

        keypoints = torch.zeros(
            (n_frames, kps_3d_format.n_keypoints, 3), device=torch.device("cpu")
        )
        visibility = torch.zeros(
            (n_frames, kps_3d_format.n_keypoints),
            dtype=torch.bool,
            device=torch.device("cpu"),
        )

        for annot in subject_kps_3d_annots:
            frame_idx = annot.frame_idx
            keypoints[frame_idx, :, :] = annot.xyz
            visibility[frame_idx, :] = annot.scores >= score_threshold

        # To OpenGL coordinate system
        keypoints = convert_world_points_from_opencv_to_opengl(keypoints)

        skeleton = SkeletonWithVisibility(
            joint_positions=keypoints.cpu().numpy(),
            joint_connections=kps_3d_format.keypoints_connectivity,
            joint_visibility=visibility.cpu().numpy(),
            colors=subject_color_rgba,
            radius=skeleton_radius,
        )
        skeleton.name = subject
        skeletons.append(skeleton)

    for view_id in camera_extrinsics.views_ids:
        view_camera_extrinsics = camera_extrinsics.filter_by_view_id(
            view_id
        ).first_or_default()
        view_camera_intrinsics = camera_intrinsics.filter_by_view_id(
            view_id
        ).first_or_default()

        K = view_camera_intrinsics.K
        Rt = view_camera_extrinsics.Rt

        # To OpenGL coordinate system
        Rt = convert_Rt_from_opencv_to_opengl(Rt)

        cam_up = OPENCV_CAMERA_UP_VECTOR.to(device=Rt.device)
        cam_up_world = Rt[:3, :3].T @ cam_up
        cam_up_world = cam_up_world / cam_up_world.norm()

        dist_coeffs = view_camera_intrinsics.distortion_coefficients
        h, w = view_camera_intrinsics.resolution_hw

        K = K.cpu().numpy()
        Rt = Rt.cpu().numpy()
        dist_coeffs = dist_coeffs.cpu().numpy()

        camera = OpenCVCamera(K=K, Rt=Rt, dist_coeffs=dist_coeffs, cols=w, rows=h)
        camera.name = view_id
        cameras.append(camera)

    headless_renderer = HeadlessRenderer()
    headless_renderer.window.init_mgl_context()
    headless_renderer.reset()
    headless_renderer.scene.origin.enabled = False

    for skeleton in skeletons:
        headless_renderer.scene.add(skeleton)
    for camera in cameras:
        headless_renderer.scene.add(camera)

    headless_renderer.playback_fps = fps
    headless_renderer.run_animations = True
    headless_renderer.save_video(video_dir=video_path)


def _create_subject_skeleton_with_confidence(
    subject_id: str,
    keypoints_3d: Keypoints3DAnnotations,
    skeleton_radius: float = 0.01,
    colormap: str = "inferno",
    convert_coordinates_to_opengl: bool = True,
    keypoints_to_show: list[int] | None = None,
    keypoints_to_hide: list[int] | None = None,
    normalize_skeleton_confidence_per_frame: bool = False,
) -> SkeletonWithConfidence:
    n_frames = keypoints_3d.n_frames

    subject_kps_3d_annots = keypoints_3d.filter_by_subject_id(subject_id)
    kps_3d_format_name = subject_kps_3d_annots.first_or_default().format

    kps_3d_format = next(
        (
            kps_3d_format
            for kps_3d_format in subject_kps_3d_annots.metadata.formats
            if kps_3d_format.name == kps_3d_format_name
        ),
        None,
    )

    keypoints = torch.zeros(
        (n_frames, kps_3d_format.n_keypoints, 3), device=torch.device("cpu")
    )
    confidences = torch.zeros(
        (n_frames, kps_3d_format.n_keypoints),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    for annot in subject_kps_3d_annots:
        frame_idx = annot.frame_idx
        keypoints[frame_idx, :, :] = annot.xyz
        confidences[frame_idx, :] = annot.scores

    # To OpenGL coordinate system
    if convert_coordinates_to_opengl:
        keypoints = convert_world_points_from_opencv_to_opengl(keypoints)

    if keypoints_to_show is not None:
        keypoints_to_show = set(keypoints_to_show)
    else:
        keypoints_to_show = set(range(kps_3d_format.n_keypoints))

    if keypoints_to_hide is not None:
        keypoints_to_hide = set(keypoints_to_hide)
    else:
        keypoints_to_hide = set()

    keypoints_indices = list(keypoints_to_show - keypoints_to_hide)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keypoints_indices)}
    joints_positions = keypoints[:, keypoints_indices]
    joints_confidences = confidences[:, keypoints_indices]
    joints_connections = [
        (index_map[i], index_map[j])
        for (i, j) in kps_3d_format.keypoints_connectivity
        if i in keypoints_indices and j in keypoints_indices
    ]

    joints_positions[joints_confidences <= torch.finfo(torch.float32).eps] = 0

    bones_confidences = torch.ones((n_frames, len(joints_connections)))
    for bone_idx, (joint_i, joint_j) in enumerate(joints_connections):
        bones_confidences[:, bone_idx] = torch.sqrt(
            joints_confidences[:, joint_i] * joints_confidences[:, joint_j]
        )

    skeleton = SkeletonWithConfidence(
        joints_positions=joints_positions.cpu().numpy(),
        joints_connections=joints_connections,
        joints_confidences=joints_confidences.cpu().numpy(),
        bones_confidences=bones_confidences.cpu().numpy(),
        colormap=colormap,
        radius=skeleton_radius,
        normalize_skeleton_confidence_per_frame=normalize_skeleton_confidence_per_frame,
    )
    skeleton.name = subject_id
    return skeleton


def _create_subject_skeleton(
    subject_id: str,
    keypoints_3d: Keypoints3DAnnotations,
    skeleton_radius: float = 0.01,
    score_threshold: float = 0.2,
    color: tuple[float, float, float, float] = (1, 0, 0, 1),
    convert_coordinates_to_opengl: bool = True,
    keypoints_to_show: list[int] | None = None,
    keypoints_to_hide: list[int] | None = None,
) -> SkeletonWithVisibility:
    n_frames = keypoints_3d.n_frames

    subject_kps_3d_annots = keypoints_3d.filter_by_subject_id(subject_id)
    kps_3d_format_name = subject_kps_3d_annots.first_or_default().format

    kps_3d_format = next(
        (
            kps_3d_format
            for kps_3d_format in subject_kps_3d_annots.metadata.formats
            if kps_3d_format.name == kps_3d_format_name
        ),
        None,
    )

    keypoints = torch.zeros(
        (n_frames, kps_3d_format.n_keypoints, 3), device=torch.device("cpu")
    )
    visibility = torch.zeros(
        (n_frames, kps_3d_format.n_keypoints),
        dtype=torch.bool,
        device=torch.device("cpu"),
    )

    for annot in subject_kps_3d_annots:
        frame_idx = annot.frame_idx
        keypoints[frame_idx, :, :] = annot.xyz
        visibility[frame_idx, :] = annot.scores >= score_threshold

    # To OpenGL coordinate system
    if convert_coordinates_to_opengl:
        keypoints = convert_world_points_from_opencv_to_opengl(keypoints)

    if keypoints_to_show is not None:
        keypoints_to_show = set(keypoints_to_show)
    else:
        keypoints_to_show = set(range(kps_3d_format.n_keypoints))

    if keypoints_to_hide is not None:
        keypoints_to_hide = set(keypoints_to_hide)
    else:
        keypoints_to_hide = set()

    keypoints_indices = list(keypoints_to_show - keypoints_to_hide)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keypoints_indices)}
    joint_positions = keypoints[:, keypoints_indices]
    joint_visibility = visibility[:, keypoints_indices]
    joint_connections = [
        (index_map[i], index_map[j])
        for (i, j) in kps_3d_format.keypoints_connectivity
        if i in keypoints_indices and j in keypoints_indices
    ]

    joint_positions[~joint_visibility] = 0

    skeleton = SkeletonWithVisibility(
        joint_positions=joint_positions.cpu().numpy(),
        joint_connections=joint_connections,
        joint_visibility=joint_visibility.cpu().numpy(),
        colors=color,
        radius=skeleton_radius,
    )
    skeleton.name = subject_id
    return skeleton


def _create_subject_joints(
    subject_id: str,
    keypoints_3d: Keypoints3DAnnotations,
    score_threshold: float = 0.2,
    sphere_radius: float = 0.01,
    keypoints_to_show: list[int] | None = None,
    keypoints_to_hide: list[int] | None = None,
    color: tuple[float, float, float, float] = (1, 0, 0, 1),
    convert_coordinates_to_opengl: bool = True,
) -> Node:
    n_frames = keypoints_3d.n_frames

    subject_kps_3d_annots = keypoints_3d.filter_by_subject_id(subject_id)
    kps_3d_format_name = subject_kps_3d_annots.first_or_default().format

    kps_3d_format = next(
        (
            kps_3d_format
            for kps_3d_format in subject_kps_3d_annots.metadata.formats
            if kps_3d_format.name == kps_3d_format_name
        ),
        None,
    )

    keypoints = torch.zeros(
        (n_frames, kps_3d_format.n_keypoints, 3), device=torch.device("cpu")
    )
    visibility = torch.zeros(
        (n_frames, kps_3d_format.n_keypoints),
        dtype=torch.bool,
        device=torch.device("cpu"),
    )

    for annot in subject_kps_3d_annots:
        frame_idx = annot.frame_idx
        keypoints[frame_idx, :, :] = annot.xyz
        visibility[frame_idx, :] = annot.scores >= score_threshold

    # To OpenGL coordinate system
    if convert_coordinates_to_opengl:
        keypoints = convert_world_points_from_opencv_to_opengl(keypoints)

    if keypoints_to_show is not None:
        keypoints_to_show = set(keypoints_to_show)
    else:
        keypoints_to_show = set(range(kps_3d_format.n_keypoints))

    if keypoints_to_hide is not None:
        keypoints_to_hide = set(keypoints_to_hide)
    else:
        keypoints_to_hide = set()

    keypoints_indices = list(keypoints_to_show - keypoints_to_hide)
    joint_positions = keypoints[:, keypoints_indices]
    joint_visibility = visibility[:, keypoints_indices]

    joint_positions[~joint_visibility] = 0
    joint_names = [
        kps_3d_format.keypoints_names[i].replace(" ", "_").replace("-", "_")
        for i in keypoints_indices
    ]

    node = Node(name=subject_id)

    for joint_idx, joint_name in enumerate(joint_names):
        sphere = Spheres(
            positions=joint_positions[:, joint_idx]
            .reshape(n_frames, 1, 3)
            .cpu()
            .numpy(),
            radius=sphere_radius,
            color=color,
        )
        print(joint_name)
        sphere.name = joint_name
        node.add(sphere)

    return node

def _create_camera(
    view_id: str,
    camera_extrinsics: CameraExtrinsicsAnnotations,
    camera_intrinsics: CameraIntrinsicsAnnotations,
    color: tuple[float, float, float, float] = (1, 0, 0, 1),
    scale: float = 1,
    convert_coordinates_to_opengl: bool = True,
) -> OpenCVCamera:
    assert (
        camera_extrinsics.frames == camera_intrinsics.frames
    ), "Camera extrinsics and camera intrinsics must have the same frames."
    assert len(camera_extrinsics.views_ids) == 1
    assert len(camera_intrinsics.views_ids) == 1
    assert camera_extrinsics.views_ids == camera_intrinsics.views_ids

    n_frames = max(camera_extrinsics.frames) + 1

    K = torch.empty((n_frames, 3, 3))
    Rt = torch.empty((n_frames, 3, 4))

    h, w = camera_intrinsics.first_or_default().resolution_hw

    for frame_idx in camera_extrinsics.frames:
        frame_camera_extrinsics = camera_extrinsics.filter_by_frame_idx(
            frame_idx
        ).first_or_default()
        frame_camera_intrinsics = camera_intrinsics.filter_by_frame_idx(
            frame_idx
        ).first_or_default()
        K[frame_idx] = frame_camera_intrinsics.K
        Rt[frame_idx] = frame_camera_extrinsics.Rt

    # To OpenGL coordinate system
    if convert_coordinates_to_opengl:
        Rt = convert_Rt_from_opencv_to_opengl(Rt)

    K = K.cpu().numpy()
    Rt = Rt.cpu().numpy()

    camera = OpenCVCamera(K=K, Rt=Rt, cols=w, rows=h)
    camera.scale = scale
    camera.mesh.color = tuple(color)
    camera.color = color
    camera.active_color = color
    camera.inactive_color = color
    camera.name = view_id
    return camera


def _compute_floor_position(
    keypoints_3d: Keypoints3DAnnotations,
    keypoints_to_show: list[int] | None = None,
    keypoints_to_hide: list[int] | None = None,
    score_threshold: float = 0.2,
    up_axis: int = 2,
) -> float:
    n_frames = keypoints_3d.n_frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kps_3d_format_name = keypoints_3d.first_or_default().format

    kps_3d_format = next(
        (
            kps_3d_format
            for kps_3d_format in keypoints_3d.metadata.formats
            if kps_3d_format.name == kps_3d_format_name
        ),
        None,
    )

    keypoints = torch.zeros((n_frames, kps_3d_format.n_keypoints, 3), device=device)
    keypoints_scores = torch.zeros((n_frames, kps_3d_format.n_keypoints), device=device)

    for annot in keypoints_3d:
        frame_idx = annot.frame_idx
        keypoints[frame_idx, :, :] = annot.xyz
        keypoints_scores[frame_idx, :] = annot.scores

    keypoints = convert_world_points_from_opencv_to_opengl(keypoints)

    if keypoints_to_show is not None:
        keypoints_to_show = set(keypoints_to_show)
    else:
        keypoints_to_show = set(range(kps_3d_format.n_keypoints))

    if keypoints_to_hide is not None:
        keypoints_to_hide = set(keypoints_to_hide)
    else:
        keypoints_to_hide = set()

    keypoints_indices = list(keypoints_to_show - keypoints_to_hide)
    keypoints = keypoints[:, keypoints_indices]
    keypoints_scores = keypoints_scores[:, keypoints_indices]

    keypoints = keypoints.reshape(-1, 3)
    keypoints_scores = keypoints_scores.reshape(-1)

    valid_kps_mask = (keypoints_scores > score_threshold) & (
        torch.isfinite(keypoints).all(dim=-1)
    )
    keypoints = keypoints[valid_kps_mask]
    keypoints_scores = keypoints_scores[valid_kps_mask]

    if len(keypoints) == 0:
        return 0

    return keypoints[..., up_axis].min().item()


def add_keypoints_3d_and_cameras(
    node: Node,
    keypoints_3d: Keypoints3DAnnotations,
    camera_extrinsics: CameraExtrinsicsAnnotations,
    camera_intrinsics: CameraIntrinsicsAnnotations,
    subjects_colors: (
        dict[str, tuple[float, float, float, float]] | tuple[float, float, float, float]
    ),
    cameras_colors: (
        dict[str, tuple[float, float, float, float]] | tuple[float, float, float, float]
    ),
    subjects_names: dict[str, str] | None = None,
    cameras_names: dict[str, str] | None = None,
    skeleton_radius: float = 0.01,
    camera_scale: float = 1,
    convert_coordinates_to_opengl: bool = True,
    keypoints_score_threshold: float = 0.2,
):
    add_keypoints_3d(
        node=node,
        keypoints_3d=keypoints_3d,
        subjects_colors=subjects_colors,
        subjects_names=subjects_names,
        skeleton_radius=skeleton_radius,
        convert_coordinates_to_opengl=convert_coordinates_to_opengl,
        score_threshold=keypoints_score_threshold,
    )
    add_cameras(
        node=node,
        camera_extrinsics=camera_extrinsics,
        camera_intrinsics=camera_intrinsics,
        cameras_colors=cameras_colors,
        cameras_names=cameras_names,
        camera_scale=camera_scale,
        convert_coordinates_to_opengl=convert_coordinates_to_opengl,
    )


def add_keypoints_3d(
    node: Node,
    keypoints_3d: Keypoints3DAnnotations,
    subjects_colors: (
        dict[str, tuple[float, float, float, float]] | tuple[float, float, float, float]
    ),
    subjects_names: dict[str, str] | None = None,
    skeleton_radius: float = 0.01,
    score_threshold: float = 0.2,
    convert_coordinates_to_opengl: bool = True,
) -> dict[str, SkeletonWithVisibility]:
    if isinstance(subjects_colors, tuple):
        subjects_colors = {
            subject_id: subjects_colors for subject_id in keypoints_3d.subjects_ids
        }
    if subjects_names is None:
        subjects_names = {
            subject_id: subject_id for subject_id in keypoints_3d.subjects_ids
        }

    skeletons = {}
    for subject_id in keypoints_3d.subjects_ids:
        subject_skeleton = _create_subject_skeleton(
            subject_id=subject_id,
            keypoints_3d=keypoints_3d,
            skeleton_radius=skeleton_radius,
            score_threshold=score_threshold,
            color=subjects_colors[subject_id],
            convert_coordinates_to_opengl=convert_coordinates_to_opengl,
        )
        subject_skeleton.name = subjects_names[subject_id]
        node.add(subject_skeleton)
        skeletons[subject_id] = subject_skeleton
    return skeletons


def add_cameras(
    node: Node,
    camera_extrinsics: CameraExtrinsicsAnnotations,
    camera_intrinsics: CameraIntrinsicsAnnotations,
    cameras_colors: (
        dict[str, tuple[float, float, float, float]] | tuple[float, float, float, float]
    ),
    cameras_names: dict[str, str] | None = None,
    camera_scale: float = 1,
    convert_coordinates_to_opengl: bool = True,
) -> dict[str, OpenCVCamera]:
    if isinstance(cameras_colors, tuple):
        cameras_colors = {
            view_id: cameras_colors for view_id in camera_extrinsics.views_ids
        }
    if cameras_names is None:
        cameras_names = {view_id: view_id for view_id in camera_extrinsics.views_ids}

    cameras = {}
    for view_id in camera_extrinsics.views_ids:
        camera = _create_camera(
            view_id=view_id,
            camera_extrinsics=camera_extrinsics.filter_by_view_id(view_id),
            camera_intrinsics=camera_intrinsics.filter_by_view_id(view_id),
            color=cameras_colors[view_id],
            scale=camera_scale,
            convert_coordinates_to_opengl=convert_coordinates_to_opengl,
        )
        camera.name = cameras_names[view_id]
        node.add(camera)
        cameras[view_id] = camera
    return cameras


def add_world_reconstruction(
    node: Node,
    world_reconstruction: WorldReconstructedSceneAnnotations,
    world_points_size: float = 0.7,
    world_z_clipping_threshold: float | None = None,
    world_points_confidence_threshold: float = 0.0,
    max_world_points_to_show: int = -1,
    convert_coordinates_to_opengl: bool = True,
    name: str = "World",
) -> PointClouds | None:
    world_reconstruction: WorldReconstructedSceneAnnotation = (
        world_reconstruction.first_or_default()
    )

    if world_reconstruction is None:
        return None

    world_points_xyz = world_reconstruction.points_xyz
    world_points_colors = world_reconstruction.points_colors
    world_points_colors = torch.cat(
        [world_points_colors, torch.ones_like(world_points_colors[..., :1])], dim=-1
    )

    world_points_conf = world_reconstruction.points_confidences

    valid_mask = world_points_conf >= world_points_confidence_threshold
    world_points_xyz = world_points_xyz[valid_mask]
    world_points_colors = world_points_colors[valid_mask]

    if world_z_clipping_threshold is not None:
        clipping_mask = world_points_xyz[..., 2] > world_z_clipping_threshold
        world_points_xyz = world_points_xyz[~clipping_mask]
        world_points_colors = world_points_colors[~clipping_mask]

    if max_world_points_to_show > 0 and world_points_xyz is not None:
        indices = torch.randperm(world_points_xyz.shape[0])[:max_world_points_to_show]
        world_points_xyz = world_points_xyz[indices]
        world_points_colors = world_points_colors[indices]

    if convert_coordinates_to_opengl:
        world_points_xyz = convert_world_points_from_opencv_to_opengl(world_points_xyz)

    if len(world_points_xyz) > 0:
        world_pcd = PointClouds(
            points=world_points_xyz.unsqueeze(0).cpu().numpy(),
            colors=world_points_colors.unsqueeze(0).cpu().numpy(),
            point_size=world_points_size,
        )
        world_pcd.name = name
        node.add(world_pcd)
        return world_pcd

    return None


def export_world_to_ply(
    path: str,
    world_reconstruction: WorldReconstructedSceneAnnotations,
    max_world_points_to_show: int = -1,
    world_z_clipping_threshold: float = None,
    world_points_confidence_threshold: float = 0.2,
    convert_coordinates_to_opengl: bool = True,
):
    world_reconstruction: WorldReconstructedSceneAnnotation = (
        world_reconstruction.first_or_default()
    )

    assert world_reconstruction is not None

    world_points_xyz = world_reconstruction.points_xyz
    world_points_colors = world_reconstruction.points_colors
    world_points_colors = torch.cat(
        [world_points_colors, torch.ones_like(world_points_colors[..., :1])], dim=-1
    )
    world_points_conf = world_reconstruction.points_confidences

    valid_mask = world_points_conf >= world_points_confidence_threshold
    world_points_xyz = world_points_xyz[valid_mask]
    world_points_colors = world_points_colors[valid_mask]
    world_points_conf = world_points_conf[valid_mask]

    if world_z_clipping_threshold is not None:
        clipping_mask = world_points_xyz[..., 2] > world_z_clipping_threshold
        world_points_xyz = world_points_xyz[~clipping_mask]
        world_points_colors = world_points_colors[~clipping_mask]

    if max_world_points_to_show > 0 and world_points_xyz is not None:
        indices = torch.randperm(world_points_xyz.shape[0])[:max_world_points_to_show]
        world_points_xyz = world_points_xyz[indices]
        world_points_colors = world_points_colors[indices]

    if convert_coordinates_to_opengl:
        world_points_xyz = convert_world_points_from_opencv_to_opengl(world_points_xyz)

    trimesh.PointCloud(
        vertices=world_points_xyz.cpu().numpy(),
        colors=world_points_colors.cpu().numpy(),
    ).export(path)


def export_usd(
    path: str,
    keypoints_3d: Keypoints3DAnnotations,
    camera_extrinsics: CameraExtrinsicsAnnotations,
    camera_intrinsics: CameraIntrinsicsAnnotations,
    fps: float,
    joints_radius: float = 0.01,
    score_threshold: float = 0.2,
    camera_scale: float = 5,
    camera_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    keypoints_to_show: list[int] | None = None,
    keypoints_to_hide: list[int] | None = None,
    export_as_directory: bool = False,
    convert_coordinates_to_opengl: bool = True,
):
    pred_joints = []
    pred_cameras = []

    for subject in keypoints_3d.subjects_ids:
        subject_joints = _create_subject_joints(
            subject_id=subject,
            keypoints_3d=keypoints_3d,
            color=get_subject_color_rgba(subject),
            sphere_radius=joints_radius,
            keypoints_to_show=keypoints_to_show,
            keypoints_to_hide=keypoints_to_hide,
            convert_coordinates_to_opengl=convert_coordinates_to_opengl,
        )

        subject_joints.name = f"pred_{subject}"
        pred_joints.append(subject_joints)

    for view_id in camera_extrinsics.views_ids:
        camera = _create_camera(
            view_id=view_id,
            camera_extrinsics=camera_extrinsics.filter_by_view_id(view_id),
            camera_intrinsics=camera_intrinsics.filter_by_view_id(view_id),
            color=camera_color + (1,),
            scale=camera_scale,
        )
        pred_cameras.append(camera)

    viewer = Viewer()
    viewer.window.init_mgl_context()
    viewer.playback_fps = fps
    viewer.run_animations = True

    pred_node = Node(name="pred")

    for skeleton in pred_joints:
        pred_node.add(skeleton)
    for camera in pred_cameras:
        pred_node.add(camera)

    viewer.scene.add(pred_node)

    up_axis = 2 if C.z_up else 1

    if keypoints_3d is not None:
        floor_position = _compute_floor_position(
            keypoints_3d, up_axis=up_axis, score_threshold=score_threshold
        )
        viewer.scene.floor.position[up_axis] = floor_position
        viewer.scene.floor.update_transform(parent_transform=viewer.scene.model_matrix)
    
    viewer.scene.floor.enabled = False
    viewer.export_usd(path, verbose=True, export_as_directory=export_as_directory)


def show_keypoints_and_cameras(
    fps: float,
    window_name: str = "Visualization 3D",
    keypoints_3d: Keypoints3DAnnotations | None = None,
    camera_extrinsics: CameraExtrinsicsAnnotations | None = None,
    camera_intrinsics: CameraIntrinsicsAnnotations | None = None,
    skeleton_radius: float = 0.01,
    score_threshold: float = 0.2,
    world_reconstruction: WorldReconstructedSceneAnnotations | None = None,
    world_points_size: float = 0.7,
    world_points_confidence_threshold: float = 0.2,
    world_z_clipping_threshold: float = None,  # threshold above which the points are not shown
    camera_scale: float = 5,
    max_world_points_to_show: int = -1,
    gt_keypoints_3d: Keypoints3DAnnotations | None = None,
    gt_camera_extrinsics: CameraExtrinsicsAnnotations | None = None,
    gt_camera_intrinsics: CameraIntrinsicsAnnotations | None = None,
    apply_similarity_transform: bool = True,
    estimate_scale: bool = True,
    camera_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    skeleton_color_override: tuple[float, float, float] = None,
    skeleton_confidence_colormap: str = "inferno",
    show_skeleton_confidence: bool = False,
    normalize_skeleton_confidence_per_frame: bool = False,
    keypoints_to_show: list[int] | None = None,
    keypoints_to_hide: list[int] | None = None,
    smpl_shape: SMPLShapeAnnotations | None = None,
    smpl_pose: SMPLPoseAnnotations | None = None,
    show_floor: bool = True,
):
    world_points_xyz = None
    world_points_colors = None

    if world_reconstruction is not None and len(world_reconstruction) > 0:
        world_reconstruction: WorldReconstructedSceneAnnotation = (
            world_reconstruction.first_or_default()
        )
        world_points_xyz = world_reconstruction.points_xyz
        world_points_colors = world_reconstruction.points_colors
        # Add alpha channel
        world_points_colors = torch.cat(
            [world_points_colors, torch.ones_like(world_points_colors[..., :1])], dim=-1
        )
        world_points_confidences = world_reconstruction.points_confidences

        if world_z_clipping_threshold is not None:
            clipping_mask = world_points_xyz[..., 2] > world_z_clipping_threshold
            world_points_xyz = world_points_xyz[~clipping_mask]
            world_points_colors = world_points_colors[~clipping_mask]
            world_points_confidences = world_points_confidences[~clipping_mask]

        assert world_points_confidences.ndim == 1

        conf_mask = (world_points_confidences >= world_points_confidence_threshold) & (
            world_points_confidences > 1e-5
        )

        world_points_xyz = world_points_xyz[conf_mask]
        world_points_colors = world_points_colors[conf_mask]
        world_points_confidences = world_points_confidences[conf_mask]

    if max_world_points_to_show > 0 and world_points_xyz is not None:
        indices = torch.randperm(world_points_xyz.shape[0])[:max_world_points_to_show]
        world_points_xyz = world_points_xyz[indices]
        world_points_colors = world_points_colors[indices]
        world_points_confidences = world_points_confidences[indices]

    if (
        gt_keypoints_3d is not None
        and len(gt_keypoints_3d) > 0
        and gt_camera_extrinsics is not None
        and len(gt_camera_extrinsics) > 0
        and gt_camera_intrinsics is not None
        and len(gt_camera_intrinsics) > 0
        and apply_similarity_transform
    ):
        if camera_extrinsics is not None:
            R, t, s = camera_extrinsics.compute_similarity_transform(
                gt_camera_extrinsics, estimate_scale=estimate_scale
            )

            camera_extrinsics = camera_extrinsics.apply_similarity_transform(R, t, s)

            if keypoints_3d is not None:
                keypoints_3d = keypoints_3d.apply_similarity_transform(R, t, s)
            
            if world_points_xyz is not None:
                world_points_xyz = apply_similarity_transform_to_points(
                    world_points_xyz.to(R.device), R, t, s
                )

    pred_skeletons = []
    pred_cameras = []
    gt_skeletons = []
    gt_cameras = []

    if keypoints_3d is not None:
        for subject in keypoints_3d.subjects_ids:

            if show_skeleton_confidence:
                subject_skeleton = _create_subject_skeleton_with_confidence(
                    subject_id=subject,
                    keypoints_3d=keypoints_3d,
                    skeleton_radius=skeleton_radius,
                    colormap=skeleton_confidence_colormap,
                    convert_coordinates_to_opengl=True,
                    keypoints_to_show=keypoints_to_show,
                    keypoints_to_hide=keypoints_to_hide,
                    normalize_skeleton_confidence_per_frame=normalize_skeleton_confidence_per_frame,
                )
            else:
                subject_skeleton = _create_subject_skeleton(
                    subject_id=subject,
                    keypoints_3d=keypoints_3d,
                    skeleton_radius=skeleton_radius,
                    score_threshold=score_threshold,
                    color=(
                        tuple(skeleton_color_override) + (1,)
                        if skeleton_color_override is not None
                        else get_subject_color_rgba(subject)
                    ),
                    convert_coordinates_to_opengl=True,
                    keypoints_to_show=keypoints_to_show,
                    keypoints_to_hide=keypoints_to_hide,
                )
            subject_skeleton.name = f"pred_{subject}"
            pred_skeletons.append(subject_skeleton)

    if gt_keypoints_3d is not None:
        for subject in gt_keypoints_3d.subjects_ids:
            subject_skeleton = _create_subject_skeleton(
                subject_id=subject,
                keypoints_3d=gt_keypoints_3d,
                skeleton_radius=skeleton_radius,
                score_threshold=score_threshold,
                color=(0, 0, 0, 1),
                convert_coordinates_to_opengl=True,
                keypoints_to_show=keypoints_to_show,
                keypoints_to_hide=keypoints_to_hide,
            )
            subject_skeleton.name = f"gt_{subject}"
            gt_skeletons.append(subject_skeleton)

    if camera_extrinsics is not None:
        for view_id in camera_extrinsics.views_ids:
            camera = _create_camera(
                view_id=view_id,
                camera_extrinsics=camera_extrinsics.filter_by_view_id(view_id),
                camera_intrinsics=camera_intrinsics.filter_by_view_id(view_id),
                color=camera_color + (1,),
                scale=camera_scale,
                convert_coordinates_to_opengl=True,
            )
            pred_cameras.append(camera)

    if gt_camera_extrinsics is not None:
        for view_id in gt_camera_extrinsics.views_ids:
            camera = _create_camera(
                view_id=view_id,
                camera_extrinsics=gt_camera_extrinsics.filter_by_view_id(view_id),
                camera_intrinsics=gt_camera_intrinsics.filter_by_view_id(view_id),
                color=(0.1, 0.1, 0.1, 1),
                scale=camera_scale,
                convert_coordinates_to_opengl=True,
            )
            gt_cameras.append(camera)

    viewer = Viewer(title=window_name)
    viewer.window.init_mgl_context()

    pred_node = Node(name="pred")

    for skeleton in pred_skeletons:
        pred_node.add(skeleton)
    for camera in pred_cameras:
        pred_node.add(camera)
    viewer.scene.add(pred_node)

    if smpl_pose is not None:

        n_frames = smpl_pose.n_frames

        subjects_ids = smpl_pose.subjects_ids
        for subject_id in subjects_ids:
            subject_poses = smpl_pose.filter_by_subject_id(subject_id)
            pose_format = subject_poses.first_or_default().format

            shape = smpl_shape.filter_by_subject_id(subject_id).first_or_default()

            if shape is None:
                betas = torch.zeros(10)
            else:
                betas = shape.betas

            smpl_layer = SMPLLayer(
                model_type="smpl" if pose_format == SMPLPoseFormat.SMPL else "smplx",
                gender="neutral",
                num_betas=betas.shape[0],
            )

            frames_poses = torch.zeros((n_frames, pose_format.num_joints() + 1, 3))
            frames_trans = torch.zeros((n_frames, 3))

            for pose in subject_poses:
                frames_poses[pose.frame_idx, :, :] = pose.pose
                frames_trans[pose.frame_idx, :] = pose.trans

            poses_root, poses_body, poses_left_hand, poses_right_hand, _ = (
                pose_format.split_poses(frames_poses)
            )

            smpl_seq = SMPLSequence(
                smpl_layer=smpl_layer,
                betas=betas,
                trans=frames_trans,
                poses_root=poses_root.reshape(n_frames, 3),
                poses_body=poses_body.reshape(n_frames, -1),
                poses_left_hand=poses_left_hand.reshape(n_frames, -1),
                poses_right_hand=poses_right_hand.reshape(n_frames, -1),
            )
            # OpenCV -> OpenGL
            smpl_seq.rotation = roma.rotvec_to_rotmat(
                torch.tensor([-90.0, 0, 0]).deg2rad()
            )
            smpl_seq.name = f"smpl_{subject_id}"
            pred_node.add(smpl_seq)

    if world_points_xyz is not None and len(world_points_xyz) > 0:
        # Convert to OpenGL coordinate system
        world_points_xyz = convert_world_points_from_opencv_to_opengl(world_points_xyz)

        world_pcd = PointClouds(
            points=world_points_xyz.unsqueeze(0).cpu().numpy(),
            colors=world_points_colors.unsqueeze(0).cpu().numpy(),
            point_size=world_points_size,
        )
        world_pcd.name = "World"
        pred_node.add(world_pcd)

    if len(gt_skeletons) > 0 or len(gt_cameras) > 0:
        gt_node = Node(name="gt")
        for skeleton in gt_skeletons:
            gt_node.add(skeleton)
        for camera in gt_cameras:
            gt_node.add(camera)
        viewer.scene.add(gt_node)

    viewer.playback_fps = fps
    viewer.run_animations = True

    up_axis = 2 if C.z_up else 1
    
    if keypoints_3d is not None:
        floor_position = _compute_floor_position(
            keypoints_3d, up_axis=up_axis, score_threshold=score_threshold
        )
        viewer.scene.floor.position[up_axis] = floor_position
        viewer.scene.floor.update_transform(parent_transform=viewer.scene.model_matrix)

    viewer.scene.floor.enabled = show_floor
    viewer.run()
