# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import rerun as rr

from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.pipeline import ViewInput
from kineo.pipeline.pipeline import Annotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.reconstructed_scene import (
    WorldReconstructedSceneAnnotations,
    WorldReconstructedSceneAnnotation,
)
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.annotations.smpl_params import SMPLShapeAnnotations, SMPLPoseAnnotations
from kineo.geometry.transformations import inverse_Rt, apply_similarity_transform_to_points
import os
import numpy as np
from dataclasses import dataclass
import torch
import warnings
from tqdm import tqdm
from kineo.visualization.utils import get_subject_color_rgba
from smplx import build_layer, SMPLLayer
import roma

try:
    import av
except ImportError:
    av = None


@dataclass(frozen=True)
class RerunExportRuntimeConfig:
    start_frame_idx: int = 0
    end_frame_idx: int = -1

    output_path_template: str = "./outputs/rerun/{sequence_name}.rrd"
    world_points_radius_m: float = 0.01
    max_world_points_to_show: int = 1_000_000
    min_world_point_confidence: float = 0.5
    world_z_clipping_m: float = None

    skeleton_joint_radius_m: float = 0.02
    skeleton_joint_radius_px: float = 14.0
    skeleton_bones_thickness_m: float = 0.01
    skeleton_bones_thickness_px: float = 9.0

    keypoints_radius_px: float = 1.0
    keypoints_radius_m: float = 0.01

    min_keypoint_score_3d: float = 0.5
    min_keypoint_score_2d: float = 0.3

    smpl_color_override: tuple[float, float, float] = None
    smpl_skeleton_2d_color_override: tuple[float, float, float] = None

    skeleton_color_override: tuple[float, float, float] = None
    image_plane_distance_m: float = 0.2
    remove_world_points_outside_scene_radius: bool = False
    scene_radius_multiplier: float = 1.5
    video_quality: int = 75
    video_view_ids: str | list[str] | None = None
    world_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    log_world_reconstruction: bool = False
    log_videos: bool = False

    log_pred_cameras: bool = False
    log_pred_keypoints_2d: bool = False
    log_pred_keypoints_3d: bool = False
    log_pred_skeletons_2d: bool = False
    log_pred_skeletons_3d: bool = False

    log_pred_smpl: bool = False
    log_pred_smpl_skeleton_2d: bool = False
    log_gt_keypoints_2d: bool = False
    log_gt_keypoints_3d: bool = False
    log_gt_skeletons_2d: bool = False
    log_gt_skeletons_3d: bool = False
    log_gt_cameras: bool = False

    log_disconnected_joints: bool = False


class RerunExportStage(PipelineStage[RerunExportRuntimeConfig]):
    def __init__(
            self,
            name: str,
            order: int,
            runtime_cfg: RerunExportRuntimeConfig,
            dynamic_runtime_cfg: dict[str, RerunExportRuntimeConfig] | None = None,
            smpl_model_path: str | None = "./body_models/smplx/SMPLX_NEUTRAL.npz",
            smpl_model_type: str = "smplx",
            smpl_num_betas: int = 10,
            smpl_gender: str = "neutral",
            smpl_use_face_contour: bool = False,
            smpl_use_pca: bool = True,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )

        if smpl_model_path is not None:
            self.smpl_layer = build_layer(
                model_path=smpl_model_path,
                model_type=smpl_model_type,
                num_betas=smpl_num_betas,
                gender=smpl_gender,
                use_face_contour=smpl_use_face_contour,
                use_pca=smpl_use_pca,
            )
            self.smpl_layer = self.smpl_layer.cpu()
            self.smpl_layer = self.smpl_layer.eval()

    def align_predictions_to_gt(
            self,
            pred_keypoints_3d: Keypoints3DAnnotations,
            pred_camera_extrinsics: CameraExtrinsicsAnnotations,
            gt_camera_extrinsics: CameraExtrinsicsAnnotations,
            pred_world_reconstruction: WorldReconstructedSceneAnnotations | None = None,
    ) -> tuple[Keypoints3DAnnotations, CameraExtrinsicsAnnotations, WorldReconstructedSceneAnnotations]:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pred_camera_extrinsics = pred_camera_extrinsics.to(device)
        gt_camera_extrinsics = gt_camera_extrinsics.to(device)
        if pred_world_reconstruction is not None:
            pred_world_reconstruction = pred_world_reconstruction.to(device)
        pred_keypoints_3d = pred_keypoints_3d.to(device)

        R, T, s = pred_camera_extrinsics.compute_similarity_transform(gt_camera_extrinsics, estimate_scale=True)
        pred_keypoints_3d = pred_keypoints_3d.apply_similarity_transform(R, T, s)
        pred_camera_extrinsics = pred_camera_extrinsics.apply_similarity_transform(R, T, s)
        if pred_world_reconstruction is not None:
            pred_world_reconstruction = pred_world_reconstruction.apply_similarity_transform(R, T, s)

        pred_keypoints_3d = pred_keypoints_3d.cpu()
        pred_camera_extrinsics = pred_camera_extrinsics.cpu()
        gt_camera_extrinsics = gt_camera_extrinsics.cpu()
        if pred_world_reconstruction is not None:
            pred_world_reconstruction = pred_world_reconstruction.cpu()

        return pred_keypoints_3d, pred_camera_extrinsics, pred_world_reconstruction

    def forward(
            self,
            sequence_name: str,
            pipeline: Pipeline,
            views: list[ViewInput],
            annotations: dict[str, Annotations],
            gt_annotations: dict[str, Annotations],
            runtime_cfg: RerunExportRuntimeConfig,
    ):
        global_time_reference: GlobalTimeReferenceAnnotation = annotations.get(
            "global_time_reference"
        )

        pred_keypoints_2d: Keypoints2DAnnotations = annotations["keypoints_2d"]
        pred_keypoints_3d: Keypoints3DAnnotations = annotations["keypoints_3d"]
        pred_camera_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "cameras_extrinsics"
        ]
        pred_camera_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "cameras_intrinsics"
        ]
        pred_world_reconstruction: WorldReconstructedSceneAnnotations = annotations.get(
            "world_reconstructed_scene"
        )
        pred_smpl_shape: SMPLShapeAnnotations = annotations.get("smpl_shape")
        pred_smpl_pose: SMPLPoseAnnotations = annotations.get("smpl_pose")

        gt_keypoints_2d: Keypoints2DAnnotations = gt_annotations.get("keypoints_2d")
        gt_keypoints_3d: Keypoints3DAnnotations = gt_annotations.get("keypoints_3d")
        gt_camera_extrinsics: CameraExtrinsicsAnnotations = gt_annotations.get("cameras_extrinsics")
        gt_camera_intrinsics: CameraIntrinsicsAnnotations = gt_annotations.get("cameras_intrinsics")

        if gt_camera_extrinsics is not None:
            pred_keypoints_3d, pred_camera_extrinsics, pred_world_reconstruction = self.align_predictions_to_gt(
                pred_keypoints_3d, pred_camera_extrinsics, gt_camera_extrinsics, pred_world_reconstruction
            )

        formatted_output_path = runtime_cfg.output_path_template.format(
            sequence_name=sequence_name
        )
        os.makedirs(os.path.dirname(formatted_output_path), exist_ok=True)
        rr.init(sequence_name)
        rr.save(formatted_output_path)

        if global_time_reference is not None:
            global_time_reference = global_time_reference.first_or_default()

            dt = torch.diff(global_time_reference.timestamps).mean().item()

            if dt > 0:
                target_fps = 1 / dt
            else:
                target_fps = 25
                warnings.warn(
                    f"Sequence {sequence_name} has a dt of {dt} seconds, which is less than 0.01 seconds. Setting target fps to 25."
                )
        else:
            target_fps = 25

        ### Ground truth ###

        if runtime_cfg.log_gt_cameras and gt_camera_extrinsics is not None and gt_camera_intrinsics is not None:
            log_cameras(
                cameras_extrinsics=gt_camera_extrinsics,
                cameras_intrinsics=gt_camera_intrinsics,
                prefix="ground_truth",
                image_plane_distance_m=runtime_cfg.image_plane_distance_m,
            )

        if runtime_cfg.log_gt_keypoints_2d and gt_keypoints_2d is not None:
            log_keypoints_2d(
                keypoints_2d=gt_keypoints_2d,
                prefix="ground_truth",
                radius=runtime_cfg.keypoints_radius_px,
                color_override=runtime_cfg.skeleton_color_override,
                min_keypoint_score_2d=0,
                fps=target_fps,
                start_frame_idx=runtime_cfg.start_frame_idx,
                end_frame_idx=runtime_cfg.end_frame_idx,
            )

        if runtime_cfg.log_gt_keypoints_3d and gt_keypoints_3d is not None:
            log_keypoints_3d(
                keypoints_3d=gt_keypoints_3d,
                prefix="ground_truth",
                radius=runtime_cfg.keypoints_radius_m,
                color_override=runtime_cfg.skeleton_color_override,
                min_keypoint_score_3d=0,
                fps=target_fps,
                start_frame_idx=runtime_cfg.start_frame_idx,
                end_frame_idx=runtime_cfg.end_frame_idx,
            )

        if runtime_cfg.log_gt_skeletons_3d and gt_keypoints_3d is not None:
            log_skeletons_3d(
                keypoints_3d=gt_keypoints_3d,
                prefix="ground_truth",
                joint_radius=runtime_cfg.skeleton_joint_radius_m,
                bones_thickness=runtime_cfg.skeleton_bones_thickness_m,
                color_override=runtime_cfg.skeleton_color_override,
                min_keypoint_score_3d=0,
                fps=target_fps,
                log_disconnected_joints=runtime_cfg.log_disconnected_joints,
                start_frame_idx=runtime_cfg.start_frame_idx,
                end_frame_idx=runtime_cfg.end_frame_idx,
            )

        if runtime_cfg.log_gt_skeletons_2d and gt_keypoints_2d is not None:
            log_skeletons_2d(
                keypoints_2d=gt_keypoints_2d,
                prefix="ground_truth",
                joint_radius=runtime_cfg.skeleton_joint_radius_px,
                bones_thickness=runtime_cfg.skeleton_bones_thickness_px,
                color_override=runtime_cfg.skeleton_color_override,
                min_keypoint_score_2d=0,
                fps=target_fps,
                log_disconnected_joints=runtime_cfg.log_disconnected_joints,
                start_frame_idx=runtime_cfg.start_frame_idx,
                end_frame_idx=runtime_cfg.end_frame_idx,
            )

        ### Predictions ###

        if runtime_cfg.log_pred_cameras:
            log_cameras(
                cameras_extrinsics=pred_camera_extrinsics,
                cameras_intrinsics=pred_camera_intrinsics,
                prefix="kineo",
                image_plane_distance_m=runtime_cfg.image_plane_distance_m,
            )

        if runtime_cfg.log_world_reconstruction and pred_world_reconstruction is not None:
            world_reconstruction = pred_world_reconstruction.first_or_default()
            log_world_reconstruction(
                cameras_extrinsics=pred_camera_extrinsics,
                world_reconstruction=world_reconstruction,
                world_points_radius_m=runtime_cfg.world_points_radius_m,
                max_world_points_to_show=runtime_cfg.max_world_points_to_show,
                min_world_point_confidence=runtime_cfg.min_world_point_confidence,
                remove_world_points_outside_scene_radius=runtime_cfg.remove_world_points_outside_scene_radius,
                scene_radius_multiplier=runtime_cfg.scene_radius_multiplier,
                world_translation=runtime_cfg.world_translation,
                world_z_clipping_m=runtime_cfg.world_z_clipping_m,
                prefix="kineo",
            )

        if runtime_cfg.log_pred_smpl and pred_smpl_shape is not None and pred_smpl_pose is not None:
            log_smpl(
                smpl_layer=self.smpl_layer,
                smpl_shape=pred_smpl_shape,
                smpl_pose=pred_smpl_pose,
                prefix="kineo",
                fps=target_fps,
                # Fetch the annotations again because we modified them in the align_predictions_to_gt method
                pred_camera_extrinsics=annotations.get("cameras_extrinsics"),
                gt_camera_extrinsics=gt_annotations.get("cameras_extrinsics"),
                smpl_color_override=runtime_cfg.smpl_color_override,
                start_frame_idx=runtime_cfg.start_frame_idx,
                end_frame_idx=runtime_cfg.end_frame_idx,
                log_pred_smpl_skeleton_2d=runtime_cfg.log_pred_smpl_skeleton_2d,
                projection_camera_extrinsics=pred_camera_extrinsics,
                projection_camera_intrinsics=pred_camera_intrinsics,
                skeleton_joint_radius_px=runtime_cfg.skeleton_joint_radius_px,
                skeleton_bones_thickness_px=runtime_cfg.skeleton_bones_thickness_px,
                smpl_skeleton_2d_color_override=runtime_cfg.smpl_skeleton_2d_color_override,
            )

        if runtime_cfg.log_pred_keypoints_3d and pred_keypoints_3d is not None:
            log_keypoints_3d(
                keypoints_3d=pred_keypoints_3d,
                prefix="kineo",
                radius=runtime_cfg.keypoints_radius_m,
                color_override=runtime_cfg.skeleton_color_override,
                min_keypoint_score_3d=runtime_cfg.min_keypoint_score_3d,
                fps=target_fps,
                start_frame_idx=runtime_cfg.start_frame_idx,
                end_frame_idx=runtime_cfg.end_frame_idx,
            )

        if runtime_cfg.log_pred_keypoints_2d and pred_keypoints_2d is not None:
            log_keypoints_2d(
                keypoints_2d=pred_keypoints_2d,
                prefix="kineo",
                radius=runtime_cfg.keypoints_radius_px,
                color_override=runtime_cfg.skeleton_color_override,
                fps=target_fps,
                min_keypoint_score_2d=runtime_cfg.min_keypoint_score_2d,
                start_frame_idx=runtime_cfg.start_frame_idx,
                end_frame_idx=runtime_cfg.end_frame_idx,
            )

        if runtime_cfg.log_pred_skeletons_3d and pred_keypoints_3d is not None:
            log_skeletons_3d(
                keypoints_3d=pred_keypoints_3d,
                prefix="kineo",
                joint_radius=runtime_cfg.skeleton_joint_radius_m,
                color_override=runtime_cfg.skeleton_color_override,
                bones_thickness=runtime_cfg.skeleton_bones_thickness_m,
                fps=target_fps,
                min_keypoint_score_3d=runtime_cfg.min_keypoint_score_3d,
                log_disconnected_joints=runtime_cfg.log_disconnected_joints,
                start_frame_idx=runtime_cfg.start_frame_idx,
                end_frame_idx=runtime_cfg.end_frame_idx,
            )

        if runtime_cfg.log_pred_skeletons_2d and pred_keypoints_2d is not None:
            log_skeletons_2d(
                keypoints_2d=pred_keypoints_2d,
                prefix="kineo",
                joint_radius=runtime_cfg.skeleton_joint_radius_px,
                color_override=runtime_cfg.skeleton_color_override,
                bones_thickness=runtime_cfg.skeleton_bones_thickness_px,
                fps=target_fps,
                min_keypoint_score_2d=runtime_cfg.min_keypoint_score_2d,
                log_disconnected_joints=runtime_cfg.log_disconnected_joints,
                start_frame_idx=runtime_cfg.start_frame_idx,
                end_frame_idx=runtime_cfg.end_frame_idx,
            )

        if runtime_cfg.log_videos:
            video_views = []

            for view_id in pred_camera_extrinsics.views_ids:
                if runtime_cfg.video_view_ids is not None and view_id not in runtime_cfg.video_view_ids:
                    continue

                video_view = next((v for v in views if v["view_id"] == view_id), None)
                if video_view is None:
                    warnings.warn(f"View {view_id} not found in views")
                    continue
                video_views.append(video_view)

            try:
                log_videos(
                    views=video_views,
                    prefix="kineo",
                    global_time_reference=global_time_reference,
                    video_quality=runtime_cfg.video_quality,
                    start_frame_idx=runtime_cfg.start_frame_idx,
                    end_frame_idx=runtime_cfg.end_frame_idx,
                )
            except Exception as e:
                warnings.warn("Unable to log video")
                print("Unable to log video")
                print(e)

        print("Exported sequence as rerun file at", formatted_output_path)


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


def log_world_reconstruction(
        cameras_extrinsics: CameraExtrinsicsAnnotations,
        world_reconstruction: WorldReconstructedSceneAnnotation,
        world_points_radius_m: float = 7.0,
        max_world_points_to_show: int = 1_000_000,
        min_world_point_confidence: float = 0.5,
        remove_world_points_outside_scene_radius: bool = True,
        scene_radius_multiplier: float = 1.5,
        world_translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
        world_z_clipping_m: float = None,
        prefix: str = "kineo",
):
    views_ids = cameras_extrinsics.views_ids
    cameras_poses = torch.zeros((len(views_ids), 3))
    for view_idx, view_id in enumerate(views_ids):
        camera_extrinsics = cameras_extrinsics.filter_by_view_id(
            view_id
        ).first_or_default()
        cameras_poses[view_idx] = inverse_Rt(camera_extrinsics.Rt)[:3, 3]

    scene_center = cameras_poses.mean(dim=0)
    scene_radius = (cameras_poses - scene_center).norm(dim=1).max()
    scene_radius = scene_radius * scene_radius_multiplier

    points_xyz = world_reconstruction.points_xyz
    points_colors = world_reconstruction.points_colors
    points_confidences = world_reconstruction.points_confidences

    # Remove non-finite points and points below confidence threshold
    valid_mask = (points_confidences > min_world_point_confidence) & (
        torch.isfinite(points_xyz).all(dim=1)
    )
    points_xyz = points_xyz[valid_mask].reshape(-1, 3)
    points_colors = points_colors[valid_mask].reshape(-1, 3)
    points_confidences = points_confidences[valid_mask].reshape(-1)

    if world_z_clipping_m is not None:
        clipping_mask = points_xyz[..., 2] > world_z_clipping_m
        points_xyz = points_xyz[~clipping_mask].reshape(-1, 3)
        points_colors = points_colors[~clipping_mask].reshape(-1, 3)
        points_confidences = points_confidences[~clipping_mask].reshape(-1)

    if remove_world_points_outside_scene_radius:
        points_inside_scene_radius = (points_xyz - scene_center).norm(
            dim=1
        ) <= scene_radius
        points_xyz = points_xyz[points_inside_scene_radius].reshape(-1, 3)
        points_colors = points_colors[points_inside_scene_radius].reshape(-1, 3)
        points_confidences = points_confidences[points_inside_scene_radius].reshape(-1)

    points_xyz = points_xyz.cpu().numpy()
    points_colors = points_colors.cpu().numpy()

    if len(points_xyz) > max_world_points_to_show:
        picked_indices = np.random.choice(
            len(points_xyz), max_world_points_to_show, replace=False
        )
        points_xyz = points_xyz[picked_indices]
        points_colors = points_colors[picked_indices]

    if points_colors.dtype != np.uint8:
        points_colors = (points_colors * 255).astype(np.uint8)
    rr.set_time("frame_idx", sequence=0)
    rr.set_time("time", timestamp=0)
    rr.log(
        f"{prefix}/world",
        rr.Transform3D(translation=world_translation)
    )
    rr.log(
        f"{prefix}/world/points",
        rr.Points3D(
            positions=points_xyz,
            colors=points_colors,
            radii=world_points_radius_m,
        ),
    )


def log_cameras(
        cameras_extrinsics: CameraExtrinsicsAnnotations,
        cameras_intrinsics: CameraIntrinsicsAnnotations,
        prefix: str = "kineo",
        image_plane_distance_m: float = 0.2,
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
                image_plane_distance_m=image_plane_distance_m,
            ),
        )
        rr.log(
            f"{prefix}/cameras/{view_id}",
            rr.Transform3D(
                translation=translation,
                mat3x3=rotation,
            ),
        )


def compute_vertex_normals(
        faces: np.ndarray,
        verts: np.ndarray,
        eps: float = 1e-8,
) -> np.ndarray:
    v0 = verts[faces[:, 0], :]
    v1 = verts[faces[:, 1], :]
    v2 = verts[faces[:, 2], :]

    face_normals = np.cross(v1 - v0, v2 - v0)

    V = verts.shape[0]
    vert_normals = np.zeros((V, 3), dtype=verts.dtype)
    for i in range(3):
        np.add.at(vert_normals, faces[:, i], face_normals)

    norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    norms[norms < eps] = 1.0
    vert_normals = vert_normals / norms

    return vert_normals


def log_smpl(
        smpl_layer: SMPLLayer,
        smpl_shape: SMPLShapeAnnotations,
        smpl_pose: SMPLPoseAnnotations,
        smpl_color_override: tuple[float, float, float] | None = None,
        prefix: str = "kineo",
        fps: float = 25,
        pred_camera_extrinsics: CameraExtrinsicsAnnotations | None = None,
        gt_camera_extrinsics: CameraExtrinsicsAnnotations | None = None,
        start_frame_idx: int = 0,
        end_frame_idx: int = -1,
        log_pred_smpl_skeleton_2d: bool = False,
        projection_camera_extrinsics: CameraExtrinsicsAnnotations | None = None,
        projection_camera_intrinsics: CameraIntrinsicsAnnotations | None = None,
        skeleton_joint_radius_px: float = 14.0,
        skeleton_bones_thickness_px: float = 9.0,
        smpl_skeleton_2d_color_override: tuple[float, float, float] | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_layer = smpl_layer.to(device)
    smpl_shape = smpl_shape.to(device)
    smpl_pose = smpl_pose.to(device)

    subjects_ids = smpl_shape.subjects_ids

    if smpl_skeleton_2d_color_override is not None:
        subject_2d_colors = {subject_id: smpl_skeleton_2d_color_override + (1,) for subject_id in subjects_ids}
    else:
        subject_2d_colors = {subject_id: get_subject_color_rgba(subject_id) for subject_id in subjects_ids}

    if smpl_color_override is not None:
        subject_colors = {subject_id: smpl_color_override + (1,) for subject_id in subjects_ids}
    else:
        subject_colors = {subject_id: get_subject_color_rgba(subject_id) for subject_id in subjects_ids}

    R = torch.eye(3)
    T = torch.zeros(3)
    s = torch.ones(1)

    if pred_camera_extrinsics is not None and gt_camera_extrinsics is not None:
        R, T, s = pred_camera_extrinsics.compute_similarity_transform(gt_camera_extrinsics, estimate_scale=True)

    R = R.to(device)
    T = T.to(device)
    s = s.to(device)

    for subject_id in subjects_ids:
        subject_smpl_shape = smpl_shape.filter_by_subject_id(subject_id).first_or_default()
        subject_smpl_shape_betas = subject_smpl_shape.betas
        subject_smpl_poses = smpl_pose.filter_by_subject_id(subject_id)

        for pose_annotation in subject_smpl_poses:
            og_frame_idx = pose_annotation.frame_idx
            frame_idx = pose_annotation.frame_idx - start_frame_idx
            if og_frame_idx < start_frame_idx or (end_frame_idx != -1 and og_frame_idx > end_frame_idx):
                continue

            frame_timestamp = frame_idx / fps

            global_orient = pose_annotation.pose[0]
            body_pose = pose_annotation.pose[1:smpl_layer.NUM_BODY_JOINTS + 1]

            smpl_result = smpl_layer.forward(
                betas=subject_smpl_shape_betas.reshape(1, -1),
                body_pose=roma.rotvec_to_rotmat(body_pose).reshape(1, -1, 3),
                global_orient=roma.rotvec_to_rotmat(global_orient).reshape(1, 3, 3),
                transl=pose_annotation.trans.reshape(1, 3),
                pose2rot=False,
            )
            vertices = apply_similarity_transform_to_points(smpl_result.vertices, R, T, s).cpu().numpy().reshape(-1, 3)

            normals = compute_vertex_normals(smpl_layer.faces, vertices)

            rr.set_time("frame_idx", sequence=frame_idx)
            rr.set_time("time", timestamp=frame_timestamp)
            rr.log(
                f"{prefix}/persons/{subject_id}/smpl/",
                rr.Mesh3D(
                    vertex_positions=vertices,
                    triangle_indices=smpl_layer.faces,
                    vertex_normals=normals,
                    albedo_factor=subject_colors[subject_id],
                ),
            )

            if log_pred_smpl_skeleton_2d and projection_camera_extrinsics is not None and projection_camera_intrinsics is not None:
                n_body_joints = smpl_layer.NUM_BODY_JOINTS + 1
                joints_3d = apply_similarity_transform_to_points(
                    smpl_result.joints[:, :n_body_joints], R, T, s
                ).squeeze(0)  # (J, 3)

                parents = smpl_layer.parents[:n_body_joints].cpu().numpy()

                for view_id in projection_camera_extrinsics.views_ids:
                    cam_ext = projection_camera_extrinsics.filter_by_view_id(view_id).first_or_default()
                    cam_int = projection_camera_intrinsics.filter_by_view_id(view_id).first_or_default()

                    Rt_cam = cam_ext.Rt.to(device)
                    K = cam_int.K.to(device)

                    joints_cam = (Rt_cam[:3, :3] @ joints_3d.T + Rt_cam[:3, 3:4]).T  # (J, 3)
                    valid = joints_cam[:, 2] > 0

                    joints_2d_hom = (K @ joints_cam.T).T  # (J, 3)
                    joints_2d = joints_2d_hom[:, :2] / joints_2d_hom[:, 2:3]  # (J, 2)

                    valid_joints_2d = joints_2d[valid].cpu().numpy()
                    rr.log(
                        f"{prefix}/cameras/{view_id}/smpl_2d_joints_{subject_id}",
                        rr.Points2D(
                            positions=valid_joints_2d,
                            colors=[int(c * 255) for c in subject_2d_colors[subject_id]],
                            radii=skeleton_joint_radius_px,
                        ),
                    )

                    line_strips_2d = []
                    for j in range(n_body_joints):
                        parent = parents[j]
                        if parent >= 0 and valid[j] and valid[parent]:
                            line_strips_2d.append([
                                joints_2d[j].cpu().numpy(),
                                joints_2d[parent].cpu().numpy(),
                            ])

                    if line_strips_2d:
                        line_strips_2d = np.array(line_strips_2d).reshape(-1, 2, 2)
                        rr.log(
                            f"{prefix}/cameras/{view_id}/smpl_2d_bones_{subject_id}",
                            rr.LineStrips2D(
                                strips=line_strips_2d,
                                colors=[int(c * 255) for c in subject_2d_colors[subject_id]],
                                radii=skeleton_bones_thickness_px,
                            ),
                        )


def log_keypoints_2d(
        keypoints_2d: Keypoints2DAnnotations,
        prefix: str = "kineo",
        radius: float = 0.01,
        color_override: tuple[float, float, float] | None = None,
        min_keypoint_score_2d: float = 0.3,
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
                subject_kps_2d_valid_indices = (kps_2d_scores[view_idx, subject_idx] > min_keypoint_score_2d).nonzero()
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
        min_keypoint_score_3d: float = 0.5,
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

            subject_kps_3d_valid_indices = (kps_3d_scores[subject_idx] > min_keypoint_score_3d).nonzero()
            subject_valid_kps_xyz = kps_xyz[subject_idx][subject_kps_3d_valid_indices]

            rr.log(
                f"{prefix}/keypoints_3d_{subject_id}",
                rr.Points3D(
                    positions=subject_valid_kps_xyz.cpu().numpy(),
                    colors=[int(c * 255) for c in subject_colors[subject_id]],
                    radii=radius,
                ),
            )


def log_skeletons_3d(
        keypoints_3d: Keypoints3DAnnotations,
        prefix: str = "kineo",
        joint_radius: float = 0.01,
        bones_thickness: float = 0.01,
        color_override: tuple[float, float, float] | None = None,
        fps: float = 25,
        min_keypoint_score_3d: float = 0.5,
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

        rr.set_time("frame_idx", sequence=frame_idx)
        rr.set_time("time", timestamp=frame_timestamp)

        for subject_idx in range(n_subjects):
            subject_id = keypoints_3d.subjects_ids[subject_idx]

            subject_kps_3d_valid_indices = (kps_3d_scores[subject_idx] > min_keypoint_score_3d).nonzero()
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


def log_skeletons_2d(
        keypoints_2d: Keypoints2DAnnotations,
        prefix: str = "kineo",
        joint_radius: float = 0.01,
        bones_thickness: float = 0.01,
        color_override: tuple[float, float, float] | None = None,
        min_keypoint_score_2d: float = 0.3,
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

        rr.set_time("frame_idx", sequence=frame_idx)
        rr.set_time("time", timestamp=frame_timestamp)

        for subject_idx in range(n_subjects):
            subject_id = keypoints_2d.subjects_ids[subject_idx]

            for view_idx in range(n_views):
                view_id = views_ids[view_idx]
                subject_kps_2d_valid_indices = (kps_2d_scores[view_idx, subject_idx] > min_keypoint_score_2d).nonzero()

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
