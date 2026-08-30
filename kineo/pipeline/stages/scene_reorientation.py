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
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.bundle_adjustment_history import (
    BundleAdjustmentHistoryAnnotation,
    BundleAdjustmentHistoryAnnotations,
    BundleAdjustmentHistoryAnnotationsMetadata,
)
from kineo.geometry.conversions import OPENCV_WORLD_UP_VECTOR, OPENCV_CAMERA_UP_VECTOR
from kineo.geometry.transformations import apply_similarity_transform_to_Rt
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class SceneReorientationRuntimeConfig:
    min_kps_quantile: float = 0.01
    min_kps_score: float = 0.2


class SceneReorientationStage(PipelineStage[SceneReorientationRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: SceneReorientationRuntimeConfig,
        dynamic_runtime_cfg: dict[str, SceneReorientationRuntimeConfig] | None = None,
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
        runtime_cfg: SceneReorientationRuntimeConfig,
    ):
        device = pipeline.device

        camera_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "cameras_extrinsics"
        ]
        camera_extrinsics = camera_extrinsics.to(device)

        keypoints_3d: Keypoints3DAnnotations = annotations["keypoints_3d"]
        keypoints_3d = keypoints_3d.to(device)

        # Camera up vector in camera space
        cam_up = OPENCV_CAMERA_UP_VECTOR.to(device=device)
        cam_up_world = torch.stack(
            [annot.R.T @ cam_up for annot in camera_extrinsics.annotations]
        )
        cam_up_world_avg = cam_up_world.mean(dim=0)
        cam_up_world_avg = cam_up_world_avg / cam_up_world_avg.norm()

        R = rotation_matrix_from_vectors(
            cam_up_world_avg, OPENCV_WORLD_UP_VECTOR.to(device)
        )
        R = R.T

        new_keypoints_3d = keypoints_3d.apply_similarity_transform(
            R=R,
            t=torch.zeros(3, device=device),
            s=torch.ones(1, device=device),
        )

        new_camera_extrinsics = camera_extrinsics.apply_similarity_transform(
            R=R,
            t=torch.zeros(3, device=device),
            s=torch.ones(1, device=device),
        )

        up_axis = torch.argmax(OPENCV_WORLD_UP_VECTOR).item()
        floor_position = _compute_floor_position(
            new_keypoints_3d,
            up_axis=up_axis,
            min_kps_score=runtime_cfg.min_kps_score,
            min_kps_quantile=runtime_cfg.min_kps_quantile,
        )
        t_vector = torch.zeros(3, device=device)
        t_vector[up_axis] = -floor_position

        # Translate the keypoints so that the floor is at y = 0
        new_keypoints_3d = new_keypoints_3d.apply_similarity_transform(
            R=torch.eye(3, device=device),
            t=t_vector,
            s=torch.ones(1, device=device),
        )

        new_camera_extrinsics = new_camera_extrinsics.apply_similarity_transform(
            R=torch.eye(3, device=device),
            t=t_vector,
            s=torch.ones(1, device=device),
        )

        annotations["keypoints_3d"] = new_keypoints_3d.cpu()
        annotations["cameras_extrinsics"] = new_camera_extrinsics.cpu()

        ba_history: BundleAdjustmentHistoryAnnotations | None = annotations.get(
            "bundle_adjustment_history"
        )
        if ba_history is not None:
            new_history = []
            for entry in ba_history:
                Rts = entry.Rts.to(device)
                # Apply rotation
                Rts = apply_similarity_transform_to_Rt(
                    Rts, R, torch.zeros(3, device=device), torch.ones(1, device=device)
                )
                # Apply floor translation
                Rts = apply_similarity_transform_to_Rt(
                    Rts, torch.eye(3, device=device), t_vector, torch.ones(1, device=device)
                )
                new_history.append(BundleAdjustmentHistoryAnnotation(
                    stage_name=entry.stage_name,
                    stage_order=entry.stage_order,
                    iteration=entry.iteration,
                    view_ids=entry.view_ids,
                    Ks=entry.Ks,
                    dist_coeffs=entry.dist_coeffs,
                    Rts=Rts.cpu(),
                    loss=entry.loss,
                ))
            annotations["bundle_adjustment_history"] = BundleAdjustmentHistoryAnnotations(
                metadata=BundleAdjustmentHistoryAnnotationsMetadata(),
                annotations=new_history,
            )


def _compute_floor_position(
    keypoints_3d: Keypoints3DAnnotations,
    up_axis: int = 1,
    min_kps_score: float = 0.2,
    min_kps_quantile: float = 0.01,
) -> float:
    n_frames = keypoints_3d.n_frames
    subject_ids = keypoints_3d.subjects_ids
    n_subjects = len(keypoints_3d.subjects_ids)
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

    subject_id_to_idx = {subject_id: i for i, subject_id in enumerate(subject_ids)}

    keypoints = torch.zeros(
        (n_frames, n_subjects, kps_3d_format.n_keypoints, 3), device=device
    )
    keypoints_scores = torch.zeros(
        (n_frames, n_subjects, kps_3d_format.n_keypoints), device=device
    )

    for annot in keypoints_3d.annotations:
        frame_idx = annot.frame_idx
        subject_idx = subject_id_to_idx[annot.subject_id]
        keypoints[frame_idx, subject_idx] = annot.xyz
        keypoints_scores[frame_idx, subject_idx] = annot.scores

    keypoints = keypoints.reshape(-1, 3)
    keypoints_scores = keypoints_scores.reshape(-1)

    valid_kps_mask = (keypoints_scores > min_kps_score) & (torch.isfinite(keypoints).all(dim=-1))
    keypoints = keypoints[valid_kps_mask]
    keypoints_scores = keypoints_scores[valid_kps_mask]

    if len(keypoints) == 0:
        return 0

    return keypoints[..., up_axis].quantile(min_kps_quantile).item()


def rotation_matrix_from_vectors(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Returns the rotation matrix that aligns vector u to vector v
    :param u: Tensor of shape (3,)
    :param v: Tensor of shape (3,)
    :return: Rotation matrix (3x3) as a tensor
    """
    u = u / u.norm()
    v = v / v.norm()

    # Cross product and dot product
    cross = torch.linalg.cross(u, v)
    dot = torch.dot(u, v)
    norm_cross = cross.norm()

    if norm_cross < 1e-8:
        # Vectors are parallel or anti-parallel
        if dot > 0.9999:
            # Same direction
            return torch.eye(3, device=u.device)
        else:
            # Opposite direction - rotate 180 degrees around any orthogonal vector
            orthogonal = torch.tensor([1.0, 0.0, 0.0], device=u.device)
            if torch.allclose(u, orthogonal):
                orthogonal = torch.tensor([0.0, 1.0, 0.0], device=u.device)
            axis = torch.linalg.cross(u, orthogonal)
            axis = axis / axis.norm()
            return rotation_matrix_from_axis_angle(axis, torch.pi)

    # Rodrigues' formula
    K = torch.tensor(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]],
        device=u.device,
    )

    R = torch.eye(3, device=u.device) + K + K @ K * ((1 - dot) / (norm_cross**2))
    return R


def rotation_matrix_from_axis_angle(axis, angle):
    """
    Computes the rotation matrix using Rodrigues' formula given an axis and angle.
    :param axis: Tensor of shape (3,)
    :param angle: Scalar angle in radians
    :return: Rotation matrix (3x3)
    """
    axis = axis / axis.norm()
    K = torch.tensor(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        device=axis.device,
    )
    R = (
        torch.eye(3, device=axis.device)
        + torch.sin(angle) * K
        + (1 - torch.cos(angle)) * K @ K
    )
    return R
