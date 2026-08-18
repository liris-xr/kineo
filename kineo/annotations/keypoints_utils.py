# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.keypoints_2d import (
    Keypoints2DAnnotations,
    Keypoints2DAnnotation,
    Keypoints2DAnnotationsMetadata,
)
from kineo.geometry.camera import (
    transform_points_from_world_to_camera,
    project_points_from_camera_to_image,
)
import torch
from kineo.torch_utils import check_shape


def project_kps3d_to_kps2d(
    kps_annotations: Keypoints3DAnnotations,
    view_ids: list[str],
    K: torch.Tensor,
    D: list[torch.Tensor],
    distortion_model: list[str],
    Rt: torch.Tensor,
) -> Keypoints2DAnnotations:
    check_shape(K, ("N", 3, 3))
    n_views = K.shape[0]
    check_shape(Rt, (n_views, 3, 4), (n_views, 4, 4))
    Rt = Rt[..., :3, :]

    if len(D) != n_views:
        raise ValueError(f"Expected D to be of length {n_views}, got {len(D)}")

    if len(distortion_model) != n_views:
        raise ValueError(
            f"Expected distortion_model to be of length {n_views}, got {len(distortion_model)}"
        )

    # Assuming all keypoints have the same format
    points_3d_world = torch.stack(
        [annotation.xyz for annotation in kps_annotations.annotations]
    ).to(device=K.device)

    annotations_2d = []

    for view_idx in range(n_views):
        view_K = K[view_idx]
        view_D = D[view_idx]
        view_distortion_model = distortion_model[view_idx]
        view_Rt = Rt[view_idx]

        points_3d_cam = transform_points_from_world_to_camera(points_3d_world, view_Rt)

        points_2d, _ = project_points_from_camera_to_image(
            points_3d=points_3d_cam,
            K=view_K,
            D=view_D,
            distortion_model=view_distortion_model,
        )

        annotations_2d.extend(
            [
                Keypoints2DAnnotation(
                    view_id=view_ids[view_idx],
                    frame_idx=orig_annotation.frame_idx,
                    subject_id=orig_annotation.subject_id,
                    xy=points_2d[i],
                    scores=orig_annotation.scores[i],
                    format=orig_annotation.format,
                )
                for i, orig_annotation in enumerate(kps_annotations.annotations)
            ]
        )

    return Keypoints2DAnnotations(
        metadata=Keypoints2DAnnotationsMetadata(
            formats=kps_annotations.metadata.formats,
        ),
        annotations=annotations_2d,
    )


def generate_kps2d_from_kps3d_and_cameras(
    kps_annotations: Keypoints3DAnnotations,
    cam_intrinsics: CameraIntrinsicsAnnotations,
    cam_extrinsics: CameraExtrinsicsAnnotations,
) -> Keypoints2DAnnotations:
    kps3d_annotations = kps_annotations.annotations
    views_ids = cam_extrinsics.views_ids

    if not kps3d_annotations:
        return Keypoints2DAnnotations(
            metadata=kps_annotations.metadata,
            annotations=[],
        )

    # Project every annotation's keypoints for a view in one batched call
    # (projection is elementwise per point). Stays on CPU like the per-point
    # version above did.
    kps_world = torch.stack([a.xyz for a in kps3d_annotations])  # (A, J, 3)
    n_annotations, n_joints, _ = kps_world.shape
    kps_world_flat = kps_world.reshape(n_annotations * n_joints, 3)

    view_kps_img: dict[str, torch.Tensor] = {}
    for view_id in views_ids:
        view_cam_extrinsics = cam_extrinsics.filter_by_view_id(
            view_id
        ).first_or_default()
        view_cam_intrinsics = cam_intrinsics.filter_by_view_id(
            view_id
        ).first_or_default()

        D = view_cam_intrinsics.distortion_coefficients
        distortion_model = view_cam_intrinsics.distortion_model
        K = view_cam_intrinsics.K
        Rt = view_cam_extrinsics.Rt

        kps_cam = transform_points_from_world_to_camera(
            points_3d_world=kps_world_flat, Rt=Rt
        )
        kps_img, _ = project_points_from_camera_to_image(
            points_3d_cam=kps_cam,
            K=K,
            D=D,
            distortion_model=distortion_model.value,
        )
        view_kps_img[view_id] = kps_img.reshape(n_annotations, n_joints, 2)

    # Emit in the original annotation-major, then view order.
    annotations: list[Keypoints2DAnnotation] = []
    for annotation_idx, annotation in enumerate(kps3d_annotations):
        for view_id in views_ids:
            annotations.append(
                Keypoints2DAnnotation(
                    view_id=view_id,
                    frame_idx=annotation.frame_idx,
                    subject_id=annotation.subject_id,
                    xy=view_kps_img[view_id][annotation_idx],
                    scores=annotation.scores.clone(),
                    format=annotation.format,
                )
            )

    return Keypoints2DAnnotations(
        metadata=kps_annotations.metadata,
        annotations=annotations,
    )
