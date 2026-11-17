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
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.camera_intrinsics import CameraDistortionModel
from kineo.annotations.bboxes_2d import (
    BBox2DAnnotations,
    BBox2DAnnotation,
    BBox2DAnnotationsMetadata,
)
from kineo.geometry.camera import (
    transform_points_from_world_to_camera,
    project_points_from_camera_to_image,
)
import torch
from tqdm import tqdm
import warnings


def _compute_subject_bbox_xyxyd(
    kps_world: torch.Tensor,
    Rt: torch.Tensor,
    K: torch.Tensor,
    D: torch.Tensor,
    distortion_model: CameraDistortionModel,
) -> tuple[torch.Tensor, float]:
    kps_cam = transform_points_from_world_to_camera(points_3d_world=kps_world, Rt=Rt)

    kps_img, kps_depth = project_points_from_camera_to_image(
        points_3d_cam=kps_cam,
        K=K,
        D=D,
        distortion_model=distortion_model.value,
    )

    min_x = kps_img[:, 0].min()
    min_y = kps_img[:, 1].min()
    max_x = kps_img[:, 0].max()
    max_y = kps_img[:, 1].max()

    bbox_xyxy = torch.tensor([min_x, min_y, max_x, max_y], device=kps_world.device)
    return bbox_xyxy, kps_depth.mean()


def _grow_bbox(bbox_xyxy: torch.Tensor, grow_factor: float) -> torch.Tensor:
    bbox_x = bbox_xyxy[0]
    bbox_y = bbox_xyxy[1]
    bbox_w = bbox_xyxy[2] - bbox_xyxy[0]
    bbox_h = bbox_xyxy[3] - bbox_xyxy[1]
    bbox_cx = bbox_x + bbox_w / 2
    bbox_cy = bbox_y + bbox_h / 2

    bbox_w = bbox_w * grow_factor
    bbox_h = bbox_h * grow_factor
    bbox_x = bbox_cx - bbox_w / 2
    bbox_y = bbox_cy - bbox_h / 2
    bbox_xyxy = torch.tensor([bbox_x, bbox_y, bbox_x + bbox_w, bbox_y + bbox_h])
    return bbox_xyxy


def generate_bboxes2d_from_kps2d(
    kps2d_annotations: Keypoints2DAnnotations,
    views_resolution_hw: dict[str, tuple[int, int]],
    category_id: int = 0,
    grow_factor: float = 1.3,
    min_bbox_visibility_ratio: float = 0.5,
) -> BBox2DAnnotations:
    """
    Generate 2D bounding boxes from 2D keypoints annotations.

    Args:
        kps2d_annotations: Keypoints2DAnnotations object.
        category_id: Category ID for the bounding boxes.
        grow_factor: Factor by which to grow the bounding boxes.
        min_bbox_visibility_ratio: Minimum proportion of bbox that is visible in the image.
    Returns:
        BBox2DAnnotations object.
    """
    annotations = []

    for annotation in kps2d_annotations.annotations:
        xy = annotation.xy  # (n_keypoints, 2)
        scores = annotation.scores  # (n_keypoints,)

        # Only use keypoints with positive scores
        valid_mask = scores > 0
        if valid_mask.sum() == 0:
            continue

        valid_xy = xy[valid_mask]

        min_x = valid_xy[:, 0].min()
        min_y = valid_xy[:, 1].min()
        max_x = valid_xy[:, 0].max()
        max_y = valid_xy[:, 1].max()

        bbox_xyxy = torch.tensor(
            [min_x, min_y, max_x, max_y], dtype=torch.float32, device=xy.device
        )
        bbox_xyxy = _grow_bbox(bbox_xyxy, grow_factor)

        resolution_hw = views_resolution_hw[annotation.view_id]

        h, w = resolution_hw
        x0, y0, x1, y1 = bbox_xyxy.tolist()
        x0_clip = max(0, min(w, x0))
        y0_clip = max(0, min(h, y0))
        x1_clip = max(0, min(w, x1))
        y1_clip = max(0, min(h, y1))
        area = max(0, x1 - x0) * max(0, y1 - y0)
        area_visible = max(0, x1_clip - x0_clip) * max(0, y1_clip - y0_clip)
        visible_ratio = area_visible / area if area > 0 else 0
        visible = visible_ratio >= min_bbox_visibility_ratio

        if not visible:
            continue

        # Use mean score of valid keypoints as bbox score
        bbox_score = scores[valid_mask].mean().item()

        bbox_ann = BBox2DAnnotation(
            view_id=annotation.view_id,
            frame_idx=annotation.frame_idx,
            subject_id=annotation.subject_id,
            category_id=category_id,
            xyxy=bbox_xyxy,
            score=bbox_score,
        )
        annotations.append(bbox_ann)

    return BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(),
        annotations=annotations,
    )


def generate_bboxes2d_from_kps3d_and_cameras(
    kps_annotations: Keypoints3DAnnotations,
    cam_intrinsics: CameraIntrinsicsAnnotations,
    cam_extrinsics: CameraExtrinsicsAnnotations,
    category_id: int = 0,
    grow_factor: float = 1.3,
    min_bbox_visibility_ratio: float = 0.5,
) -> BBox2DAnnotations:
    """
    Generate 2D bounding boxes from 2D keypoints.

    Args:
        kps_annotations: Keypoints2DAnnotations object.
        category_id: Category ID for the bounding boxes.
        grow_factor: Factor by which to grow the bounding boxes.
        min_bbox_visibility_ratio: Minimum proportion of bbox that is visible in the image.
            For example, 0.3 means that the bbox needs to be at least 30% visible in the image. If its occluded or outside the image, it will be discarded.
    """

    annotations: list[BBox2DAnnotation] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = kps_annotations.frames

    views_ids = cam_extrinsics.views_ids

    for frame_idx in tqdm(frames, leave=False, desc="Generating bboxes"):
        frame_kps = kps_annotations.filter_by_frame_idx(frame_idx)

        if len(frame_kps.annotations) == 0:
            continue

        for view_id in views_ids:
            view_cam_extrinsics = cam_extrinsics.filter_by_view_id(
                view_id
            ).first_or_default()

            view_cam_intrinsics = cam_intrinsics.filter_by_view_id(
                view_id
            ).first_or_default()

            if view_cam_extrinsics is None or view_cam_intrinsics is None:
                continue

            view_h, view_w = view_cam_intrinsics.resolution_hw
            D = view_cam_intrinsics.distortion_coefficients
            distortion_model = view_cam_intrinsics.distortion_model
            K = view_cam_intrinsics.K
            Rt = view_cam_extrinsics.Rt

            bboxes = []

            for subject_id in frame_kps.subjects_ids:
                subject_kps = frame_kps.filter_by_subject_id(
                    subject_id
                ).first_or_default()

                if subject_kps is None:
                    continue

                kps_world = subject_kps.xyz
                bbox_xyxy, bbox_depth = _compute_subject_bbox_xyxyd(
                    kps_world.to(device),
                    Rt.to(device),
                    K.to(device),
                    D.to(device),
                    distortion_model,
                )

                if bbox_depth <= 0:
                    continue

                bbox_w = bbox_xyxy[2] - bbox_xyxy[0]
                bbox_h = bbox_xyxy[3] - bbox_xyxy[1]
                bbox_area = bbox_w * bbox_h
                bbox_occluded = torch.zeros(
                    (view_h, view_w), dtype=torch.bool, device=device
                )

                bbox_x1 = torch.clamp(bbox_xyxy[0], 0, view_w).floor().int()
                bbox_y1 = torch.clamp(bbox_xyxy[1], 0, view_h).floor().int()
                bbox_x2 = torch.clamp(bbox_xyxy[2], 0, view_w).ceil().int()
                bbox_y2 = torch.clamp(bbox_xyxy[3], 0, view_h).ceil().int()
                bbox_occluded[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = True

                bboxes.append(
                    {
                        "subject_id": subject_id,
                        "bbox_xyxy": bbox_xyxy,
                        "bbox_depth": bbox_depth,
                        "bbox_occluded": bbox_occluded,
                        "bbox_area": bbox_area,
                    }
                )

            for bbox_i in range(len(bboxes)):
                bbox_i_subject_id = bboxes[bbox_i]["subject_id"]
                bbox_i_xyxy = bboxes[bbox_i]["bbox_xyxy"]
                bbox_i_depth = bboxes[bbox_i]["bbox_depth"]
                bbox_i_occluded = bboxes[bbox_i]["bbox_occluded"]

                for bbox_j in range(len(bboxes)):
                    if bbox_i == bbox_j:
                        continue

                    bbox_j_xyxy = bboxes[bbox_j]["bbox_xyxy"]
                    bbox_j_depth = bboxes[bbox_j]["bbox_depth"]
                    bbox_j_x1 = torch.clamp(bbox_j_xyxy[0], 0, view_w).floor().int()
                    bbox_j_y1 = torch.clamp(bbox_j_xyxy[1], 0, view_h).floor().int()
                    bbox_j_x2 = torch.clamp(bbox_j_xyxy[2], 0, view_w).ceil().int()
                    bbox_j_y2 = torch.clamp(bbox_j_xyxy[3], 0, view_h).ceil().int()

                    # The bbox is in front of the other bbox. We hide the overlapping part
                    if bbox_j_depth < bbox_i_depth:
                        bbox_i_occluded[bbox_j_y1:bbox_j_y2, bbox_j_x1:bbox_j_x2] = (
                            False
                        )

                bbox_area = bboxes[bbox_i]["bbox_area"]
                bbox_area_visible = bbox_i_occluded.sum()
                bbox_area_visible_ratio = bbox_area_visible / (bbox_area + 1e-6)

                if bbox_area_visible_ratio < min_bbox_visibility_ratio:
                    continue

                bbox_xyxy = _grow_bbox(bbox_i_xyxy, grow_factor)

                if torch.isnan(bbox_xyxy).any():
                    warnings.warn(
                        f"Bbox is nan in frame {frame_idx} for view {view_id} and subject {bbox_i_subject_id}"
                    )
                    continue

                annotations.append(
                    BBox2DAnnotation(
                        view_id=view_id,
                        frame_idx=frame_idx,
                        subject_id=bbox_i_subject_id,
                        category_id=category_id,
                        xyxy=bbox_xyxy.cpu(),
                        score=1.0,
                    )
                )

    return BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(),
        annotations=annotations,
    )
