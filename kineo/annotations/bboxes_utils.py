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
from kineo.annotations.bboxes_2d import (
    BBox2DAnnotations,
    BBox2DAnnotation,
    BBox2DAnnotationsMetadata,
)
from kineo.geometry.camera import (
    transform_points_from_world_to_camera,
    project_points_from_camera_to_image,
)
import math
import torch
from tqdm import tqdm
import warnings


def _union_area_int(rects: list[tuple[int, int, int, int]]) -> int:
    """Total pixel area covered by a union of axis-aligned integer rects.

    Rects are (x1, y1, x2, y2) with x2 > x1 and y2 > y1. Uses coordinate
    compression, so overlaps are counted once. Exact on the integer pixel grid,
    matching a boolean-mask OR of the same rects.
    """
    if not rects:
        return 0

    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    ys = sorted({r[1] for r in rects} | {r[3] for r in rects})

    area = 0
    for xi in range(len(xs) - 1):
        x0, x1 = xs[xi], xs[xi + 1]
        for yi in range(len(ys) - 1):
            y0, y1 = ys[yi], ys[yi + 1]
            for rx1, ry1, rx2, ry2 in rects:
                if rx1 <= x0 and x1 <= rx2 and ry1 <= y0 and y1 <= ry2:
                    area += (x1 - x0) * (y1 - y0)
                    break
    return area


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = kps_annotations.frames
    views_ids = cam_extrinsics.views_ids

    # Group keypoints by frame once (subject order preserved); reused per view so
    # the per-frame/per-subject filtering is not repeated for every camera.
    frame_subjects: list[tuple[int, list[tuple[str, torch.Tensor]]]] = []
    for frame_idx in frames:
        frame_kps = kps_annotations.filter_by_frame_idx(frame_idx)
        if len(frame_kps.annotations) == 0:
            continue
        subjects: list[tuple[str, torch.Tensor]] = []
        for subject_id in frame_kps.subjects_ids:
            subject_kps = frame_kps.filter_by_subject_id(subject_id).first_or_default()
            if subject_kps is None:
                continue
            subjects.append((subject_id, subject_kps.xyz))
        if subjects:
            frame_subjects.append((frame_idx, subjects))

    # Annotations are collected per (frame, view) and emitted in the original
    # frame -> view -> subject order at the end, so the output is byte-identical.
    results: dict[tuple[int, str], list[BBox2DAnnotation]] = {}

    for view_id in tqdm(views_ids, leave=False, desc="Generating bboxes"):
        view_cam_extrinsics = cam_extrinsics.filter_by_view_id(
            view_id
        ).first_or_default()
        view_cam_intrinsics = cam_intrinsics.filter_by_view_id(
            view_id
        ).first_or_default()

        if view_cam_extrinsics is None or view_cam_intrinsics is None:
            continue

        view_h, view_w = view_cam_intrinsics.resolution_hw
        D = view_cam_intrinsics.distortion_coefficients.to(device)
        distortion_model = view_cam_intrinsics.distortion_model
        K = view_cam_intrinsics.K.to(device)
        Rt = view_cam_extrinsics.Rt.to(device)

        # Flatten every (frame, subject) keypoint set of this view.
        entry_frames: list[int] = []
        entry_subjects: list[str] = []
        kps_list: list[torch.Tensor] = []
        for frame_idx, subjects in frame_subjects:
            for subject_id, xyz in subjects:
                entry_frames.append(frame_idx)
                entry_subjects.append(subject_id)
                kps_list.append(xyz)

        if not kps_list:
            continue

        # One batched projection for the whole view. Projection is elementwise
        # per keypoint, so this is bit-identical to projecting each subject
        # separately; only the per-subject min/max/mean reductions below matter.
        kps_world = torch.stack(kps_list).to(device)  # (N, J, 3)
        n_entries, n_joints, _ = kps_world.shape
        kps_cam = transform_points_from_world_to_camera(
            points_3d_world=kps_world.reshape(n_entries * n_joints, 3), Rt=Rt
        )
        kps_img, kps_depth = project_points_from_camera_to_image(
            points_3d_cam=kps_cam,
            K=K,
            D=D,
            distortion_model=distortion_model.value,
        )
        kps_img = kps_img.reshape(n_entries, n_joints, 2)
        kps_depth = kps_depth.reshape(n_entries, n_joints)

        min_xy = kps_img.min(dim=1).values  # (N, 2)
        max_xy = kps_img.max(dim=1).values  # (N, 2)
        depth = kps_depth.mean(dim=1)  # (N,)
        bbox_wh = max_xy - min_xy
        bbox_area = (bbox_wh[:, 0] * bbox_wh[:, 1]).cpu()  # (N,) float32

        # The occlusion logic below is scalar integer work; run it on CPU.
        bbox_xyxy = torch.cat([min_xy, max_xy], dim=1).cpu()  # (N, 4)
        min_c = min_xy.tolist()
        max_c = max_xy.tolist()
        depth_c = depth.tolist()

        per_frame: dict[int, list[int]] = {}
        for n in range(n_entries):
            per_frame.setdefault(entry_frames[n], []).append(n)

        for frame_idx, idxs in per_frame.items():
            # Drop subjects behind the camera first: they get no bbox and do not
            # occlude the others (matches the old `bbox_depth <= 0` skip).
            frame_bboxes: list[tuple[int, str, tuple[int, int, int, int], float]] = []
            for n in idxs:
                if depth_c[n] <= 0:
                    continue
                x0, y0 = min_c[n]
                x2, y2 = max_c[n]
                rect = (
                    math.floor(min(view_w, max(0.0, x0))),
                    math.floor(min(view_h, max(0.0, y0))),
                    math.ceil(min(view_w, max(0.0, x2))),
                    math.ceil(min(view_h, max(0.0, y2))),
                )
                frame_bboxes.append((n, entry_subjects[n], rect, depth_c[n]))

            frame_annotations: list[BBox2DAnnotation] = []
            for i in range(len(frame_bboxes)):
                i_n, i_subject_id, (i_x1, i_y1, i_x2, i_y2), i_depth = frame_bboxes[i]
                i_area_px = max(0, i_x2 - i_x1) * max(0, i_y2 - i_y1)

                # Union of nearer subjects' rects clipped to bbox_i = its occluded
                # pixel count (equivalent to the old image-sized mask).
                occluders: list[tuple[int, int, int, int]] = []
                for j in range(len(frame_bboxes)):
                    if i == j:
                        continue
                    _, _, (j_x1, j_y1, j_x2, j_y2), j_depth = frame_bboxes[j]
                    if j_depth < i_depth:
                        ox1 = max(i_x1, j_x1)
                        oy1 = max(i_y1, j_y1)
                        ox2 = min(i_x2, j_x2)
                        oy2 = min(i_y2, j_y2)
                        if ox2 > ox1 and oy2 > oy1:
                            occluders.append((ox1, oy1, ox2, oy2))

                bbox_area_visible = i_area_px - _union_area_int(occluders)
                bbox_area_visible_ratio = bbox_area_visible / (bbox_area[i_n] + 1e-6)

                if bbox_area_visible_ratio < min_bbox_visibility_ratio:
                    continue

                grown_xyxy = _grow_bbox(bbox_xyxy[i_n], grow_factor)

                if torch.isnan(grown_xyxy).any():
                    warnings.warn(
                        f"Bbox is nan in frame {frame_idx} for view {view_id} and subject {i_subject_id}"
                    )
                    continue

                frame_annotations.append(
                    BBox2DAnnotation(
                        view_id=view_id,
                        frame_idx=frame_idx,
                        subject_id=i_subject_id,
                        category_id=category_id,
                        xyxy=grown_xyxy.cpu(),
                        score=1.0,
                    )
                )

            results[(frame_idx, view_id)] = frame_annotations

    annotations: list[BBox2DAnnotation] = []
    for frame_idx, _ in frame_subjects:
        for view_id in views_ids:
            annotations.extend(results.get((frame_idx, view_id), []))

    return BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(),
        annotations=annotations,
    )
