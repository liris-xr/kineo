# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import torch
import torchvision.ops._utils
from kineo.geometry.camera import (
    transform_points_from_world_to_camera,
    project_points_from_camera_to_image,
)
from kineo.geometry.conversions import convert_points_basis, OPENCV_WORLD_BASIS
from kineo.torch_utils import check_shape
import torchvision
from tqdm import tqdm


def standardize_keypoints(
    keypoints: torch.Tensor,
    src_world_basis: torch.Tensor | None = None,
    src_world_unit_in_meters: float = 1.0,
):
    if torch.cuda.is_available():
        keypoints = keypoints.cuda()

    # Scale keypoints to meters
    if src_world_unit_in_meters != 1.0:
        keypoints = keypoints * src_world_unit_in_meters

    if src_world_basis is not None:
        keypoints = convert_points_basis(
            keypoints,
            src_world_basis.to(keypoints.device),
            OPENCV_WORLD_BASIS.to(keypoints.device),
        )

    return keypoints.cpu()


def compute_keypoints_3d_cam(
    keypoints_3d_world: torch.Tensor, Rt: torch.Tensor
) -> torch.Tensor:
    """
    Transform the 3D keypoints from world coordinates to camera coordinates.
    """
    check_shape(keypoints_3d_world, ("F", "S", "K", "3"))
    check_shape(Rt, (3, 4))

    keypoints_3d_world_homogeneous = torch.cat(
        [keypoints_3d_world, torch.ones_like(keypoints_3d_world[..., :1])], dim=-1
    )

    if torch.cuda.is_available():
        keypoints_3d_world_homogeneous = keypoints_3d_world_homogeneous.cuda()
        Rt = Rt.cuda()

    keypoints_3d_cam = torch.einsum(
        "ij,fnkj->fnki", Rt, keypoints_3d_world_homogeneous
    )[..., :3]

    return keypoints_3d_cam.cpu()


def compute_keypoints_2d(
    keypoints_3d_world: torch.Tensor,
    Rt: torch.Tensor,
    K: torch.Tensor,
    distortion_coefficients: torch.Tensor,
) -> torch.Tensor:
    """
    Project the 3D keypoints from world coordinates to 2D image coordinates.
    """
    orig_device = keypoints_3d_world.device
    orig_shape = keypoints_3d_world.shape
    check_shape(keypoints_3d_world, [("F", "S", "K", "3"), ("F", "K", "3")])
    check_shape(Rt, (3, 4))
    check_shape(K, (3, 3))
    check_shape(distortion_coefficients, ("D"))

    if keypoints_3d_world.ndim == 3:
        # Add the S dimension
        keypoints_3d_world = keypoints_3d_world.unsqueeze(1)

    n_frames = keypoints_3d_world.shape[0]

    if torch.cuda.is_available():
        keypoints_3d_world = keypoints_3d_world.cuda()
        Rt = Rt.cuda()
        K = K.cuda()
        distortion_coefficients = distortion_coefficients.cuda()

    # transform_points_from_world_to_camera and project_points_from_camera_to_image expect (F, P, 3) shape
    keypoints_3d_cam = transform_points_from_world_to_camera(
        keypoints_3d_world.view(n_frames, -1, 3), Rt
    )

    p_i = distortion_coefficients[..., [2, 3]]
    k_i = distortion_coefficients[
        ..., [0, 1] + list(range(4, distortion_coefficients.shape[0]))
    ]

    keypoints_2d, _ = project_points_from_camera_to_image(
        keypoints_3d_cam, K=K, k_i=k_i, p_i=p_i
    )

    return keypoints_2d.reshape(orig_shape[:-1] + (2,)).to(orig_device)


def compute_bboxes_xyxy(
    poses_2d: torch.Tensor,
    padding_x: int,
    padding_y: int,
    image_size_hw: tuple[int, int],
    clamp_to_image_size: bool = True,
) -> torch.Tensor:
    x1 = torch.min(poses_2d[..., 0], dim=-1).values
    x2 = torch.max(poses_2d[..., 0], dim=-1).values
    y1 = torch.min(poses_2d[..., 1], dim=-1).values
    y2 = torch.max(poses_2d[..., 1], dim=-1).values

    if clamp_to_image_size:
        x1 = torch.clamp(x1, min=0)
        x2 = torch.clamp(x2, max=image_size_hw[1] - 1)
        y1 = torch.clamp(y1, min=0)
        y2 = torch.clamp(y2, max=image_size_hw[0] - 1)

    x1 = x1 - padding_x
    x2 = x2 + padding_x
    y1 = y1 - padding_y
    y2 = y2 + padding_y

    if clamp_to_image_size:
        x1 = torch.clamp(x1, min=0)
        x2 = torch.clamp(x2, max=image_size_hw[1] - 1)
        y1 = torch.clamp(y1, min=0)
        y2 = torch.clamp(y2, max=image_size_hw[0] - 1)

    return torch.stack([x1, y1, x2, y2], dim=-1)


def suppress_overlapping_bboxes(
    bboxes_xywh: torch.Tensor,
    distance: torch.Tensor,
    overlap_threshold: float = 0.75,
) -> torch.Tensor:
    """
    Suppress overlapping bboxes based on the overlap threshold.
    When two bboxes exceed the overlap threshold, the bbox with lower distance is kept.

    Args:
        bboxes_xywh: (F, S, 4) tensor of bounding boxes in xywh format. With F being the number of frames, S being the number of subjects.
        distance: (F, S) tensor of distance of each bounding box to the camera
        overlap_threshold: overlap threshold for suppression (default: 0.75)

    Returns:
        keep: (F, S) tensor of boolean values indicating whether the bbox should be kept
    """

    check_shape(bboxes_xywh, ("F", "S", "4"))
    check_shape(distance, ("F", "S"))

    n_frames, n_subjects, _ = bboxes_xywh.shape

    bboxes_xyxy = torchvision.ops.box_convert(
        bboxes_xywh.view(-1, 4), in_fmt="xywh", out_fmt="xyxy"
    ).reshape_as(bboxes_xywh)

    keep = torch.ones(
        (n_frames, n_subjects),
        dtype=torch.bool,
        device=bboxes_xyxy.device,
    )

    # Sort subjects by distance (closest first)
    sorted_subjects_ids_by_distance = torch.argsort(distance, descending=False, dim=-1)

    pbar = tqdm(
        total=n_frames * n_subjects, desc="Suppressing overlapping bboxes", leave=False
    )

    for frame_idx in range(n_frames):
        for subject_idx in range(n_subjects):
            subject_id = sorted_subjects_ids_by_distance[frame_idx, subject_idx]

            if not keep[frame_idx, subject_id]:
                pbar.update(1)
                continue  # already suppressed

            # All the subjects that are further away from the current subject are potentially occluded.
            potentially_occluded_idxs = sorted_subjects_ids_by_distance[
                frame_idx, subject_idx + 1 :
            ]

            if potentially_occluded_idxs.numel() == 0:
                pbar.update(1)
                continue

            current_subject_bbox = bboxes_xyxy[frame_idx, subject_id].unsqueeze(0)
            potentially_occluded_bboxes = bboxes_xyxy[
                frame_idx, potentially_occluded_idxs
            ]

            lt = torch.max(
                current_subject_bbox[:, :2], potentially_occluded_bboxes[:, :2]
            )
            rb = torch.min(
                current_subject_bbox[:, 2:], potentially_occluded_bboxes[:, 2:]
            )

            intersection_bbox = torch.cat([lt, rb], dim=-1)
            intersection_bbox_area = torchvision.ops.box_area(intersection_bbox)

            # If the overlap between the potentially occluded bboxes and the intersection bbox is greater than the overlap threshold, then the current subject is occluded.
            # The simplest example for this is when the bbox of the occluded subject is completely inside the current subject bbox, in that case the overlap is 1.
            potentially_occluded_bboxes_area = torchvision.ops.box_area(
                potentially_occluded_bboxes
            )
            overlap = intersection_bbox_area / potentially_occluded_bboxes_area

            suppress = overlap > overlap_threshold
            keep[frame_idx, potentially_occluded_idxs[suppress]] = False
            pbar.update(1)

    pbar.close()
    return keep
