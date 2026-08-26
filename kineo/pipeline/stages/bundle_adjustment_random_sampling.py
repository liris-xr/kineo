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
from tqdm import tqdm

from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import Annotations
from kineo.annotations.camera_intrinsics import (
    CameraIntrinsicsAnnotations,
    CameraDistortionModel,
)
from kineo.annotations.keypoints_2d import (
    Keypoints2DAnnotations,
    stage_keypoints_2d,
)
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.geometry.triangulation import triangulate_points
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.geometry.transformations import undistort_points
from kineo.annotations.bundle_adjustment_keypoints import (
    BundleAdjustmentKeypointsAnnotations,
    BundleAdjustmentKeypointsAnnotation,
    BundleAdjustmentKeypointsAnnotationsMetadata,
)

from dataclasses import dataclass


@dataclass(frozen=True)
class BundleAdjustmentRandomSamplingRuntimeConfig:
    keypoints_indices: list[int] | None = None
    # Per view, not per sequence: a rig's cameras each need their own share of
    # the observations, and a shared budget would thin it as views are added.
    # -1 keeps every candidate.
    n_kp_samples_per_view: int = 50
    min_kp_score: float = 0.4


class BundleAdjustmentRandomSamplingStage(
    PipelineStage[BundleAdjustmentRandomSamplingRuntimeConfig]
):
    """Draws the bundle adjustment observations uniformly at random.

    Each view draws from its own confident observations and the selections are
    unioned, so the draw is uniform over candidates rather than spread over the
    image plane. :class:`BundleAdjustmentFpsSamplingStage` is the counterpart
    that samples for coverage instead.
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: BundleAdjustmentRandomSamplingRuntimeConfig,
        dynamic_runtime_cfg: dict[str, BundleAdjustmentRandomSamplingRuntimeConfig]
        | None = None,
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
        runtime_cfg: BundleAdjustmentRandomSamplingRuntimeConfig,
    ):
        device = pipeline.device

        cameras_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "camera_intrinsics"
        ]

        cameras_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "camera_extrinsics"
        ]

        # Assuming kps2d are on a common time reference (same number of keypoints for each view)
        kps_2d: Keypoints2DAnnotations = annotations["keypoints_2d"]

        global_time_reference: GlobalTimeReferenceAnnotation = annotations[
            "global_time_reference"
        ].first_or_default()

        if global_time_reference is None:
            raise ValueError("Global time reference is not available")

        views_ids = [views["view_id"] for views in views]

        if not set(views_ids) <= set(cameras_intrinsics.views_ids) or not set(views_ids) <= set(cameras_extrinsics.views_ids):
            raise ValueError(
                "Views ids must be included in the cameras_intrinsics and cameras_extrinsics"
            )

        subjects_ids = kps_2d.subjects_ids
        n_views = len(views_ids)
        n_keypoints = kps_2d.metadata.formats[0].n_keypoints
        distortion_model = cameras_intrinsics.first_or_default().distortion_model

        Ks = torch.zeros(n_views, 3, 3, device=device)

        if distortion_model == CameraDistortionModel.BROWN_CONRADY:
            dist_coeffs = torch.zeros(n_views, 5, device=device)
        elif distortion_model == CameraDistortionModel.OPENCV_FISHEYE:
            dist_coeffs = torch.zeros(n_views, 4, device=device)
        else:
            raise ValueError(f"Unsupported distortion model: {distortion_model}")

        Rts = torch.zeros(n_views, 3, 4, device=device)
        cameras_resolutions_hw = []

        for view_idx, view_id in enumerate(views_ids):
            cam_intrinsics = cameras_intrinsics.filter_by_view_id(
                view_id
            ).first_or_default()
            cam_extrinsics = cameras_extrinsics.filter_by_view_id(
                view_id
            ).first_or_default()

            K = cam_intrinsics.K
            D = cam_intrinsics.distortion_coefficients
            Rt = cam_extrinsics.Rt

            Ks[view_idx] = K
            dist_coeffs[view_idx] = D
            Rts[view_idx] = Rt
            cameras_resolutions_hw.append(cam_intrinsics.resolution_hw)

        view_id_to_idx = {
            view_id: view_idx for view_idx, view_id in enumerate(views_ids)
        }
        subject_id_to_idx = {
            subject_id: subject_idx
            for subject_idx, subject_id in enumerate(subjects_ids)
        }

        if runtime_cfg.keypoints_indices is not None:
            keypoints_indices = torch.as_tensor(runtime_cfg.keypoints_indices)
        else:
            keypoints_indices = torch.arange(n_keypoints)

        world_timestamps = global_time_reference.timestamps
        n_frames = len(world_timestamps)

        kps_xy, kps_scores = stage_keypoints_2d(
            kps_2d=kps_2d,
            view_id_to_idx=view_id_to_idx,
            subject_id_to_idx=subject_id_to_idx,
            n_frames=n_frames,
            keypoints_indices=keypoints_indices,
        )

        # Staged frame-major, sampled view-major; the flat candidate axis stays
        # ordered (frame, subject, keypoint) either way.
        kps_xy = kps_xy.permute(1, 0, 2, 3, 4).reshape(n_views, -1, 2).to(device)
        kps_scores = kps_scores.permute(1, 0, 2, 3).reshape(n_views, -1).to(device)

        # Candidates are the keypoints with at least two views with score > min_kp_score
        confident_mask = kps_scores > runtime_cfg.min_kp_score
        candidates_indices = torch.where(confident_mask.sum(dim=0) >= 2)[0]

        n_candidates = len(candidates_indices)
        per_view_budget = (
            n_candidates
            if runtime_cfg.n_kp_samples_per_view == -1
            else min(runtime_cfg.n_kp_samples_per_view, n_candidates)
        )

        generator = torch.Generator(device=device)
        generator.manual_seed(pipeline.seed)

        views_orders = []

        for view_idx in tqdm(
            range(n_views), desc="Sampling candidates per view", leave=False
        ):
            view_candidates = torch.where(
                confident_mask[view_idx, candidates_indices]
            )[0]
            order = torch.randperm(
                len(view_candidates), generator=generator, device=device
            )[:per_view_budget]
            # Read back once: deduplicating the union is a per-element Python
            # loop, and indexing a device tensor per element would synchronize
            # on every lookup.
            views_orders.append(view_candidates[order].tolist())

        # Every view's draw is kept, so the union is a plain deduplication.
        union = dict.fromkeys(idx for order in views_orders for idx in order)
        picked_indices = candidates_indices[
            torch.as_tensor(list(union), dtype=torch.long, device=device)
        ]

        n_samples = len(picked_indices)
        # Shape is (n_views, n_kps, 2)
        sampled_frame_kps_xy = kps_xy[:, picked_indices]
        # Shape is (n_views, n_kps)
        sampled_frame_kps_scores = kps_scores[:, picked_indices]

        sampled_frame_kps_xy_undistorted = sampled_frame_kps_xy.clone()

        for view_idx in tqdm(
            range(n_views), desc="Undistorting keypoints", leave=False
        ):
            if torch.allclose(
                dist_coeffs[view_idx], torch.zeros_like(dist_coeffs[view_idx])
            ):
                continue

            sampled_frame_kps_xy_undistorted[view_idx] = undistort_points(
                points=sampled_frame_kps_xy[view_idx, :],
                K=Ks[view_idx],
                D=dist_coeffs[view_idx],
                distortion_model=distortion_model.value,
            )

        Ps = torch.einsum("cij,cjk->cik", Ks, Rts)

        sampled_frame_kps_3d = triangulate_points(
            Ps=Ps.reshape(n_views, 3, 4),
            points=sampled_frame_kps_xy_undistorted.reshape(n_views, n_samples, 2),
            points_weights=sampled_frame_kps_scores.reshape(n_views, n_samples),
        )

        annotations["bundle_adjustment_keypoints"] = (
            BundleAdjustmentKeypointsAnnotations(
                metadata=BundleAdjustmentKeypointsAnnotationsMetadata(),
                annotations=[
                    BundleAdjustmentKeypointsAnnotation(
                        view_ids=views_ids,
                        kps_2d_xy=sampled_frame_kps_xy,
                        kps_2d_scores=sampled_frame_kps_scores,
                        kps_3d=sampled_frame_kps_3d,
                    )
                ],
            ).cpu()
        )
