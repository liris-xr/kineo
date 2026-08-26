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
from kineo.geometry.camera import transform_points_from_world_to_camera
from kineo.sampling import (
    farthest_point_sampling,
    normalized_uv,
    random_point_sampling,
    valid_observations_mask,
)
from kineo.geometry.triangulation import (
    triangulate_points,
    triangulate_points_in_chunks,
    triangulation_quality_mask,
)
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.geometry.transformations import undistort_points
from kineo.annotations.bundle_adjustment_keypoints import (
    BundleAdjustmentKeypointsAnnotations,
    BundleAdjustmentKeypointsAnnotation,
    BundleAdjustmentKeypointsAnnotationsMetadata,
)

from dataclasses import dataclass


def normalized_depths(
    points_3d: torch.Tensor,
    Rts: torch.Tensor,
    quantile: float = 0.01,
) -> torch.Tensor:
    """Per-view depths rescaled into the unit interval.

    The depth axis has to share a scale with the normalized image coordinates
    it is sampled alongside, and the sequence's depth range is not known ahead
    of time. Rescaling by the inner quantile range rather than by the extremes
    keeps a single far outlier from compressing every other point into one
    bucket, which would leave farthest-point sampling with nothing to separate.

    Args:
        points_3d: Points in the world frame with shape (n_points, 3).
        Rts: World-to-camera extrinsics with shape (n_views, 3, 4).
        quantile: Lower tail clipped away; the upper tail is its complement.

    Returns:
        Depths in ``[0, 1]`` with shape (n_views, n_points).
    """
    depths = transform_points_from_world_to_camera(points_3d, Rts)[..., 2]
    finite = torch.nan_to_num(depths, nan=0.0, posinf=0.0, neginf=0.0)

    low = torch.quantile(finite, quantile, dim=-1, keepdim=True)
    high = torch.quantile(finite, 1.0 - quantile, dim=-1, keepdim=True)

    return ((finite - low) / (high - low).clamp_min(1e-12)).clamp(0.0, 1.0)


@dataclass(frozen=True)
class BundleAdjustmentSamplingRuntimeConfig:
    keypoints_indices: list[int] | None = None
    # Per view, not per sequence: a rig's cameras each need their own image
    # coverage, and a shared budget would thin it as views are added. -1 keeps
    # every candidate.
    n_kp_samples_per_view: int = 50
    min_kp_score: float = 0.4
    sampler: str = "fps"
    # Off-frame and non-finite keypoints are the extremes FPS chases, so the
    # rejection is on by default. Turning it off is the uniform-baseline arm,
    # and isolates the filter from the sampler in the ablation.
    reject_invalid_observations: bool = True
    w_uv: float = 1.0
    w_t: float = 1.0
    w_d: float = 0.0
    filter_negative_depth: bool = False
    min_parallax_deg: float | None = None
    max_reproj_error: float | None = None
    triangulation_chunk_bytes: int = 128 * 1024 * 1024


class BundleAdjustmentSamplingStage(
    PipelineStage[BundleAdjustmentSamplingRuntimeConfig]
):
    """Samples the bundle adjustment observations, per view.

    ``sampler="fps"`` runs farthest-point sampling over ``[u, v, t]`` per view
    and unions the selections, so the observation set spans each camera's image
    plane and the sequence duration. ``sampler="random"`` draws uniformly over
    the same candidates instead, which is the baseline the coverage sampler is
    measured against.

    ``reject_invalid_observations`` gates off-frame and non-finite keypoints
    ahead of either sampler, so the filter and the sampler vary independently.
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: BundleAdjustmentSamplingRuntimeConfig,
        dynamic_runtime_cfg: dict[str, BundleAdjustmentSamplingRuntimeConfig]
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
        runtime_cfg: BundleAdjustmentSamplingRuntimeConfig,
    ):
        if runtime_cfg.sampler not in ("fps", "random"):
            raise ValueError(
                f"Unsupported sampler: {runtime_cfg.sampler}, "
                'expected "fps" or "random"'
            )

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
        n_subjects = len(subjects_ids)
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

        n_selected_keypoints = len(keypoints_indices)

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
        n_confident_candidates = int((confident_mask.sum(dim=0) >= 2).sum())

        resolutions_hw = torch.as_tensor(
            cameras_resolutions_hw, dtype=kps_xy.dtype, device=device
        )
        observations_mask = confident_mask

        if runtime_cfg.reject_invalid_observations:
            observations_mask = observations_mask & valid_observations_mask(
                kps_xy, resolutions_hw
            )

        candidates_indices = torch.where(observations_mask.sum(dim=0) >= 2)[0]
        n_unfiltered_candidates = len(candidates_indices)

        Ps = torch.einsum("cij,cjk->cik", Ks, Rts)

        # Both the quality gate and the depth sampling axis are functions of the
        # triangulated point, so the candidates have to be triangulated up front
        # rather than after the draw. Neither is free, so a configuration that
        # asks for neither keeps the cheaper sample-then-triangulate path.
        needs_candidates_geometry = (
            runtime_cfg.filter_negative_depth
            or runtime_cfg.min_parallax_deg is not None
            or runtime_cfg.max_reproj_error is not None
            or runtime_cfg.w_d > 0.0
        )

        # Drives candidate selection and the sampling features.
        sampling_mask = observations_mask
        # Drives the weights handed to the bundle adjustment. It starts fully
        # permissive so that an entry the gate never judged, because it was not
        # a real observation to begin with, keeps whatever weight it had.
        emitted_observations_mask = torch.ones_like(observations_mask)
        candidates_depths = None

        if needs_candidates_geometry:
            candidates_kps_xy = kps_xy[:, candidates_indices]
            candidates_observations_mask = observations_mask[:, candidates_indices]

            candidates_kps_3d = triangulate_points_in_chunks(
                Ps=Ps,
                points=self._undistorted(
                    candidates_kps_xy, Ks, dist_coeffs, distortion_model
                ),
                points_weights=kps_scores[:, candidates_indices],
                max_chunk_bytes=runtime_cfg.triangulation_chunk_bytes,
            )

            gated_mask = triangulation_quality_mask(
                points_3d=candidates_kps_3d,
                points_2d=candidates_kps_xy,
                Ks=Ks,
                Rts=Rts,
                Ds=dist_coeffs,
                distortion_model=distortion_model.value,
                observations_mask=candidates_observations_mask,
                min_parallax_deg=runtime_cfg.min_parallax_deg,
                max_reproj_error=runtime_cfg.max_reproj_error,
                reject_negative_depth=runtime_cfg.filter_negative_depth,
            )

            sampling_mask = observations_mask.clone()
            sampling_mask[:, candidates_indices] = gated_mask
            emitted_observations_mask[:, candidates_indices] = (
                gated_mask | ~candidates_observations_mask
            )

            surviving = torch.where(gated_mask.sum(dim=0) >= 2)[0]
            candidates_indices = candidates_indices[surviving]

            if runtime_cfg.w_d > 0.0:
                candidates_depths = normalized_depths(candidates_kps_3d, Rts)[
                    :, surviving
                ]

        n_candidates = len(candidates_indices)

        generator = torch.Generator(device=device)
        generator.manual_seed(pipeline.seed)

        picked_indices = candidates_indices[
            self._sample_candidates_per_view(
                kps_xy=kps_xy,
                observations_mask=sampling_mask,
                candidates_indices=candidates_indices,
                candidates_depths=candidates_depths,
                resolutions_hw=resolutions_hw,
                n_subjects=n_subjects,
                n_selected_keypoints=n_selected_keypoints,
                n_frames=n_frames,
                runtime_cfg=runtime_cfg,
                generator=generator,
            )
        ]

        n_samples = len(picked_indices)

        print(
            f"BA sampling [{runtime_cfg.sampler}]: {n_confident_candidates} "
            f"confident candidates -> {n_unfiltered_candidates} after "
            f"invalid-observation rejection -> {n_candidates} after the quality "
            f"gate -> {n_samples} sampled."
        )

        # Shape is (n_views, n_kps, 2)
        sampled_frame_kps_xy = kps_xy[:, picked_indices]
        # Shape is (n_views, n_kps). A gate-rejected observation is zeroed here
        # rather than dropped: the bundle adjustment weights by this score, so a
        # zero removes it from the objective while keeping the tensors dense.
        sampled_frame_kps_scores = (
            kps_scores[:, picked_indices]
            * emitted_observations_mask[:, picked_indices]
        )

        # Triangulated from the emitted weights, so a gate-rejected observation
        # no longer pulls on the point that seeds the bundle adjustment.
        sampled_frame_kps_3d = triangulate_points(
            Ps=Ps.reshape(n_views, 3, 4),
            points=self._undistorted(
                sampled_frame_kps_xy, Ks, dist_coeffs, distortion_model
            ).reshape(n_views, n_samples, 2),
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

    def _undistorted(
        self,
        kps_xy: torch.Tensor,
        Ks: torch.Tensor,
        dist_coeffs: torch.Tensor,
        distortion_model,
    ) -> torch.Tensor:
        """Undistorts keypoints view by view, skipping undistorted views.

        Args:
            kps_xy: Keypoints with shape (n_views, n_points, 2).
            Ks: Camera intrinsics with shape (n_views, 3, 3).
            dist_coeffs: Distortion coefficients, (n_views, n_coefficients).
            distortion_model: Distortion model of the rig.

        Returns:
            Undistorted keypoints with the shape of ``kps_xy``.
        """
        undistorted = kps_xy.clone()

        for view_idx in tqdm(
            range(kps_xy.shape[0]), desc="Undistorting keypoints", leave=False
        ):
            if torch.allclose(
                dist_coeffs[view_idx], torch.zeros_like(dist_coeffs[view_idx])
            ):
                continue

            undistorted[view_idx] = undistort_points(
                points=kps_xy[view_idx, :],
                K=Ks[view_idx],
                D=dist_coeffs[view_idx],
                distortion_model=distortion_model.value,
            )

        return undistorted

    def _sample_candidates_per_view(
        self,
        kps_xy: torch.Tensor,
        observations_mask: torch.Tensor,
        candidates_indices: torch.Tensor,
        candidates_depths: torch.Tensor | None,
        resolutions_hw: torch.Tensor,
        n_subjects: int,
        n_selected_keypoints: int,
        n_frames: int,
        runtime_cfg: BundleAdjustmentSamplingRuntimeConfig,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Samples each view over its own observations and unions the results.

        A candidate observed by different view subsets has no well-defined
        distance in a feature stacked across views, so each view samples within
        its own surviving observations and the selections are unioned.

        Sizing the draw per view keeps per-camera coverage constant as a rig
        grows, and stops a view from ranking candidates the union never
        consumes.

        Args:
            kps_xy: Keypoints of shape (n_views, n_flat, 2).
            observations_mask: Surviving observations, (n_views, n_flat) bool.
            candidates_indices: Flat indices of the surviving candidates.
            candidates_depths: Normalized per-view depths of the candidates,
                (n_views, n_candidates), or None to sample without a depth
                axis.
            resolutions_hw: Per-view (height, width).
            n_subjects: Number of subjects in the flattened candidate axis.
            n_selected_keypoints: Number of keypoints per subject.
            n_frames: Number of frames on the global time reference.
            runtime_cfg: Stage runtime configuration.
            generator: Seeded generator.

        Returns:
            Long tensor of indices into ``candidates_indices``.
        """
        n_views = kps_xy.shape[0]
        n_candidates = len(candidates_indices)
        per_view_budget = (
            n_candidates
            if runtime_cfg.n_kp_samples_per_view == -1
            else min(runtime_cfg.n_kp_samples_per_view, n_candidates)
        )

        # Every view ranks the same candidate axis and differs only in which
        # entries it observes, so the views form a uniform batch.
        view_candidates_mask = observations_mask[:, candidates_indices]

        if runtime_cfg.sampler == "random":
            views_orders = [
                view_order.tolist()
                for view_order in random_point_sampling(
                    view_candidates_mask, per_view_budget, generator
                )
            ]
        else:
            uvs = normalized_uv(kps_xy[:, candidates_indices], resolutions_hw)

            # Normalized time of each candidate, shared by every view.
            frame_indices = candidates_indices // (
                n_subjects * n_selected_keypoints
            )
            ts = frame_indices.to(kps_xy.dtype) / max(n_frames - 1, 1)

            axes = [
                runtime_cfg.w_uv * uvs[..., 0],
                runtime_cfg.w_uv * uvs[..., 1],
                runtime_cfg.w_t * ts.unsqueeze(0).expand(n_views, -1),
            ]

            if candidates_depths is not None:
                # Two points on the same ray share a (u, v) and are near
                # duplicates without this axis, while constraining the geometry
                # very differently.
                axes.append(runtime_cfg.w_d * candidates_depths)

            selected, selected_valid = farthest_point_sampling(
                torch.stack(axes, dim=-1),
                per_view_budget,
                generator,
                view_candidates_mask,
            )
            # Read back once: deduplicating the union is a per-element Python
            # loop, and indexing a device tensor per element would synchronize
            # on every lookup.
            views_orders = [
                row[keep].tolist() for row, keep in zip(selected, selected_valid)
            ]

        # Every view's draw is kept, so the union is a plain deduplication.
        union = dict.fromkeys(idx for order in views_orders for idx in order)

        return torch.as_tensor(
            list(union), dtype=torch.long, device=candidates_indices.device
        )
