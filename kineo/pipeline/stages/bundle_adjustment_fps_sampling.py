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
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
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


def farthest_point_sampling(
    features: torch.Tensor,
    n_samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Greedy farthest-point sampling in Euclidean feature space.

    The first point is drawn uniformly at random; every subsequent point is the
    one maximizing the distance to the already selected set.

    Args:
        features: Point features of shape (n_points, n_dims).
        n_samples: Number of points to select. Clamped to ``n_points``.
        generator: Seeded generator used to draw the first point.

    Returns:
        Long tensor of shape (min(n_samples, n_points),) holding the selected
        row indices of ``features``, in selection order.
    """
    n_points = features.shape[0]
    n_samples = min(max(n_samples, 0), n_points)

    if n_samples == 0:
        return torch.empty(0, dtype=torch.long, device=features.device)

    selected = torch.empty(n_samples, dtype=torch.long, device=features.device)
    selected[0] = torch.randint(
        n_points, (1,), generator=generator, device=features.device
    )

    # Squared distance from every point to the closest selected one. Already
    # selected points are driven negative so argmax cannot pick them twice,
    # which a duplicated feature row would otherwise allow.
    min_sq_dists = torch.full(
        (n_points,), float("inf"), device=features.device, dtype=features.dtype
    )

    for i in range(1, n_samples):
        sq_dists = torch.sum(
            (features - features[selected[i - 1]]) ** 2, dim=1
        )
        min_sq_dists = torch.minimum(min_sq_dists, sq_dists)
        min_sq_dists[selected[i - 1]] = -1.0
        selected[i] = torch.argmax(min_sq_dists)

    return selected


def valid_observations_mask(
    kps_xy: torch.Tensor,
    resolutions_hw: torch.Tensor,
) -> torch.Tensor:
    """Flags the keypoints that are finite and inside their own view's frame.

    Args:
        kps_xy: Keypoint positions of shape (n_views, n_candidates, 2).
        resolutions_hw: Per-view (height, width) of shape (n_views, 2).

    Returns:
        Bool tensor of shape (n_views, n_candidates), False where a keypoint is
        non-finite or falls outside ``[0, W] x [0, H]`` of its own view.
    """
    heights = resolutions_hw[:, 0].unsqueeze(1)
    widths = resolutions_hw[:, 1].unsqueeze(1)

    us = kps_xy[..., 0]
    vs = kps_xy[..., 1]

    in_frame = (us >= 0) & (us <= widths) & (vs >= 0) & (vs <= heights)
    return in_frame & torch.isfinite(kps_xy).all(dim=-1)


@dataclass(frozen=True)
class BundleAdjustmentFpsSamplingRuntimeConfig:
    keypoints_indices: list[int] | None = None
    n_kp_samples: int = 1000
    min_kp_score: float = 0.4
    sampler: str = "fps"
    w_uv: float = 1.0
    w_t: float = 1.0


class BundleAdjustmentFpsSamplingStage(
    PipelineStage[BundleAdjustmentFpsSamplingRuntimeConfig]
):
    """
    Samples the bundle adjustment observations with image-plane and time coverage.

    Drop-in alternative to :class:`BundleAdjustmentSamplingStage`. Keypoints that
    are off-frame or non-finite are rejected, then farthest-point sampling over
    ``[u, v, t]`` is run per view and the selections are unioned, so the
    observation set spans each camera's image plane and the sequence duration
    instead of being an unstructured uniform draw.
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: BundleAdjustmentFpsSamplingRuntimeConfig,
        dynamic_runtime_cfg: dict[str, BundleAdjustmentFpsSamplingRuntimeConfig]
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
        runtime_cfg: BundleAdjustmentFpsSamplingRuntimeConfig,
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

        kps_xy = torch.zeros(
            n_views, n_frames, n_subjects, n_selected_keypoints, 2, device=device
        )
        kps_scores = torch.zeros(
            n_views, n_frames, n_subjects, n_selected_keypoints, device=device
        )

        for annot in tqdm(
            kps_2d.annotations, desc="Collecting 2D keypoints", leave=False
        ):
            view_idx = view_id_to_idx[annot.view_id]
            subject_idx = subject_id_to_idx[annot.subject_id]
            kps_xy[view_idx, annot.frame_idx, subject_idx] = annot.xy[keypoints_indices]
            kps_scores[view_idx, annot.frame_idx, subject_idx] = annot.scores[
                keypoints_indices
            ]

        kps_xy = kps_xy.reshape(n_views, -1, 2)
        kps_scores = kps_scores.reshape(n_views, -1)

        # Candidates are the keypoints with at least two views with score > min_kp_score
        confident_mask = kps_scores > runtime_cfg.min_kp_score
        n_confident_candidates = int((confident_mask.sum(dim=0) >= 2).sum())

        resolutions_hw = torch.as_tensor(
            cameras_resolutions_hw, dtype=kps_xy.dtype, device=device
        )
        # An off-frame or non-finite keypoint is exactly the kind of extreme that
        # would dominate the FPS selection, so it is dropped before sampling.
        observations_mask = confident_mask & valid_observations_mask(
            kps_xy, resolutions_hw
        )

        candidates_indices = torch.where(observations_mask.sum(dim=0) >= 2)[0]
        n_candidates = len(candidates_indices)

        generator = torch.Generator(device=device)
        generator.manual_seed(pipeline.seed)

        if runtime_cfg.n_kp_samples == -1:
            picked_indices = candidates_indices
        elif runtime_cfg.sampler == "random":
            picked_indices = candidates_indices[
                torch.randperm(n_candidates, generator=generator, device=device)[
                    : runtime_cfg.n_kp_samples
                ]
            ]
        else:
            picked_indices = candidates_indices[
                self._farthest_point_sample_candidates(
                    kps_xy=kps_xy,
                    observations_mask=observations_mask,
                    candidates_indices=candidates_indices,
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
            f"confident candidates -> {n_candidates} after invalid-observation "
            f"rejection -> {n_samples} sampled."
        )

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

    def _farthest_point_sample_candidates(
        self,
        kps_xy: torch.Tensor,
        observations_mask: torch.Tensor,
        candidates_indices: torch.Tensor,
        resolutions_hw: torch.Tensor,
        n_subjects: int,
        n_selected_keypoints: int,
        n_frames: int,
        runtime_cfg: BundleAdjustmentFpsSamplingRuntimeConfig,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Runs FPS over ``[u, v, t]`` per view and unions the selections.

        A candidate observed by different view subsets has no well-defined
        distance in a feature stacked across views, so each view samples over its
        own surviving observations. The per-view selections are then interleaved
        round-robin, which both spreads the budget evenly over the views and
        tops up the union when views select the same candidate.

        Args:
            kps_xy: Keypoints of shape (n_views, n_flat, 2).
            observations_mask: Surviving observations, (n_views, n_flat) bool.
            candidates_indices: Flat indices of the surviving candidates.
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
        n_samples = min(runtime_cfg.n_kp_samples, n_candidates)

        # Normalized time of each surviving candidate, shared by every view.
        frame_indices = candidates_indices // (n_subjects * n_selected_keypoints)
        ts = frame_indices.to(kps_xy.dtype) / max(n_frames - 1, 1)

        views_orders = []

        for view_idx in tqdm(
            range(n_views), desc="Farthest point sampling", leave=False
        ):
            # Indices into candidates_indices, i.e. the filtered candidate space.
            view_candidates = torch.where(
                observations_mask[view_idx, candidates_indices]
            )[0]

            if len(view_candidates) == 0:
                views_orders.append(view_candidates)
                continue

            uvs = kps_xy[view_idx, candidates_indices[view_candidates]]
            height, width = resolutions_hw[view_idx]

            features = torch.stack(
                [
                    runtime_cfg.w_uv * uvs[:, 0] / width,
                    runtime_cfg.w_uv * uvs[:, 1] / height,
                    runtime_cfg.w_t * ts[view_candidates],
                ],
                dim=1,
            )

            order = farthest_point_sampling(features, n_samples, generator)
            # Interleaving is a per-element Python loop, so the orders are read
            # back once here rather than synchronizing on every lookup.
            views_orders.append(view_candidates[order].tolist())

        picked = set()
        picked_order = []

        for rank in range(max((len(order) for order in views_orders), default=0)):
            if len(picked_order) == n_samples:
                break

            for order in views_orders:
                if rank >= len(order) or order[rank] in picked:
                    continue

                picked.add(order[rank])
                picked_order.append(order[rank])

                if len(picked_order) == n_samples:
                    break

        return torch.as_tensor(
            picked_order, dtype=torch.long, device=candidates_indices.device
        )
