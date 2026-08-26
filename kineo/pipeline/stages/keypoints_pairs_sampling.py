# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import itertools
from dataclasses import dataclass

import torch
from tqdm import tqdm

from kineo.annotations import Annotations
from kineo.annotations import Keypoints2DAnnotations
from kineo.annotations.calibration_points_pairs import (
    CalibrationPointsAnnotation,
    CalibrationPointsAnnotations,
    CalibrationPointsAnnotationsMetadata,
)
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.annotations.keypoints_2d import stage_keypoints_2d
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.pipeline.pipeline import Pipeline, PipelineStage
from kineo.sampling import (
    farthest_point_sampling,
    normalized_uv,
    random_point_sampling,
    valid_observations_mask,
)

# Image-plane coordinates of both views plus the time axis.
N_SAMPLING_AXES = 5


def pair_sampling_features(
    kps_xy: torch.Tensor,
    resolutions_hw: torch.Tensor,
    pairs: torch.Tensor,
    ts: torch.Tensor,
    w_uv: float,
    w_t: float,
) -> torch.Tensor:
    """Builds the ``[u_i, v_i, u_j, v_j, t]`` feature of every candidate pair.

    The sampled object is a correspondence rather than a keypoint, so it is
    placed by where it lands in *both* images: two correspondences sharing a
    position in the first view but not the second are near duplicates under a
    single-view feature while constraining the epipolar geometry differently.

    ``w_uv`` is split across the four image axes so that the image-plane block
    does not outweigh the single time axis purely because it is spread over
    more dimensions.

    Args:
        kps_xy: Keypoints of shape (n_views, n_flat, 2).
        resolutions_hw: Per-view (height, width) of shape (n_views, 2).
        pairs: View index pairs of shape (n_pairs, 2).
        ts: Normalized time of every candidate, shape (n_flat,).
        w_uv: Weight of the image-plane block.
        w_t: Weight of the time axis.

    Returns:
        Float tensor of shape (n_pairs, n_flat, 5).
    """
    views_i, views_j = pairs[:, 0], pairs[:, 1]

    normalized = normalized_uv(kps_xy, resolutions_hw)

    uv_weight = w_uv / 2.0
    n_pairs, n_flat = len(pairs), kps_xy.shape[1]

    return torch.cat(
        [
            uv_weight * normalized[views_i],
            uv_weight * normalized[views_j],
            (w_t * ts).expand(n_pairs, n_flat).unsqueeze(-1),
        ],
        dim=-1,
    )


def pair_candidates_mask(
    kps_scores: torch.Tensor,
    observations_mask: torch.Tensor,
    pairs: torch.Tensor,
    pair_avg_conf_score_thr: float,
) -> torch.Tensor:
    """Flags the candidates each view pair may draw from.

    Args:
        kps_scores: Keypoint scores of shape (n_views, n_flat).
        observations_mask: Valid observations, (n_views, n_flat) bool.
        pairs: View index pairs of shape (n_pairs, 2).
        pair_avg_conf_score_thr: Lower bound on the geometric mean of the two
            views' scores.

    Returns:
        Bool tensor of shape (n_pairs, n_flat).
    """
    views_i, views_j = pairs[:, 0], pairs[:, 1]

    pair_avg_conf_scores = torch.sqrt(kps_scores[views_i] * kps_scores[views_j])

    return (
        (pair_avg_conf_scores > pair_avg_conf_score_thr)
        & observations_mask[views_i]
        & observations_mask[views_j]
    )


def pairs_chunk_size(n_pairs: int, n_flat: int, max_chunk_bytes: int) -> int:
    """Number of pairs whose feature block fits in the byte budget.

    A 20-camera rig over a long sequence stacks 190 pairs against tens of
    thousands of candidates, which is hundreds of megabytes of features that
    only ever get read one chunk at a time.

    The budget covers the feature tensor alone. Each greedy step materializes a
    difference tensor of the same shape, so expect a peak a few times this —
    around 490 MB at the 128 MiB default on a 20-camera rig.

    Chunking is a memory knob only: rows draw their first point from the
    generator in row order regardless of where the chunk boundaries fall, so
    the selection is invariant to it.

    Args:
        n_pairs: Total number of view pairs.
        n_flat: Number of candidates on the flattened axis.
        max_chunk_bytes: Byte budget for one chunk's feature tensor.

    Returns:
        Chunk size in pairs, at least 1 and at most ``n_pairs``.
    """
    bytes_per_pair = n_flat * N_SAMPLING_AXES * 4
    return max(1, min(n_pairs, max_chunk_bytes // max(bytes_per_pair, 1)))


@dataclass(frozen=True)
class KeypointsPairsSamplingRuntimeConfig:
    max_points_pairs: int = 2000
    pair_avg_conf_score_thr: float = 0.75
    keypoints_indices: list[int] | None = None
    sampler: str = "fps"
    # Off-frame and non-finite keypoints are the extremes FPS chases, so the
    # rejection is on by default. Turning it off is the uniform-baseline arm,
    # and isolates the filter from the sampler in the ablation.
    reject_invalid_observations: bool = True
    w_uv: float = 1.0
    w_t: float = 1.0
    max_chunk_bytes: int = 128 * 1024 * 1024


class KeypointsPairsSamplingStage(
    PipelineStage[KeypointsPairsSamplingRuntimeConfig]
):
    """Samples the calibration point pairs, per view pair.

    ``sampler="fps"`` runs farthest-point sampling over
    ``[u_i, v_i, u_j, v_j, t]``, so each pair's correspondences span both image
    planes and the sequence duration. ``sampler="random"`` draws uniformly over
    the same candidates instead, which is the baseline the coverage sampler is
    measured against.

    ``reject_invalid_observations`` gates off-frame and non-finite keypoints
    ahead of either sampler, so the filter and the sampler vary independently.
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: KeypointsPairsSamplingRuntimeConfig,
        dynamic_runtime_cfg: dict[str, KeypointsPairsSamplingRuntimeConfig]
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
        runtime_cfg: KeypointsPairsSamplingRuntimeConfig,
    ):
        if runtime_cfg.sampler not in ("fps", "random"):
            raise ValueError(
                f"Unsupported sampler: {runtime_cfg.sampler}, "
                'expected "fps" or "random"'
            )

        device = pipeline.device
        views_ids = [view["view_id"] for view in views]
        n_views = len(views_ids)
        pairs = torch.as_tensor(
            list(itertools.combinations(range(n_views), 2)), device=device
        )

        # Assuming keypoints are synchronized and time is expressed in global time
        kps_2d: Keypoints2DAnnotations = annotations["keypoints_2d"]

        cameras_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "camera_intrinsics"
        ]

        global_time_reference: GlobalTimeReferenceAnnotation = annotations[
            "global_time_reference"
        ].first_or_default()

        if global_time_reference is None:
            raise ValueError("Global time reference is not available")

        if not set(views_ids) <= set(cameras_intrinsics.views_ids):
            raise ValueError("Views ids must be included in the cameras_intrinsics")

        subjects_ids = kps_2d.subjects_ids
        n_frames = len(global_time_reference.timestamps)
        n_subjects = len(subjects_ids)

        subject_id_to_idx = {
            subject_id: idx for idx, subject_id in enumerate(subjects_ids)
        }
        view_id_to_idx = {view_id: idx for idx, view_id in enumerate(views_ids)}

        # Assuming the same number of keypoints for all subjects
        n_keypoints = kps_2d.metadata.formats[0].n_keypoints

        if runtime_cfg.keypoints_indices is not None:
            keypoints_indices = torch.as_tensor(
                runtime_cfg.keypoints_indices
            ).reshape(-1)
        else:
            keypoints_indices = torch.arange(n_keypoints).reshape(-1)

        n_selected_keypoints = len(keypoints_indices)

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
        n_flat = kps_xy.shape[1]

        resolutions_hw = torch.as_tensor(
            [
                cameras_intrinsics.filter_by_view_id(view_id)
                .first_or_default()
                .resolution_hw
                for view_id in views_ids
            ],
            dtype=kps_xy.dtype,
            device=device,
        )

        observations_mask = (
            valid_observations_mask(kps_xy, resolutions_hw)
            if runtime_cfg.reject_invalid_observations
            else torch.ones(n_views, n_flat, dtype=torch.bool, device=device)
        )

        frame_indices = torch.arange(n_flat, device=device) // (
            n_subjects * n_selected_keypoints
        )
        ts = frame_indices.to(kps_xy.dtype) / max(n_frames - 1, 1)

        # Drawing from the global RNG would make the sample depend on whatever
        # consumed it earlier, so a cached stage upstream would silently change
        # which pairs are picked.
        generator = torch.Generator(device=device)
        generator.manual_seed(pipeline.seed)

        picked_per_pair = self._sample_pairs(
            kps_xy=kps_xy,
            kps_scores=kps_scores,
            observations_mask=observations_mask,
            resolutions_hw=resolutions_hw,
            pairs=pairs,
            ts=ts,
            runtime_cfg=runtime_cfg,
            generator=generator,
        )

        calibration_points_annotations: list[CalibrationPointsAnnotation] = []

        for (view_i, view_j), picked_indices in zip(pairs.tolist(), picked_per_pair):
            pair_avg_conf_scores = torch.sqrt(
                kps_scores[view_i, picked_indices]
                * kps_scores[view_j, picked_indices]
            ).cpu()
            pair_points_i = kps_xy[view_i, picked_indices].cpu()
            pair_points_j = kps_xy[view_j, picked_indices].cpu()
            pair_frame_indices = frame_indices[picked_indices].cpu()

            calibration_points_annotations.append(
                CalibrationPointsAnnotation(
                    view1_id=views_ids[view_i],
                    view2_id=views_ids[view_j],
                    points1=pair_points_i,
                    points2=pair_points_j,
                    confidence_scores=pair_avg_conf_scores,
                    frame_indices=pair_frame_indices,
                )
            )
            calibration_points_annotations.append(
                CalibrationPointsAnnotation(
                    view1_id=views_ids[view_j],
                    view2_id=views_ids[view_i],
                    points1=pair_points_j,
                    points2=pair_points_i,
                    confidence_scores=pair_avg_conf_scores,
                    frame_indices=pair_frame_indices,
                )
            )

        annotations["calibration_points"] = CalibrationPointsAnnotations(
            metadata=CalibrationPointsAnnotationsMetadata(),
            annotations=calibration_points_annotations,
        ).cpu()

    def _sample_pairs(
        self,
        kps_xy: torch.Tensor,
        kps_scores: torch.Tensor,
        observations_mask: torch.Tensor,
        resolutions_hw: torch.Tensor,
        pairs: torch.Tensor,
        ts: torch.Tensor,
        runtime_cfg: KeypointsPairsSamplingRuntimeConfig,
        generator: torch.Generator,
    ) -> list[torch.Tensor]:
        """Draws each view pair's correspondences from its own candidate set.

        Pairs are sampled in chunks rather than one at a time: the greedy loop
        is launch-bound, so stepping every pair in a chunk together costs one
        round of kernels per sample instead of one per sample per pair.

        Args:
            kps_xy: Keypoints of shape (n_views, n_flat, 2).
            kps_scores: Keypoint scores of shape (n_views, n_flat).
            observations_mask: Valid observations, (n_views, n_flat) bool.
            resolutions_hw: Per-view (height, width).
            pairs: View index pairs of shape (n_pairs, 2).
            ts: Normalized time of every candidate, shape (n_flat,).
            runtime_cfg: Stage runtime configuration.
            generator: Seeded generator.

        Returns:
            One long tensor of flat candidate indices per pair, in pair order.
        """
        n_pairs, n_flat = len(pairs), kps_xy.shape[1]
        budget = (
            n_flat
            if runtime_cfg.max_points_pairs == -1
            else runtime_cfg.max_points_pairs
        )
        chunk_size = pairs_chunk_size(n_pairs, n_flat, runtime_cfg.max_chunk_bytes)

        picked_per_pair: list[torch.Tensor] = []

        for start in tqdm(
            range(0, n_pairs, chunk_size),
            total=-(-n_pairs // chunk_size),
            leave=False,
            desc="Sampling keypoints pairs",
        ):
            chunk_pairs = pairs[start : start + chunk_size]
            candidates_mask = pair_candidates_mask(
                kps_scores=kps_scores,
                observations_mask=observations_mask,
                pairs=chunk_pairs,
                pair_avg_conf_score_thr=runtime_cfg.pair_avg_conf_score_thr,
            )

            if runtime_cfg.sampler == "random":
                picked_per_pair.extend(
                    random_point_sampling(candidates_mask, budget, generator)
                )
                continue

            features = pair_sampling_features(
                kps_xy=kps_xy,
                resolutions_hw=resolutions_hw,
                pairs=chunk_pairs,
                ts=ts,
                w_uv=runtime_cfg.w_uv,
                w_t=runtime_cfg.w_t,
            )
            selected, selected_valid = farthest_point_sampling(
                features, budget, generator, candidates_mask
            )
            picked_per_pair.extend(
                row[keep] for row, keep in zip(selected, selected_valid)
            )

        return picked_per_pair
