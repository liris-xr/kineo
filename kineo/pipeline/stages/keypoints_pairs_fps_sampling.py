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

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import Annotations
from kineo.pipeline.pipeline import Pipeline
from kineo.annotations import Keypoints2DAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.calibration_points_pairs import (
    CalibrationPointsAnnotations,
    CalibrationPointsAnnotationsMetadata,
    CalibrationPointsAnnotation,
)
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.sampling import farthest_point_sampling, normalized_uv
import itertools
from tqdm import tqdm
from dataclasses import dataclass

# u, v in both views plus time
N_SAMPLING_AXES = 5


def pair_sampling_features(
    kps_xy: torch.Tensor,
    resolutions_hw: torch.Tensor,
    pairs: torch.Tensor,
    ts: torch.Tensor,
    image_plane_weight: float,
    time_weight: float,
) -> torch.Tensor:
    views_i, views_j = pairs[:, 0], pairs[:, 1]
    normalized = normalized_uv(kps_xy, resolutions_hw)
    # Halved to balance the 4 uv axes against time
    uv_weight = image_plane_weight / 2.0
    n_pairs, n_flat = len(pairs), kps_xy.shape[1]

    return torch.cat(
        [
            uv_weight * normalized[views_i],
            uv_weight * normalized[views_j],
            (time_weight * ts).expand(n_pairs, n_flat).unsqueeze(-1),
        ],
        dim=-1,
    )


def pairs_chunk_size(n_pairs: int, n_flat: int, max_chunk_bytes: int) -> int:
    bytes_per_pair = n_flat * N_SAMPLING_AXES * 4
    return max(1, min(n_pairs, max_chunk_bytes // max(bytes_per_pair, 1)))


@dataclass(frozen=True)
class KeypointsPairsFPSSamplingRuntimeConfig:
    max_points_pairs: int = 2000
    pair_avg_conf_score_thr: float = 0.75
    keypoints_indices: list[int] | None = None
    image_plane_weight: float = 1.0
    time_weight: float = 1.0
    features_chunk_bytes: int = 128 * 1024 * 1024


class KeypointsPairsFPSSamplingStage(
    PipelineStage[KeypointsPairsFPSSamplingRuntimeConfig]
):
    """KeypointsPairsSamplingStage with farthest-point instead of random draws."""

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: KeypointsPairsFPSSamplingRuntimeConfig,
        dynamic_runtime_cfg: dict[str, KeypointsPairsFPSSamplingRuntimeConfig]
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
        runtime_cfg: KeypointsPairsFPSSamplingRuntimeConfig,
    ):
        device = pipeline.device
        n_views = len(views)
        pairs = torch.as_tensor(
            list(itertools.combinations(range(n_views), 2)), device=device
        )

        # Assuming keypoints are synchronized and time is expressed in global time
        kps_2d: Keypoints2DAnnotations = annotations["keypoints_2d"]

        cameras_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "cameras_intrinsics"
        ]

        global_time_reference: GlobalTimeReferenceAnnotation = annotations[
            "global_time_reference"
        ].first_or_default()

        if global_time_reference is None:
            raise ValueError("Global time reference is not available")

        subjects_ids = kps_2d.subjects_ids
        n_frames = len(global_time_reference.timestamps)
        n_subjects = len(subjects_ids)
        views_ids = [view["view_id"] for view in views]

        if not set(views_ids) <= set(cameras_intrinsics.views_ids):
            raise ValueError("Views ids must be included in the cameras_intrinsics")

        subject_id_to_idx = {
            subject_id: idx for idx, subject_id in enumerate(subjects_ids)
        }
        view_id_to_idx = {view_id: idx for idx, view_id in enumerate(views_ids)}

        # Assuming the same number of keypoints for all subjects
        n_keypoints = kps_2d.metadata.formats[0].n_keypoints

        calibration_points_annotations: list[CalibrationPointsAnnotation] = []

        if runtime_cfg.keypoints_indices is not None:
            keypoints_indices = torch.as_tensor(runtime_cfg.keypoints_indices).reshape(
                -1
            )
        else:
            keypoints_indices = torch.arange(n_keypoints).reshape(-1)

        n_selected_keypoints = len(keypoints_indices)

        kps_xy = torch.zeros(
            (n_frames, n_views, n_subjects, n_selected_keypoints, 2),
            dtype=torch.float32,
            device=device,
        )
        kps_scores = torch.zeros(
            (n_frames, n_views, n_subjects, n_selected_keypoints),
            dtype=torch.float32,
            device=device,
        )

        for annot in tqdm(
            kps_2d.annotations, desc="Collecting 2D keypoints", leave=False
        ):
            view_idx = view_id_to_idx[annot.view_id]
            subject_idx = subject_id_to_idx[annot.subject_id]
            kps_xy[annot.frame_idx, view_idx, subject_idx] = annot.xy[keypoints_indices]
            kps_scores[annot.frame_idx, view_idx, subject_idx] = annot.scores[
                keypoints_indices
            ]

        # Candidates are ordered (frame, subject, keypoint)
        kps_xy = kps_xy.permute(1, 0, 2, 3, 4).reshape(n_views, -1, 2)
        kps_scores = kps_scores.permute(1, 0, 2, 3).reshape(n_views, -1)
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

        frame_indices = torch.arange(n_flat, device=device) // (
            n_subjects * n_selected_keypoints
        )
        ts = frame_indices.to(kps_xy.dtype) / max(n_frames - 1, 1)

        # For deterministic behavior
        generator = torch.Generator(device=device)
        generator.manual_seed(pipeline.seed)

        budget = (
            n_flat
            if runtime_cfg.max_points_pairs == -1
            else runtime_cfg.max_points_pairs
        )
        chunk_size = pairs_chunk_size(
            len(pairs), n_flat, runtime_cfg.features_chunk_bytes
        )

        for start in tqdm(
            range(0, len(pairs), chunk_size),
            total=-(-len(pairs) // chunk_size),
            leave=False,
            desc="Sampling keypoints pairs",
        ):
            chunk_pairs = pairs[start : start + chunk_size]
            views_i, views_j = chunk_pairs[:, 0], chunk_pairs[:, 1]
            pair_avg_conf_scores = torch.sqrt(kps_scores[views_i] * kps_scores[views_j])
            candidates_mask = pair_avg_conf_scores > runtime_cfg.pair_avg_conf_score_thr

            features = pair_sampling_features(
                kps_xy=kps_xy,
                resolutions_hw=resolutions_hw,
                pairs=chunk_pairs,
                ts=ts,
                image_plane_weight=runtime_cfg.image_plane_weight,
                time_weight=runtime_cfg.time_weight,
            )
            selected, selected_valid = farthest_point_sampling(
                features, budget, generator, candidates_mask
            )

            for row, (view_i, view_j) in enumerate(chunk_pairs.tolist()):
                picked_indices = selected[row][selected_valid[row]]
                pair_points_i = kps_xy[view_i, picked_indices].cpu()
                pair_points_j = kps_xy[view_j, picked_indices].cpu()
                pair_picked_scores = pair_avg_conf_scores[row, picked_indices].cpu()
                pair_frame_indices = frame_indices[picked_indices].cpu()

                calibration_points_annotations.append(
                    CalibrationPointsAnnotation(
                        view1_id=views_ids[view_i],
                        view2_id=views_ids[view_j],
                        points1=pair_points_i,
                        points2=pair_points_j,
                        confidence_scores=pair_picked_scores,
                        frame_indices=pair_frame_indices,
                    )
                )
                calibration_points_annotations.append(
                    CalibrationPointsAnnotation(
                        view1_id=views_ids[view_j],
                        view2_id=views_ids[view_i],
                        points1=pair_points_j,
                        points2=pair_points_i,
                        confidence_scores=pair_picked_scores,
                        frame_indices=pair_frame_indices,
                    )
                )

        calibration_points_annotations = CalibrationPointsAnnotations(
            metadata=CalibrationPointsAnnotationsMetadata(),
            annotations=calibration_points_annotations,
        ).cpu()

        annotations["calibration_points"] = calibration_points_annotations
