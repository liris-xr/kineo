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
from kineo.annotations.keypoints_2d import stage_keypoints_2d
from kineo.annotations.calibration_points_pairs import (
    CalibrationPointsAnnotations,
    CalibrationPointsAnnotationsMetadata,
    CalibrationPointsAnnotation,
)
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
import itertools
from tqdm import tqdm
from dataclasses import dataclass


@dataclass(frozen=True)
class KeypointsPairsSamplingRuntimeConfig:
    max_points_pairs: int = 2000
    pair_avg_conf_score_thr: float = 0.75
    keypoints_indices: list[int] | None = None


class KeypointsPairsSamplingStage(PipelineStage[KeypointsPairsSamplingRuntimeConfig]):
    """
    Produces :class:`KeypointsPairSelectionAnnotations`.
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
        device = pipeline.device
        n_views = len(views)
        pairs = list(itertools.combinations(range(n_views), 2))

        # Assuming keypoints are synchronized and time is expressed in global time
        kps_2d: Keypoints2DAnnotations = annotations["keypoints_2d"]

        global_time_reference: GlobalTimeReferenceAnnotation = annotations[
            "global_time_reference"
        ].first_or_default()

        if global_time_reference is None:
            raise ValueError("Global time reference is not available")

        subjects_ids = kps_2d.subjects_ids
        n_frames = len(global_time_reference.timestamps)
        n_subjects = len(subjects_ids)
        views_ids = [view["view_id"] for view in views]

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

        # Only the scores are moved: the pair loop below reads kps_xy on the
        # host.
        kps_xy, kps_scores = stage_keypoints_2d(
            kps_2d=kps_2d,
            view_id_to_idx=view_id_to_idx,
            subject_id_to_idx=subject_id_to_idx,
            n_frames=n_frames,
            keypoints_indices=keypoints_indices,
        )
        kps_scores = kps_scores.to(device)

        for view_i, view_j in tqdm(
            pairs, total=len(pairs), leave=False, desc="Sampling keypoints pairs"
        ):
            kps_xy_view_i = kps_xy[:, view_i].reshape(-1, 2)
            kps_xy_view_j = kps_xy[:, view_j].reshape(-1, 2)
            kps_scores_view_i = kps_scores[:, view_i].reshape(-1)
            kps_scores_view_j = kps_scores[:, view_j].reshape(-1)
            pair_avg_conf_scores = torch.sqrt(kps_scores_view_i * kps_scores_view_j)

            # Sample the best candidates by confidence score, and sample a random subset of them
            valid_pick_mask = pair_avg_conf_scores > runtime_cfg.pair_avg_conf_score_thr
            valid_pick_indices = torch.where(valid_pick_mask)[0]

            if runtime_cfg.max_points_pairs == -1:
                picked_indices = valid_pick_indices
            else:
                # Drawing from the global RNG would make the sample depend on
                # whatever consumed it earlier, so a cached stage upstream would
                # silently change which pairs are picked.
                generator = torch.Generator(device=valid_pick_indices.device)
                generator.manual_seed(pipeline.seed)
                picked_indices = valid_pick_indices[
                    torch.randperm(
                        valid_pick_indices.shape[0],
                        generator=generator,
                        device=valid_pick_indices.device,
                    )[: runtime_cfg.max_points_pairs]
                ]

            picked_indices = picked_indices.cpu()
            pair_points_i = kps_xy_view_i[picked_indices]
            pair_points_j = kps_xy_view_j[picked_indices]
            pair_avg_conf_scores = pair_avg_conf_scores[picked_indices]
            pair_frame_indices = picked_indices // (n_subjects * n_selected_keypoints)

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

        calibration_points_annotations = CalibrationPointsAnnotations(
            metadata=CalibrationPointsAnnotationsMetadata(),
            annotations=calibration_points_annotations,
        ).cpu()

        annotations["calibration_points"] = calibration_points_annotations
