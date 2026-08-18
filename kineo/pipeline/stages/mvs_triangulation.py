# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from math import ceil
import torch
from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import Annotations
from kineo.geometry.transformations import undistort_points
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.keypoints_3d import (
    Keypoints3DAnnotations,
    Keypoints3DAnnotationsMetadata,
    Keypoints3DAnnotation,
)
from kineo.annotations.global_time_reference import (
    GlobalTimeReferenceAnnotation,
)
from kineo.geometry.triangulation import triangulate_points
from kineo.geometry.metrics import pairwise_reprojection_consensus_score
from kineo.annotations.camera_intrinsics import CameraDistortionModel
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MVSTriangulationRuntimeConfig:
    triangulation_chunk_size: int = 200
    use_eigendecomposition: bool = False


class MVSTriangulationStage(PipelineStage[MVSTriangulationRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: MVSTriangulationRuntimeConfig = MVSTriangulationRuntimeConfig(),
        dynamic_runtime_cfg: Optional[dict[str, MVSTriangulationRuntimeConfig]] = None,
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
        runtime_cfg: MVSTriangulationRuntimeConfig,
    ):
        device = pipeline.device
        n_views = len(views)

        camera_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "camera_intrinsics"
        ]
        camera_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "camera_extrinsics"
        ]

        kps_2d: Keypoints2DAnnotations = annotations["keypoints_2d"]

        global_time_reference: GlobalTimeReferenceAnnotation = annotations[
            "global_time_reference"
        ].first_or_default()

        if global_time_reference is None:
            raise ValueError("Global time reference is not available")

        views_ids = camera_intrinsics.views_ids
        subjects_ids = kps_2d.subjects_ids
        n_frames = global_time_reference.timestamps.numel()
        n_views = len(views_ids)
        n_subjects = len(subjects_ids)

        kps_format = kps_2d.metadata.formats[0]
        n_keypoints = kps_format.n_keypoints

        view_id_to_idx = {
            view_id: view_idx for view_idx, view_id in enumerate(views_ids)
        }
        subject_id_to_idx = {
            subject_id: subject_idx
            for subject_idx, subject_id in enumerate(subjects_ids)
        }

        distortion_model = camera_intrinsics.first_or_default().distortion_model

        Rts = torch.zeros((n_views, 3, 4), device=device)
        Ks = torch.zeros((n_views, 3, 3), device=device)
        Ps = torch.zeros((n_views, 3, 4), device=device)

        if distortion_model == CameraDistortionModel.BROWN_CONRADY:
            dist_coeffs = torch.zeros((n_views, 5), device=device)
        elif distortion_model == CameraDistortionModel.OPENCV_FISHEYE:
            dist_coeffs = torch.zeros((n_views, 4), device=device)
        else:
            raise ValueError(f"Invalid distortion model: {distortion_model}")

        cameras_resolutions_hw = []

        for view_idx in range(n_views):
            view_camera_intrinsics = camera_intrinsics.filter_by_view_id(
                views[view_idx]["view_id"]
            ).first_or_default()

            view_camera_extrinsics = camera_extrinsics.filter_by_view_id(
                views[view_idx]["view_id"]
            ).first_or_default()

            Rts[view_idx] = torch.cat(
                [view_camera_extrinsics.R, view_camera_extrinsics.t.unsqueeze(1)],
                dim=1,
            )
            Ks[view_idx] = view_camera_intrinsics.K
            Ps[view_idx] = Ks[view_idx] @ Rts[view_idx]
            dist_coeffs[view_idx] = view_camera_intrinsics.distortion_coefficients
            cameras_resolutions_hw.append(views[view_idx]["frame_loader"].resolution_hw)

        kps_3d_annotations: list[Keypoints3DAnnotation] = []

        all_kps_2d = torch.zeros(
            (n_frames, n_views, n_subjects, n_keypoints, 2), device=device
        )
        all_kps_2d_scores = torch.zeros(
            (n_frames, n_views, n_subjects, n_keypoints), device=device
        )

        for annot in tqdm(
            kps_2d.annotations, desc="Collecting 2D keypoints", leave=False
        ):
            view_idx = view_id_to_idx[annot.view_id]
            subject_idx = subject_id_to_idx[annot.subject_id]
            all_kps_2d[annot.frame_idx, view_idx, subject_idx] = annot.xy
            all_kps_2d_scores[annot.frame_idx, view_idx, subject_idx] = annot.scores

        all_kps_2d_undistorted = all_kps_2d.clone()

        for view_idx, view_id in tqdm(
            enumerate(views_ids), desc="Undistorting 2D keypoints", leave=False
        ):
            all_kps_2d_undistorted[:, view_idx] = undistort_points(
                points=all_kps_2d[:, view_idx].reshape(-1, 2),
                K=Ks[view_idx],
                D=dist_coeffs[view_idx],
                distortion_model=distortion_model.value,
            ).reshape(n_frames, n_subjects, n_keypoints, 2)

        # Replace NaN and Inf with 0 and set their scores to 0 (to avoid error with SVD)
        # These can happen due to points at (0, 0) being undistorted to NaN or Inf
        valid_kps_2d_mask = torch.isfinite(all_kps_2d_undistorted).all(dim=-1)
        all_kps_2d_undistorted[~valid_kps_2d_mask] = 0
        all_kps_2d_scores[~valid_kps_2d_mask] = 0

        all_kps_3d = torch.zeros((n_frames, n_subjects, n_keypoints, 3), device=device)
        all_kps_3d_scores = torch.zeros(
            (n_frames, n_subjects, n_keypoints), device=device
        )

        for subject_idx, subject_id in enumerate(subjects_ids):
            points_2d = all_kps_2d_undistorted[..., subject_idx, :, :].reshape(
                -1, n_views, n_keypoints, 2
            )
            points_2d_distorted = all_kps_2d[..., subject_idx, :, :].reshape(
                -1, n_views, n_keypoints, 2
            )
            points_2d_scores = all_kps_2d_scores[..., subject_idx, :].reshape(
                -1, n_views, n_keypoints
            )

            n_chunks = int(ceil(n_frames / runtime_cfg.triangulation_chunk_size))

            for chunk_idx in tqdm(
                range(0, n_chunks),
                desc=f"Triangulating 3D keypoints for subject {subject_id}",
                leave=False,
                unit="chunk",
            ):
                chunk_start_frame = chunk_idx * runtime_cfg.triangulation_chunk_size
                chunk_end_frame = min(
                    chunk_start_frame + runtime_cfg.triangulation_chunk_size, n_frames
                )
                chunk_frames = torch.arange(chunk_start_frame, chunk_end_frame)

                chunk_points_2d = points_2d[chunk_frames]
                chunk_points_2d_scores = points_2d_scores[chunk_frames]
                chunk_points_3d = triangulate_points(
                    Ps=Ps,
                    points=chunk_points_2d,
                    points_weights=chunk_points_2d_scores,
                    use_eigendecomposition=runtime_cfg.use_eigendecomposition,
                ).reshape(-1, n_keypoints, 3)

                chunk_points_3d_scores = pairwise_reprojection_consensus_score(
                    kps_3d=chunk_points_3d,
                    kps_2d=points_2d_distorted[chunk_frames],
                    kps_2d_scores=chunk_points_2d_scores,
                    Rts=Rts,
                    Ks=Ks,
                    Ds=dist_coeffs,
                    distortion_model=distortion_model.value,
                )

                all_kps_3d[chunk_frames, subject_idx, :, :] = chunk_points_3d
                all_kps_3d_scores[chunk_frames, subject_idx, :] = chunk_points_3d_scores

        valid_kps_3d_mask = torch.isfinite(all_kps_3d).all(dim=-1)

        all_kps_3d[~valid_kps_3d_mask] = 0
        all_kps_3d_scores[~valid_kps_3d_mask] = 0

        for subject_idx, subject_id in enumerate(subjects_ids):
            for f in range(n_frames):
                kps_3d_annotation = Keypoints3DAnnotation(
                    subject_id=subject_id,
                    frame_idx=f,
                    xyz=all_kps_3d[f, subject_idx],
                    scores=all_kps_3d_scores[f, subject_idx],
                    format=kps_format.name,
                )
                kps_3d_annotations.append(kps_3d_annotation)

        annotations["keypoints_3d"] = Keypoints3DAnnotations(
            metadata=Keypoints3DAnnotationsMetadata(
                formats=[kps_format],
            ),
            annotations=kps_3d_annotations,
        ).cpu()
