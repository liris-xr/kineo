# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import math

import torch

from kineo.annotations.camera_extrinsics import (
    CameraExtrinsicsAnnotation,
    CameraExtrinsicsAnnotations,
    CameraExtrinsicsAnnotationsMetadata,
)
from kineo.annotations.keypoints_3d import (
    Keypoints3DAnnotation,
    Keypoints3DAnnotations,
    Keypoints3DAnnotationsMetadata,
)
from kineo.annotations import COCO_17_KEYPOINTS_FORMAT
from kineo.eval.human_metrics import compute_human_metrics, flatten_human_metrics

SUBJECT = "dancer"
N_KEYPOINTS = COCO_17_KEYPOINTS_FORMAT.n_keypoints


def _keypoints(frames: list[int]) -> Keypoints3DAnnotations:
    return Keypoints3DAnnotations(
        metadata=Keypoints3DAnnotationsMetadata(formats=[COCO_17_KEYPOINTS_FORMAT]),
        annotations=[
            Keypoints3DAnnotation(
                frame_idx=frame_idx,
                subject_id=SUBJECT,
                xyz=torch.full((N_KEYPOINTS, 3), float(frame_idx)),
                scores=torch.ones(N_KEYPOINTS),
                format=COCO_17_KEYPOINTS_FORMAT.name,
            )
            for frame_idx in frames
        ],
    )


def _extrinsics(view_ids: list[str]) -> CameraExtrinsicsAnnotations:
    return CameraExtrinsicsAnnotations(
        metadata=CameraExtrinsicsAnnotationsMetadata(),
        annotations=[
            CameraExtrinsicsAnnotation(
                view_id=view_id,
                frame_idx=0,
                R=torch.eye(3),
                t=torch.tensor([float(i), 0.0, 0.0]),
            )
            for i, view_id in enumerate(view_ids)
        ],
    )


def _crowd(xyz_by_subject: dict[str, torch.Tensor]) -> Keypoints3DAnnotations:
    return Keypoints3DAnnotations(
        metadata=Keypoints3DAnnotationsMetadata(formats=[COCO_17_KEYPOINTS_FORMAT]),
        annotations=[
            Keypoints3DAnnotation(
                frame_idx=0,
                subject_id=subject_id,
                xyz=xyz,
                scores=torch.ones(N_KEYPOINTS),
                format=COCO_17_KEYPOINTS_FORMAT.name,
            )
            for subject_id, xyz in xyz_by_subject.items()
        ],
    )


def test_ga_mpjpe_ignores_a_similarity_shared_by_every_subject():
    torch.manual_seed(0)
    gt = {
        "a": torch.rand(N_KEYPOINTS, 3),
        "b": torch.rand(N_KEYPOINTS, 3) + torch.tensor([2.0, 0.0, 0.0]),
    }
    # A rotation about z, a scaling and a translation of the whole group.
    R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = {
        subject_id: 3.0 * xyz @ R.T + torch.tensor([5.0, -1.0, 2.0])
        for subject_id, xyz in gt.items()
    }

    views = ["cam01", "cam02"]
    metrics = compute_human_metrics(
        gt_keypoints_3d_annotations=_crowd(gt),
        gt_cam_extrinsics_annotations=_extrinsics(views),
        pred_keypoints_3d_annotations=_crowd(pred),
        pred_cam_extrinsics_annotations=_extrinsics(views),
    )

    flattened = flatten_human_metrics(metrics)
    assert max(flattened["ga-mpjpe"]) < 1e-4
    assert max(flattened["w-mpjpe"]) > 1.0


def test_ga_mpjpe_scores_where_a_subject_stands_relative_to_the_others():
    torch.manual_seed(0)
    gt = {
        "a": torch.rand(N_KEYPOINTS, 3),
        "b": torch.rand(N_KEYPOINTS, 3) + torch.tensor([2.0, 0.0, 0.0]),
    }
    pred = dict(gt)
    pred["b"] = pred["b"] + torch.tensor([1.0, 0.0, 0.0])

    views = ["cam01", "cam02"]
    metrics = compute_human_metrics(
        gt_keypoints_3d_annotations=_crowd(gt),
        gt_cam_extrinsics_annotations=_extrinsics(views),
        pred_keypoints_3d_annotations=_crowd(pred),
        pred_cam_extrinsics_annotations=_extrinsics(views),
    )

    flattened = flatten_human_metrics(metrics)
    assert max(flattened["ga-mpjpe"]) > 0.1
    # PA-MPJPE fits each subject on its own, so the displacement is absorbed.
    assert max(flattened["pa-mpjpe"]) < 1e-4


def test_a_frame_no_prediction_answers_for_is_not_scored():
    views = ["cam01", "cam02"]
    metrics = compute_human_metrics(
        gt_keypoints_3d_annotations=_keypoints([0, 1, 2]),
        gt_cam_extrinsics_annotations=_extrinsics(views),
        pred_keypoints_3d_annotations=_keypoints([0, 1]),
        pred_cam_extrinsics_annotations=_extrinsics(views),
    )

    assert [frame[0]["reconstructed"] for frame in metrics.values()] == [
        True,
        True,
        False,
    ]

    unreconstructed = metrics[2][0]["joints"]
    assert all(math.isnan(joint["w-mpjpe"]) for joint in unreconstructed)
    assert all(math.isnan(joint["pa-mpjpe"]) for joint in unreconstructed)


def test_reconstruction_is_reported_next_to_the_errors():
    views = ["cam01", "cam02"]
    metrics = compute_human_metrics(
        gt_keypoints_3d_annotations=_keypoints([0, 1, 2, 3]),
        gt_cam_extrinsics_annotations=_extrinsics(views),
        pred_keypoints_3d_annotations=_keypoints([0, 1]),
        pred_cam_extrinsics_annotations=_extrinsics(views),
    )

    flattened = flatten_human_metrics(metrics)

    assert flattened["reconstructed"] == [1.0, 1.0, 0.0, 0.0]
    assert sum(math.isnan(error) for error in flattened["w-mpjpe"]) == (
        2 * N_KEYPOINTS
    )
