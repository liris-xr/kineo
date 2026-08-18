import math
import torch
import roma

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
from kineo.pipeline.stages.nlf.skeleton_keypoints_format import SMPL_22_KEYPOINTS_FORMAT
from kineo.eval.human_metrics import compute_human_metrics, flatten_human_metrics

_CENTERS = {"cam01": [1.0, 0.0, 0.0], "cam02": [0.0, 1.0, 0.0],
            "cam03": [0.0, 0.0, 1.0], "cam04": [1.0, 1.0, 0.0]}


def _world2cam(center, rotvec):
    R_wc = roma.rotvec_to_rotmat(torch.tensor(rotvec))
    R = R_wc.transpose(0, 1)
    return R, -(R @ torch.tensor(center))


def _extrinsics(poses):
    anns = []
    for view_id, frame_idx, rotvec in poses:
        R, t = _world2cam(_CENTERS[view_id], rotvec)
        anns.append(CameraExtrinsicsAnnotation(view_id, frame_idx, R, t))
    return CameraExtrinsicsAnnotations(CameraExtrinsicsAnnotationsMetadata(), anns)


def _keypoints():
    n_kp = SMPL_22_KEYPOINTS_FORMAT.n_keypoints
    anns = [
        Keypoints3DAnnotation(
            frame_idx=f,
            subject_id="aria01",
            xyz=torch.full((n_kp, 3), float(f)) + torch.arange(n_kp * 3).reshape(n_kp, 3) * 0.01,
            scores=torch.ones(n_kp, dtype=torch.float32),
            format="smpl_22",
        )
        for f in (0, 1)
    ]
    return Keypoints3DAnnotations(
        metadata=Keypoints3DAnnotationsMetadata(formats=[SMPL_22_KEYPOINTS_FORMAT]),
        annotations=anns,
    )


def test_non_static_gt_camera_does_not_crash_and_is_finite():
    # cam04 non-static (two GT poses); previously tripped `assert len == n_views`.
    knock = math.radians(10.0)
    gt_ext = _extrinsics(
        [(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS] + [("cam04", 1, [0.0, 0.0, knock])]
    )
    pred_ext = _extrinsics([(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS])
    kps = _keypoints()

    metrics = compute_human_metrics(kps, gt_ext, kps, pred_ext)
    flat = flatten_human_metrics(metrics)

    assert len(flat["w-mpjpe"]) > 0
    assert all(math.isfinite(v) for v in flat["w-mpjpe"])
    assert all(math.isfinite(v) for v in flat["pa-mpjpe"])
    # gt kps == pred kps with aligned cameras -> near-zero error.
    assert max(flat["w-mpjpe"]) < 1e-3


def _keypoints_with(frames, zero_score_kp=None):
    """GT-shaped predictions, optionally with one keypoint marked unpredicted."""
    n_kp = SMPL_22_KEYPOINTS_FORMAT.n_keypoints
    anns = []
    for f in frames:
        scores = torch.ones(n_kp, dtype=torch.float32)
        if zero_score_kp is not None:
            scores[zero_score_kp] = 0.0
        anns.append(
            Keypoints3DAnnotation(
                frame_idx=f,
                subject_id="aria01",
                xyz=torch.full((n_kp, 3), float(f))
                + torch.arange(n_kp * 3).reshape(n_kp, 3) * 0.01,
                scores=scores,
                format="smpl_22",
            )
        )
    return Keypoints3DAnnotations(
        metadata=Keypoints3DAnnotationsMetadata(formats=[SMPL_22_KEYPOINTS_FORMAT]),
        annotations=anns,
    )


def test_unpredicted_keypoints_are_nan_not_origin_errors():
    ext = _extrinsics([(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS])
    gt = _keypoints()
    # Keypoint 3 was zeroed by the pipeline (score 0), so it is not a prediction.
    pred = _keypoints_with(frames=(0, 1), zero_score_kp=3)

    flat = flatten_human_metrics(compute_human_metrics(gt, ext, pred, ext))

    w = torch.tensor(flat["w-mpjpe"])
    assert torch.isnan(w).sum() == 2, "one masked keypoint per frame expected"
    assert w.nanmean() < 1e-3, "remaining keypoints match, so the mean stays ~0"
    # The whole pose is dropped from PA-MPJPE, its Procrustes fit is contaminated.
    assert torch.isnan(torch.tensor(flat["pa-mpjpe"])).all()


def test_gt_frames_without_a_prediction_are_nan():
    ext = _extrinsics([(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS])
    gt = _keypoints()  # frames 0 and 1
    pred = _keypoints_with(frames=(0,))  # frame 1 never predicted

    flat = flatten_human_metrics(compute_human_metrics(gt, ext, pred, ext))

    n_kp = SMPL_22_KEYPOINTS_FORMAT.n_keypoints
    w = torch.tensor(flat["w-mpjpe"])
    assert torch.isnan(w).sum() == n_kp, "the unpredicted frame is fully masked"
    assert w.nanmean() < 1e-3
