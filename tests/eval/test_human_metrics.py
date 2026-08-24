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
from kineo.eval.human_metrics import (
    compute_human_metrics,
    flatten_human_metrics,
)

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
    # The Procrustes fit weights the unpredicted keypoint out, so the pose
    # survives: only that keypoint is masked, and the rest still align exactly.
    pa = torch.tensor(flat["pa-mpjpe"])
    assert torch.isnan(pa).sum() == 2, "only the masked keypoint is dropped"
    assert pa.nanmean() < 1e-3, "excluding it leaves an uncontaminated fit"


def test_gt_frames_without_a_prediction_are_nan():
    ext = _extrinsics([(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS])
    gt = _keypoints()  # frames 0 and 1
    pred = _keypoints_with(frames=(0,))  # frame 1 never predicted

    flat = flatten_human_metrics(compute_human_metrics(gt, ext, pred, ext))

    n_kp = SMPL_22_KEYPOINTS_FORMAT.n_keypoints
    w = torch.tensor(flat["w-mpjpe"])
    assert torch.isnan(w).sum() == n_kp, "the unpredicted frame is fully masked"
    assert w.nanmean() < 1e-3


def _keypoints_with_unreconstructed(unreconstructed_per_frame: int):
    """Predictions where some keypoints carry a zero score (not triangulated)."""
    n_kp = SMPL_22_KEYPOINTS_FORMAT.n_keypoints
    anns = []
    for frame_idx in (0, 1):
        scores = torch.ones(n_kp, dtype=torch.float32)
        scores[:unreconstructed_per_frame] = 0.0
        anns.append(
            Keypoints3DAnnotation(
                frame_idx=frame_idx,
                subject_id="aria01",
                xyz=torch.full((n_kp, 3), float(frame_idx))
                + torch.arange(n_kp * 3).reshape(n_kp, 3) * 0.01,
                scores=scores,
                format="smpl_22",
            )
        )
    return Keypoints3DAnnotations(
        metadata=Keypoints3DAnnotationsMetadata(formats=[SMPL_22_KEYPOINTS_FORMAT]),
        annotations=anns,
    )


def test_pa_mpjpe_needs_three_keypoints_to_score_a_pose():
    # A similarity transform needs three points to be determined; with fewer it
    # absorbs any error, so the whole pose must be dropped rather than scored.
    # The counts are literal on purpose: deriving them from
    # MIN_PROCRUSTES_KEYPOINTS would move them in lockstep with the guard and
    # the test could never catch it changing.
    n_kp = SMPL_22_KEYPOINTS_FORMAT.n_keypoints
    ext = _extrinsics([(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS])
    gt = _keypoints()

    def pa_for(n_scored):
        flat = flatten_human_metrics(
            compute_human_metrics(
                gt,
                ext,
                _keypoints_with_unreconstructed(
                    unreconstructed_per_frame=n_kp - n_scored
                ),
                ext,
            )
        )
        return torch.tensor(flat["pa-mpjpe"])

    three = pa_for(3)
    assert not torch.isnan(three).all(), "three keypoints determine the fit"
    assert torch.isnan(three).sum() == 2 * (n_kp - 3)

    assert torch.isnan(pa_for(2)).all(), "two keypoints cannot determine a fit"


def test_reconstruction_rate_counts_unreconstructed_keypoints():
    # HSfM fits a parametric body model, so every keypoint always exists; we
    # triangulate, so a keypoint seen by fewer than two views has no prediction
    # at all. Those are dropped from w-mpjpe, which would silently flatter the
    # 2-view setting (27% of its keypoints) unless the rate is reported too.
    n_kp = SMPL_22_KEYPOINTS_FORMAT.n_keypoints
    ext = _extrinsics([(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS])
    gt = _keypoints()
    pred = _keypoints_with_unreconstructed(unreconstructed_per_frame=5)

    flat = flatten_human_metrics(compute_human_metrics(gt, ext, pred, ext))

    rate = flat["reconstruction-rate"]
    assert len(rate) == len(flat["w-mpjpe"])
    # Misses must score 0, never NaN, or nanmean would skip them and always
    # report a perfect rate.
    assert all(not math.isnan(v) for v in rate)
    assert sum(rate) / len(rate) == 100.0 * (n_kp - 5) / n_kp


def test_reconstruction_rate_is_one_when_everything_is_predicted():
    ext = _extrinsics([(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS])
    kps = _keypoints()

    flat = flatten_human_metrics(compute_human_metrics(kps, ext, kps, ext))

    assert all(v == 100.0 for v in flat["reconstruction-rate"])


def _two_view_gauge(G):
    """Two cameras on the world z axis, in a world rotated by G."""
    anns = []
    for view_id, center in (("cam01", [0.0, 0.0, 1.0]), ("cam02", [0.0, 0.0, 3.0])):
        R = roma.rotvec_to_rotmat(torch.tensor([0.3, 0.0, 0.0])).transpose(0, 1)
        R = R @ G.transpose(0, 1)
        anns.append(
            CameraExtrinsicsAnnotation(
                view_id=view_id,
                frame_idx=0,
                R=R,
                t=-(R @ (G @ torch.tensor(center))),
            )
        )
    return CameraExtrinsicsAnnotations(CameraExtrinsicsAnnotationsMetadata(), anns)


def _rotated_keypoints(G):
    n_kp = SMPL_22_KEYPOINTS_FORMAT.n_keypoints
    anns = []
    for f in (0, 1):
        xyz = torch.full((n_kp, 3), float(f)) + torch.arange(n_kp * 3).reshape(n_kp, 3) * 0.01
        anns.append(
            Keypoints3DAnnotation(
                frame_idx=f,
                subject_id="aria01",
                xyz=xyz @ G.transpose(0, 1),
                scores=torch.ones(n_kp, dtype=torch.float32),
                format="smpl_22",
            )
        )
    return Keypoints3DAnnotations(
        metadata=Keypoints3DAnnotationsMetadata(formats=[SMPL_22_KEYPOINTS_FORMAT]),
        annotations=anns,
    )


def test_two_view_world_alignment_uses_camera_orientations():
    # Both centres sit on the z axis, so rotating the world about z leaves them
    # put: a position-only fit cannot see the rotation and leaves the whole human
    # set flipped. The camera orientations pin it, so this exact reconstruction
    # must score ~0.
    identity = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, 0.0]))
    flip_about_z = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, math.pi]))

    gt_ext, gt_kps = _two_view_gauge(identity), _rotated_keypoints(identity)
    pred_ext, pred_kps = _two_view_gauge(flip_about_z), _rotated_keypoints(flip_about_z)

    flat = flatten_human_metrics(
        compute_human_metrics(gt_kps, gt_ext, pred_kps, pred_ext)
    )

    assert max(flat["w-mpjpe"]) < 1e-3


def test_ga_mpjpe_aligns_every_person_jointly():
    # GA-MPJPE applies one Sim(3) to all people at once, so it sits between
    # w-mpjpe (no alignment of its own) and pa-mpjpe (one fit per person).
    ext = _extrinsics([(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS])
    kps = _keypoints()

    flat = flatten_human_metrics(compute_human_metrics(kps, ext, kps, ext))

    assert "ga-mpjpe" in flat
    assert len(flat["ga-mpjpe"]) == len(flat["w-mpjpe"])
    assert max(flat["ga-mpjpe"]) < 1e-3
