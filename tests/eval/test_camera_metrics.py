import math
import torch
import roma

from kineo.annotations.camera_extrinsics import (
    CameraExtrinsicsAnnotation,
    CameraExtrinsicsAnnotations,
    CameraExtrinsicsAnnotationsMetadata,
)
from kineo.annotations.camera_intrinsics import (
    CameraDistortionModel,
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotations,
    CameraIntrinsicsAnnotationsMetadata,
)
from kineo.eval.camera_metrics import compute_camera_metrics

_CENTERS = {
    "cam01": [1.0, 0.0, 0.0],
    "cam02": [0.0, 1.0, 0.0],
    "cam03": [0.0, 0.0, 1.0],
    "cam04": [1.0, 1.0, 0.0],
}


def _world2cam(center, rotvec):
    R_wc = roma.rotvec_to_rotmat(torch.tensor(rotvec))  # world-from-cam
    R = R_wc.transpose(0, 1)
    t = -(R @ torch.tensor(center))
    return R, t


def _extrinsics(poses) -> CameraExtrinsicsAnnotations:
    # poses: list of (view_id, frame_idx, rotvec)
    anns = []
    for view_id, frame_idx, rotvec in poses:
        R, t = _world2cam(_CENTERS[view_id], rotvec)
        anns.append(
            CameraExtrinsicsAnnotation(view_id=view_id, frame_idx=frame_idx, R=R, t=t)
        )
    return CameraExtrinsicsAnnotations(CameraExtrinsicsAnnotationsMetadata(), anns)


def _intrinsics(view_ids) -> CameraIntrinsicsAnnotations:
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    anns = [
        CameraIntrinsicsAnnotation(
            view_id=v,
            frame_idx=0,
            K=K,
            distortion_coefficients=torch.zeros(4),
            distortion_model=CameraDistortionModel.OPENCV_FISHEYE,
            resolution_hw=(480, 640),
        )
        for v in view_ids
    ]
    return CameraIntrinsicsAnnotations(CameraIntrinsicsAnnotationsMetadata(), anns)


def test_static_sequence_uses_single_frame_and_matches_perfectly():
    poses = [(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS]
    gt_ext = _extrinsics(poses)
    pred_ext = _extrinsics(poses)
    intr = _intrinsics(list(_CENTERS))

    out = compute_camera_metrics(intr, gt_ext, intr, pred_ext)

    assert list(out.keys()) == [0]  # single-frame path unchanged
    for cam in out[0]["views"]:
        assert cam["TE"] < 1e-4
    for pair in out[0]["pairs"]:
        assert pair["AE"] < 1e-2


def test_non_static_camera_scored_against_active_segment_per_frame():
    knock = math.radians(10.0)
    # cam04 non-static: pose A (onset 0), pose B (onset 100, +10deg yaw, same center).
    gt_ext = _extrinsics(
        [
            ("cam01", 0, [0.0, 0.0, 0.0]),
            ("cam02", 0, [0.0, 0.0, 0.0]),
            ("cam03", 0, [0.0, 0.0, 0.0]),
            ("cam04", 0, [0.0, 0.0, 0.0]),
            ("cam04", 100, [0.0, 0.0, knock]),
        ]
    )
    # Static method output: cam04 recovered as pose B (the majority segment).
    pred_ext = _extrinsics(
        [
            ("cam01", 0, [0.0, 0.0, 0.0]),
            ("cam02", 0, [0.0, 0.0, 0.0]),
            ("cam03", 0, [0.0, 0.0, 0.0]),
            ("cam04", 0, [0.0, 0.0, knock]),
        ]
    )
    intr = _intrinsics(list(_CENTERS))

    out = compute_camera_metrics(intr, gt_ext, intr, pred_ext)

    assert sorted(out.keys()) == [0, 100]  # frame set now from GT onsets

    def pairs_with_cam04(frame):
        return [p for p in out[frame]["pairs"] if "cam04" in p["pair_ids"]]

    def pairs_without_cam04(frame):
        return [p for p in out[frame]["pairs"] if "cam04" not in p["pair_ids"]]

    # Frame 0: GT=A, pred=B. AE is pairwise, so the 10 deg error shows up in
    # exactly the pairs cam04 belongs to and nowhere else.
    assert all(9.0 < p["AE"] < 11.0 for p in pairs_with_cam04(0))
    assert all(p["AE"] < 0.5 for p in pairs_without_cam04(0))

    # Frame 100: GT=B, pred=B.
    for p in out[100]["pairs"]:
        assert p["AE"] < 0.5


def _gauge_rotated(view_ids, centers, rotvecs, G):
    """Extrinsics for a world rotated by G: C -> G C, and world2cam R -> R G^T."""
    anns = []
    for view_id, center, rotvec in zip(view_ids, centers, rotvecs):
        R = roma.rotvec_to_rotmat(torch.tensor(rotvec)).transpose(0, 1) @ G.transpose(0, 1)
        t = -(R @ (G @ torch.tensor(center)))
        anns.append(CameraExtrinsicsAnnotation(view_id, 0, R, t))
    return CameraExtrinsicsAnnotations(CameraExtrinsicsAnnotationsMetadata(), anns)


def test_pairwise_rra_is_invariant_to_an_unobservable_gauge_rotation():
    # The 2-view collapse: both centres lie on the z axis, so rotating the world
    # about z leaves them fixed. Centre-based alignment cannot see that rotation,
    # so AE reports ~180 deg while the reconstruction is in fact exact. RRA must
    # be perfect, since a global rotation cancels in R_i^T R_j.
    view_ids = ["cam01", "cam02"]
    centers = [[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]]
    # Distinct orientations, or the conjugation would cancel for the wrong reason.
    rotvecs = [[math.radians(20.0), 0.0, 0.0], [0.0, math.radians(35.0), 0.0]]
    identity = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, 0.0]))
    flip_about_z = roma.rotvec_to_rotmat(torch.tensor([0.0, 0.0, math.pi]))

    gt_ext = _gauge_rotated(view_ids, centers, rotvecs, identity)
    pred_ext = _gauge_rotated(view_ids, centers, rotvecs, flip_about_z)
    intr = _intrinsics(view_ids)

    out = compute_camera_metrics(intr, gt_ext, intr, pred_ext)

    pairs = out[0]["pairs"]
    assert len(pairs) == 1  # 2 cameras -> exactly one pair
    assert pairs[0]["AE"] < 1e-3
    assert pairs[0]["RRA05"] == 1.0


def test_pairwise_rra_counts_every_camera_pair():
    poses = [(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS]
    gt_ext = _extrinsics(poses)
    intr = _intrinsics(list(_CENTERS))

    out = compute_camera_metrics(intr, gt_ext, intr, _extrinsics(poses))

    n_cams = len(_CENTERS)
    assert len(out[0]["pairs"]) == n_cams * (n_cams - 1) // 2
    assert all(p["RRA05"] == 1.0 for p in out[0]["pairs"])


def test_pairwise_rra_detects_a_genuinely_wrong_relative_rotation():
    gt_ext = _extrinsics([(v, 0, [0.0, 0.0, 0.0]) for v in ("cam01", "cam02")])
    # Only cam02 is turned: the relative rotation really is wrong by ~20 deg.
    pred_ext = _extrinsics(
        [("cam01", 0, [0.0, 0.0, 0.0]), ("cam02", 0, [0.0, 0.0, math.radians(20.0)])]
    )
    intr = _intrinsics(["cam01", "cam02"])

    out = compute_camera_metrics(intr, gt_ext, intr, pred_ext)

    pair = out[0]["pairs"][0]
    assert pair["RRA05"] == 0.0
    assert pair["RRA30"] == 1.0
    assert 19.0 < pair["AE"] < 21.0


def test_ae_is_not_reported_per_camera():
    # AE is a property of a camera pair, not of a single camera (HSfM supp.
    # S.4.1), so it must not appear alongside the per-view position metrics.
    poses = [(v, 0, [0.0, 0.0, 0.0]) for v in _CENTERS]
    intr = _intrinsics(list(_CENTERS))

    out = compute_camera_metrics(intr, _extrinsics(poses), intr, _extrinsics(poses))

    assert all("AE" not in cam for cam in out[0]["views"])
    assert all("AE" in pair for pair in out[0]["pairs"])


def test_scale_aligned_position_metrics_are_undefined_below_three_cameras():
    # A similarity transform has 7 DOF and n centres give 3n constraints, so with
    # two cameras it fits them exactly: s-TE is 0 and s-CCA is 1 for any
    # prediction at all, including a wrong one. Report undefined, not perfect.
    gt_ext = _extrinsics([(v, 0, [0.0, 0.0, 0.0]) for v in ("cam01", "cam02")])
    # Deliberately wrong: cam02 turned by 40 deg.
    pred_ext = _extrinsics(
        [("cam01", 0, [0.0, 0.0, 0.0]), ("cam02", 0, [0.0, 0.0, math.radians(40.0)])]
    )
    intr = _intrinsics(["cam01", "cam02"])

    out = compute_camera_metrics(intr, gt_ext, intr, pred_ext)

    for cam in out[0]["views"]:
        assert math.isnan(cam["s-TE"])
        for threshold in ("05", "10", "15", "20", "25", "30"):
            assert math.isnan(cam[f"s-CCA{threshold}"])
        # Rigid alignment keeps 6 DOF against 6 constraints but preserves the
        # inter-centre distance, so these stay meaningful.
        assert not math.isnan(cam["TE"])
        assert not math.isnan(cam["CCA10"])


def test_scale_aligned_position_metrics_are_defined_from_three_cameras():
    poses = [(v, 0, [0.0, 0.0, 0.0]) for v in ("cam01", "cam02", "cam03")]
    gt_ext = _extrinsics(poses)
    intr = _intrinsics(["cam01", "cam02", "cam03"])

    out = compute_camera_metrics(intr, gt_ext, intr, _extrinsics(poses))

    for cam in out[0]["views"]:
        assert not math.isnan(cam["s-TE"])
        assert not math.isnan(cam["s-CCA10"])
