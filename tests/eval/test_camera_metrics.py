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
    for cam in out[0]:
        assert cam["AE"] < 1e-2
        assert cam["TE"] < 1e-4


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

    def cam04(frame):
        return next(c for c in out[frame] if c["view_id"] == "cam04")

    assert cam04(0)["AE"] > 8.0  # frame 0: GT=A, pred=B -> ~10deg
    assert cam04(100)["AE"] < 0.5  # frame 100: GT=B, pred=B -> ~0
    for frame in (0, 100):
        for c in out[frame]:
            if c["view_id"] != "cam04":
                assert c["AE"] < 0.5  # static cameras unaffected
