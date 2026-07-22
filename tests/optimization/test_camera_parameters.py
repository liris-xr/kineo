import torch

from kineo.optimization.camera_parameters import CameraIntrinsicsParameters


def test_intrinsics_K_roundtrip_idempotent_shared_focal_nonsquare():
    # Non-square 16:9 image, single shared focal (fx_and_fy=False, as bundle
    # adjustment uses when optimizing focal length). Setting K then reading it
    # back must be idempotent. The old setter averaged fx/W and fy/H (normalized
    # by different dimensions), scaling the focal by (1 + W/H)/2 on every
    # set->get round-trip -> the focal ballooned across BA passes (tagging_014:
    # 1805 -> 2508, a uniform ~12 deg vfov error).
    H, W = 2160, 3840
    params = CameraIntrinsicsParameters(
        image_size_hw_px=(H, W), batch_size=1, fx_and_fy=False
    )
    f = 1805.0
    K = torch.tensor(
        [[[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]]]
    )

    params.K = K
    K_out = params.K
    assert torch.allclose(K_out[0, 0, 0], torch.tensor(f), atol=1.0)
    assert torch.allclose(K_out[0, 1, 1], torch.tensor(f), atol=1.0)

    # Repeated set->get (mimics the multi-pass BA boundaries) stays fixed.
    params.K = K_out
    K_out2 = params.K
    assert torch.allclose(K_out2, K_out, atol=1e-3)
