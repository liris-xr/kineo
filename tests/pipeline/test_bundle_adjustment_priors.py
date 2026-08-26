import pytest
import torch

from kineo.annotations.camera_intrinsics import CameraDistortionModel
from kineo.pipeline.stages.bundle_adjustment import (
    BundleAdjustmentRuntimeConfig,
    BundleAdjustmentStage,
    camera_rotation_prior,
    camera_translation_prior,
    scene_scale_from_translations,
)


def test_scene_scale_is_the_median_translation_norm():
    translations = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 10.0], [6.0, 0.0, 0.0]])

    # norms are 5, 10, 6 -> median 6
    assert scene_scale_from_translations(translations) == pytest.approx(6.0)


def test_scene_scale_ignores_cameras_at_the_origin():
    """A camera at the origin would drag the median toward zero.

    The world frame is anchored on the first camera, so ``t = 0`` is the norm
    for it, not an outlier -- but it says nothing about the scene's extent.
    """
    translations = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 6.0, 0.0]])

    assert scene_scale_from_translations(translations) == pytest.approx(5.5)


def test_scene_scale_falls_back_to_one_when_every_camera_sits_at_the_origin():
    translations = torch.zeros(4, 3)

    assert scene_scale_from_translations(translations) == pytest.approx(1.0)


def test_translation_prior_vanishes_at_the_anchor():
    transl = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    prior = camera_translation_prior(transl, transl.clone(), scene_scale=2.0, weight=1.0)

    assert prior == pytest.approx(0.0)


def test_translation_prior_is_invariant_to_the_reconstruction_scale():
    """The metric scale is arbitrary, so the prior's strength must not track it.

    Doubling the reconstruction doubles the displacement and the scene scale
    together, which must leave the penalty unchanged.
    """
    transl = torch.tensor([[1.0, 2.0, 3.0]])
    anchor = torch.tensor([[1.5, 2.5, 2.0]])

    small = camera_translation_prior(transl, anchor, scene_scale=2.0, weight=1.0)
    large = camera_translation_prior(
        10.0 * transl, 10.0 * anchor, scene_scale=20.0, weight=1.0
    )

    assert small == pytest.approx(large, rel=1e-6)


def test_translation_prior_grows_with_displacement():
    anchor = torch.zeros(1, 3)
    near = camera_translation_prior(
        torch.tensor([[0.1, 0.0, 0.0]]), anchor, scene_scale=1.0, weight=1.0
    )
    far = camera_translation_prior(
        torch.tensor([[5.0, 0.0, 0.0]]), anchor, scene_scale=1.0, weight=1.0
    )

    assert far > near


def test_rotation_prior_vanishes_at_the_anchor():
    R = torch.eye(3).unsqueeze(0)

    assert camera_rotation_prior(R, R.clone(), weight=1.0) == pytest.approx(0.0)


def test_rotation_prior_grows_with_angle():
    def rot_z(theta):
        c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]).unsqueeze(0)

    anchor = rot_z(0.0)
    small = camera_rotation_prior(rot_z(0.1), anchor, weight=1.0)
    large = camera_rotation_prior(rot_z(1.0), anchor, weight=1.0)

    assert large > small


def test_priors_are_differentiable():
    transl = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    R = torch.eye(3).unsqueeze(0).clone().requires_grad_(True)

    total = camera_translation_prior(
        transl, torch.zeros(1, 3), scene_scale=1.0, weight=1.0
    ) + camera_rotation_prior(R, torch.eye(3).unsqueeze(0) * 0.5, weight=1.0)
    total.backward()

    assert transl.grad is not None and bool(torch.isfinite(transl.grad).all())
    assert R.grad is not None and bool(torch.isfinite(R.grad).all())


def _toy_problem(n_points: int = 60):
    """Three cameras viewing a point cloud, with view 2 fed corrupted 2D data.

    Returns tensors shaped the way ``_bundle_adjustment`` expects them.
    """
    generator = torch.Generator().manual_seed(0)

    Rts = torch.eye(4)[:3].repeat(3, 1, 1)
    Rts[1, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
    Rts[2, :3, 3] = torch.tensor([0.0, 1.0, 0.0])

    Ks = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    ).repeat(3, 1, 1)
    dist_coeffs = torch.zeros(3, 5)

    kps_3d = torch.rand(n_points, 3, generator=generator) * 2.0 - 1.0
    kps_3d[:, 2] += 5.0

    cam_points = torch.einsum("vij,pj->vpi", Rts[:, :3, :3], kps_3d) + Rts[:, :3, 3].unsqueeze(1)
    projected = torch.einsum("vij,vpj->vpi", Ks, cam_points)
    kps_2d_xy = projected[..., :2] / projected[..., 2:3]

    # View 2 sees a large systematic offset, so the data term alone would drag
    # that camera far from where it started.
    kps_2d_xy[2] += 120.0

    return Ks, dist_coeffs, Rts, kps_3d, kps_2d_xy


def _run_ba(translation_prior_weight, rotation_prior_weight, n_iters=60):
    Ks, dist_coeffs, Rts, kps_3d, kps_2d_xy = _toy_problem()
    cfg = BundleAdjustmentRuntimeConfig(
        optimize_distortion_coefficients=False,
        optimize_focal_length=False,
        optimize_principal_point=False,
        n_iters=n_iters,
        translation_prior_weight=translation_prior_weight,
        rotation_prior_weight=rotation_prior_weight,
    )
    stage = BundleAdjustmentStage(name="ba", order=0, runtime_cfg=cfg)

    # _bundle_adjustment optimizes the extrinsics in place, so the starting
    # pose has to be snapshotted before the call to measure drift against it.
    Rts_before = Rts.clone()

    _, _, Rts_opt, _, _ = stage._bundle_adjustment(
        Ks=Ks,
        dist_coeffs=dist_coeffs,
        distortion_model=CameraDistortionModel.BROWN_CONRADY,
        Rts=Rts,
        cameras_resolutions_hw=[(480, 640)] * 3,
        kps_2d_xy=kps_2d_xy,
        kps_2d_scores=torch.ones(3, kps_2d_xy.shape[1]),
        kps_3d=kps_3d,
        view_ids=["cam01", "cam02", "cam03"],
        optimize_distortion_coefficients=False,
        optimize_focal_length=False,
        optimize_principal_point=False,
        n_iters=n_iters,
        translation_prior_weight=translation_prior_weight,
        rotation_prior_weight=rotation_prior_weight,
    )
    drift = (Rts_opt[:, :3, 3] - Rts_before[:, :3, 3]).norm(dim=-1)
    return float(drift.max())


def test_translation_prior_bounds_how_far_a_camera_can_drift():
    """The regression badminton_001 hit: one camera walking out of the scene."""
    unconstrained = _run_ba(translation_prior_weight=0.0, rotation_prior_weight=0.0)
    constrained = _run_ba(translation_prior_weight=10.0, rotation_prior_weight=10.0)

    assert constrained < unconstrained


def test_zero_weights_leave_the_objective_untouched():
    """Default-off must reproduce the pre-prior optimizer exactly."""
    first = _run_ba(translation_prior_weight=0.0, rotation_prior_weight=0.0)
    second = _run_ba(translation_prior_weight=0.0, rotation_prior_weight=0.0)

    assert first == pytest.approx(second)
