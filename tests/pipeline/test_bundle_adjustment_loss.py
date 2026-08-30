import torch

from kineo.optimization.utils import reprojection_loss

DISTORTION_MODEL = "brown_conrady"


def _loss(zs: list[float], cost: float = 100.0):
    """One camera at the origin looking down +z, with a point at each depth."""
    K = torch.tensor([[[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]])
    Rt = torch.eye(4)[:3].unsqueeze(0).requires_grad_(True)
    kps_3d = torch.tensor([[[0.5, 0.5, z] for z in zs]])

    loss, n_valid = reprojection_loss(
        kps_3d,
        torch.full((1, len(zs), 2), 320.0),
        torch.ones(1, len(zs)),
        K,
        Rt,
        torch.zeros(1, 5),
        DISTORTION_MODEL,
        reproj_huber_delta_px=1.0,
        invalid_observation_cost_px=cost,
    )
    loss.backward()
    return loss, int(n_valid), Rt.grad


def test_an_invalid_observation_is_charged_the_fixed_cost():
    # Charged rather than dropped, so the solve cannot lower its loss by
    # pushing observations out of the gate.
    loss, n_valid, _ = _loss([-2.0] * 4, cost=100.0)

    assert n_valid == 0
    assert torch.allclose(loss, torch.tensor(100.0))


def test_a_camera_with_everything_behind_it_gets_no_gradient():
    # Known limitation: the charge is flat, so once every observation of a
    # camera is behind it the camera receives no gradient and cannot recover.
    # Cheirality has to be right before bundle adjustment starts.
    _, n_valid, grad = _loss([-2.0] * 4)

    assert n_valid == 0
    assert grad.abs().sum() == 0


def test_the_charge_does_not_depend_on_how_far_behind_the_point_is():
    near, _, _ = _loss([-2.0] * 4)
    far, _, _ = _loss([-2000.0] * 4)

    assert torch.allclose(near, far)


def test_observations_in_front_are_charged_only_their_residual():
    with_cost, n_valid, _ = _loss([2.0] * 4)
    without_cost, _, _ = _loss([2.0] * 4, cost=0.0)

    assert n_valid == 4
    assert torch.allclose(with_cost, without_cost)
