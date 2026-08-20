import torch

from kineo.geometry.metrics import compute_reprojection_residuals


def _scene(depths: list[float]):
    """A two-view scene whose points sit at the requested camera depths."""
    n_views = 2
    K = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    Ks = K.expand(n_views, 3, 3)
    Ds = torch.zeros(n_views, 5)

    Rts = torch.eye(4)[:3, :].expand(n_views, 3, 4).clone()
    Rts[1, 0, 3] = 1.0
    Rts.requires_grad_(True)

    kps_3d = torch.tensor([[0.1, 0.05, d] for d in depths])
    kps_2d = torch.full((n_views, len(depths), 2), 320.0)
    return kps_3d, kps_2d, Ks, Rts, Ds


def _masked_loss_backward(kps_3d, kps_2d, Ks, Rts, Ds):
    """Mirror the bundle adjustment closure: Huber, drop invalid, mean."""
    residuals, depth = compute_reprojection_residuals(
        kps_3d=kps_3d,
        kps_2d=kps_2d,
        Ks=Ks,
        Rts=Rts,
        Ds=Ds,
        distortion_model="brown_conrady",
    )
    residuals_huber = torch.nn.functional.huber_loss(
        input=residuals, target=torch.zeros_like(residuals), reduction="none", delta=1.0
    )
    valid = residuals_huber.isfinite() & (depth.squeeze(-1).abs() > 1e-6)
    loss = residuals_huber[valid].mean()
    loss.backward()
    return loss, valid


def test_point_on_the_camera_plane_does_not_poison_the_gradient():
    # Regression guard for fencing_005: a single point at depth 0 made the
    # perspective divide emit inf. The forward loss stayed healthy because the
    # point was masked out, but the backward pass multiplied 0 by the inf's NaN
    # jacobian and wiped every camera parameter.
    kps_3d, kps_2d, Ks, Rts, Ds = _scene([5.0, 6.0, 7.0, 0.0])

    loss, valid = _masked_loss_backward(kps_3d, kps_2d, Ks, Rts, Ds)

    assert torch.isfinite(loss)
    assert not valid[:, -1].any(), "the depth-0 point must be excluded"
    assert torch.isfinite(Rts.grad).all(), "depth-0 point poisoned the gradient"


def test_healthy_points_keep_their_gradient_when_a_bad_point_is_present():
    # The rescue must be local: adding a degenerate point must not perturb the
    # gradient contributed by the well-conditioned ones.
    healthy = _scene([5.0, 6.0, 7.0])
    loss_clean, _ = _masked_loss_backward(*healthy)
    grad_clean = healthy[3].grad.clone()

    mixed = _scene([5.0, 6.0, 7.0, 0.0])
    loss_mixed, _ = _masked_loss_backward(*mixed)
    grad_mixed = mixed[3].grad.clone()

    assert torch.allclose(loss_clean, loss_mixed)
    assert torch.allclose(grad_clean, grad_mixed)
