import kornia
import torch

from kineo.optimization.utils import (
    batched_epipolar_huber_loss,
)

HUBER_DELTA = 1.0


def _reference_loop_loss(Rts, Ks, edge_index, points1, points2, valid):
    """The per-edge loop the batched implementation replaces."""
    total = torch.tensor(0.0)
    for e in range(edge_index.shape[0]):
        i, j = int(edge_index[e, 0]), int(edge_index[e, 1])
        keep = valid[e]
        E = kornia.geometry.essential_from_Rt(
            R1=Rts[i][:3, :3],
            t1=Rts[i][:3, 3:],
            R2=Rts[j][:3, :3],
            t2=Rts[j][:3, 3:],
        )
        F = kornia.geometry.epipolar.fundamental_from_essential(E, Ks[i], Ks[j])
        sampson = kornia.geometry.epipolar.sampson_epipolar_distance(
            pts1=points1[e][keep],
            pts2=points2[e][keep],
            Fm=F.unsqueeze(0),
            squared=False,
        ).squeeze(0)
        total = total + torch.nn.functional.huber_loss(
            input=sampson,
            target=torch.zeros_like(sampson),
            reduction="none",
            delta=HUBER_DELTA,
        ).mean()
    return total / edge_index.shape[0]


def _scene(n_views=5, n_points=40, ragged=False):
    torch.manual_seed(0)
    Rts = torch.eye(4).repeat(n_views, 1, 1)[:, :3, :].clone()
    for v in range(1, n_views):
        Rts[v, :3, :3] = kornia.geometry.axis_angle_to_rotation_matrix(
            (torch.randn(1, 3) * 0.2)
        )[0]
        Rts[v, :3, 3] = torch.randn(3) * 0.5 + torch.tensor([1.0, 0.0, 0.0])
    Ks = torch.eye(3).repeat(n_views, 1, 1)
    Ks[:, 0, 0] = Ks[:, 1, 1] = 900.0
    Ks[:, 0, 2], Ks[:, 1, 2] = 640.0, 360.0
    edges = [(i, j) for i in range(n_views) for j in range(i + 1, n_views)]
    edge_index = torch.tensor(edges, dtype=torch.long)
    n_edges = edge_index.shape[0]
    points1 = torch.rand(n_edges, n_points, 2) * 800 + 100
    points2 = torch.rand(n_edges, n_points, 2) * 800 + 100
    valid = torch.ones(n_edges, n_points, dtype=torch.bool)
    if ragged:
        # Pairs keep different numbers of correspondences; the padding must not
        # leak into any edge's mean.
        for e in range(n_edges):
            valid[e, n_points - (e % 7) :] = False
    return Rts, Ks, edge_index, points1, points2, valid


def test_batched_loss_matches_the_per_edge_loop():
    Rts, Ks, edge_index, p1, p2, valid = _scene()

    batched = batched_epipolar_huber_loss(
        Rts, Ks, edge_index, p1, p2, valid, HUBER_DELTA
    )
    reference = _reference_loop_loss(Rts, Ks, edge_index, p1, p2, valid)

    assert torch.allclose(batched, reference, rtol=1e-5, atol=1e-6)


def test_padded_correspondences_do_not_leak_into_the_loss():
    Rts, Ks, edge_index, p1, p2, valid = _scene(ragged=True)

    batched = batched_epipolar_huber_loss(
        Rts, Ks, edge_index, p1, p2, valid, HUBER_DELTA
    )
    reference = _reference_loop_loss(Rts, Ks, edge_index, p1, p2, valid)

    assert torch.allclose(batched, reference, rtol=1e-5, atol=1e-6)


def test_loss_is_differentiable_wrt_the_camera_poses():
    Rts, Ks, edge_index, p1, p2, valid = _scene()
    Rts = Rts.clone().requires_grad_(True)

    batched_epipolar_huber_loss(
        Rts, Ks, edge_index, p1, p2, valid, HUBER_DELTA
    ).backward()

    assert Rts.grad is not None
    assert torch.isfinite(Rts.grad).all()
