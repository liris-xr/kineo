import torch

from kineo.geometry.triangulation import triangulate_points


def _scene(n_views=6, n_points=25, seed=0):
    """Cameras on a ring looking at points in front of them."""
    torch.manual_seed(seed)
    K = torch.tensor([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    points_3d = torch.randn(n_points, 3) * 0.6 + torch.tensor([0.0, 0.0, 6.0])

    Ps, Rts = [], []
    for view_idx in range(n_views):
        angle = 0.35 * (view_idx - (n_views - 1) / 2)
        cos, sin = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        R = torch.tensor([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
        t = torch.tensor([-1.2 * (view_idx - (n_views - 1) / 2), 0.0, 0.0])
        Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)
        Rts.append(Rt)
        Ps.append(K @ Rt)
    Ps = torch.stack(Ps)

    projected = []
    for Rt in Rts:
        cam = points_3d @ Rt[:, :3].T + Rt[:, 3]
        uv = (cam @ K.T)
        projected.append(uv[:, :2] / uv[:, 2:])
    points_2d = torch.stack(projected)
    return Ps, points_2d, points_3d


def test_triangulation_recovers_the_original_points():
    Ps, points_2d, points_3d = _scene()

    recovered = triangulate_points(Ps=Ps, points=points_2d)

    assert torch.allclose(recovered, points_3d, atol=1e-3)


def test_zero_weight_views_are_excluded_from_the_solve():
    Ps, points_2d, points_3d = _scene()
    # Corrupt one view, then weight it out: the result must be unaffected.
    corrupted = points_2d.clone()
    corrupted[2] += 250.0
    weights = torch.ones(points_2d.shape[0], points_2d.shape[1])
    weights[2] = 0.0

    recovered = triangulate_points(
        Ps=Ps, points=corrupted, points_weights=weights
    )

    assert torch.allclose(recovered, points_3d, atol=1e-3)


def test_points_seen_by_fewer_than_two_views_are_nan():
    Ps, points_2d, _ = _scene()
    weights = torch.ones(points_2d.shape[0], points_2d.shape[1])
    weights[1:, 0] = 0.0  # point 0 survives in a single view

    recovered = triangulate_points(
        Ps=Ps, points=points_2d, points_weights=weights
    )

    assert torch.isnan(recovered[0]).all()
    assert torch.isfinite(recovered[1:]).all()


def test_batched_frames_match_per_frame_triangulation():
    Ps, points_2d, _ = _scene()
    frames = torch.stack([points_2d, points_2d + 0.05, points_2d - 0.05])

    batched = triangulate_points(Ps=Ps, points=frames)
    per_frame = torch.stack(
        [triangulate_points(Ps=Ps, points=frames[f]) for f in range(3)]
    )

    assert torch.allclose(batched, per_frame, atol=1e-5)
