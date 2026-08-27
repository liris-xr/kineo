import torch

from kineo.sampling import (
    farthest_point_sampling,
    normalized_uv,
    uniform_point_sampling,
    valid_observations_mask,
)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _four_clusters(n_per_cluster: int = 25) -> torch.Tensor:
    """Four tight clusters at the corners of a 100x100 box."""
    generator = _generator(0)
    centers = torch.tensor(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]]
    )
    noise = torch.rand(
        4, n_per_cluster, 2, generator=generator
    )  # in [0, 1), far below the inter-cluster distance
    return (centers[:, None, :] + noise).reshape(-1, 2)


def _reference_fps(features, n_samples, generator, valid_mask):
    """Naive per-row greedy loop, the shape the batched version replaced."""
    orders = []
    for row in range(features.shape[0]):
        row_points = torch.where(valid_mask[row])[0]
        points = features[row, row_points]
        n = min(n_samples, len(row_points))
        selected = torch.empty(n, dtype=torch.long)
        selected[0] = torch.randint(len(row_points), (1,), generator=generator)
        min_sq = torch.full((len(row_points),), float("inf"))
        for i in range(1, n):
            sq = ((points - points[selected[i - 1]]) ** 2).sum(dim=-1)
            min_sq = torch.minimum(min_sq, sq)
            min_sq[selected[i - 1]] = -1.0
            selected[i] = min_sq.argmax()
        orders.append(row_points[selected].tolist())
    return orders


def test_fps_covers_every_cluster():
    features = _four_clusters().unsqueeze(0)
    n_per_cluster = features.shape[1] // 4

    for seed in range(5):
        selected, _ = farthest_point_sampling(features, 4, _generator(seed))
        cluster_ids = torch.unique(selected[0] // n_per_cluster)
        assert cluster_ids.numel() == 4, f"seed {seed} missed a cluster"


def test_fps_is_deterministic_for_a_given_seed():
    features = torch.rand(1, 200, 3, generator=_generator(7))

    first, _ = farthest_point_sampling(features, 10, _generator(19))
    second, _ = farthest_point_sampling(features, 10, _generator(19))
    other, _ = farthest_point_sampling(features, 10, _generator(20))

    assert torch.equal(first, second)
    assert first[0, 0] != other[0, 0]


def test_fps_returns_every_index_when_asked_for_more_than_available():
    features = torch.rand(1, 17, 2, generator=_generator(3))

    selected, valid = farthest_point_sampling(features, 40, _generator(1))

    assert selected.dtype == torch.long
    assert bool(valid.all())
    assert torch.equal(torch.sort(selected[0]).values, torch.arange(17))


def test_fps_with_zero_samples_returns_empty():
    features = torch.rand(1, 17, 2, generator=_generator(3))

    selected, valid = farthest_point_sampling(features, 0, _generator(1))

    assert selected.dtype == torch.long
    assert selected.numel() == 0 and valid.numel() == 0


def test_batched_fps_matches_a_sequential_reference():
    n_rows, n_points = 4, 60
    features = torch.rand(n_rows, n_points, 3, generator=_generator(5))
    valid_mask = torch.rand(n_rows, n_points, generator=_generator(6)) > 0.3

    selected, keep = farthest_point_sampling(
        features, 12, _generator(19), valid_mask
    )
    reference = _reference_fps(features, 12, _generator(19), valid_mask)

    for row in range(n_rows):
        assert selected[row][keep[row]].tolist() == reference[row], row


def test_batched_fps_only_selects_points_a_row_may_see():
    n_rows, n_points = 3, 40
    features = torch.rand(n_rows, n_points, 2, generator=_generator(8))
    valid_mask = torch.zeros(n_rows, n_points, dtype=torch.bool)
    for row in range(n_rows):
        valid_mask[row, row::n_rows] = True

    selected, keep = farthest_point_sampling(
        features, 10, _generator(2), valid_mask
    )

    for row in range(n_rows):
        assert bool(valid_mask[row][selected[row][keep[row]]].all())


def test_batched_fps_flags_rows_that_run_out_of_points():
    features = torch.rand(2, 30, 2, generator=_generator(4))
    valid_mask = torch.ones(2, 30, dtype=torch.bool)
    valid_mask[1, 5:] = False  # only 5 selectable points in the second row

    selected, keep = farthest_point_sampling(
        features, 12, _generator(3), valid_mask
    )

    assert int(keep[0].sum()) == 12
    assert int(keep[1].sum()) == 5
    assert len(set(selected[1][keep[1]].tolist())) == 5


def test_valid_observations_mask_rejects_off_frame_and_non_finite():
    resolutions_hw = torch.tensor([[100.0, 200.0]])
    kps_xy = torch.tensor(
        [
            [
                [100.0, 50.0],  # inside
                [0.0, 0.0],  # on the lower bound
                [200.0, 100.0],  # on the upper bound
                [-1.0, 50.0],  # u below 0
                [201.0, 50.0],  # u above W
                [100.0, -1.0],  # v below 0
                [100.0, 101.0],  # v above H
                [float("nan"), 50.0],
                [100.0, float("inf")],
            ]
        ]
    )

    mask = valid_observations_mask(kps_xy, resolutions_hw)

    expected = torch.tensor(
        [[True, True, True, False, False, False, False, False, False]]
    )
    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


def test_valid_observations_mask_uses_per_view_resolutions():
    # Same observation, valid in the large view, off-frame in the small one.
    resolutions_hw = torch.tensor([[1000.0, 1000.0], [100.0, 100.0]])
    kps_xy = torch.tensor([[[500.0, 500.0]], [[500.0, 500.0]]])

    mask = valid_observations_mask(kps_xy, resolutions_hw)

    assert torch.equal(mask, torch.tensor([[True], [False]]))


def test_uniform_sampling_only_draws_points_a_row_may_see():
    valid_mask = torch.zeros(3, 40, dtype=torch.bool)
    for row in range(3):
        valid_mask[row, row::3] = True

    picked = uniform_point_sampling(valid_mask, 10, _generator(2))

    for row in range(3):
        assert len(picked[row]) == 10
        assert bool(valid_mask[row][picked[row]].all())
        assert len(set(picked[row].tolist())) == 10


def test_uniform_sampling_yields_every_point_of_a_short_row():
    valid_mask = torch.ones(2, 30, dtype=torch.bool)
    valid_mask[1, 5:] = False

    picked = uniform_point_sampling(valid_mask, 12, _generator(3))

    assert len(picked[0]) == 12
    assert torch.equal(torch.sort(picked[1]).values, torch.arange(5))


def test_uniform_sampling_is_deterministic_for_a_given_seed():
    valid_mask = torch.rand(4, 60, generator=_generator(6)) > 0.3

    first = uniform_point_sampling(valid_mask, 12, _generator(19))
    second = uniform_point_sampling(valid_mask, 12, _generator(19))
    other = uniform_point_sampling(valid_mask, 12, _generator(20))

    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert any(not torch.equal(a, b) for a, b in zip(first, other))


def test_normalized_uv_scales_each_view_by_its_own_resolution():
    resolutions_hw = torch.tensor([[100.0, 200.0], [400.0, 800.0]])
    kps_xy = torch.tensor([[[50.0, 25.0]], [[200.0, 100.0]]])

    uvs = normalized_uv(kps_xy, resolutions_hw)

    # u divides by W and v by H, so the same fractions come out of both views.
    expected = torch.tensor([[[0.25, 0.25]], [[0.25, 0.25]]])
    assert torch.allclose(uvs, expected)
