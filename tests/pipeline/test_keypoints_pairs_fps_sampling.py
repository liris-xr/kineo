import itertools

import torch

from kineo.pipeline.stages.keypoints_pairs_fps_sampling import (
    KeypointsPairsFpsSamplingRuntimeConfig,
    KeypointsPairsFpsSamplingStage,
    pair_candidates_mask,
    pair_sampling_features,
    pairs_chunk_size,
)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _pairs(n_views: int) -> torch.Tensor:
    return torch.as_tensor(list(itertools.combinations(range(n_views), 2)))


def test_features_normalize_each_view_by_its_own_resolution():
    # Two views with different sensor sizes, each keypoint at its own corner.
    kps_xy = torch.tensor(
        [
            [[0.0, 0.0], [640.0, 480.0]],
            [[0.0, 0.0], [1920.0, 1080.0]],
        ]
    )
    resolutions_hw = torch.tensor([[480.0, 640.0], [1080.0, 1920.0]])
    ts = torch.zeros(2)

    features = pair_sampling_features(
        kps_xy=kps_xy,
        resolutions_hw=resolutions_hw,
        pairs=_pairs(2),
        ts=ts,
        w_uv=1.0,
        w_t=1.0,
    )

    assert features.shape == (1, 2, 5)
    assert torch.allclose(features[0, 0], torch.zeros(5))
    # Both views hit their own bottom-right corner, so despite the 3x sensor
    # difference both normalize to the same feature.
    assert torch.allclose(features[0, 1, :2], features[0, 1, 2:4])


def test_features_carry_the_second_view_independently():
    # View 0 is pinned; only view 1 moves. A reference-view-only feature would
    # collapse these two candidates onto the same point.
    kps_xy = torch.tensor(
        [
            [[50.0, 50.0], [50.0, 50.0]],
            [[0.0, 0.0], [100.0, 100.0]],
        ]
    )
    resolutions_hw = torch.tensor([[100.0, 100.0], [100.0, 100.0]])

    features = pair_sampling_features(
        kps_xy=kps_xy,
        resolutions_hw=resolutions_hw,
        pairs=_pairs(2),
        ts=torch.zeros(2),
        w_uv=1.0,
        w_t=1.0,
    )

    assert torch.allclose(features[0, 0, :2], features[0, 1, :2])
    assert not torch.allclose(features[0, 0, 2:4], features[0, 1, 2:4])


def test_image_block_and_time_axis_carry_their_own_weight():
    """The four image axes together must weigh ``w_uv``, not ``2 * w_uv``.

    Distance is Euclidean, so spreading the image-plane block over four axes
    would otherwise make it dominate the single time axis for free.
    """
    # Both views at their bottom-right corner: every normalized axis is 1, so
    # the image block's norm is exactly its weight.
    kps_xy = torch.tensor([[[100.0, 100.0]], [[100.0, 100.0]]])
    resolutions_hw = torch.tensor([[100.0, 100.0], [100.0, 100.0]])

    weighted = pair_sampling_features(
        kps_xy=kps_xy,
        resolutions_hw=resolutions_hw,
        pairs=_pairs(2),
        ts=torch.tensor([1.0]),
        w_uv=2.0,
        w_t=4.0,
    )

    assert torch.allclose(
        torch.linalg.norm(weighted[0, 0, :4]), torch.tensor(2.0)
    )
    assert torch.allclose(weighted[0, 0, 4], torch.tensor(4.0))


def test_time_axis_drops_out_when_its_weight_is_zero():
    kps_xy = torch.zeros(2, 3, 2)
    resolutions_hw = torch.tensor([[100.0, 100.0], [100.0, 100.0]])

    features = pair_sampling_features(
        kps_xy=kps_xy,
        resolutions_hw=resolutions_hw,
        pairs=_pairs(2),
        ts=torch.tensor([0.0, 0.5, 1.0]),
        w_uv=1.0,
        w_t=0.0,
    )

    assert torch.allclose(features[..., 4], torch.zeros(1, 3))


def test_candidates_require_both_views_above_the_threshold():
    kps_scores = torch.tensor([[0.9, 0.9, 0.1], [0.9, 0.1, 0.1]])
    observations_mask = torch.ones(2, 3, dtype=torch.bool)

    mask = pair_candidates_mask(
        kps_scores=kps_scores,
        observations_mask=observations_mask,
        pairs=_pairs(2),
        pair_avg_conf_score_thr=0.6,
    )

    # sqrt(0.9*0.9)=0.9 passes; sqrt(0.9*0.1)=0.3 and sqrt(0.1*0.1)=0.1 do not.
    assert mask.tolist() == [[True, False, False]]


def test_candidates_reject_an_observation_either_view_cannot_see():
    kps_scores = torch.full((2, 3), 0.9)
    observations_mask = torch.ones(2, 3, dtype=torch.bool)
    observations_mask[0, 1] = False  # off-frame in the first view
    observations_mask[1, 2] = False  # off-frame in the second view

    mask = pair_candidates_mask(
        kps_scores=kps_scores,
        observations_mask=observations_mask,
        pairs=_pairs(2),
        pair_avg_conf_score_thr=0.6,
    )

    assert mask.tolist() == [[True, False, False]]


def test_candidates_are_built_for_every_pair_in_order():
    kps_scores = torch.tensor([[0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
    observations_mask = torch.ones(3, 2, dtype=torch.bool)

    mask = pair_candidates_mask(
        kps_scores=kps_scores,
        observations_mask=observations_mask,
        pairs=_pairs(3),
        pair_avg_conf_score_thr=0.6,
    )

    # Pairs are (0,1), (0,2), (1,2) in itertools.combinations order.
    assert mask.tolist() == [
        [True, False],
        [False, False],
        [False, True],
    ]


def test_chunk_size_bounds_the_feature_tensor():
    # 100 candidates x 5 axes x 4 bytes = 2000 bytes per pair.
    assert pairs_chunk_size(n_pairs=190, n_flat=100, max_chunk_bytes=2000) == 1
    assert pairs_chunk_size(n_pairs=190, n_flat=100, max_chunk_bytes=20000) == 10
    assert pairs_chunk_size(n_pairs=190, n_flat=100, max_chunk_bytes=10**9) == 190


def test_chunk_size_never_returns_zero():
    assert pairs_chunk_size(n_pairs=190, n_flat=10**6, max_chunk_bytes=1) == 1


def test_fps_over_the_joint_space_separates_correspondences_view_i_cannot():
    """The joint feature must resolve candidates a reference view cannot.

    Every candidate sits at the same point in view 0 and at a different corner
    in view 1, so a ``[u_i, v_i, t]`` feature sees one cluster while the joint
    ``[u_i, v_i, u_j, v_j, t]`` feature sees four.
    """
    from kineo.pipeline.stages.bundle_adjustment_fps_sampling import (
        farthest_point_sampling,
    )

    corners = torch.tensor(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]]
    )
    n_per_corner = 20
    noise = torch.rand(4, n_per_corner, 2, generator=_generator(0))
    view_j = (corners[:, None, :] + noise).reshape(-1, 2)
    view_i = torch.full_like(view_j, 50.0)

    kps_xy = torch.stack([view_i, view_j])
    resolutions_hw = torch.tensor([[100.0, 100.0], [100.0, 100.0]])

    features = pair_sampling_features(
        kps_xy=kps_xy,
        resolutions_hw=resolutions_hw,
        pairs=_pairs(2),
        ts=torch.zeros(view_j.shape[0]),
        w_uv=1.0,
        w_t=1.0,
    )

    selected, keep = farthest_point_sampling(features, 4, _generator(3))

    corner_ids = torch.unique(selected[0][keep[0]] // n_per_corner)
    assert corner_ids.numel() == 4


def _sample(sampler, max_chunk_bytes, n_views=4, n_flat=60, seed=19):
    """Runs the stage hot path over a synthetic rig."""
    cfg = KeypointsPairsFpsSamplingRuntimeConfig(
        max_points_pairs=8,
        pair_avg_conf_score_thr=0.5,
        sampler=sampler,
        max_chunk_bytes=max_chunk_bytes,
    )
    stage = KeypointsPairsFpsSamplingStage(name="t", order=0, runtime_cfg=cfg)

    kps_xy = torch.rand(n_views, n_flat, 2, generator=_generator(1)) * 100.0
    kps_scores = torch.rand(n_views, n_flat, generator=_generator(2))
    observations_mask = torch.rand(n_views, n_flat, generator=_generator(3)) > 0.1
    resolutions_hw = torch.full((n_views, 2), 100.0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    return stage._sample_pairs(
        kps_xy=kps_xy,
        kps_scores=kps_scores,
        observations_mask=observations_mask,
        resolutions_hw=resolutions_hw,
        pairs=_pairs(n_views),
        ts=torch.linspace(0.0, 1.0, n_flat),
        runtime_cfg=cfg,
        generator=generator,
    )


def test_chunking_does_not_change_which_pairs_are_picked():
    """Chunk size is a memory knob; it must not move the selection.

    Rows draw their first point in row order whatever the chunk boundaries
    are, so a run split into chunks must reproduce the unchunked one exactly.
    """
    for sampler in ("fps", "random"):
        unchunked = _sample(sampler, max_chunk_bytes=10**9)
        chunked = _sample(sampler, max_chunk_bytes=1)  # one pair per chunk

        assert len(unchunked) == len(chunked) == 6
        for picked, other in zip(unchunked, chunked):
            assert torch.equal(picked, other), sampler


def test_sampling_is_deterministic_for_a_given_seed():
    for sampler in ("fps", "random"):
        first = _sample(sampler, max_chunk_bytes=10**9, seed=19)
        again = _sample(sampler, max_chunk_bytes=10**9, seed=19)
        other = _sample(sampler, max_chunk_bytes=10**9, seed=20)

        assert all(torch.equal(a, b) for a, b in zip(first, again)), sampler
        assert not all(torch.equal(a, b) for a, b in zip(first, other)), sampler


def test_both_samplers_draw_from_the_same_candidate_set():
    """The filter is shared, so only the selection rule differs in the A/B."""
    n_views, n_flat = 4, 60
    kps_scores = torch.rand(n_views, n_flat, generator=_generator(2))
    observations_mask = torch.rand(n_views, n_flat, generator=_generator(3)) > 0.1
    candidates = pair_candidates_mask(
        kps_scores=kps_scores,
        observations_mask=observations_mask,
        pairs=_pairs(n_views),
        pair_avg_conf_score_thr=0.5,
    )

    for sampler in ("fps", "random"):
        for pair_idx, picked in enumerate(_sample(sampler, max_chunk_bytes=10**9)):
            assert bool(candidates[pair_idx][picked].all()), sampler


def test_an_unsupported_sampler_is_rejected():
    cfg = KeypointsPairsFpsSamplingRuntimeConfig(sampler="nearest")
    stage = KeypointsPairsFpsSamplingStage(name="t", order=0, runtime_cfg=cfg)

    try:
        stage.forward(
            sequence_name="s",
            pipeline=None,
            views=[],
            annotations={},
            gt_annotations={},
            runtime_cfg=cfg,
        )
    except ValueError as error:
        assert "Unsupported sampler" in str(error)
    else:
        raise AssertionError("expected a ValueError")
