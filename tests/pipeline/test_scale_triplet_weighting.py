import torch

from kineo.pipeline.stages.sfm_camera_extrinsics_initialization import (
    edge_cost_triplet_weights,
)


def test_edge_cost_weights_span_one_to_zero_across_the_cost_range():
    # The closure squares-roots these weights, so returning (1 - normalised
    # cost) reproduces the sqrt(1 - cost) factor the pre-IRLS solve folded
    # into its design matrix.
    costs = torch.tensor([2.0, 4.0, 6.0])
    valid = torch.tensor([True, True, True])

    weights = edge_cost_triplet_weights(costs, valid)

    assert torch.allclose(weights, torch.tensor([1.0, 0.5, 0.0]), atol=1e-6)


def test_edge_cost_weights_zero_out_invalid_triplets():
    costs = torch.tensor([2.0, 100.0, 6.0])
    valid = torch.tensor([True, False, True])

    weights = edge_cost_triplet_weights(costs, valid)

    assert weights[1].item() == 0.0
    # The invalid triplet must not stretch the normalisation of the rest.
    assert torch.allclose(weights[[0, 2]], torch.tensor([1.0, 0.0]), atol=1e-6)
