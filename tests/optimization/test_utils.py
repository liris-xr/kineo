import torch

from kineo.optimization.utils import huber_weights


def test_huber_weights_inlier_region_is_one():
    r = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(huber_weights(r, 1.0), torch.ones(3))


def test_huber_weights_outlier_region_downweights():
    r = torch.tensor([2.0, 4.0])
    assert torch.allclose(huber_weights(r, 1.0), torch.tensor([0.5, 0.25]))


def test_huber_weights_handles_sign_and_zero():
    r = torch.tensor([-2.0, 0.0])
    assert torch.allclose(huber_weights(r, 1.0), torch.tensor([0.5, 1.0]))
