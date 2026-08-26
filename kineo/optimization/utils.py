# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import warnings

import torch

def _gather_flat_grad(optimizer: torch.optim.Optimizer):
    views = []

    params: list[torch.Tensor] = []

    for group in optimizer.param_groups:
        params.extend(group["params"])

    for p in params:
        if p.grad is None:
            view = p.new(p.numel()).zero_()
        elif p.grad.is_sparse:
            view = p.grad.to_dense().view(-1)
        else:
            view = p.grad.view(-1)
        if torch.is_complex(view):
            view = torch.view_as_real(view).view(-1)
        views.append(view)
    return torch.cat(views, 0)

def optimizer_should_stop(
    optimizer: torch.optim.Optimizer,
    loss: torch.Tensor,
    prev_losses: list[torch.Tensor],
    patience: int = 10,
    tolerance_grad: float = 1e-05,
    tolerance_change: float = 1e-09,
) -> bool:
    """
    Check if the optimizer should stop.

    Args:
        optimizer: The optimizer to check.
        loss: The current loss.
        prev_losses: The previous losses.
        patience: The number of previous losses to consider.
        tolerance_grad: The tolerance for the gradient.
        tolerance_change: The tolerance for the change in loss.
    """
    flat_grad = _gather_flat_grad(optimizer)
    opt_cond = flat_grad.abs().max() <= tolerance_grad

    if opt_cond:
        return True

    if len(prev_losses) == 0:
        return False

    prev_losses = torch.stack(prev_losses)

    if (
        len(prev_losses) >= patience
        and (
            torch.abs(loss - prev_losses[-patience:]) < tolerance_change
        ).all()
    ):
        # A frozen loss means either a minimum or a line search that can no
        # longer find a step. Both end the optimization, but only the first is
        # a converged result, so say which one this was.
        grad_max = flat_grad.abs().max()
        if grad_max > tolerance_grad:
            warnings.warn(
                f"Optimizer stopped without converging: the loss did not move "
                f"over {patience} steps but |grad|_max={grad_max:.3e} still "
                f"exceeds tolerance_grad={tolerance_grad:.1e}. The line search "
                f"stalled at a non-stationary point."
            )
        return True

    return False


def huber_weights(
    residuals: torch.Tensor, delta: float | torch.Tensor
) -> torch.Tensor:
    """IRLS weights for the Huber loss.

    Args:
        residuals: Per-datum residuals of any shape.
        delta: Huber transition threshold (scalar or broadcastable tensor).

    Returns:
        Weights of the same shape: 1 where |residual| <= delta, else
        delta / |residual|.
    """
    r = residuals.abs()
    return torch.where(r <= delta, torch.ones_like(r), delta / r.clamp_min(1e-12))