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

import kornia
import torch

from kineo.geometry.camera import MIN_PROJECTION_DEPTH
from kineo.geometry.metrics import compute_reprojection_residuals

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


def reprojection_loss(
    kps_3d: torch.Tensor,
    kps_2d_xy: torch.Tensor,
    kps_2d_scores: torch.Tensor,
    Ks: torch.Tensor,
    Rts: torch.Tensor,
    dist_coeffs: torch.Tensor,
    distortion_model: str,
    reproj_huber_delta_px: float,
    invalid_observation_cost_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted Huber reprojection loss, charging the rejected observations.

    Charging them, and averaging over every observation rather than the
    surviving ones, stops a pass from lowering its loss by pushing its worst
    observations out of the gate.

    Returns:
        The loss, and the number of observations that contributed to it.
    """
    n_views = Ks.shape[0]

    residuals, depth = compute_reprojection_residuals(
        kps_3d=kps_3d,
        kps_2d=kps_2d_xy,
        Ks=Ks,
        Rts=Rts,
        Ds=dist_coeffs,
        distortion_model=distortion_model,
    )
    depth = depth.squeeze(-1)

    residuals_huber = torch.nn.functional.huber_loss(
        input=residuals,
        target=torch.zeros_like(residuals),
        reduction="none",
        delta=reproj_huber_delta_px,
    )

    weights = kps_2d_scores.view(n_views, -1)

    # Behind the camera the projection is mirrored, so a cheirality-violating
    # pose can score a small error and be kept. The finiteness test catches
    # distortion overflowing at a large normalized radius.
    valid_mask = residuals_huber.isfinite() & (depth > MIN_PROJECTION_DEPTH)

    observation_costs = torch.where(
        valid_mask, residuals_huber, invalid_observation_cost_px
    )
    loss = (observation_costs * weights).sum() / weights.sum().clamp_min(
        torch.finfo(weights.dtype).eps
    )
    return loss, valid_mask.sum()


def batched_epipolar_huber_loss(
    Rts: torch.Tensor,
    Ks: torch.Tensor,
    edge_index: torch.Tensor,
    points1: torch.Tensor,
    points2: torch.Tensor,
    valid: torch.Tensor,
    sampson_huber_delta_px: float,
) -> torch.Tensor:
    """Mean over edges of each edge's mean Huber-ed Sampson distance.

    Batched over edges: a 20-camera rig has 190 of them and LBFGS re-evaluates
    this every line-search step, so per-edge kernel launches dominate.

    Args:
        Rts: Absolute camera poses, shape (V, 3, 4).
        Ks: Camera intrinsic matrices, shape (V, 3, 3).
        edge_index: View pairs, shape (E, 2).
        points1: Correspondences in the first view, padded, shape (E, N, 2).
        points2: Correspondences in the second view, padded, shape (E, N, 2).
        valid: Which correspondences are real rather than padding, shape (E, N).
        sampson_huber_delta_px: Huber threshold on the Sampson distance.

    Returns:
        Scalar loss.
    """
    view_i, view_j = edge_index[:, 0], edge_index[:, 1]

    E_mat = kornia.geometry.essential_from_Rt(
        R1=Rts[view_i][:, :3, :3],
        t1=Rts[view_i][:, :3, 3:],
        R2=Rts[view_j][:, :3, :3],
        t2=Rts[view_j][:, :3, 3:],
    )
    F_mat = kornia.geometry.epipolar.fundamental_from_essential(
        E_mat, Ks[view_i], Ks[view_j]
    )
    sampson = kornia.geometry.epipolar.sampson_epipolar_distance(
        pts1=points1, pts2=points2, Fm=F_mat, squared=False
    )
    huber = torch.nn.functional.huber_loss(
        input=sampson,
        target=torch.zeros_like(sampson),
        reduction="none",
        delta=sampson_huber_delta_px,
    )
    # Padding must not enter any edge's mean, so mask before reducing.
    weights = valid.to(huber.dtype)
    per_edge = (huber * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    return per_edge.mean()
