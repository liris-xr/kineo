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
from tqdm import tqdm

from kineo.maths import weighted_median
from kineo.optimization.utils import optimizer_should_stop
from kineo.optimization.utils import reprojection_loss
from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import Annotations
from kineo.annotations.camera_intrinsics import (
    CameraIntrinsicsAnnotations,
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotationsMetadata,
    CameraDistortionModel,
)
from kineo.annotations.camera_extrinsics import (
    CameraExtrinsicsAnnotations,
    CameraExtrinsicsAnnotation,
    CameraExtrinsicsAnnotationsMetadata,
)
from kineo.optimization.manifolds import UnitDirection
from kineo.optimization.camera_parameters import (
    CameraIntrinsicsParameters,
    CameraExtrinsicsParameters,
)
from kineo.geometry.camera import MIN_PROJECTION_DEPTH
from kineo.geometry.metrics import compute_reprojection_residuals

from kineo.annotations.bundle_adjustment_keypoints import (
    BundleAdjustmentKeypointsAnnotation,
    BundleAdjustmentKeypointsAnnotations,
    BundleAdjustmentKeypointsAnnotationsMetadata,
)
from kineo.annotations.bundle_adjustment_history import (
    BundleAdjustmentHistoryAnnotation,
    BundleAdjustmentHistoryAnnotations,
    BundleAdjustmentHistoryAnnotationsMetadata,
)
from dataclasses import dataclass


@dataclass(frozen=True)
class BundleAdjustmentRuntimeConfig:
    # True/False optimizes all/no distortion coefficients. A list of indices
    # optimizes only those, fixing the rest (Brown-Conrady order (k1,k2,p1,p2,k3)).
    optimize_distortion_coefficients: bool | list[int] = True
    optimize_focal_length: bool = True
    optimize_principal_point: bool = False
    optimize_rotation: bool = True
    optimize_translation: bool = True
    shared_intrinsics: bool = False
    reproj_huber_delta_px: float = 10.0
    n_iters: int = 10
    tolerance_grad: float = 1e-05
    tolerance_change: float = 1e-09
    patience: int = 5
    dist_coeffs_regularization_weight: float = 1.0
    # Cost for an invalid observation (behind the camera, non-finite residual).
    invalid_observation_cost_px: float = 100.0
    use_lbfgs: bool = True
    lr: float = 1.0


def _warn_if_robust_loss_saturated(
    kps_3d: torch.Tensor,
    kps_2d_xy: torch.Tensor,
    kps_2d_scores: torch.Tensor,
    Ks: torch.Tensor,
    Rts: torch.Tensor,
    dist_coeffs: torch.Tensor,
    distortion_model: CameraDistortionModel,
    reproj_huber_delta_px: float,
) -> None:
    """Warn when the Huber loss starts with no inliers left to distinguish.

    Huber is linear past ``reproj_huber_delta_px``, so when the bulk of the observations
    already sits above it every residual is treated as an outlier: the objective
    is L1 almost everywhere and the refinement has no quadratic basin to settle
    into. That is worth surfacing, because the symptom otherwise shows up only
    as a mediocre fit that still looks like it converged.

    The median is weighted by the observation scores, because that is what the
    objective sees: gate-rejected observations carry a zero score and reproject
    thousands of pixels away, so counting them would fire this warning on data
    the optimizer correctly ignores.
    """
    errors, depth = compute_reprojection_residuals(
        kps_3d=kps_3d,
        kps_2d=kps_2d_xy,
        Ks=Ks,
        Rts=Rts,
        Ds=dist_coeffs,
        distortion_model=distortion_model.value,
    )
    keep = (
        errors.isfinite()
        & (depth.squeeze(-1) > MIN_PROJECTION_DEPTH)
        & (kps_2d_scores > 0)
    )
    if not keep.any():
        return

    median = float(weighted_median(errors[keep], kps_2d_scores[keep])[0])
    if median <= reproj_huber_delta_px:
        return

    warnings.warn(
        f"Bundle adjustment robust loss is saturated: median reprojection error "
        f"is {median:.1f}px against reproj_huber_delta_px={reproj_huber_delta_px}px, so the bulk of "
        f"the observations is past the knee and the loss is L1 almost "
        f"everywhere. Check the correspondences and the initial geometry.",
        stacklevel=2,
    )


def _warn_if_observations_were_ejected(
    n_valid_first: int, n_valid_last: int, drop_ratio_thr: float = 0.05
) -> None:
    """Warn when the solve ends on materially fewer observations than it began.

    Rejected observations are charged rather than dropped, so shedding one is
    not cheaper than fitting it. Losing them anyway means the geometry moved
    points through the camera plane, and nothing else records that.

    ``n_valid_last`` is the last closure *evaluation*, which under a strong
    Wolfe line search may be a rejected trial point rather than the accepted
    iterate: accurate enough for a threshold, not for a per-iteration curve.

    Args:
        n_valid_first: Observations the first evaluation admitted.
        n_valid_last: Observations the last evaluation admitted.
        drop_ratio_thr: Fraction of the initial count that may be lost before
            this warns.
    """
    if n_valid_first == 0:
        return

    dropped = n_valid_first - n_valid_last
    if dropped <= drop_ratio_thr * n_valid_first:
        return

    warnings.warn(
        f"Bundle adjustment ended on {n_valid_last} valid observations against "
        f"{n_valid_first} at the first evaluation ({dropped} lost, "
        f"{100 * dropped / n_valid_first:.1f}%). Points moved through the "
        f"camera plane during the solve; check the initial geometry.",
        stacklevel=2,
    )


class BundleAdjustmentStage(PipelineStage[BundleAdjustmentRuntimeConfig]):
    """
    Graph-based stage for refining the camera extrinsics parameters.

    Uses :class:`CameraIntrinsicsAnnotations` and :class:`Keypoints2DAnnotations` from the previous stages and produces :class:`CameraExtrinsicsAnnotations`.
    """

    def __init__(
            self,
            name: str,
            order: int,
            runtime_cfg: BundleAdjustmentRuntimeConfig,
            dynamic_runtime_cfg: dict[str, BundleAdjustmentRuntimeConfig] | None = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )

    def forward(
            self,
            sequence_name: str,
            pipeline: Pipeline,
            views: list[ViewInput],
            annotations: dict[str, Annotations],
            gt_annotations: dict[str, Annotations],
            runtime_cfg: BundleAdjustmentRuntimeConfig,
    ):
        device = pipeline.device

        cameras_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "cameras_intrinsics"
        ]

        cameras_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "cameras_extrinsics"
        ]

        bundle_adjustment_keypoints: BundleAdjustmentKeypointsAnnotation = (
            annotations["bundle_adjustment_keypoints"]
        ).first_or_default()

        views_ids = [views["view_id"] for views in views]

        if not set(views_ids) <= set(cameras_intrinsics.views_ids) or not set(
                views_ids
        ) <= set(cameras_extrinsics.views_ids):
            raise ValueError(
                "Views ids must be included in the cameras_intrinsics and cameras_extrinsics"
            )

        n_views = len(views_ids)
        distortion_model = cameras_intrinsics.first_or_default().distortion_model

        if distortion_model == CameraDistortionModel.BROWN_CONRADY:
            dist_coeffs = torch.zeros(n_views, 5, device=device)
        elif distortion_model == CameraDistortionModel.OPENCV_FISHEYE:
            dist_coeffs = torch.zeros(n_views, 4, device=device)
        else:
            raise ValueError(f"Unsupported distortion model: {distortion_model}")

        Ks = torch.zeros(n_views, 3, 3, device=device)
        Rts = torch.zeros(n_views, 3, 4, device=device)
        cameras_resolutions_hw = []

        for view_idx, view_id in enumerate(views_ids):
            cam_intrinsics = cameras_intrinsics.filter_by_view_id(
                view_id
            ).first_or_default()
            cam_extrinsics = cameras_extrinsics.filter_by_view_id(
                view_id
            ).first_or_default()

            K = cam_intrinsics.K
            D = cam_intrinsics.distortion_coefficients
            Rt = cam_extrinsics.Rt

            Ks[view_idx] = K
            dist_coeffs[view_idx] = D
            Rts[view_idx] = Rt
            cameras_resolutions_hw.append(cam_intrinsics.resolution_hw)

        sampled_frame_kps_xy = bundle_adjustment_keypoints.kps_2d_xy.to(device)
        sampled_frame_kps_scores = bundle_adjustment_keypoints.kps_2d_scores.to(device)
        sampled_frame_kps_3d = bundle_adjustment_keypoints.kps_3d.to(device)

        Ks, dist_coeffs, Rts, kps_3d_opt, history_entries = self._bundle_adjustment(
            Ks=Ks,
            dist_coeffs=dist_coeffs,
            distortion_model=distortion_model,
            Rts=Rts,
            cameras_resolutions_hw=cameras_resolutions_hw,
            kps_2d_xy=sampled_frame_kps_xy,
            kps_2d_scores=sampled_frame_kps_scores,
            kps_3d=sampled_frame_kps_3d,
            view_ids=views_ids,
            optimize_distortion_coefficients=runtime_cfg.optimize_distortion_coefficients,
            optimize_focal_length=runtime_cfg.optimize_focal_length,
            optimize_principal_point=runtime_cfg.optimize_principal_point,
            optimize_rotation=runtime_cfg.optimize_rotation,
            optimize_translation=runtime_cfg.optimize_translation,
            n_iters=runtime_cfg.n_iters,
            shared_intrinsics=runtime_cfg.shared_intrinsics,
            reproj_huber_delta_px=runtime_cfg.reproj_huber_delta_px,
            invalid_observation_cost_px=runtime_cfg.invalid_observation_cost_px,
            tolerance_grad=runtime_cfg.tolerance_grad,
            tolerance_change=runtime_cfg.tolerance_change,
            patience=runtime_cfg.patience,
            dist_coeffs_regularization_weight=runtime_cfg.dist_coeffs_regularization_weight,
            use_lbfgs=runtime_cfg.use_lbfgs,
            lr=runtime_cfg.lr,
        )

        annotations["cameras_extrinsics"] = CameraExtrinsicsAnnotations(
            metadata=CameraExtrinsicsAnnotationsMetadata(),
            annotations=[
                CameraExtrinsicsAnnotation(
                    view_id=view_id,
                    frame_idx=0,
                    R=Rts[view_idx][:3, :3],
                    t=Rts[view_idx][:3, 3],
                )
                for view_idx, view_id in enumerate(views_ids)
            ],
        )

        annotations["cameras_intrinsics"] = CameraIntrinsicsAnnotations(
            metadata=CameraIntrinsicsAnnotationsMetadata(),
            annotations=[
                CameraIntrinsicsAnnotation(
                    view_id=view_id,
                    frame_idx=0,
                    K=Ks[view_idx],
                    distortion_coefficients=dist_coeffs[view_idx],
                    distortion_model=distortion_model,
                    resolution_hw=cameras_resolutions_hw[view_idx],
                )
                for view_idx, view_id in enumerate(views_ids)
            ],
        )

        annotations["bundle_adjustment_keypoints"] = (
            BundleAdjustmentKeypointsAnnotations(
                metadata=BundleAdjustmentKeypointsAnnotationsMetadata(),
                annotations=[
                    BundleAdjustmentKeypointsAnnotation(
                        view_ids=views_ids,
                        kps_2d_xy=sampled_frame_kps_xy.clone(),
                        kps_2d_scores=sampled_frame_kps_scores.clone(),
                        kps_3d=kps_3d_opt.clone(),
                    )
                ],
            ).cpu()
        )

        existing_history: BundleAdjustmentHistoryAnnotations | None = annotations.get(
            "bundle_adjustment_history"
        )
        existing_entries = list(existing_history) if existing_history is not None else []
        existing_entries.extend(history_entries)
        annotations["bundle_adjustment_history"] = BundleAdjustmentHistoryAnnotations(
            metadata=BundleAdjustmentHistoryAnnotationsMetadata(),
            annotations=existing_entries,
        )


    def _bundle_adjustment(
            self,
            Ks: torch.Tensor,
            dist_coeffs: torch.Tensor,
            distortion_model: CameraDistortionModel,
            Rts: torch.Tensor,
            cameras_resolutions_hw: list[tuple[int, int]],
            kps_2d_xy: torch.Tensor,
            kps_2d_scores: torch.Tensor,
            kps_3d: torch.Tensor,
            view_ids: list[str],
            optimize_distortion_coefficients: bool | list[int] = True,
            optimize_focal_length: bool = True,
            optimize_principal_point: bool = True,
            optimize_rotation: bool = True,
            optimize_translation: bool = True,
            n_iters: int = 5,
            shared_intrinsics: bool = False,
            reproj_huber_delta_px: float = 10.0,
            invalid_observation_cost_px: float = 100.0,
            tolerance_grad: float = 1e-05,
            tolerance_change: float = 1e-09,
            patience: int = 5,
            dist_coeffs_regularization_weight: float = 1.0,
            use_lbfgs: bool = True,
            lr: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[BundleAdjustmentHistoryAnnotation]]:

        if (
                not optimize_distortion_coefficients
                and not optimize_focal_length
                and not optimize_principal_point
                and not optimize_rotation
                and not optimize_translation
        ):
            return Ks, dist_coeffs, Rts, kps_3d, []

        # BA is small and sequential (LBFGS line search); CPU is faster
        input_device = Ks.device
        Ks = Ks.cpu()
        dist_coeffs = dist_coeffs.cpu()
        Rts = Rts.cpu()
        kps_2d_xy = kps_2d_xy.cpu()
        kps_2d_scores = kps_2d_scores.cpu()
        kps_3d = kps_3d.cpu()
        device = torch.device("cpu")

        n_views, n_samples, _ = kps_2d_xy.shape

        _warn_if_robust_loss_saturated(
            kps_3d=kps_3d,
            kps_2d_xy=kps_2d_xy,
            kps_2d_scores=kps_2d_scores,
            Ks=Ks,
            Rts=Rts,
            dist_coeffs=dist_coeffs,
            distortion_model=distortion_model,
            reproj_huber_delta_px=reproj_huber_delta_px,
        )

        kps_3d_opt = kps_3d.clone()
        kps_3d_opt.requires_grad = True

        # We lock the first camera in place to restrict the solutions
        origin_camera_extrinsics = CameraExtrinsicsParameters(
            batch_size=1,
            device=device,
        )
        origin_camera_extrinsics.Rt = Rts[0].unsqueeze(0)

        # Locking the first camera still leaves the overall scale free:
        # scaling every other translation and every point together leaves the
        # reprojections untouched. That direction never moves the loss, but it
        # does enter the quasi-Newton curvature pairs, where a zero s^T y
        # degrades the inverse-Hessian estimate. One camera therefore keeps its
        # distance to the origin and optimizes only its direction. The farthest
        # camera is chosen, since pinning the scale to a short baseline would
        # tie it to a poorly determined length.
        gauge_fixed = optimize_translation and n_views > 2
        scale_reference_view = (
            int(Rts[1:, :3, 3].norm(dim=-1).argmax()) + 1 if gauge_fixed else 1
        )
        free_views = [
            view for view in range(1, n_views)
            if not gauge_fixed or view != scale_reference_view
        ]

        scale_reference_extrinsics = CameraExtrinsicsParameters(
            batch_size=1,
            device=device,
        )
        scale_reference_length = torch.zeros((), device=device)
        scale_reference_direction = None
        if gauge_fixed:
            scale_reference_extrinsics.Rt = Rts[scale_reference_view].unsqueeze(0)
            scale_reference_length = (
                Rts[scale_reference_view, :3, 3]
                .norm()
                .clamp_min(torch.finfo(Rts.dtype).eps)
            )
            scale_reference_direction = UnitDirection(
                Rts[scale_reference_view, :3, 3].unsqueeze(0)
            ).to(device)

        other_cameras_extrinsics = CameraExtrinsicsParameters(
            batch_size=len(free_views),
            device=device,
        )
        other_cameras_extrinsics.Rt = Rts[free_views]

        # Assembled as [origin, scale reference, free cameras], so the poses are
        # gathered back into the caller's view order with a single index.
        view_order = torch.empty(n_views, dtype=torch.long, device=device)
        view_order[
            torch.tensor(
                ([0, scale_reference_view] if gauge_fixed else [0]) + free_views,
                device=device,
            )
        ] = torch.arange(n_views, device=device)

        camera_intrinsics = CameraIntrinsicsParameters(
            batch_size=n_views,
            image_size_hw_px=cameras_resolutions_hw,
            fx_and_fy=False if optimize_focal_length else True,
            shared_parameters=shared_intrinsics,
            distortion_model=distortion_model.value,
            device=device,
        )

        camera_intrinsics.K = Ks
        camera_intrinsics.distortion_coefficients = dist_coeffs

        camera_intrinsics.set_optimized_parameters(
            distortion_coefficients=optimize_distortion_coefficients,
            focal_length=optimize_focal_length,
            principal_point=optimize_principal_point,
        )

        other_cameras_extrinsics.set_optimized_parameters(
            rotation=optimize_rotation,
            translation=optimize_translation,
        )

        if gauge_fixed:
            # Its translation is rebuilt from the fixed length and the direction
            # below, so leaving it optimized here would hand the optimizer a
            # parameter with no gradient.
            scale_reference_extrinsics.set_optimized_parameters(
                rotation=optimize_rotation,
                translation=False,
            )

        extrinsics_parameters = list(other_cameras_extrinsics.optimized_parameters)
        if gauge_fixed:
            extrinsics_parameters += list(
                scale_reference_extrinsics.optimized_parameters
            ) + [scale_reference_direction.tangent]

        if optimize_distortion_coefficients:
            assert (
                    distortion_model == CameraDistortionModel.BROWN_CONRADY
            ), "Only brown_conrady distortion model is supported for now"

        had_finite_loss = False
        n_valid_first: int | None = None
        n_valid_last = 0
        # The guard below returns without a backward pass, leaving the zeros
        # that zero_grad() wrote. Track it so a gradient that was never computed
        # is not mistaken for a stationary point.
        last_eval_produced_grad = False
        history_entries: list[BundleAdjustmentHistoryAnnotation] = []

        def current_parameters():
            if shared_intrinsics:
                Ks = camera_intrinsics.K.expand(n_views, -1, -1)
                dist_coeffs = camera_intrinsics.distortion_coefficients.expand(n_views, -1)
            else:
                Ks = camera_intrinsics.K.reshape(n_views, 3, 3)
                dist_coeffs = camera_intrinsics.distortion_coefficients.reshape(n_views, -1)

            if gauge_fixed:
                reference_Rt = torch.cat(
                    [
                        scale_reference_extrinsics.Rt[:, :3, :3],
                        (
                            scale_reference_length * scale_reference_direction()
                        ).unsqueeze(-1),
                    ],
                    dim=-1,
                )
                blocks = [
                    origin_camera_extrinsics.Rt,
                    reference_Rt,
                    other_cameras_extrinsics.Rt,
                ]
            else:
                blocks = [
                    origin_camera_extrinsics.Rt,
                    other_cameras_extrinsics.Rt,
                ]
            Rts = torch.cat(blocks, dim=0)[view_order]
            return Ks, dist_coeffs, Rts

        def record_history(iteration: int, loss: float | None) -> None:
            Ks, dist_coeffs, Rts = current_parameters()
            history_entries.append(BundleAdjustmentHistoryAnnotation(
                stage_name=self.name,
                stage_order=self.order,
                iteration=iteration,
                view_ids=view_ids,
                Ks=Ks.detach().clone().cpu(),
                dist_coeffs=dist_coeffs.detach().clone().cpu(),
                Rts=Rts.detach().clone().cpu(),
                loss=loss,
            ))

        def opt_closure():
            nonlocal had_finite_loss, n_valid_first, n_valid_last
            nonlocal last_eval_produced_grad
            optimizer.zero_grad()
            last_eval_produced_grad = False

            Ks, dist_coeffs, Rts = current_parameters()

            loss, n_valid_t = reprojection_loss(
                kps_3d_opt,
                kps_2d_xy,
                kps_2d_scores,
                Ks,
                Rts,
                dist_coeffs,
                distortion_model.value,
                reproj_huber_delta_px,
                invalid_observation_cost_px,
            )
            n_valid = int(n_valid_t)

            if n_valid_first is None:
                n_valid_first = n_valid
            n_valid_last = n_valid

            total_loss = loss

            if optimize_distortion_coefficients:
                dist_coeffs_regularization = (dist_coeffs ** 2).mean()
                total_loss += dist_coeffs_regularization_weight * dist_coeffs_regularization

            if n_valid == 0 or not torch.isfinite(total_loss):
                if not had_finite_loss:
                    raise ValueError(
                        "Bundle adjustment initial evaluation is non-finite "
                        "(degenerate geometry or no valid keypoints)."
                    )
                return torch.full_like(total_loss, 1e12)

            had_finite_loss = True
            total_loss.backward()
            last_eval_produced_grad = True

            return total_loss

        if use_lbfgs:
            optimizer = torch.optim.LBFGS(
                camera_intrinsics.optimized_parameters
                + extrinsics_parameters
                + [kps_3d_opt],
                lr=lr,
                line_search_fn="strong_wolfe",
            )
        else:
            optimizer = torch.optim.AdamW(
                camera_intrinsics.optimized_parameters
                + extrinsics_parameters
                + [kps_3d_opt],
                lr=lr,
            )

        pbar = tqdm(range(n_iters), desc="Refining camera parameters", leave=False)

        prev_losses = []
        record_history(iteration=0, loss=None)

        for iter in pbar:
            loss = optimizer.step(opt_closure)

            if not last_eval_produced_grad:
                opt_closure()
            pbar.set_postfix(loss=loss.item())
            record_history(iteration=iter + 1, loss=loss.item())

            if optimizer_should_stop(
                    optimizer,
                    loss,
                    prev_losses,
                    patience=patience,
                    tolerance_grad=tolerance_grad,
                    tolerance_change=tolerance_change,
            ):
                break

            prev_losses.append(loss.detach())

        pbar.close()
        _warn_if_observations_were_ejected(n_valid_first or 0, n_valid_last)

        Ks_opt = camera_intrinsics.K.detach().expand(n_views, -1, -1).clone()
        dist_coeffs_opt = (
            camera_intrinsics.distortion_coefficients.detach().expand(n_views, -1).clone()
        )
        # Same assembly as the closure, so the scale reference is rebuilt from
        # its fixed length and optimized direction rather than read back raw.
        Rts_opt = current_parameters()[2].detach().clone()
        kps_3d_opt = kps_3d_opt.detach().clone()

        for name, tensor in (
            ("intrinsics", Ks_opt),
            ("distortion coefficients", dist_coeffs_opt),
            ("extrinsics", Rts_opt),
            ("keypoints", kps_3d_opt),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Bundle adjustment produced non-finite {name}."
                )

        return (
            Ks_opt.to(input_device),
            dist_coeffs_opt.to(input_device),
            Rts_opt.to(input_device),
            kps_3d_opt.to(input_device),
            history_entries,
        )
