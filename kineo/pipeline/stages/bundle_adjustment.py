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

from kineo.optimization.utils import optimizer_should_stop
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
from kineo.optimization.camera_parameters import (
    CameraIntrinsicsParameters,
    CameraExtrinsicsParameters,
)
from kineo.geometry.camera import MIN_PROJECTION_DEPTH
from kineo.geometry.metrics import compute_reprojection_residuals
from kineo.maths import weighted_mean
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
    reproj_huber_delta_px: float = 1.0
    n_iters: int = 10
    tolerance_grad: float = 1e-05
    tolerance_change: float = 1e-09
    patience: int = 5
    dist_coeffs_regularization_weight: float = 1.0
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
    huber_delta: float,
) -> None:
    """Warn when the Huber loss starts with no inliers left to distinguish.

    Huber is linear past ``huber_delta``, so when the bulk of the observations
    already sits above it every residual is treated as an outlier: the objective
    is L1 almost everywhere and the refinement has no quadratic basin to settle
    into. That is worth surfacing, because the symptom otherwise shows up only
    as a mediocre fit that still looks like it converged.

    The median is weighted by the observation scores, because that is what the
    objective sees. Gate-rejected observations are kept in the tensors with a
    zero score and reproject thousands of pixels away; counting them would fire
    this warning on data the optimizer correctly ignores.
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

    kept_errors, kept_weights = errors[keep], kps_2d_scores[keep]
    order = kept_errors.argsort()
    cumulative = kept_weights[order].cumsum(0) / kept_weights.sum()
    median = float(kept_errors[order][int(torch.searchsorted(cumulative, 0.5))])
    if median <= huber_delta:
        return

    warnings.warn(
        f"Bundle adjustment robust loss is saturated: median reprojection error "
        f"is {median:.1f}px against huber_delta={huber_delta}px, so the bulk of "
        f"the observations is past the knee and the loss is L1 almost "
        f"everywhere. Check the correspondences and the initial geometry.",
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
        # -------------------------------------------------------------------------
        # TODO: Consider replacing this dense optimization with sparse bundle adjustment
        # using a solver like Ceres. This could significantly improve efficiency and
        # scalability for large-scale multi-view setups.
        # -------------------------------------------------------------------------

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
            huber_delta=runtime_cfg.reproj_huber_delta_px,
            tolerance_grad=runtime_cfg.tolerance_grad,
            tolerance_change=runtime_cfg.tolerance_change,
            patience=runtime_cfg.patience,
            dist_coeffs_regularization_weight=runtime_cfg.dist_coeffs_regularization_weight,
            use_lbfgs=runtime_cfg.use_lbfgs,
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
            huber_delta: float = 1.0,
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

        # BA is small and sequential (LBFGS line search); CPU avoids the
        # per-closure host<->device sync and runs faster with on-par accuracy.
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
            huber_delta=huber_delta,
        )

        kps_3d_opt = kps_3d.clone()
        kps_3d_opt.requires_grad = True

        # We lock the first camera in place to restrict the solutions
        origin_camera_extrinsics = CameraExtrinsicsParameters(
            batch_size=1,
            device=device,
        )
        origin_camera_extrinsics.Rt = Rts[0].unsqueeze(0)

        other_cameras_extrinsics = CameraExtrinsicsParameters(
            batch_size=n_views - 1,
            device=device,
        )
        other_cameras_extrinsics.Rt = Rts[1:]

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

        if optimize_distortion_coefficients:
            assert (
                    distortion_model == CameraDistortionModel.BROWN_CONRADY
            ), "Only brown_conrady distortion model is supported for now"

        had_finite_loss = False

        def opt_closure():
            nonlocal had_finite_loss
            optimizer.zero_grad()

            if shared_intrinsics:
                Ks = camera_intrinsics.K.expand(n_views, -1, -1)
                dist_coeffs = camera_intrinsics.distortion_coefficients.expand(n_views, -1)
            else:
                Ks = camera_intrinsics.K.reshape(n_views, 3, 3)
                dist_coeffs = camera_intrinsics.distortion_coefficients.reshape(n_views, -1)

            Rts = torch.cat(
                [origin_camera_extrinsics.Rt, other_cameras_extrinsics.Rt], dim=0
            )

            residuals, depth = compute_reprojection_residuals(
                kps_3d=kps_3d_opt,
                kps_2d=kps_2d_xy,
                Ks=Ks,
                Rts=Rts,
                Ds=dist_coeffs,
                distortion_model=distortion_model.value,
            )

            residuals_huber = torch.nn.functional.huber_loss(
                input=residuals,
                target=torch.zeros_like(residuals),
                reduction="none",
                delta=huber_delta,
            )

            weights = kps_2d_scores.view(n_views, -1)

            # A point on or behind the camera plane projects to a bounded but
            # meaningless residual: on the plane it dwarfs every real
            # observation, and behind it the projection is mirrored, so a
            # cheirality-violating pose can score a small error and be kept.
            valid_mask = residuals_huber.isfinite() & (
                depth.squeeze(-1) > MIN_PROJECTION_DEPTH
            )
            n_valid = int(valid_mask.sum())
            residuals_huber = residuals_huber[valid_mask]
            weights = weights[valid_mask]
            reprojection_loss = weighted_mean(residuals_huber, weights)

            total_loss = reprojection_loss

            if optimize_distortion_coefficients:
                dist_coeffs_regularization = (dist_coeffs ** 2).mean()
                total_loss += dist_coeffs_regularization_weight * dist_coeffs_regularization

            # A non-finite step mid-optimization (e.g. points through the camera
            # plane) gets a large finite loss so LBFGS line-search backtracks. But
            # a non-finite / empty very first evaluation means the initial geometry
            # is degenerate: there is no gradient to backtrack from, so fail loudly
            # instead of silently returning the unrefined cameras.
            if n_valid == 0 or not torch.isfinite(total_loss):
                if not had_finite_loss:
                    raise ValueError(
                        "Bundle adjustment initial evaluation is non-finite "
                        "(degenerate geometry or no valid keypoints)."
                    )
                return torch.full_like(total_loss, 1e12)

            had_finite_loss = True
            total_loss.backward()

            return total_loss

        if use_lbfgs:
            optimizer = torch.optim.LBFGS(
                camera_intrinsics.optimized_parameters
                + other_cameras_extrinsics.optimized_parameters
                + [kps_3d_opt],
                lr=lr,
                line_search_fn="strong_wolfe",
            )
        else:
            optimizer = torch.optim.AdamW(
                camera_intrinsics.optimized_parameters
                + other_cameras_extrinsics.optimized_parameters
                + [kps_3d_opt],
                lr=lr,
            )

        pbar = tqdm(range(n_iters), desc="Refining camera parameters", leave=False)

        prev_losses = []
        history_entries = []

        # Capture initial state for history
        if shared_intrinsics:
            _init_Ks = camera_intrinsics.K.expand(n_views, -1, -1)
            _init_dist_coeffs = camera_intrinsics.distortion_coefficients.expand(n_views, -1)
        else:
            _init_Ks = camera_intrinsics.K.reshape(n_views, 3, 3)
            _init_dist_coeffs = camera_intrinsics.distortion_coefficients.reshape(n_views, -1)

        _init_Rts = torch.cat(
            [origin_camera_extrinsics.Rt, other_cameras_extrinsics.Rt], dim=0
        )

        history_entries.append(BundleAdjustmentHistoryAnnotation(
            stage_name=self.name,
            stage_order=self.order,
            iteration=0,
            view_ids=view_ids,
            Ks=_init_Ks.detach().clone().cpu(),
            dist_coeffs=_init_dist_coeffs.detach().clone().cpu(),
            Rts=_init_Rts.detach().clone().cpu(),
            loss=None,
        ))

        for iter in pbar:
            loss = optimizer.step(opt_closure)
            pbar.set_postfix(loss=loss.item())

            if shared_intrinsics:
                _iter_Ks = camera_intrinsics.K.expand(n_views, -1, -1)
                _iter_dist_coeffs = camera_intrinsics.distortion_coefficients.expand(n_views, -1)
            else:
                _iter_Ks = camera_intrinsics.K.reshape(n_views, 3, 3)
                _iter_dist_coeffs = camera_intrinsics.distortion_coefficients.reshape(n_views, -1)

            _iter_Rts = torch.cat(
                [origin_camera_extrinsics.Rt, other_cameras_extrinsics.Rt], dim=0
            )

            history_entries.append(BundleAdjustmentHistoryAnnotation(
                stage_name=self.name,
                stage_order=self.order,
                iteration=iter + 1,
                view_ids=view_ids,
                Ks=_iter_Ks.detach().clone().cpu(),
                dist_coeffs=_iter_dist_coeffs.detach().clone().cpu(),
                Rts=_iter_Rts.detach().clone().cpu(),
                loss=loss.item(),
            ))

            if optimizer_should_stop(
                    optimizer,
                    loss,
                    prev_losses,
                    patience=patience,
                    tolerance_grad=tolerance_grad,
                    tolerance_change=tolerance_change,
            ):
                # Stop early if no improvement can be made
                break

            prev_losses.append(loss.detach())

        pbar.close()

        Ks_opt = camera_intrinsics.K.detach().expand(n_views, -1, -1).clone()
        dist_coeffs_opt = (
            camera_intrinsics.distortion_coefficients.detach().expand(n_views, -1).clone()
        )
        Rts_opt = (
            torch.cat([origin_camera_extrinsics.Rt, other_cameras_extrinsics.Rt], dim=0)
            .detach()
            .clone()
        )
        kps_3d_opt = kps_3d_opt.detach().clone()

        # LBFGS commits whatever the last step produced, so a non-finite
        # parameter escapes here and only surfaces stages later as an unrelated
        # decomposition failure.
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
