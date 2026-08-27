# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import torch
import networkx as nx
import concurrent.futures
import itertools
import os
import warnings
import cv2
import kornia
from tqdm import tqdm

from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import Annotations
from kineo.annotations.camera_intrinsics import (
    CameraIntrinsicsAnnotations,
)
from kineo.annotations.calibration_points_pairs import CalibrationPointsAnnotations
from kineo.annotations.camera_extrinsics import (
    CameraExtrinsicsAnnotations,
    CameraExtrinsicsAnnotation,
    CameraExtrinsicsAnnotationsMetadata,
)
from kineo.geometry.camera import (
    inverse_Rt,
    transform_points_from_world_to_camera,
)
from kineo.geometry.triangulation import triangulate_points
from kineo.geometry.rotation_averaging import (
    average_rotations,
    edge_closure_rates,
)
from kineo.optimization.camera_parameters import CameraExtrinsicsParameters
from kineo.geometry.transformations import undistort_points

import math
import numpy as np
from dataclasses import dataclass

from kineo.optimization.utils import huber_weights, optimizer_should_stop


@dataclass(frozen=True)
class SfMCameraExtrinsicsInitializationRuntimeConfig:
    ransac_confidence: float = 0.99
    ransac_reproj_threshold_px: float = 3.0
    n_refinement_iters: int = 0
    tolerance_grad: float = 1e-05
    tolerance_change: float = 1e-09
    use_lbfgs: bool = True
    n_relative_scale_factors_iters: int = 10
    scale_triplet_weighting: str = "irls"
    n_scale_irls_iters: int = 5
    sampson_huber_delta_px: float = 1.0
    rotation_outlier_thresh_deg: float = 7.0
    n_rotation_averaging_iters: int = 100
    # Chain translations along a tree ranked by full-pose loop closure rather
    # than rotation closure alone. An edge can be rotation-consistent and still
    # have no observable scale, and chaining through one collapses the child
    # camera onto its parent. False restores the rotation tree.
    use_pose_closure_tree: bool = True
    # Loop-closure tolerance, relative to the largest term in the loop.
    max_closure_residual_ratio: float = 0.10


class SfMCameraExtrinsicsInitializationStage(
    PipelineStage[SfMCameraExtrinsicsInitializationRuntimeConfig]
):
    """
    Graph-based stage for initializing the camera extrinsics parameters.

    Uses :class:`CameraIntrinsicsAnnotations` and :class:`Keypoints2DAnnotations` from the previous stages and produces :class:`CameraExtrinsicsAnnotations`.
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: SfMCameraExtrinsicsInitializationRuntimeConfig,
        dynamic_runtime_cfg: (
            dict[str, SfMCameraExtrinsicsInitializationRuntimeConfig] | None
        ) = None,
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
        runtime_cfg: SfMCameraExtrinsicsInitializationRuntimeConfig,
    ):
        # The estimation below is a 190-iteration loop over small tensors built
        # around cv2.recoverPose, which is CPU-only. On GPU every iteration pays
        # launch and transfer latency that dwarfs the arithmetic, so the whole
        # estimation runs on CPU. The stage's annotations are stored on CPU
        # either way, so nothing downstream sees the difference.
        device = torch.device("cpu")

        calibration_points_annotations: CalibrationPointsAnnotations = annotations[
            "calibration_points"
        ]

        if calibration_points_annotations is None:
            raise ValueError("No calibration points annotations found.")

        cameras_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "cameras_intrinsics"
        ]

        n_views = len(views)
        view_names = [view["view_id"] for view in views]
        pairs = list(itertools.combinations(range(n_views), 2))

        points1 = []
        points2 = []
        cameras_resolutions_hw = []

        Ks_init = torch.zeros((n_views, 3, 3), device=device)
        dist_coeffs_init = []
        distortion_models = []

        for view_idx, view in enumerate(views):
            view_id = view["view_id"]
            camera_intrinsics = cameras_intrinsics.filter_by_view_id(
                view_id
            ).first_or_default()

            if camera_intrinsics is None:
                raise ValueError(
                    f"No camera intrinsics annotation found for view {view_id}"
                )

            Ks_init[view_idx] = camera_intrinsics.K
            dist_coeffs_init.append(camera_intrinsics.distortion_coefficients)
            distortion_models.append(camera_intrinsics.distortion_model.value)
            cameras_resolutions_hw.append(view["frame_loader"].resolution_hw)

        dist_coeffs_init = torch.stack(dist_coeffs_init)

        Ks_init = Ks_init.to(device)
        dist_coeffs_init = dist_coeffs_init.to(device)

        for view_i, view_j in pairs:
            pair_calibration_points_annotation = (
                calibration_points_annotations.get_annotation(
                    (views[view_i]["view_id"], views[view_j]["view_id"])
                )
            )
            if pair_calibration_points_annotation is None:
                raise ValueError(
                    f"No calibration points annotation found for view pair ({views[view_i]['view_id']}, {views[view_j]['view_id']})"
                )

            points1.append(pair_calibration_points_annotation.points1)
            points2.append(pair_calibration_points_annotation.points2)

        Rts_init = _estimate_camera_extrinsics(
            view_names=view_names,
            points1=points1,
            points2=points2,
            Ks_init=Ks_init,
            dist_coeffs_init=dist_coeffs_init,
            distortion_models=distortion_models,
            ransac_confidence=runtime_cfg.ransac_confidence,
            ransac_reproj_threshold_px=runtime_cfg.ransac_reproj_threshold_px,
            n_refinement_iters=runtime_cfg.n_refinement_iters,
            use_lbfgs=runtime_cfg.use_lbfgs,
            tolerance_grad=runtime_cfg.tolerance_grad,
            tolerance_change=runtime_cfg.tolerance_change,
            n_relative_scale_factors_iters=runtime_cfg.n_relative_scale_factors_iters,
        scale_triplet_weighting=runtime_cfg.scale_triplet_weighting,
        n_scale_irls_iters=runtime_cfg.n_scale_irls_iters,
            sampson_huber_delta_px=runtime_cfg.sampson_huber_delta_px,
            rotation_outlier_thresh_deg=runtime_cfg.rotation_outlier_thresh_deg,
            n_rotation_averaging_iters=runtime_cfg.n_rotation_averaging_iters,
            use_pose_closure_tree=runtime_cfg.use_pose_closure_tree,
            max_closure_residual_ratio=runtime_cfg.max_closure_residual_ratio,
        )

        camera_extrinsics_annotations = []

        for view_idx in range(n_views):
            view_id = views[view_idx]["view_id"]

            camera_extrinsics_annotations.append(
                CameraExtrinsicsAnnotation(
                    view_id=view_id,
                    frame_idx=0,
                    R=Rts_init[view_idx][:3, :3],
                    t=Rts_init[view_idx][:3, 3],
                )
            )

        annotations["cameras_extrinsics"] = CameraExtrinsicsAnnotations(
            metadata=CameraExtrinsicsAnnotationsMetadata(),
            annotations=camera_extrinsics_annotations,
        ).cpu()


def _estimate_camera_extrinsics(
    view_names: list[str],
    points1: list[torch.Tensor],
    points2: list[torch.Tensor],
    Ks_init: torch.Tensor,
    dist_coeffs_init: torch.Tensor,
    distortion_models: str,
    ransac_confidence: float = 0.999,
    ransac_reproj_threshold_px: float = 3.0,
    n_refinement_iters: int = 100,
    use_lbfgs: bool = True,
    tolerance_grad: float = 1e-05,
    tolerance_change: float = 1e-09,
    n_relative_scale_factors_iters: int = 10,
    scale_triplet_weighting: str = "irls",
    n_scale_irls_iters: int = 5,
    sampson_huber_delta_px: float = 1.0,
    rotation_outlier_thresh_deg: float = 7.0,
    n_rotation_averaging_iters: int = 100,
    use_pose_closure_tree: bool = True,
    max_closure_residual_ratio: float = 0.10,
) -> torch.Tensor:
    """
    Estimate camera extrinsics from 2D keypoint correspondences across multiple views.

    This function implements a complete pipeline for camera pose estimation:
    1. Computes pairwise relative camera transformations
    2. Optimizes the scale factors between views
    3. Computes absolute camera poses in a global coordinate system

    The estimation is robust to outliers through RANSAC-based pose estimation
    and uses loop closure constraints to optimize the scale factors.

    Args:
        Ks: torch.Tensor of shape (n_cameras, 3, 3)
            Camera intrinsic matrices
        distCoeffs: torch.Tensor of shape (n_cameras, D)
            Camera distortion coefficients
        kps2d_xy: torch.Tensor of shape (n_cameras, n_frames, n_kps, 2)
            2D keypoint coordinates in image space
        kps2d_scores: torch.Tensor of shape (n_cameras, n_frames, n_kps)
            Confidence scores for each keypoint
        kps2d_timestamps: torch.Tensor of shape (n_frames,)
            Timestamps for each frame
        cameras_resolutions_hw: list[tuple[int, int]]
            Resolution of each camera
        Rts: torch.Tensor of shape (n_cameras, 3, 4)
            Initial camera poses
        correspondence_score_threshold: float
            Threshold for the correspondence score
        confidence: float
            Confidence level for the RANSAC-based pose estimation
        ransac_reproj_threshold_px: float
            Reprojection threshold for the RANSAC-based pose estimation
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - Ks: torch.Tensor of shape (n_cameras, 3, 3)
                Camera intrinsic matrices
            - dist_coeffs: torch.Tensor of shape (n_cameras, D)
                Camera distortion coefficients
            - Rts: torch.Tensor of shape (n_cameras, 3, 4)
                Absolute camera transformation matrices
    """
    graph = _initialize_graph(
        view_names=view_names,
        Ks=Ks_init,
        dist_coeffs=dist_coeffs_init,
        distortion_models=distortion_models,
        points1=points1,
        points2=points2,
        ransac_confidence=ransac_confidence,
        ransac_reproj_threshold_px=ransac_reproj_threshold_px,
    )

    graph = _average_rotations(
        graph,
        n_iters=n_rotation_averaging_iters,
        huber_delta_rad=math.radians(rotation_outlier_thresh_deg),
    )

    graph = _compute_relative_scale_factors(
        graph,
        n_iters=n_relative_scale_factors_iters,
        n_irls_iters=n_scale_irls_iters,
        weighting=scale_triplet_weighting,
        max_closure_residual_ratio=max_closure_residual_ratio,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
    )

    graph = _compute_absolute_Rts(
        graph, use_pose_closure_tree=use_pose_closure_tree
    )

    graph = _refine_camera_extrinsics(
        graph,
        n_iters=n_refinement_iters,
        use_lbfgs=use_lbfgs,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        sampson_huber_delta_px=sampson_huber_delta_px,
    )

    graph = _enforce_cheirality(graph)

    device = graph.nodes[0]["K"].device
    n_views = len(graph.nodes())
    Rts_init = torch.empty((n_views, 3, 4), dtype=torch.float32, device=device)

    for view_idx in range(n_views):
        Rts_init[view_idx] = graph.nodes[view_idx]["Rt"]

    return Rts_init


def _enforce_cheirality(graph: nx.DiGraph) -> nx.DiGraph:
    """Flips the reconstruction when it triangulates behind the cameras.

    Negating every camera translation together with every 3D point leaves all
    reprojections unchanged, so neither the epipolar cost minimized above nor
    the reprojection cost minimized by bundle adjustment can tell the two
    apart. RANSAC selects its inliers on that same sign-blind criterion, so the
    cheirality test inside recoverPose can pass on a corrupted inlier set.
    Only the full correspondence set distinguishes them.

    Args:
        graph: Graph whose nodes carry the absolute camera poses.

    Returns:
        The graph, with every node translation negated if the reconstruction
        was mirrored.
    """
    n_views = len(graph.nodes())
    Ks = torch.stack([graph.nodes[i]["K"] for i in range(n_views)])
    Rts = torch.stack([graph.nodes[i]["Rt"] for i in range(n_views)])
    Ps = Ks @ Rts

    in_front = 0
    behind = 0

    for view_i, view_j in itertools.combinations(range(n_views), 2):
        if not _pair_has_correspondences(graph, view_i, view_j):
            continue

        points = torch.stack(
            [
                graph.edges[view_i, view_j]["points1"],
                graph.edges[view_i, view_j]["points2"],
            ]
        )
        points_3d = triangulate_points(Ps=Ps[[view_i, view_j]], points=points)
        valid = torch.isfinite(points_3d).all(dim=-1)

        for view_idx in (view_i, view_j):
            depths = transform_points_from_world_to_camera(
                points_3d[valid], Rts[view_idx]
            )[..., 2]
            in_front += int((depths > 0).sum())
            behind += int((depths <= 0).sum())

    if behind > in_front:
        warnings.warn(
            f"Reconstruction triangulates behind the cameras "
            f"({behind}/{behind + in_front} points); flipping to its mirror."
        )
        for view_idx in range(n_views):
            graph.nodes[view_idx]["Rt"][:3, 3] *= -1

    return graph


def _pair_has_correspondences(graph: nx.DiGraph, view_i: int, view_j: int) -> bool:
    """Whether a view pair yielded a usable relative pose.

    Pairs with too few correspondences stay in the graph as placeholders with an
    infinite cost, an identity relative pose and no point attributes, so anything
    reading the correspondences has to skip them.

    Args:
        graph: Graph built by _initialize_graph.
        view_i: Index of the first view of the pair.
        view_j: Index of the second view of the pair.

    Returns:
        False when the pair failed pairwise pose estimation.
    """
    cost = torch.as_tensor(graph[view_i][view_j]["cost"])
    return not bool(torch.isinf(cost))


def batched_epipolar_huber_loss(
    Rts: torch.Tensor,
    Ks: torch.Tensor,
    edge_index: torch.Tensor,
    points1: torch.Tensor,
    points2: torch.Tensor,
    valid: torch.Tensor,
    huber_delta: float,
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
        huber_delta: Huber threshold on the Sampson distance.

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
        pts1=points1, pts2=points2, Fm=F_mat
    )
    huber = torch.nn.functional.huber_loss(
        input=sampson,
        target=torch.zeros_like(sampson),
        reduction="none",
        delta=huber_delta,
    )
    # Padding must not enter any edge's mean, so mask before reducing.
    weights = valid.to(huber.dtype)
    per_edge = (huber * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
    return per_edge.mean()


def _refine_camera_extrinsics(
    graph: nx.DiGraph,
    n_iters: int = 100,
    tolerance_grad: float = 1e-05,
    tolerance_change: float = 1e-09,
    use_lbfgs: bool = True,
    sampson_huber_delta_px: float = 1.0,
) -> nx.DiGraph:
    device = graph.nodes[0]["K"].device
    n_views = len(graph.nodes())

    Rts = torch.empty((n_views, 3, 4), dtype=torch.float32, device=device)

    for view_idx in range(n_views):
        Rts[view_idx] = graph.nodes[view_idx]["Rt"]

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

    # Remove self loops
    graph_undirected = graph.to_undirected()

    edges_no_selfloops = [
        (view_i, view_j)
        for view_i, view_j in graph_undirected.edges()
        if view_i != view_j and _pair_has_correspondences(graph, view_i, view_j)
    ]

    if len(edges_no_selfloops) == 0:
        warnings.warn(
            "Skipping camera extrinsics refinement: no usable correspondences "
            "between any pair of views."
        )
        return graph

    print(f"Refining camera extrinsics using {len(edges_no_selfloops)} edges.")

    # Pairs keep different numbers of points, so short edges are padded with a
    # repeat of their first point and masked out of the reduction.
    edge_index = torch.tensor(
        edges_no_selfloops, dtype=torch.long, device=device
    )
    Ks_all = torch.stack(
        [graph.nodes[view_idx]["K"] for view_idx in range(n_views)]
    )
    edge_points1 = [
        graph.edges[view_i, view_j]["points1"]
        for view_i, view_j in edges_no_selfloops
    ]
    edge_points2 = [
        graph.edges[view_i, view_j]["points2"]
        for view_i, view_j in edges_no_selfloops
    ]
    n_points_max = max(points.shape[0] for points in edge_points1)
    n_edges = len(edges_no_selfloops)
    points1_padded = torch.zeros(
        (n_edges, n_points_max, 2), dtype=torch.float32, device=device
    )
    points2_padded = torch.zeros_like(points1_padded)
    points_valid = torch.zeros(
        (n_edges, n_points_max), dtype=torch.bool, device=device
    )
    for edge_idx, (points_i, points_j) in enumerate(
        zip(edge_points1, edge_points2)
    ):
        n_points = points_i.shape[0]
        points1_padded[edge_idx, :n_points] = points_i
        points2_padded[edge_idx, :n_points] = points_j
        points1_padded[edge_idx, n_points:] = points_i[0]
        points2_padded[edge_idx, n_points:] = points_j[0]
        points_valid[edge_idx, :n_points] = True


    had_finite_loss = False

    def opt_closure():
        nonlocal had_finite_loss
        optimizer.zero_grad()

        Rts = torch.cat(
            [origin_camera_extrinsics.Rt, other_cameras_extrinsics.Rt], dim=0
        )

        total_loss = batched_epipolar_huber_loss(
            Rts=Rts,
            Ks=Ks_all,
            edge_index=edge_index,
            points1=points1_padded,
            points2=points2_padded,
            valid=points_valid,
            huber_delta=sampson_huber_delta_px,
        )
        # A non-finite step mid-optimization (e.g. degenerate pose, |t|->0) gets a
        # large finite loss so LBFGS line-search backtracks. But a non-finite very
        # first evaluation means the seed geometry itself is degenerate: there is
        # no gradient to backtrack from, so fail loudly instead of silently
        # returning the unrefined extrinsics.
        if not torch.isfinite(total_loss):
            if not had_finite_loss:
                raise ValueError(
                    "Initial camera extrinsics produce a non-finite refinement "
                    "loss (degenerate seed geometry)."
                )
            return torch.full_like(total_loss, 1e12)
        had_finite_loss = True
        total_loss.backward()
        return total_loss

    other_cameras_extrinsics.set_optimized_parameters(
        rotation=True,
        translation=True,
    )

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(
            other_cameras_extrinsics.optimized_parameters,
            lr=1,
            line_search_fn="strong_wolfe",
        )
    else:
        optimizer = torch.optim.AdamW(
            other_cameras_extrinsics.optimized_parameters,
            lr=1e-3,
        )

    pbar = tqdm(range(n_iters), desc="Refining init camera poses", leave=False)

    prev_losses = []

    for iter in pbar:
        loss = optimizer.step(opt_closure)
        pbar.set_postfix(loss=loss.item())

        if optimizer_should_stop(
            optimizer,
            loss,
            prev_losses,
            patience=5,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
        ):
            # Stop early if no improvement can be made
            break

        prev_losses.append(loss.detach())

    pbar.close()

    Rts = torch.cat([origin_camera_extrinsics.Rt, other_cameras_extrinsics.Rt], dim=0)

    # Update the graph
    for view_idx in range(n_views):
        graph.nodes[view_idx]["Rt"] = Rts[view_idx].detach()

    for view_i, view_j in graph.edges():
        if view_i == view_j or not _pair_has_correspondences(graph, view_i, view_j):
            continue

        rel_R, rel_t = kornia.geometry.relative_camera_motion(
            R1=Rts[view_i][:3, :3],
            t1=Rts[view_i][:3, 3:],
            R2=Rts[view_j][:3, :3],
            t2=Rts[view_j][:3, 3:],
        )
        rel_Rt = torch.cat([rel_R, rel_t], dim=-1)

        E = kornia.geometry.essential_from_Rt(
            R1=Rts[view_i][:3, :3],
            t1=Rts[view_i][:3, 3:],
            R2=Rts[view_j][:3, :3],
            t2=Rts[view_j][:3, 3:],
        )

        K_i = graph.nodes[view_i]["K"]
        K_j = graph.nodes[view_j]["K"]

        undistorted_points1 = graph.edges[view_i, view_j]["points1"]
        undistorted_points2 = graph.edges[view_j, view_i]["points2"]
        undistorted_points1_norm = kornia.geometry.normalize_points_with_intrinsics(
            undistorted_points1, camera_matrix=K_i
        )
        undistorted_points2_norm = kornia.geometry.normalize_points_with_intrinsics(
            undistorted_points2, camera_matrix=K_j
        )

        cost = (
            kornia.geometry.sampson_epipolar_distance(
                pts1=undistorted_points1_norm,
                pts2=undistorted_points2_norm,
                Fm=E.unsqueeze(0),
            )
            .mean()
            .item()
        )

        graph.edges[view_i, view_j]["Rt"] = rel_Rt
        graph.edges[view_j, view_i]["Rt"] = inverse_Rt(rel_Rt)
        graph.edges[view_i, view_j]["cost"] = cost
        graph.edges[view_j, view_i]["cost"] = cost

    return graph


def _initialize_graph(
    view_names: list[str],
    Ks: torch.Tensor,
    dist_coeffs: torch.Tensor,
    distortion_models: list[str],
    points1: list[torch.Tensor],
    points2: list[torch.Tensor],
    ransac_confidence: float = 0.99,
    ransac_reproj_threshold_px: float = 3.0,
) -> nx.DiGraph:
    """
    Compute the relative camera transformations between all pairs of views using 2D keypoint correspondences.
    The transformation is defined up to scale, i.e. X_2 = R_12 @ X_1 + λ_12 * t_12 where λ_12 is a positive scalar.

    This function uses RANSAC-based pose estimation to compute the essential matrix and recover the relative
    camera pose between each pair of views. The process includes:
    1. Selecting high-quality keypoint pairs across views
    2. Computing the essential matrix using RANSAC
    3. Recovering the relative rotation and translation
    4. Computing the error using Sampson distance

    Args:
        Ks: torch.Tensor of shape (n_views, 3, 3)
            Camera intrinsic matrices for each view
        dist_coeffs: list[torch.Tensor]
            Distortion coefficients for each camera (k1, k2, p1, p2, k3) for Brown-Conrady or (k1, k2, k3, k4) for OpenCV fisheye model
        distortion_models: list[str]
            Distortion model for each camera (Brown-Conrady or OpenCV fisheye)
        view_pair_correspondences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            List of tuples containing the keypoints and their correspondence scores for each view pair
        ransac_confidence: float, default=0.99
            Confidence level for RANSAC inlier selection (0-1)
        ransac_reproj_threshold_px: float, default=1.0
            Maximum reprojection error in pixels for RANSAC inlier selection

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - Rt_ij: torch.Tensor of shape (n_views, n_views, 3, 4)
                Relative camera transformations between all view pairs
            - errors: torch.Tensor of shape (n_views, n_views)
                Mean Sampson distance error for each view pair
    """

    device = Ks.device
    n_views = Ks.shape[0]

    graph = nx.DiGraph()
    for view_idx in range(n_views):
        graph.add_node(
            view_idx,
            view_idx=view_idx,
            view_name=view_names[view_idx],
            K=Ks[view_idx],
            distCoeffs=dist_coeffs[view_idx],
        )
        graph.add_edge(
            view_idx, view_idx, cost=0, Rt=torch.eye(4, device=device)[:3, :]
        )

    pairs = list(itertools.combinations(range(n_views), 2))

    # cv2.recoverPose releases the GIL and draws from a call-local RNG, not
    # cv2.theRNG(), so threading the 190 pairs of a 20-view rig is bit-exact.
    # Only the RANSAC is threaded; torch stays on the calling thread.
    undistorted_points: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    pose_futures: dict[int, concurrent.futures.Future] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for pair_idx, (view_i, view_j) in enumerate(pairs):
            pair_points1 = points1[pair_idx].to(device)
            pair_points2 = points2[pair_idx].to(device)
            # pair_confidence_scores = confidence_scores[pair_idx]

            n_points = pair_points1.shape[0]

            if n_points < 5:
                warnings.warn(
                    f"Not enough points in pair ({view_i}, {view_j}). Expected at least 5, got {n_points}"
                )
                graph.add_edge(
                    view_i,
                    view_j,
                    cost=torch.tensor(math.inf, device=device),
                    Rt=torch.eye(4, device=device)[:3, :],
                )
                graph.add_edge(
                    view_j,
                    view_i,
                    cost=torch.tensor(math.inf, device=device),
                    Rt=torch.eye(4, device=device)[:3, :],
                )
                continue

            undistorted_points1 = undistort_points(
                points=pair_points1.to(device),
                K=Ks[view_i].to(device),
                D=dist_coeffs[view_i].to(device),
                distortion_model=distortion_models[view_i],
            )
            undistorted_points2 = undistort_points(
                points=pair_points2.to(device),
                K=Ks[view_j].to(device),
                D=dist_coeffs[view_j].to(device),
                distortion_model=distortion_models[view_j],
            )
            undistorted_points[pair_idx] = (undistorted_points1, undistorted_points2)

            pose_futures[pair_idx] = executor.submit(
                cv2.recoverPose,
                points1=undistorted_points1.cpu().numpy(),
                points2=undistorted_points2.cpu().numpy(),
                cameraMatrix1=Ks[view_i].cpu().numpy(),
                distCoeffs1=np.zeros(5),
                cameraMatrix2=Ks[view_j].cpu().numpy(),
                distCoeffs2=np.zeros(5),
                prob=ransac_confidence,
                threshold=ransac_reproj_threshold_px,
                method=cv2.FM_RANSAC,
            )

        for pair_idx, (view_i, view_j) in tqdm(
            enumerate(pairs),
            total=len(pairs),
            leave=False,
            desc="Computing pairwise relative extrinsics",
        ):
            if pair_idx not in pose_futures:
                continue

            undistorted_points1, undistorted_points2 = undistorted_points[pair_idx]
            _, E, R, t, inliers_mask = pose_futures[pair_idx].result()

            inliers_mask = torch.from_numpy(inliers_mask).squeeze(-1).bool().to(device)
            E = torch.from_numpy(E).to(dtype=torch.float32, device=device)
            R = torch.from_numpy(R).to(dtype=torch.float32, device=device)
            t = torch.from_numpy(t).to(dtype=torch.float32, device=device)
            t = t / torch.norm(t)

            pair_points1_norm = kornia.geometry.normalize_points_with_intrinsics(
                undistorted_points1, camera_matrix=Ks[view_i].to(device)
            )
            pair_points2_norm = kornia.geometry.normalize_points_with_intrinsics(
                undistorted_points2, camera_matrix=Ks[view_j].to(device)
            )

            best_error = (
                kornia.geometry.sampson_epipolar_distance(
                    pts1=pair_points1_norm,
                    pts2=pair_points2_norm,
                    Fm=E.unsqueeze(0),
                )
                .mean()
                .item()
            )

            best_candidate_Rt = torch.cat([R, t], dim=-1)

            graph.add_edge(
                view_i,
                view_j,
                points1=undistorted_points1,
                points2=undistorted_points2,
                cost=best_error,
                Rt=best_candidate_Rt,
                inliers_mask=inliers_mask,
            )
            graph.add_edge(
                view_j,
                view_i,
                points1=undistorted_points2,
                points2=undistorted_points1,
                cost=best_error,
                Rt=inverse_Rt(best_candidate_Rt),
                inliers_mask=inliers_mask,
            )

    return graph


def translation_spanning_tree(consistency: "nx.Graph") -> "nx.Graph":
    """Spanning tree for chaining translations.

    Ranked by full-pose loop closure rather than rotation closure: an edge can
    be perfectly rotation-consistent and still have no observable scale, and
    chaining a translation through one places the child camera on top of its
    parent. One cost, in the same ``-log(rate)`` units as the rotation tree, so
    there is nothing to blend and no weight to calibrate.

    Args:
        consistency: Undirected graph carrying ``pose_closure_cost`` per edge.

    Returns:
        A minimum spanning tree over the same nodes.
    """
    return nx.minimum_spanning_tree(consistency, weight="pose_closure_cost")


def edge_pose_closure_rates(
    node_R: dict,
    edges: dict,
    n_views: int,
    rel_thresh: float = 0.10,
) -> dict:
    """Fraction of each edge's triplets whose full pose loop closes.

    The rotation-only rate in :func:`edge_closure_rates` is silent about
    whether an edge's scale was ever observable, so a scale-degenerate edge
    scores near-perfect and is free to carry translations. Closing the whole
    SE(3) loop instead folds both failure modes into one rate:

        lambda_ij R_ki R_jk t_ij + lambda_jk R_ki t_jk + lambda_ki t_ki ~ 0

    A broken rotation breaks this loop too, so nothing the rotation rate
    catches is lost. Unlike a collapse detector it also catches a scale that is
    merely *wrong* rather than zero, which is what survives once a prior stops
    the scale from collapsing.

    The residual is measured against the largest term, so the rate is
    invariant to the reconstruction's arbitrary overall scale.

    Args:
        node_R: Absolute rotation per view index.
        edges: Maps (i, j) to (R_ij, t_hat_ij, scale_ij).
        n_views: Number of views.
        rel_thresh: Closure tolerance, relative to the largest loop term.

    Returns:
        Maps ``frozenset((i, j))`` to a rate in [0, 1]. Edges appearing in no
        complete triplet score 1, matching the rotation rate's convention.
    """
    good: dict = {}
    total: dict = {}

    for i, j, k in itertools.combinations(range(n_views), 3):
        legs = ((i, j), (j, k), (k, i))
        if not all(e in edges for e in legs):
            continue

        R_ki = node_R[i] @ node_R[k].transpose(-1, -2)
        R_jk = node_R[k] @ node_R[j].transpose(-1, -2)

        terms = []
        magnitudes = []
        for (a, b), pre in zip(legs, (R_ki @ R_jk, R_ki, None)):
            _, t_hat, scale = edges[(a, b)]
            leg = scale * (t_hat if pre is None else pre @ t_hat)
            terms.append(leg)
            magnitudes.append(float(leg.norm()))

        residual = float(torch.stack(terms).sum(dim=0).norm())
        closed = float(residual <= rel_thresh * max(max(magnitudes), 1e-12))

        for a, b in legs:
            key = frozenset((a, b))
            good[key] = good.get(key, 0.0) + closed
            total[key] = total.get(key, 0.0) + 1.0

    rates = {}
    for (a, b) in edges:
        key = frozenset((a, b))
        rates[key] = good[key] / total[key] if total.get(key) else 1.0
    return rates


def _average_rotations(
    graph: nx.DiGraph,
    n_iters: int,
    huber_delta_rad: float,
) -> nx.DiGraph:
    """Robustly average edge rotations into absolute node rotations.

    Seeds absolute rotations by chaining edge rotations along a
    cycle-consistency-weighted MST from view 0, then refines with
    average_rotations over all edges. No edges are rejected: outliers are
    down-weighted, not removed (soft, threshold-free robustness). The seed MST
    routes along the most loop-consistent edges so a single outlier edge cannot
    corrupt it. Writes the result into each node's "Rt" rotation block; the
    translation block is left zero for the later translation-placement pass.

    Args:
        graph: SfM graph with pairwise "Rt" edges.
        n_iters: Sweeps for each of the L1 and IRLS averaging phases.
        huber_delta_rad: Huber threshold (radians) for the IRLS phase; also the
            loop-closure tolerance used to score edge cycle-consistency.

    Returns:
        The graph with absolute rotations stored in node "Rt".
    """
    n_views = graph.number_of_nodes()
    device = graph.nodes[0]["K"].device

    directed = [(i, j) for i, j in graph.edges() if i != j]
    node_pairs = torch.tensor(directed, dtype=torch.long)
    rel_rotations = torch.stack(
        [graph.edges[i, j]["Rt"][:3, :3] for i, j in directed]
    )

    # Cycle-consistency edge trust (SOTA: reweight, do not reject). Seed the
    # absolute rotations along an MST that prefers loop-consistent edges, so a
    # single gross-outlier edge cannot corrupt the seed. Cached on the graph and
    # reused by _compute_absolute_Rts so translation placement routes the same
    # way. All edges are kept; residual outliers are handled by the robust
    # averaging below.
    rates = edge_closure_rates(
        node_pairs, rel_rotations.cpu(), n_views, huber_delta_rad
    )
    undirected_rate: dict[frozenset, float] = {}
    for e, (i, j) in enumerate(directed):
        key = frozenset((i, j))
        undirected_rate[key] = min(
            undirected_rate.get(key, 1.0), float(rates[e])
        )
    consistency = nx.Graph()
    consistency.add_nodes_from(range(n_views))
    for key, rate in undirected_rate.items():
        i, j = tuple(key)
        # Cost, not a rate: an edge whose rotation loops mostly fail scores
        # near -log(1e-3) ~ 6.9, one whose loops all close scores ~0.
        consistency.add_edge(
            i, j, rotation_consistency_cost=-math.log(rate + 1e-3)
        )
    mst = nx.minimum_spanning_tree(
        consistency, weight="rotation_consistency_cost"
    )
    graph.graph["mst"] = mst
    # Kept so translation placement can rerank the same edges once the scales
    # are known.
    graph.graph["consistency"] = consistency

    R_seed = torch.eye(3, device=device).repeat(n_views, 1, 1)
    for view_idx in range(1, n_views):
        path = nx.shortest_path(mst, source=0, target=view_idx)
        R_acc = torch.eye(3, device=device)
        for a, b in zip(path[:-1], path[1:]):
            R_acc = graph.edges[a, b]["Rt"][:3, :3] @ R_acc
        R_seed[view_idx] = R_acc

    R_abs = average_rotations(
        node_pairs=node_pairs,
        rel_rotations=rel_rotations,
        R_seed=R_seed,
        n_l1_iters=n_iters,
        n_irls_iters=n_iters,
        huber_delta_rad=huber_delta_rad,
    )

    for view_idx in range(n_views):
        Rt = torch.zeros((3, 4), device=device)
        Rt[:3, :3] = R_abs[view_idx]
        graph.nodes[view_idx]["Rt"] = Rt

    return graph


def normalize_points(points: np.ndarray):
    """
    Normalize 2D homogeneous points for numerical stability.

    Args:
        points: array of shape (N, 2) or (N, 3) representing points in
                Euclidean or homogeneous coordinates.

    Returns:
        points_normalized: normalized points in homogeneous coordinates (N, 3)
        T: the 3x3 normalization transformation matrix
    """
    points = np.asarray(points)

    if points.shape[1] == 2:
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    else:
        points_h = points.copy()

    centroid = np.mean(points_h[:, :2], axis=0)
    shifted = points_h[:, :2] - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1)))
    scale = np.sqrt(2) / mean_dist

    T = np.array(
        [[scale, 0, -scale * centroid[0]], [0, scale, -scale * centroid[1]], [0, 0, 1]]
    )

    points_normalized = (T @ points_h.T).T
    return points_normalized[:, :2], T


def _refine_fundamental_matrix(
    F: np.ndarray, points1: np.ndarray, points2: np.ndarray
) -> np.ndarray:
    """
    Refine the fundamental matrix using the camera intrinsics.
    """

    from scipy.optimize import least_squares

    normalized_points1, T1 = normalize_points(points1)
    normalized_points2, T2 = normalize_points(points2)

    def enforce_rank_2(F: np.ndarray) -> np.ndarray:
        U, S, V = np.linalg.svd(F)
        S[2] = 0
        return U @ np.diag(S) @ V

    def residual(f: np.ndarray) -> np.ndarray:
        F = f.reshape(3, 3)
        sym_error = (
            kornia.geometry.epipolar.symmetrical_epipolar_distance(
                torch.from_numpy(normalized_points1).float(),
                torch.from_numpy(normalized_points2).float(),
                torch.from_numpy(F).float().reshape(1, 3, 3),
                squared=True,
            )
            .squeeze(0)
            .numpy()
        )

        return sym_error

    f0 = enforce_rank_2(F).flatten()
    res = least_squares(residual, f0, loss="soft_l1", f_scale=1.0)
    F_norm_refined = res.x.reshape(3, 3)
    F_refined = T2.T @ F_norm_refined @ T1
    F_refined = enforce_rank_2(F_refined)
    F_refined = F_refined / F_refined[-1, -1]

    # Compute error before and after refinement
    sym_error_before = (
        kornia.geometry.epipolar.symmetrical_epipolar_distance(
            torch.from_numpy(points1).float(),
            torch.from_numpy(points2).float(),
            torch.from_numpy(F).float().reshape(1, 3, 3),
            squared=True,
        )
        .squeeze(0)
        .numpy()
    )
    sym_error_after = (
        kornia.geometry.epipolar.symmetrical_epipolar_distance(
            torch.from_numpy(points1).float(),
            torch.from_numpy(points2).float(),
            torch.from_numpy(F_refined).float().reshape(1, 3, 3),
            squared=True,
        )
        .squeeze(0)
        .numpy()
    )
    print(
        f"Sym error before refinement: {sym_error_before.mean()}, after refinement: {sym_error_after.mean()}"
    )

    if sym_error_after.mean() < sym_error_before.mean():
        print("Improved error for fundamental matrix")
        return F_refined

    return F_refined


SCALE_TRIPLET_WEIGHTINGS = ("irls", "edge_cost", "uniform")


def edge_cost_triplet_weights(
    costs: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Weights a triplet by how cheap its three edges were, best to worst.

    The pre-IRLS scale weighting. Costs only compare within a sequence, hence
    the min-max normalisation; invalid triplets stay out of it so one unusable
    triplet cannot flatten the rest. The caller square-roots these, reproducing
    that solve's sqrt(1 - cost) factor.

    Args:
        costs: Mean edge cost per triplet, shape (L,).
        valid: Whether each triplet's edges all yielded a relative pose.

    Returns:
        Weights in [0, 1] with shape (L,), zero wherever valid is False.
    """
    weights = torch.zeros_like(costs)

    if not bool(valid.any()):
        return weights

    valid_costs = costs[valid]
    lo, hi = valid_costs.min(), valid_costs.max()
    normalized = (
        (valid_costs - lo) / (hi - lo)
        if hi > lo
        else torch.zeros_like(valid_costs)
    )
    weights[valid] = (1.0 - normalized).clamp_min(0.0)
    return weights


def _compute_relative_scale_factors(
    graph: nx.DiGraph,
    n_iters: int = 10,
    n_irls_iters: int = 5,
    weighting: str = "irls",
    tolerance_grad: float = 1e-05,
    tolerance_change: float = 1e-09,
    max_closure_residual_ratio: float = 0.10,
) -> nx.DiGraph:
    """
    Compute the scale factors for the relative transformations in the graph.

    Minimizes the error in loop closure constraints while maintaining the relative geometry between cameras.

    Args:
        graph: nx.DiGraph
            Graph containing the camera transformations

    Returns:
        nx.DiGraph
            Updated graph with the scale factors for the relative transformations
    """
    n_views = graph.number_of_nodes()
    # Only surviving edges after outlier rejection carry a scale; pairs and
    # triplets referencing a removed edge are skipped.
    pairs = [
        (i, j)
        for i, j in itertools.combinations(range(n_views), 2)
        if graph.has_edge(i, j)
    ]
    n_pairs = len(pairs)
    triplets = [
        (i, j, k)
        for i, j, k in itertools.combinations(range(n_views), 3)
        if graph.has_edge(i, j)
        and graph.has_edge(j, k)
        and graph.has_edge(k, i)
    ]
    n_triplets = len(triplets)
    device = graph.nodes[0]["K"].device

    if n_pairs == 1:
        graph[0][1]["lambda_idx"] = 0
        graph[1][0]["lambda_idx"] = 0
        graph[0][1]["scale"] = 1.0
        graph[1][0]["scale"] = 1.0
        return graph

    for pair_idx, (i, j) in enumerate(pairs):
        graph[i][j]["lambda_idx"] = pair_idx
        graph[j][i]["lambda_idx"] = pair_idx

    # Unweighted translation loop-closure design matrix. Rotation coefficients use
    # the averaged absolute node rotations (trustworthy) rather than raw edge
    # rotations. A triplet is invalid when any of its edges failed pose estimation.
    L, M = n_triplets, n_pairs
    if weighting not in SCALE_TRIPLET_WEIGHTINGS:
        raise ValueError(
            f"Unknown scale triplet weighting: {weighting}. "
            f"Expected one of {SCALE_TRIPLET_WEIGHTINGS}."
        )

    A_base = torch.zeros((3, L, M), device=device)
    valid = torch.zeros(L, dtype=torch.bool, device=device)
    triplet_costs = torch.zeros(L, device=device)

    for triplet_idx, (i, j, k) in enumerate(triplets):
        if any(
            torch.isinf(torch.as_tensor(graph[a][b]["cost"]))
            for a, b in ((i, j), (j, k), (k, i))
        ):
            continue
        valid[triplet_idx] = True
        triplet_costs[triplet_idx] = sum(
            torch.as_tensor(graph[a][b]["cost"], device=device)
            for a, b in ((i, j), (j, k), (k, i))
        ) / 3.0

        R_i = graph.nodes[i]["Rt"][:3, :3]
        R_j = graph.nodes[j]["Rt"][:3, :3]
        R_k = graph.nodes[k]["Rt"][:3, :3]
        R_ki = R_i @ R_k.transpose(-1, -2)
        R_jk = R_k @ R_j.transpose(-1, -2)

        t_ij = graph[i][j]["Rt"][:3, 3]
        t_jk = graph[j][k]["Rt"][:3, 3]
        t_ki = graph[k][i]["Rt"][:3, 3]
        t_ij_hat = t_ij / torch.norm(t_ij)
        t_jk_hat = t_jk / torch.norm(t_jk)
        t_ki_hat = t_ki / torch.norm(t_ki)

        A_base[:, triplet_idx, graph[i][j]["lambda_idx"]] = R_ki @ R_jk @ t_ij_hat
        A_base[:, triplet_idx, graph[j][k]["lambda_idx"]] = R_ki @ t_jk_hat
        A_base[:, triplet_idx, graph[k][i]["lambda_idx"]] = t_ki_hat

    log_lmb = torch.zeros((M,), device=device, requires_grad=True)
    if weighting == "edge_cost":
        weights = edge_cost_triplet_weights(triplet_costs, valid)
    else:
        weights = valid.float()
    # Only IRLS revisits its weights; the others solve once.
    n_rounds = n_irls_iters if weighting == "irls" else 1

    def closure():
        optimizer.zero_grad()
        lmb = torch.exp(log_lmb)
        A = torch.sqrt(weights).view(1, L, 1) * A_base
        sq_error = ((A @ lmb).norm(p=2) ** 2) / (3 * L)
        # Left as-is deliberately. Replacing this with a log-space or
        # one-sided prior stops the collapse but costs accuracy: an
        # unobservable scale then takes a plausible-but-wrong value that
        # neither the closure rate nor a collapse detector can flag.
        # Measured on badminton_001: AE 0.79 -> 3.76.
        reg_loss = torch.abs(1 - lmb).mean()
        total_loss = sq_error + 0.001 * reg_loss
        total_loss.backward()
        return total_loss

    pbar = tqdm(
        range(n_rounds), desc="Computing relative scale factors", leave=False
    )
    for _ in pbar:
        optimizer = torch.optim.LBFGS(
            [log_lmb], lr=1, line_search_fn="strong_wolfe"
        )
        prev_losses = []
        for _ in range(n_iters):
            loss = optimizer.step(closure)
            pbar.set_postfix(loss=loss.item())
            if optimizer_should_stop(
                optimizer,
                loss,
                prev_losses,
                patience=5,
                tolerance_grad=tolerance_grad,
                tolerance_change=tolerance_change,
            ):
                break
            prev_losses.append(loss.detach())

        if weighting != "irls":
            continue

        # Reweight triplets by their closure residual (Huber IRLS, adaptive scale).
        with torch.no_grad():
            residuals = (A_base @ torch.exp(log_lmb)).norm(dim=0)
            if valid.any():
                delta = 1.345 * residuals[valid].median().clamp_min(1e-9)
                weights = valid.float() * huber_weights(residuals, delta)
    pbar.close()

    lmb = torch.exp(log_lmb.detach())
    for pair_idx, (view_i, view_j) in enumerate(pairs):
        graph[view_i][view_j]["scale"] = lmb[pair_idx]
        graph[view_j][view_i]["scale"] = lmb[pair_idx]

    # Whether each edge's full pose loop closes now that the scales exist. The
    # rotation rate computed earlier could not see the scales at all.
    node_R = {n: d["Rt"][:3, :3] for n, d in graph.nodes(data=True) if "Rt" in d}
    pose_edges = {
        (i, j): (d["Rt"][:3, :3], d["Rt"][:3, 3], float(d.get("scale", 0.0)))
        for i, j, d in graph.edges(data=True)
        if i != j and "scale" in d
    }
    rates = edge_pose_closure_rates(
        node_R, pose_edges, n_views, rel_thresh=max_closure_residual_ratio
    )
    for key, rate in rates.items():
        view_i, view_j = tuple(key)
        cost = -math.log(rate + 1e-3)
        if graph.has_edge(view_i, view_j):
            graph[view_i][view_j]["pose_closure_cost"] = cost
            graph[view_j][view_i]["pose_closure_cost"] = cost

    return graph


def _compute_absolute_Rts(
    graph: nx.DiGraph,
    use_pose_closure_tree: bool = True,
) -> nx.DiGraph:
    """
    Place camera translations along the MST using the fixed averaged rotations.

    Composes the absolute camera poses by chaining transformations along the MST.

    Args:
        graph: nx.DiGraph
            Graph containing the camera transformations

    Returns:
        nx.DiGraph
            Updated graph with absolute camera poses stored in the nodes "Rt" attribute.
    """
    n_views = graph.number_of_nodes()
    device = graph.nodes[0]["K"].device
    # Reuse the cycle-consistency-weighted MST built by _average_rotations so
    # translation placement routes along the same trustworthy edges.
    consistency = graph.graph.get("consistency")
    if consistency is not None and use_pose_closure_tree:
        for i, j in consistency.edges:
            consistency[i][j]["pose_closure_cost"] = graph[i][j].get(
                "pose_closure_cost", 0.0
            )
        mst = translation_spanning_tree(consistency)
    else:
        mst = graph.graph["mst"]

    graph.nodes[0]["Rt"][:3, 3] = torch.zeros(3, device=device)

    for view_idx in range(1, n_views):
        path = nx.shortest_path(mst, source=0, target=view_idx)
        t = torch.zeros((3, 1), device=device)
        for a, b in zip(path[:-1], path[1:]):
            R_a = graph.nodes[a]["Rt"][:3, :3]
            R_b = graph.nodes[b]["Rt"][:3, :3]
            R_ab = R_b @ R_a.transpose(-1, -2)
            t_hat = graph.edges[a, b]["Rt"][:3, 3:]
            t = R_ab @ t + graph.edges[a, b]["scale"] * t_hat
        graph.nodes[view_idx]["Rt"][:3, 3] = t.squeeze(-1)

    return graph
