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
import itertools
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
from kineo.geometry.camera import inverse_Rt, multiply_Rt
from kineo.geometry.rotation_averaging import edge_closure_rates
from kineo.optimization.camera_parameters import CameraExtrinsicsParameters
from kineo.geometry.transformations import undistort_points

import math
import numpy as np
from dataclasses import dataclass

from kineo.optimization.utils import optimizer_should_stop


@dataclass(frozen=True)
class SfMCameraExtrinsicsInitializationRuntimeConfig:
    ransac_confidence: float = 0.99
    ransac_reproj_threshold: float = 3.0
    n_refinement_iters: int = 0
    tolerance_grad: float = 1e-05
    tolerance_change: float = 1e-09
    use_lbfgs: bool = True
    n_relative_scale_factors_iters: int = 10
    refinement_huber_delta: float = 1.0


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
        device = pipeline.device

        calibration_points_annotations: CalibrationPointsAnnotations = annotations[
            "calibration_points"
        ]

        if calibration_points_annotations is None:
            raise ValueError("No calibration points annotations found.")

        cameras_intrinsics: CameraIntrinsicsAnnotations = annotations[
            "camera_intrinsics"
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
            ransac_reproj_threshold=runtime_cfg.ransac_reproj_threshold,
            n_refinement_iters=runtime_cfg.n_refinement_iters,
            use_lbfgs=runtime_cfg.use_lbfgs,
            tolerance_grad=runtime_cfg.tolerance_grad,
            tolerance_change=runtime_cfg.tolerance_change,
            n_relative_scale_factors_iters=runtime_cfg.n_relative_scale_factors_iters,
            refinement_huber_delta=runtime_cfg.refinement_huber_delta,
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

        annotations["camera_extrinsics"] = CameraExtrinsicsAnnotations(
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
    ransac_reproj_threshold: float = 3.0,
    n_refinement_iters: int = 100,
    use_lbfgs: bool = True,
    tolerance_grad: float = 1e-05,
    tolerance_change: float = 1e-09,
    n_relative_scale_factors_iters: int = 10,
    refinement_huber_delta: float = 1.0,
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
        ransac_reproj_threshold: float
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
        ransac_reproj_threshold=ransac_reproj_threshold,
    )

    graph = _compute_relative_scale_factors(
        graph,
        n_iters=n_relative_scale_factors_iters,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
    )

    graph = _compute_absolute_Rts(graph)

    graph = _refine_camera_extrinsics(
        graph,
        n_iters=n_refinement_iters,
        use_lbfgs=use_lbfgs,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        refinement_huber_delta=refinement_huber_delta,
    )

    device = graph.nodes[0]["K"].device
    n_views = len(graph.nodes())
    Rts_init = torch.empty((n_views, 3, 4), dtype=torch.float32, device=device)

    for view_idx in range(n_views):
        Rts_init[view_idx] = graph.nodes[view_idx]["Rt"]

    return Rts_init


def _refine_camera_extrinsics(
    graph: nx.DiGraph,
    n_iters: int = 100,
    tolerance_grad: float = 1e-05,
    tolerance_change: float = 1e-09,
    use_lbfgs: bool = True,
    refinement_huber_delta: float = 1.0,
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

    # pruned_graph = _prune_graph(graph)

    # Remove self loops
    graph_undirected = graph.to_undirected()

    edges_no_selfloops = [
        (view_i, view_j)
        for view_i, view_j in graph_undirected.edges()
        if view_i != view_j
    ]

    print(f"Refining camera extrinsics using {len(edges_no_selfloops)} edges.")

    def opt_closure():
        optimizer.zero_grad()

        Rts = torch.cat(
            [origin_camera_extrinsics.Rt, other_cameras_extrinsics.Rt], dim=0
        )

        sampson_loss = torch.tensor(0.0, device=device)

        for view_i, view_j in edges_no_selfloops:
            # Take the points from the original graph, because the undirected one might have dropped the attributes
            undistorted_points1 = graph.edges[view_i, view_j]["points1"]
            undistorted_points2 = graph.edges[view_i, view_j]["points2"]
            K_i = graph.nodes[view_i]["K"]
            K_j = graph.nodes[view_j]["K"]

            E = kornia.geometry.essential_from_Rt(
                R1=Rts[view_i][:3, :3],
                t1=Rts[view_i][:3, 3:],
                R2=Rts[view_j][:3, :3],
                t2=Rts[view_j][:3, 3:],
            )

            F = kornia.geometry.epipolar.fundamental_from_essential(E, K_i, K_j)

            sampson_dist = kornia.geometry.epipolar.sampson_epipolar_distance(
                pts1=undistorted_points1,
                pts2=undistorted_points2,
                Fm=F.unsqueeze(0),
            ).squeeze(0)

            huber_loss = torch.nn.functional.huber_loss(
                input=sampson_dist,
                target=torch.zeros_like(sampson_dist),
                reduction="none",
                delta=refinement_huber_delta,
            )

            sampson_loss += huber_loss.mean()

        total_loss = sampson_loss / len(edges_no_selfloops)
        # Non-finite step (e.g. degenerate pose, |t|->0): large loss so LBFGS backtracks.
        if not torch.isfinite(total_loss):
            return torch.full_like(total_loss, 1e12)
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
        if view_i == view_j:
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
    ransac_reproj_threshold: float = 3.0,
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
        ransac_reproj_threshold: float, default=1.0
            Maximum reprojection error in pixels for RANSAC inlier selection
        max_points_pairs: int, default=2000
            Maximum number of point pairs to use for pose estimation

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

    for pair_idx, (view_i, view_j) in tqdm(
        enumerate(pairs),
        total=len(pairs),
        leave=False,
        desc="Computing pairwise relative extrinsics",
    ):
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

        _, E, R, t, inliers_mask = cv2.recoverPose(
            points1=undistorted_points1.cpu().numpy(),
            points2=undistorted_points2.cpu().numpy(),
            cameraMatrix1=Ks[view_i].cpu().numpy(),
            distCoeffs1=np.zeros(5),
            cameraMatrix2=Ks[view_j].cpu().numpy(),
            distCoeffs2=np.zeros(5),
            prob=ransac_confidence,
            threshold=ransac_reproj_threshold,
            method=cv2.FM_RANSAC,
        )
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


def _reject_rotation_outlier_edges(
    graph: nx.DiGraph,
    thresh_deg: float,
    min_close_rate: float,
) -> nx.DiGraph:
    """Drop gross rotation-outlier edges via triplet loop-closure voting.

    Removes undirected edges whose loop-closure rate (see
    kineo.geometry.rotation_averaging.edge_closure_rates) falls below
    min_close_rate, worst-first, skipping any removal that would disconnect the
    graph. Blunt pre-filter that protects the seed and early averaging; the fine
    down-weighting is done later by rotation averaging.

    Args:
        graph: SfM graph with per-edge "Rt" relative poses.
        thresh_deg: Loop-closure tolerance in degrees.
        min_close_rate: Minimum fraction of closing triplets to keep an edge.

    Returns:
        The graph with outlier edges removed.
    """
    n_views = graph.number_of_nodes()
    directed = [(i, j) for i, j in graph.edges() if i != j]
    if not directed:
        return graph

    node_pairs = torch.tensor(directed, dtype=torch.long)
    rel_rotations = torch.stack(
        [graph.edges[i, j]["Rt"][:3, :3].cpu() for i, j in directed]
    )
    rates = edge_closure_rates(
        node_pairs, rel_rotations, n_views, math.radians(thresh_deg)
    )

    undirected_rate: dict[frozenset, float] = {}
    for e, (i, j) in enumerate(directed):
        key = frozenset((i, j))
        undirected_rate[key] = min(undirected_rate.get(key, 1.0), float(rates[e]))

    candidates = sorted(
        (rate, tuple(sorted(key)))
        for key, rate in undirected_rate.items()
        if rate < min_close_rate
    )

    work = graph.to_undirected()
    work.remove_edges_from(list(nx.selfloop_edges(work)))

    for _, (i, j) in candidates:
        work.remove_edge(i, j)
        if nx.is_connected(work):
            graph.remove_edge(i, j)
            graph.remove_edge(j, i)
        else:
            work.add_edge(i, j)

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


def _prune_graph(G: nx.DiGraph) -> nx.DiGraph:
    """
    Prune the graph by removing edges. Keeps the graph rigid and with its original MST.

    In practice, we do the opposite, i.e. build a rigid graph using Henneberg I algorithm.
    """
    undirected_graph: nx.Graph = G.to_undirected()
    selfloop_edges = list(nx.selfloop_edges(undirected_graph))
    undirected_graph.remove_edges_from(selfloop_edges)

    n = undirected_graph.number_of_nodes()

    mst: nx.Graph = nx.minimum_spanning_tree(undirected_graph, weight="cost")

    available_edges = list(undirected_graph.edges())
    # MST edges are first, then sorted by cost
    available_edges.sort(
        key=lambda x: (x not in mst.edges(), undirected_graph[x[0]][x[1]]["cost"])
    )

    L = nx.Graph()

    nodes = list(nx.dfs_preorder_nodes(mst))

    # We now construct the rigid graph following Henneberg construction
    # We greedily connect new nodes to two existing nodes in the graph. In that case, we
    # always pick the MST edges and then the remaining edges sorted by cost.
    for i in range(n):
        node = nodes[i]

        L.add_node(node)
        # print(f"Added node {node}")

        if i == 0:
            continue

        existing_vertices = set(L.nodes())

        # Connect by new vertex using two available edges
        candidate_edges = [
            e
            for e in available_edges
            if (e[0] == node and e[1] in existing_vertices)
            or (e[1] == node and e[0] in existing_vertices)
        ]
        first_edge = candidate_edges.pop(0)
        available_edges.remove(first_edge)
        L.add_edge(first_edge[0], first_edge[1])
        # print(f"Added edge {first_edge[0]}-{first_edge[1]}")

        if i == 1:
            continue

        second_edge = candidate_edges.pop(0)
        available_edges.remove(second_edge)
        L.add_edge(second_edge[0], second_edge[1])
        # print(f"Added edge {second_edge[0]}-{second_edge[1]}")

    mst_edges = list(mst.edges())
    L.add_edges_from(mst_edges)
    L.add_edges_from(selfloop_edges)

    # Now remove edges that are not in L in G.
    G = G.copy()
    G_edges = list(G.edges())

    for edge in G_edges:
        if (edge[0], edge[1]) not in L.edges() and (edge[1], edge[0]) not in L.edges():
            try:
                G.remove_edge(edge[0], edge[1])
                G.remove_edge(edge[1], edge[0])
            except nx.NetworkXError:
                pass
    return G


def _compute_relative_scale_factors(
    graph: nx.DiGraph,
    n_iters: int = 10,
    tolerance_grad: float = 1e-05,
    tolerance_change: float = 1e-09,
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
    pairs = list(itertools.combinations(range(n_views), 2))
    n_pairs = len(pairs)
    n_triplets = len(list(itertools.combinations(range(n_views), 3)))

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

    triplet_costs = torch.zeros((n_triplets,), device=device)

    for triplet_idx, (i, j, k) in enumerate(itertools.combinations(range(n_views), 3)):
        cost_ij = torch.as_tensor(graph[i][j]["cost"])
        cost_jk = torch.as_tensor(graph[j][k]["cost"])
        cost_ki = torch.as_tensor(graph[k][i]["cost"])

        # Check if any of the edges are inf
        if torch.isinf(cost_ij) or torch.isinf(cost_jk) or torch.isinf(cost_ki):
            triplet_costs[triplet_idx] = torch.inf
            continue

        triplet_costs[triplet_idx] = (cost_ij + cost_jk + cost_ki) / 3

    non_inf_mask = ~torch.isinf(triplet_costs)
    triplet_cost_min = triplet_costs[non_inf_mask].min()
    triplet_cost_max = triplet_costs[non_inf_mask].max()

    if triplet_cost_min == triplet_cost_max:
        # Degenrate case where all triplets have the same cost
        triplet_costs[non_inf_mask] = 0.0
    else:
        # Normalize the costs to be between 0 and 1
        triplet_costs[non_inf_mask] = (
            triplet_costs[non_inf_mask] - triplet_cost_min
        ) / (triplet_cost_max - triplet_cost_min)

    L = n_triplets
    M = n_pairs
    A = torch.zeros((3, L, M), device=device)

    for triplet_idx, (i, j, k) in enumerate(itertools.combinations(range(n_views), 3)):
        lambda_ij_idx = graph[i][j]["lambda_idx"]
        lambda_jk_idx = graph[j][k]["lambda_idx"]
        lambda_ki_idx = graph[k][i]["lambda_idx"]

        triplet_cost = triplet_costs[triplet_idx]

        if torch.isinf(triplet_cost):
            warnings.warn(f"Triplet ({i}, {j}, {k}) has infinite cost.")
            A[:, triplet_idx, lambda_ij_idx] = 0
            A[:, triplet_idx, lambda_jk_idx] = 0
            A[:, triplet_idx, lambda_ki_idx] = 0
            continue

        R_ki = graph[k][i]["Rt"][:3, :3]
        R_jk = graph[j][k]["Rt"][:3, :3]
        t_ij = graph[i][j]["Rt"][:3, 3]
        t_jk = graph[j][k]["Rt"][:3, 3]
        t_ki = graph[k][i]["Rt"][:3, 3]
        t_ij_hat = t_ij / torch.norm(t_ij)
        t_jk_hat = t_jk / torch.norm(t_jk)
        t_ki_hat = t_ki / torch.norm(t_ki)

        triplet_weight_sq = torch.sqrt(torch.clamp(1 - triplet_cost, min=0.0))  # W^1/2
        A[:, triplet_idx, lambda_ij_idx] = triplet_weight_sq * (R_ki @ R_jk @ t_ij_hat)
        A[:, triplet_idx, lambda_jk_idx] = triplet_weight_sq * (R_ki @ t_jk_hat)
        A[:, triplet_idx, lambda_ki_idx] = triplet_weight_sq * (t_ki_hat)

    def closure():
        optimizer.zero_grad()
        lmb = torch.exp(log_lmb)
        sq_error = ((A @ lmb).norm(p=2) ** 2) / (3 * L)
        reg_loss = torch.abs(1 - lmb).mean()
        total_loss = sq_error + 0.001 * reg_loss
        total_loss.backward()
        return total_loss

    log_lmb = torch.zeros((M,), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_lmb], lr=1, line_search_fn="strong_wolfe")

    prev_losses = []

    pbar = tqdm(range(n_iters), desc="Computing relative scale factors", leave=False)

    for iter in pbar:
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

    pbar.close()

    lmb = torch.exp(log_lmb.detach())

    for pair_idx, (view_i, view_j) in enumerate(pairs):
        graph[view_i][view_j]["scale"] = lmb[pair_idx]
        graph[view_j][view_i]["scale"] = lmb[pair_idx]

    return graph


def _compose_path_Rt_rel(graph: nx.DiGraph, path: list[int]) -> torch.Tensor:
    """
    Compose a sequence of relative camera transformations along a path.

    Given a path of camera indices and their pairwise relative transformations,
    this function computes the cumulative transformation by chaining the individual
    transformations along the path. The translations are scaled according to the
    provided scale factors.

    Args:
        graph: nx.DiGraph
            Graph containing the camera transformations
        path: list[int]
            Sequence of camera indices representing the path

    Returns:
        torch.Tensor of shape (3, 4)
            Composed camera transformation matrix representing the cumulative
            transformation along the path
    """
    device = graph.nodes[0]["K"].device
    Rt_rel = torch.eye(4, device=device)[:3, :]

    for i in range(len(path) - 1):
        scale_ij = graph[path[i]][path[i + 1]]["scale"]
        Rt_ij = graph[path[i]][path[i + 1]]["Rt"]

        rotation = Rt_ij[:3, :3]
        translation = Rt_ij[:3, 3:]
        scaled_translation = scale_ij * translation

        RT = torch.cat(
            [rotation, scaled_translation],
            dim=-1,
        )

        Rt_rel = multiply_Rt(RT, Rt_rel)

    return Rt_rel


def _compute_absolute_Rts(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Compute absolute camera poses from relative transformations.

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

    Rts = torch.eye(4, device=device)[:3, :].repeat(n_views, 1, 1)
    mst = nx.minimum_spanning_tree(graph.to_undirected(), weight="cost")

    graph.nodes[0]["Rt"] = Rts[0]

    for view_idx in range(1, n_views):
        path = nx.shortest_path(mst, source=0, target=view_idx)
        Rt_0i = _compose_path_Rt_rel(graph, path)
        Rts[view_idx] = multiply_Rt(Rt_0i, Rts[0])
        graph.nodes[view_idx]["Rt"] = Rts[view_idx]

    return graph
