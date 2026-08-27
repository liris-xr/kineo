import torch

from kineo.geometry.camera import positive_depth_mask


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


# --- Stage-level tests -------------------------------------------------------

N_VIEWS = 3
N_FRAMES = 20
N_KEYPOINTS = 5
RESOLUTION_HW = (100, 200)
MIN_KP_SCORE = 0.6

# Planted slots, addressed as (frame_idx, keypoint_idx). With one subject the
# flat candidate index is frame_idx * N_KEYPOINTS + keypoint_idx.
OFF_FRAME_SLOT = (3, 1)  # view 0 off-frame, view 2 below threshold
SINGLE_VIEW_SLOT = (7, 2)  # only view 0 above threshold
OFF_FRAME_SENTINEL = (11.0, 12.0)
SINGLE_VIEW_SENTINEL = (13.0, 14.0)


class _StubPipeline:
    """The two pipeline attributes the stage reads."""

    def __init__(self, seed: int = 19):
        self.device = torch.device("cpu")
        self.seed = seed


def _synthetic_annotations():
    from kineo.annotations.camera_extrinsics import (
        CameraExtrinsicsAnnotation,
        CameraExtrinsicsAnnotations,
        CameraExtrinsicsAnnotationsMetadata,
    )
    from kineo.annotations.camera_intrinsics import (
        CameraDistortionModel,
        CameraIntrinsicsAnnotation,
        CameraIntrinsicsAnnotations,
        CameraIntrinsicsAnnotationsMetadata,
    )
    from kineo.annotations.global_time_reference import (
        GlobalTimeReferenceAnnotation,
        GlobalTimeReferenceAnnotations,
        GlobalTimeReferenceAnnotationsMetadata,
    )
    from kineo.annotations.keypoints_2d import (
        Keypoints2DAnnotation,
        Keypoints2DAnnotations,
        Keypoints2DAnnotationsMetadata,
    )
    from kineo.annotations.keypoints_format import KeypointsFormat

    generator = _generator(11)
    views_ids = [f"view_{i}" for i in range(N_VIEWS)]
    height, width = RESOLUTION_HW

    K = torch.tensor(
        [[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0.0, 0.0, 1.0]]
    )

    intrinsics = []
    extrinsics = []

    for view_idx, view_id in enumerate(views_ids):
        intrinsics.append(
            CameraIntrinsicsAnnotation(
                view_id=view_id,
                frame_idx=0,
                K=K.clone(),
                distortion_coefficients=torch.zeros(5),
                distortion_model=CameraDistortionModel.BROWN_CONRADY,
                resolution_hw=RESOLUTION_HW,
            )
        )
        extrinsics.append(
            CameraExtrinsicsAnnotation(
                view_id=view_id,
                frame_idx=0,
                R=torch.eye(3),
                t=torch.tensor([float(view_idx) - 1.0, 0.0, 5.0]),
            )
        )

    keypoints_format = KeypointsFormat(
        name="synthetic",
        n_keypoints=N_KEYPOINTS,
        keypoints_names=[f"kp_{i}" for i in range(N_KEYPOINTS)],
        keypoints_connectivity=[],
    )

    # Kept well inside the frame so the planted sentinels cannot collide.
    xy = torch.rand(
        N_VIEWS, N_FRAMES, N_KEYPOINTS, 2, generator=generator
    ) * torch.tensor([100.0, 50.0]) + torch.tensor([50.0, 25.0])
    scores = torch.full((N_VIEWS, N_FRAMES, N_KEYPOINTS), 0.9)

    off_frame_f, off_frame_k = OFF_FRAME_SLOT
    xy[0, off_frame_f, off_frame_k] = torch.tensor([5000.0, 5000.0])
    xy[1, off_frame_f, off_frame_k] = torch.tensor(OFF_FRAME_SENTINEL)
    scores[2, off_frame_f, off_frame_k] = 0.1

    single_view_f, single_view_k = SINGLE_VIEW_SLOT
    xy[0, single_view_f, single_view_k] = torch.tensor(SINGLE_VIEW_SENTINEL)
    scores[1, single_view_f, single_view_k] = 0.1
    scores[2, single_view_f, single_view_k] = 0.1

    keypoints = [
        Keypoints2DAnnotation(
            view_id=view_id,
            frame_idx=frame_idx,
            subject_id="subject_0",
            xy=xy[view_idx, frame_idx].clone(),
            scores=scores[view_idx, frame_idx].clone(),
            format="synthetic",
        )
        for view_idx, view_id in enumerate(views_ids)
        for frame_idx in range(N_FRAMES)
    ]

    annotations = {
        "camera_intrinsics": CameraIntrinsicsAnnotations(
            metadata=CameraIntrinsicsAnnotationsMetadata(),
            annotations=intrinsics,
        ),
        "camera_extrinsics": CameraExtrinsicsAnnotations(
            metadata=CameraExtrinsicsAnnotationsMetadata(),
            annotations=extrinsics,
        ),
        "keypoints_2d": Keypoints2DAnnotations(
            metadata=Keypoints2DAnnotationsMetadata(formats=[keypoints_format]),
            annotations=keypoints,
        ),
        "global_time_reference": GlobalTimeReferenceAnnotations(
            metadata=GlobalTimeReferenceAnnotationsMetadata(),
            annotations=[
                GlobalTimeReferenceAnnotation(
                    timestamps=torch.arange(N_FRAMES, dtype=torch.float32) / 20.0,
                    closest_local_frame_idx={
                        view_id: torch.arange(N_FRAMES) for view_id in views_ids
                    },
                )
            ],
        ),
    }
    views = [{"view_id": view_id} for view_id in views_ids]
    return views, annotations


def _run_stage(
    n_kp_samples_per_view: int,
    sampler: str = "farthest_point",
    filter_negative_depth: bool = False,
    min_parallax_deg: float | None = None,
    max_reproj_error: float | None = None,
    w_d: float = 0.0,
    filter_off_frame_keypoints: bool = True,
):
    from kineo.pipeline.stages.bundle_adjustment_sampling import (
        BundleAdjustmentSamplingRuntimeConfig,
        BundleAdjustmentSamplingStage,
    )

    views, annotations = _synthetic_annotations()
    runtime_cfg = BundleAdjustmentSamplingRuntimeConfig(
        n_kp_samples_per_view=n_kp_samples_per_view,
        min_kp_score=MIN_KP_SCORE,
        sampler=sampler,
        filter_off_frame_keypoints=filter_off_frame_keypoints,
        filter_negative_depth=filter_negative_depth,
        min_parallax_deg=min_parallax_deg,
        max_reproj_error=max_reproj_error,
        w_d=w_d,
    )
    stage = BundleAdjustmentSamplingStage(
        name="Bundle Adjustment FPS Sampling", order=55, runtime_cfg=runtime_cfg
    )
    stage.forward(
        sequence_name="synthetic",
        pipeline=_StubPipeline(),
        views=views,
        annotations=annotations,
        gt_annotations={},
        runtime_cfg=runtime_cfg,
    )
    return annotations["bundle_adjustment_keypoints"].first_or_default()


def _contains_point(points: torch.Tensor, point: tuple[float, float]) -> bool:
    return bool(
        (points == torch.tensor(point)).all(dim=-1).any()
    )


def test_stage_rejects_off_frame_and_single_view_candidates():
    annotation = _run_stage(n_kp_samples_per_view=-1)

    assert not _contains_point(annotation.kps_2d_xy[1], OFF_FRAME_SENTINEL)
    assert not _contains_point(annotation.kps_2d_xy[0], SINGLE_VIEW_SENTINEL)


def test_disabling_the_filter_keeps_an_off_frame_observation():
    # The uniform-baseline arm: no rejection, so the planted off-frame slot
    # reaches the bundle adjustment. Only the single-view slot is still dropped,
    # since two views above threshold is a candidacy rule, not a filter.
    annotation = _run_stage(
        n_kp_samples_per_view=-1, filter_off_frame_keypoints=False
    )

    assert _contains_point(annotation.kps_2d_xy[1], OFF_FRAME_SENTINEL)
    assert not _contains_point(annotation.kps_2d_xy[0], SINGLE_VIEW_SENTINEL)


def test_stage_keeps_every_surviving_candidate_when_unbounded():
    annotation = _run_stage(n_kp_samples_per_view=-1)

    # Every slot survives except the two planted ones.
    assert annotation.kps_2d_xy.shape[1] == N_FRAMES * N_KEYPOINTS - 2


def test_fps_spreads_wider_than_uniform_on_the_same_input():
    n_kp_samples = 20
    height, width = RESOLUTION_HW
    scale = torch.tensor([width, height], dtype=torch.float32)

    def mean_nearest_neighbour_distance(annotation) -> float:
        points = annotation.kps_2d_xy[0] / scale
        dists = torch.cdist(points, points)
        dists.fill_diagonal_(float("inf"))
        return float(dists.min(dim=1).values.mean())

    fps = mean_nearest_neighbour_distance(
        _run_stage(n_kp_samples, "farthest_point")
    )
    uniform = mean_nearest_neighbour_distance(
        _run_stage(n_kp_samples, "uniform")
    )

    assert fps > uniform


def test_stage_rejects_an_unknown_sampler():
    import pytest

    with pytest.raises(ValueError, match="Unsupported sampler"):
        _run_stage(n_kp_samples_per_view=10, sampler="grid")


# --- Cheirality filter -------------------------------------------------------


def test_positive_depth_mask_separates_front_from_behind():
    # Identity rotation, camera at the world origin: depth is just z.
    Rts = torch.eye(3, 4).unsqueeze(0)
    kps_3d = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0], [0.0, 0.0, 0.0]])

    mask = positive_depth_mask(kps_3d, Rts)

    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.tensor([[True, False, False]]))


def test_positive_depth_mask_is_per_view():
    # Second view is translated so the same point lands behind it.
    Rts = torch.stack([torch.eye(3, 4), torch.eye(3, 4)])
    Rts[1, 2, 3] = -10.0
    kps_3d = torch.tensor([[0.0, 0.0, 5.0]])

    mask = positive_depth_mask(kps_3d, Rts)

    assert torch.equal(mask, torch.tensor([[True], [False]]))


def _rig_extrinsics() -> torch.Tensor:
    """The (n_views, 3, 4) extrinsics the synthetic rig is built with."""
    return torch.stack(
        [
            torch.cat(
                [torch.eye(3), torch.tensor([i - 1.0, 0.0, 5.0]).reshape(3, 1)],
                dim=1,
            )
            for i in range(N_VIEWS)
        ]
    )


def _weighted_observations(annotation) -> torch.Tensor:
    return annotation.kps_2d_scores > 0


def test_stage_emits_observations_behind_their_view_without_the_gate():
    # Each view's 2D is drawn independently, so the rig triangulates to an
    # incoherent cloud. That is what gives the gate something to reject here;
    # it says nothing about how often this happens on real sequences.
    annotation = _run_stage(n_kp_samples_per_view=-1)

    in_front = positive_depth_mask(annotation.kps_3d, _rig_extrinsics())

    assert bool((_weighted_observations(annotation) & ~in_front).any())


def test_cheirality_gate_drops_observations_behind_their_view():
    annotation = _run_stage(n_kp_samples_per_view=-1, filter_negative_depth=True)

    in_front = positive_depth_mask(annotation.kps_3d, _rig_extrinsics())

    assert bool((_weighted_observations(annotation) & ~in_front).sum() == 0)
    assert bool((_weighted_observations(annotation).sum(dim=0) >= 2).all())


def test_reprojection_gate_bounds_the_residual_of_every_kept_observation():
    from kineo.geometry.metrics import compute_normalized_reprojection_residuals

    max_reproj_error = 0.01
    annotation = _run_stage(n_kp_samples_per_view=-1, max_reproj_error=max_reproj_error)

    height, width = RESOLUTION_HW
    Ks = torch.tensor(
        [[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0.0, 0.0, 1.0]]
    ).expand(N_VIEWS, 3, 3)
    residuals, _ = compute_normalized_reprojection_residuals(
        kps_3d=annotation.kps_3d,
        kps_2d=annotation.kps_2d_xy,
        Ks=Ks,
        Rts=_rig_extrinsics(),
        Ds=torch.zeros(N_VIEWS, 5),
        distortion_model="brown_conrady",
    )

    kept = _weighted_observations(annotation)
    assert bool(kept.any())
    # The emitted points are retriangulated from the gated weights, so this
    # rechecks the residual against the point the stage actually exports.
    assert float(residuals[kept].max()) < 2.0 * max_reproj_error


def test_the_gate_only_ever_shrinks_the_candidate_set():
    ungated = _run_stage(n_kp_samples_per_view=-1)
    gated = _run_stage(
        n_kp_samples_per_view=-1,
        filter_negative_depth=True,
        min_parallax_deg=1.0,
        max_reproj_error=0.01,
    )

    assert 0 < gated.kps_3d.shape[0] < ungated.kps_3d.shape[0]


def test_depth_axis_changes_the_selection():
    per_view = 20
    without = _run_stage(n_kp_samples_per_view=per_view, w_d=0.0)
    with_depth = _run_stage(n_kp_samples_per_view=per_view, w_d=1.0)

    def selection(annotation):
        return {tuple(p) for p in annotation.kps_2d_xy[0].tolist()}

    # The unions need not be the same size: the depth axis changes which
    # candidates each view picks, and so how much the views overlap.
    assert selection(with_depth) != selection(without)
    for annotation in (without, with_depth):
        assert per_view <= annotation.kps_2d_xy.shape[1] <= per_view * N_VIEWS


def test_gate_keeps_the_annotation_contract():
    annotation = _run_stage(n_kp_samples_per_view=-1, filter_negative_depth=True)

    n_samples = annotation.kps_2d_xy.shape[1]
    assert 0 < n_samples < N_FRAMES * N_KEYPOINTS
    assert annotation.kps_2d_scores.shape == (N_VIEWS, n_samples)
    assert annotation.kps_3d.shape == (n_samples, 3)


# --- Per-view budget ---------------------------------------------------------


def _observed_counts(annotation) -> torch.Tensor:
    """How many emitted points each view actually observes."""
    return (annotation.kps_2d_scores > 0).sum(dim=1)


def test_per_view_budget_gives_every_view_its_own_coverage():
    per_view = 10
    annotation = _run_stage(n_kp_samples_per_view=per_view)

    assert bool((_observed_counts(annotation) >= per_view).all())


def test_per_view_budget_applies_to_the_uniform_sampler_too():
    per_view = 10
    annotation = _run_stage(
        n_kp_samples_per_view=per_view, sampler="uniform"
    )

    assert bool((_observed_counts(annotation) >= per_view).all())
    assert annotation.kps_2d_xy.shape[1] <= per_view * N_VIEWS



def test_stage_emits_the_annotation_contract():
    per_view = 10
    annotation = _run_stage(n_kp_samples_per_view=per_view)

    n_samples = annotation.kps_2d_xy.shape[1]
    assert per_view <= n_samples <= per_view * N_VIEWS
    assert annotation.kps_2d_xy.shape == (N_VIEWS, n_samples, 2)
    assert annotation.kps_2d_scores.shape == (N_VIEWS, n_samples)
    assert annotation.kps_3d.shape == (n_samples, 3)
    assert annotation.view_ids == [f"view_{i}" for i in range(N_VIEWS)]


def test_every_view_keeps_its_own_budget_in_the_union():
    per_view = 10

    for sampler in ("farthest_point", "uniform"):
        annotation = _run_stage(n_kp_samples_per_view=per_view, sampler=sampler)
        observed = (annotation.kps_2d_scores > 0).sum(dim=1)
        assert bool((observed >= per_view).all()), sampler
