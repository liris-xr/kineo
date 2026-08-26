import torch

from kineo.geometry.camera import (
    camera_centers_from_extrinsics,
    positive_depth_mask,
    project_points_from_camera_to_image,
    transform_points_from_world_to_camera,
)
from kineo.geometry.triangulation import (
    triangulate_points,
    triangulate_points_in_chunks,
    triangulation_quality_mask,
)

FOCAL = 500.0
PRINCIPAL_POINT = (100.0, 50.0)


def _rig(centers: list[tuple[float, float, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Intrinsics and extrinsics for axis-aligned cameras at ``centers``."""
    K = torch.tensor(
        [
            [FOCAL, 0.0, PRINCIPAL_POINT[0]],
            [0.0, FOCAL, PRINCIPAL_POINT[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    Ks = K.expand(len(centers), 3, 3).clone()
    Rts = torch.stack(
        [
            torch.cat([torch.eye(3), -torch.tensor(c).reshape(3, 1)], dim=1)
            for c in centers
        ]
    )
    return Ks, Rts


def _project(points_3d: torch.Tensor, Ks: torch.Tensor, Rts: torch.Tensor) -> torch.Tensor:
    points_cam = transform_points_from_world_to_camera(points_3d, Rts)
    Ds = torch.zeros(Ks.shape[0], 5)
    points_2d, _ = project_points_from_camera_to_image(
        points_cam, Ks, Ds, "brown_conrady"
    )
    return points_2d


def _quality_mask(points_3d, points_2d, Ks, Rts, **kwargs):
    return triangulation_quality_mask(
        points_3d=points_3d,
        points_2d=points_2d,
        Ks=Ks,
        Rts=Rts,
        Ds=torch.zeros(Ks.shape[0], 5),
        distortion_model="brown_conrady",
        observations_mask=torch.ones(
            Ks.shape[0], points_3d.shape[0], dtype=torch.bool
        ),
        **kwargs,
    )


def test_camera_centers_invert_the_extrinsics():
    centers = [(0.0, 0.0, -5.0), (1.0, 2.0, -5.0)]
    _, Rts = _rig(centers)

    assert torch.allclose(
        camera_centers_from_extrinsics(Rts), torch.tensor(centers)
    )


def test_positive_depth_mask_is_per_view():
    _, Rts = _rig([(0.0, 0.0, -5.0), (0.0, 0.0, 5.0)])
    # Between the two cameras, so in front of the first and behind the second.
    points_3d = torch.tensor([[0.0, 0.0, 0.0]])

    assert torch.equal(
        positive_depth_mask(points_3d, Rts), torch.tensor([[True], [False]])
    )


def test_chunked_triangulation_matches_the_unchunked_result():
    Ks, Rts = _rig([(0.0, 0.0, -5.0), (2.0, 0.0, -5.0), (-2.0, 1.0, -5.0)])
    generator = torch.Generator().manual_seed(4)
    points_3d = torch.rand(37, 3, generator=generator) * 2.0
    points_2d = _project(points_3d, Ks, Rts)
    Ps = torch.einsum("cij,cjk->cik", Ks, Rts)

    reference = triangulate_points(Ps, points_2d)

    # 3 views -> 3 pairs -> 16*3*4 bytes per point, so these budgets straddle
    # the single-chunk case, several chunks, and slicing turned off.
    for max_chunk_bytes in (1, 192, 400, 1 << 20, 0):
        chunked = triangulate_points_in_chunks(
            Ps, points_2d, max_chunk_bytes=max_chunk_bytes
        )
        assert torch.allclose(chunked, reference, atol=1e-5), max_chunk_bytes


def test_quality_mask_keeps_a_well_conditioned_point():
    Ks, Rts = _rig([(0.0, 0.0, -5.0), (2.0, 0.0, -5.0), (-2.0, 0.0, -5.0)])
    points_3d = torch.tensor([[0.0, 0.0, 0.0]])
    points_2d = _project(points_3d, Ks, Rts)

    mask = _quality_mask(
        points_3d,
        points_2d,
        Ks,
        Rts,
        min_parallax_deg=5.0,
        max_reproj_error=0.005,
        reject_negative_depth=True,
    )

    assert bool(mask.all())


def test_quality_mask_rejects_only_the_view_a_point_is_behind():
    Ks, Rts = _rig([(0.0, 0.0, -5.0), (0.0, 0.0, 5.0)])
    points_3d = torch.tensor([[0.0, 0.0, 0.0]])
    points_2d = _project(points_3d, Ks, Rts)

    mask = _quality_mask(points_3d, points_2d, Ks, Rts, reject_negative_depth=True)

    assert torch.equal(mask, torch.tensor([[True], [False]]))


def test_quality_mask_rejects_an_observation_that_does_not_reproject():
    Ks, Rts = _rig([(0.0, 0.0, -5.0), (2.0, 0.0, -5.0), (-2.0, 0.0, -5.0)])
    points_3d = torch.tensor([[0.0, 0.0, 0.0]])
    points_2d = _project(points_3d, Ks, Rts)
    points_2d[1] += 100.0

    mask = _quality_mask(points_3d, points_2d, Ks, Rts, max_reproj_error=0.005)

    assert torch.equal(mask, torch.tensor([[True], [False], [True]]))


def test_quality_mask_rejects_a_point_with_too_little_parallax():
    # Two nearly coincident cameras looking at a distant point.
    Ks, Rts = _rig([(0.0, 0.0, -5.0), (0.1, 0.0, -5.0)])
    points_3d = torch.tensor([[0.0, 0.0, 500.0]])
    points_2d = _project(points_3d, Ks, Rts)

    mask = _quality_mask(points_3d, points_2d, Ks, Rts, min_parallax_deg=5.0)

    assert not bool(mask.any())


def test_parallax_is_measured_over_the_views_that_survive_the_other_gates():
    # Two close cameras plus a distant one. The wide baseline to the third view
    # is the only thing giving this point a workable angle.
    Ks, Rts = _rig([(0.0, 0.0, -5.0), (0.1, 0.0, -5.0), (10.0, 0.0, -5.0)])
    points_3d = torch.tensor([[0.0, 0.0, 5.0]])
    points_2d = _project(points_3d, Ks, Rts)

    kept = _quality_mask(points_3d, points_2d, Ks, Rts, min_parallax_deg=5.0)
    assert bool(kept.all())

    # Corrupting the wide-baseline observation must cost the point its angle,
    # not just that one view.
    points_2d[2] += 100.0
    dropped = _quality_mask(
        points_3d,
        points_2d,
        Ks,
        Rts,
        min_parallax_deg=5.0,
        max_reproj_error=0.005,
    )
    assert not bool(dropped.any())


def test_triangulation_is_independent_of_how_points_are_grouped():
    # mvs_triangulation folds (subjects, keypoints) into one point axis; that
    # is only sound because each point is triangulated on its own.
    Ks, Rts = _rig([(0.0, 0.0, -5.0), (2.0, 0.0, -5.0), (-2.0, 1.0, -5.0)])
    generator = torch.Generator().manual_seed(9)
    n_frames, n_subjects, n_keypoints = 4, 3, 5
    points_3d = torch.rand(
        n_frames, n_subjects, n_keypoints, 3, generator=generator
    ) * 2.0
    Ps = torch.einsum("cij,cjk->cik", Ks, Rts)

    points_2d = torch.stack(
        [
            torch.stack([_project(points_3d[f, s], Ks, Rts) for s in range(n_subjects)])
            for f in range(n_frames)
        ]
    )  # (F, S, C, K, 2)
    points_2d = points_2d.permute(0, 2, 1, 3, 4)  # (F, C, S, K, 2)

    per_subject = torch.stack(
        [triangulate_points(Ps, points_2d[:, :, s]) for s in range(n_subjects)],
        dim=1,
    )
    folded = triangulate_points(
        Ps, points_2d.reshape(n_frames, len(Rts), n_subjects * n_keypoints, 2)
    ).reshape(n_frames, n_subjects, n_keypoints, 3)

    assert torch.allclose(folded, per_subject, atol=1e-4)
    assert torch.allclose(folded, points_3d, atol=1e-4)


def test_parallax_broadcasts_camera_centers_over_a_frame_batch():
    _, Rts = _rig([(0.0, 0.0, -5.0), (2.0, 0.0, -5.0)])
    from kineo.geometry.triangulation import triangulation_parallax_angles

    points_3d = torch.rand(4, 7, 3) * 2.0
    angles = triangulation_parallax_angles(
        points_3d, camera_centers_from_extrinsics(Rts)
    )

    assert angles.shape == (4, 7)


def test_byte_budget_shrinks_the_chunk_as_views_are_added():
    from kineo.geometry.triangulation import triangulate_points_in_chunks

    budget = 1 << 20
    generator = torch.Generator().manual_seed(2)

    counts = []
    for n_views in (3, 6):
        centers = [(float(i), 0.0, -5.0) for i in range(n_views)]
        Ks, Rts = _rig(centers)
        points_3d = torch.rand(64, 3, generator=generator) * 2.0
        points_2d = _project(points_3d, Ks, Rts)
        Ps = torch.einsum("cij,cjk->cik", Ks, Rts)

        n_pairs = n_views * (n_views - 1) // 2
        counts.append(budget // (16 * n_pairs * points_2d.element_size()))

        # Still correct at either width.
        assert torch.allclose(
            triangulate_points_in_chunks(Ps, points_2d, max_chunk_bytes=budget),
            triangulate_points(Ps, points_2d),
            atol=1e-4,
        )

    # 6 views has 5x the pairs of 3, so the same budget buys ~5x fewer points.
    assert counts[0] > counts[1]
