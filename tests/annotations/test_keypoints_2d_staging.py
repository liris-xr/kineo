import torch

from kineo.annotations.keypoints_2d import (
    Keypoints2DAnnotation,
    Keypoints2DAnnotations,
    Keypoints2DAnnotationsMetadata,
    stage_keypoints_2d,
)
from kineo.annotations.keypoints_format import KeypointsFormat

N_FRAMES, N_VIEWS, N_SUBJECTS, N_KEYPOINTS = 6, 3, 2, 5
VIEW_ID_TO_IDX = {f"view_{i}": i for i in range(N_VIEWS)}
SUBJECT_ID_TO_IDX = {f"subject_{i}": i for i in range(N_SUBJECTS)}


def _annotations(skip: set[tuple[int, int, int]] | None = None):
    skip = skip or set()
    generator = torch.Generator().manual_seed(3)
    fmt = KeypointsFormat(
        name="synthetic",
        n_keypoints=N_KEYPOINTS,
        keypoints_names=[f"kp_{i}" for i in range(N_KEYPOINTS)],
        keypoints_connectivity=[],
    )
    annots = [
        Keypoints2DAnnotation(
            view_id=f"view_{v}",
            frame_idx=f,
            subject_id=f"subject_{s}",
            xy=torch.rand(N_KEYPOINTS, 2, generator=generator) * 100.0,
            scores=torch.rand(N_KEYPOINTS, generator=generator),
            format="synthetic",
        )
        for f in range(N_FRAMES)
        for v in range(N_VIEWS)
        for s in range(N_SUBJECTS)
        if (f, v, s) not in skip
    ]
    return Keypoints2DAnnotations(
        metadata=Keypoints2DAnnotationsMetadata(formats=[fmt]), annotations=annots
    )


def _per_annotation(kps_2d, keypoints_indices):
    """The loop the staging helper replaced."""
    xy = torch.zeros(
        N_FRAMES, N_VIEWS, N_SUBJECTS, len(keypoints_indices), 2
    )
    scores = torch.zeros(N_FRAMES, N_VIEWS, N_SUBJECTS, len(keypoints_indices))
    for annot in kps_2d.annotations:
        v = VIEW_ID_TO_IDX[annot.view_id]
        s = SUBJECT_ID_TO_IDX[annot.subject_id]
        xy[annot.frame_idx, v, s] = annot.xy[keypoints_indices]
        scores[annot.frame_idx, v, s] = annot.scores[keypoints_indices]
    return xy, scores


def _stage(kps_2d, keypoints_indices=None):
    return stage_keypoints_2d(
        kps_2d=kps_2d,
        view_id_to_idx=VIEW_ID_TO_IDX,
        subject_id_to_idx=SUBJECT_ID_TO_IDX,
        n_frames=N_FRAMES,
        keypoints_indices=keypoints_indices,
    )


def test_staging_matches_the_per_annotation_loop():
    kps_2d = _annotations()

    xy, scores = _stage(kps_2d)
    xy_ref, scores_ref = _per_annotation(kps_2d, torch.arange(N_KEYPOINTS))

    assert torch.equal(xy, xy_ref)
    assert torch.equal(scores, scores_ref)


def test_staging_honours_the_keypoints_subset_and_its_order():
    kps_2d = _annotations()
    keypoints_indices = torch.tensor([3, 0, 4])

    xy, scores = _stage(kps_2d, keypoints_indices)
    xy_ref, scores_ref = _per_annotation(kps_2d, keypoints_indices)

    assert xy.shape == (N_FRAMES, N_VIEWS, N_SUBJECTS, 3, 2)
    assert torch.equal(xy, xy_ref)
    assert torch.equal(scores, scores_ref)


def test_missing_entries_stay_zero():
    skip = {(0, 0, 0), (2, 1, 1)}
    kps_2d = _annotations(skip)

    xy, scores = _stage(kps_2d)

    for f, v, s in skip:
        assert torch.equal(xy[f, v, s], torch.zeros(N_KEYPOINTS, 2))
        assert torch.equal(scores[f, v, s], torch.zeros(N_KEYPOINTS))
    assert bool((scores.sum(dim=-1) > 0).sum() == N_FRAMES * N_VIEWS * N_SUBJECTS - len(skip))


def test_staging_with_no_annotations_returns_zeros():
    kps_2d = _annotations({(f, v, s) for f in range(N_FRAMES)
                           for v in range(N_VIEWS) for s in range(N_SUBJECTS)})

    xy, scores = _stage(kps_2d)

    assert xy.shape == (N_FRAMES, N_VIEWS, N_SUBJECTS, N_KEYPOINTS, 2)
    assert not bool(xy.any())
    assert not bool(scores.any())
