import pytest
import torch

from kineo.annotations import (
    BBox2DAnnotation,
    BBox2DAnnotations,
    BBox2DAnnotationsMetadata,
    Keypoints2DAnnotation,
    Keypoints2DAnnotations,
    Keypoints2DAnnotationsMetadata,
    KeypointsFormat,
)
from kineo.pipeline import per_view_cache

TOY_FORMAT = KeypointsFormat(
    name="toy_2",
    n_keypoints=2,
    keypoints_names=["head", "foot"],
    keypoints_connectivity=[(0, 1)],
)
KEYPOINTS_METADATA = Keypoints2DAnnotationsMetadata(formats=[TOY_FORMAT])


def make_views(*view_ids):
    return [{"view_id": view_id} for view_id in view_ids]


def make_keypoints(view_ids, frame_idxs=(0,)):
    return Keypoints2DAnnotations(
        metadata=KEYPOINTS_METADATA,
        annotations=[
            Keypoints2DAnnotation(
                view_id=view_id,
                frame_idx=frame_idx,
                subject_id="subject",
                xy=torch.zeros((2, 2), dtype=torch.float32),
                scores=torch.ones(2, dtype=torch.float32),
                format=TOY_FORMAT.name,
            )
            for view_id in view_ids
            for frame_idx in frame_idxs
        ],
    )


def make_bboxes(view_ids):
    return BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(),
        annotations=[
            BBox2DAnnotation(
                view_id=view_id,
                frame_idx=0,
                subject_id="subject",
                xyxy=torch.zeros(4, dtype=torch.float32),
                score=1.0,
                category_id=0,
            )
            for view_id in view_ids
        ],
    )


class RecordingInference:
    """Stands in for a stage's inference, remembering what it was asked for."""

    def __init__(self, builders):
        self.builders = builders
        self.calls = []

    def __call__(self, views):
        view_ids = [view["view_id"] for view in views]
        self.calls.append(view_ids)
        return {key: builder(view_ids) for key, builder in self.builders.items()}


@pytest.fixture
def template(tmp_path):
    return str(tmp_path / "{sequence_name}" / "{annotation_key}" / "{view_id}.pkl")


@pytest.fixture
def keypoints_specs():
    return {
        "keypoints_2d": per_view_cache.PerViewCacheSpec(
            annotations_cls=Keypoints2DAnnotations,
            metadata=KEYPOINTS_METADATA,
        )
    }


def run(views, specs, infer, template, use_cache=True, sequence_name="seq"):
    return per_view_cache.load_or_infer_per_view(
        views=views,
        specs=specs,
        infer_missing=infer,
        sequence_name=sequence_name,
        cache_output_path_template=template,
        use_cache=use_cache,
    )


def test_cold_cache_infers_every_view(template, keypoints_specs, tmp_path):
    infer = RecordingInference({"keypoints_2d": make_keypoints})

    result = run(make_views("cam1", "cam2"), keypoints_specs, infer, template)

    assert infer.calls == [["cam1", "cam2"]]
    assert result["keypoints_2d"].view_ids == ["cam1", "cam2"]
    assert (tmp_path / "seq" / "keypoints_2d" / "cam1.pkl").exists()
    assert (tmp_path / "seq" / "keypoints_2d" / "cam2.pkl").exists()


def test_two_views_then_all_views_only_infers_the_new_ones(template, keypoints_specs):
    infer = RecordingInference({"keypoints_2d": make_keypoints})

    run(make_views("cam1", "cam2"), keypoints_specs, infer, template)
    result = run(make_views("cam1", "cam2", "cam3"), keypoints_specs, infer, template)

    assert infer.calls == [["cam1", "cam2"], ["cam3"]]
    assert result["keypoints_2d"].view_ids == ["cam1", "cam2", "cam3"]


def test_all_views_then_two_views_infers_nothing(template, keypoints_specs):
    infer = RecordingInference({"keypoints_2d": make_keypoints})

    run(make_views("cam1", "cam2", "cam3"), keypoints_specs, infer, template)
    result = run(make_views("cam1", "cam3"), keypoints_specs, infer, template)

    assert infer.calls == [["cam1", "cam2", "cam3"]]
    assert result["keypoints_2d"].view_ids == ["cam1", "cam3"]


def test_result_follows_the_requested_view_order(template, keypoints_specs):
    infer = RecordingInference({"keypoints_2d": make_keypoints})

    run(make_views("cam2"), keypoints_specs, infer, template)
    result = run(make_views("cam1", "cam2", "cam3"), keypoints_specs, infer, template)

    ordered = [annotation.view_id for annotation in result["keypoints_2d"]]
    assert ordered == ["cam1", "cam2", "cam3"]


def test_partial_hit_keeps_every_annotation_of_a_view(template, keypoints_specs):
    def builder(view_ids):
        return make_keypoints(view_ids, frame_idxs=(0, 1, 2))

    infer = RecordingInference({"keypoints_2d": builder})

    run(make_views("cam1"), keypoints_specs, infer, template)
    result = run(make_views("cam1", "cam2"), keypoints_specs, infer, template)

    assert len(result["keypoints_2d"]) == 6


def test_a_view_yielding_nothing_is_not_reinferred(template, keypoints_specs):
    def builder(view_ids):
        return make_keypoints([view_id for view_id in view_ids if view_id != "cam2"])

    infer = RecordingInference({"keypoints_2d": builder})

    run(make_views("cam1", "cam2"), keypoints_specs, infer, template)
    result = run(make_views("cam1", "cam2"), keypoints_specs, infer, template)

    assert infer.calls == [["cam1", "cam2"]]
    assert result["keypoints_2d"].view_ids == ["cam1"]


def test_a_view_is_a_hit_only_when_every_key_has_a_file(template, tmp_path):
    specs = {
        "keypoints_2d": per_view_cache.PerViewCacheSpec(
            annotations_cls=Keypoints2DAnnotations, metadata=KEYPOINTS_METADATA
        ),
        "bboxes_2d": per_view_cache.PerViewCacheSpec(
            annotations_cls=BBox2DAnnotations, metadata=BBox2DAnnotationsMetadata()
        ),
    }
    infer = RecordingInference(
        {"keypoints_2d": make_keypoints, "bboxes_2d": make_bboxes}
    )

    run(make_views("cam1"), specs, infer, template)
    (tmp_path / "seq" / "bboxes_2d" / "cam1.pkl").unlink()
    run(make_views("cam1"), specs, infer, template)

    assert infer.calls == [["cam1"], ["cam1"]]


def test_disabled_cache_writes_nothing(template, keypoints_specs, tmp_path):
    infer = RecordingInference({"keypoints_2d": make_keypoints})

    run(make_views("cam1"), keypoints_specs, infer, template, use_cache=False)

    assert not (tmp_path / "seq").exists()


def test_metadata_mismatch_is_rejected(template, keypoints_specs):
    infer = RecordingInference({"keypoints_2d": make_keypoints})
    run(make_views("cam1"), keypoints_specs, infer, template)

    other_format = KeypointsFormat(
        name="toy_3",
        n_keypoints=3,
        keypoints_names=["head", "hip", "foot"],
        keypoints_connectivity=[(0, 1), (1, 2)],
    )
    other_specs = {
        "keypoints_2d": per_view_cache.PerViewCacheSpec(
            annotations_cls=Keypoints2DAnnotations,
            metadata=Keypoints2DAnnotationsMetadata(formats=[other_format]),
        )
    }

    with pytest.raises(ValueError, match="incompatible configuration"):
        run(make_views("cam1"), other_specs, infer, template)


def test_template_without_view_id_is_rejected(keypoints_specs, tmp_path):
    infer = RecordingInference({"keypoints_2d": make_keypoints})
    template = str(tmp_path / "{sequence_name}" / "{annotation_key}.pkl")

    with pytest.raises(ValueError, match="view_id"):
        run(make_views("cam1"), keypoints_specs, infer, template)


def test_logs_a_full_hit(template, keypoints_specs, capsys):
    infer = RecordingInference({"keypoints_2d": make_keypoints})
    run(make_views("cam1", "cam2"), keypoints_specs, infer, template)
    capsys.readouterr()

    run(make_views("cam1", "cam2"), keypoints_specs, infer, template)

    assert "[cache] seq keypoints_2d: 2/2 views hit" in capsys.readouterr().out


def test_logs_the_views_it_has_to_infer(template, keypoints_specs, capsys):
    infer = RecordingInference({"keypoints_2d": make_keypoints})
    run(make_views("cam1"), keypoints_specs, infer, template)
    capsys.readouterr()

    run(make_views("cam1", "cam2"), keypoints_specs, infer, template)

    assert (
        "[cache] seq keypoints_2d: 1/2 views hit, inferring cam2"
        in capsys.readouterr().out
    )


def test_logs_that_the_cache_is_disabled(template, keypoints_specs, capsys):
    infer = RecordingInference({"keypoints_2d": make_keypoints})

    run(make_views("cam1"), keypoints_specs, infer, template, use_cache=False)

    assert "[cache] seq keypoints_2d: disabled" in capsys.readouterr().out
