import pytest

from kineo.datasets.egohumans.hsfm_eval_views import load_hsfm_view_selection

ALL_VIEWS = [f"cam{i:02d}" for i in range(1, 21)]


def test_selection_matches_the_paper_four_view_setting():
    assert load_hsfm_view_selection("tennis_001", 4, ALL_VIEWS) == [
        "cam04",
        "cam09",
        "cam12",
        "cam20",
    ]


def test_all_sentinel_resolves_to_every_available_view():
    available = ["cam01", "cam02", "cam03"]
    assert load_hsfm_view_selection("tagging_002", 8, available) == available


def test_sequence_override_wins_over_activity_selection():
    # badminton_049 lacks cam01, so the paper's 4-view set is overridden.
    available = [f"cam{i:02d}" for i in range(2, 16) if i != 8]
    assert load_hsfm_view_selection("badminton_049", 4, available) == [
        "cam02",
        "cam03",
        "cam05",
        "cam07",
    ]
    # An unaffected badminton sequence keeps the paper's set.
    assert load_hsfm_view_selection("badminton_001", 4, ALL_VIEWS) == [
        "cam01",
        "cam02",
        "cam05",
        "cam07",
    ]


def test_missing_camera_raises():
    with pytest.raises(ValueError, match="cam20"):
        load_hsfm_view_selection("tennis_001", 4, ["cam04", "cam09", "cam12"])


def test_unknown_activity_raises():
    with pytest.raises(ValueError, match="curling"):
        load_hsfm_view_selection("curling_001", 4, ALL_VIEWS)


def test_unknown_view_count_raises():
    with pytest.raises(ValueError, match="3 views"):
        load_hsfm_view_selection("tennis_001", 3, ALL_VIEWS)
