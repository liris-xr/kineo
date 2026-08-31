# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import numpy as np
import pytest

from kineo.datasets.chime6 import chime6_preprocess


def test_parse_time():
    assert chime6_preprocess.parse_time("00:00:00.00") == 0.0
    assert chime6_preprocess.parse_time("01:02:03.45") == pytest.approx(3723.45)


def test_concurrency_segments_covers_session_from_zero():
    starts = np.array([10.0, 30.0])
    ends = np.array([20.0, 40.0])

    seg_starts, seg_ends, counts = chime6_preprocess.concurrency_segments(
        starts, ends
    )

    assert seg_starts[0] == 0.0
    assert seg_ends[-1] == 40.0
    np.testing.assert_allclose(seg_starts[1:], seg_ends[:-1])


def test_concurrency_segments_counts_overlap():
    # [0,10) silent, [10,15) P1, [15,20) P1+P2, [20,25) P2, [25,30) silent,
    # [30,40) P3.
    starts = np.array([10.0, 15.0, 30.0])
    ends = np.array([20.0, 25.0, 40.0])

    seg_starts, seg_ends, counts = chime6_preprocess.concurrency_segments(
        starts, ends
    )

    expected = {
        (0.0, 10.0): 0,
        (10.0, 15.0): 1,
        (15.0, 20.0): 2,
        (20.0, 25.0): 1,
        (25.0, 30.0): 0,
        (30.0, 40.0): 1,
    }
    assert dict(zip(zip(seg_starts, seg_ends), counts)) == expected


def test_concurrency_segments_adjacent_utterances_do_not_overlap():
    starts = np.array([10.0, 20.0])
    ends = np.array([20.0, 30.0])

    _, _, counts = chime6_preprocess.concurrency_segments(starts, ends)

    assert counts.max() == 1


def test_composition_rows_sum_to_one():
    starts = np.array([10.0, 15.0, 30.0])
    ends = np.array([20.0, 25.0, 40.0])
    seg_starts, seg_ends, counts = chime6_preprocess.concurrency_segments(
        starts, ends
    )

    comp = chime6_preprocess.composition(
        seg_starts, seg_ends, counts, np.array([0.0, 5.0, 20.0]), 20.0
    )

    np.testing.assert_allclose(comp.sum(axis=1), 1.0)


def test_composition_splits_a_known_window():
    starts = np.array([10.0, 15.0, 30.0])
    ends = np.array([20.0, 25.0, 40.0])
    seg_starts, seg_ends, counts = chime6_preprocess.concurrency_segments(
        starts, ends
    )

    # [0,20): 10 s silent, 5 s single, 5 s overlapping.
    comp = chime6_preprocess.composition(
        seg_starts, seg_ends, counts, np.array([0.0]), 20.0
    )

    np.testing.assert_allclose(comp[0], [0.5, 0.25, 0.25])


def test_select_cell_rejects_mutually_overlapping_candidates():
    # Stride 30 s on 60 s windows: neighbours share 50% of their span, which
    # MAX_MUTUAL_OVERLAP admits, but candidate 1 shares 75% with candidate 0.
    scores = np.array([1.0, 0.9, 0.8, 0.7])
    qualifies = np.ones(4, dtype=bool)

    selected = chime6_preprocess.select_cell(
        scores, qualifies, stride=15.0, length=60.0, count=2
    )

    assert selected == [0, 2]


def test_select_cell_ranks_qualifying_candidates_first():
    scores = np.array([0.9, 0.1])
    qualifies = np.array([False, True])

    selected = chime6_preprocess.select_cell(
        scores, qualifies, stride=60.0, length=60.0, count=2
    )

    assert selected[0] == 1


def test_build_views_keeps_only_units_with_audio():
    positions = {
        "sessions": {
            "S02": {
                "units": {
                    "U01": {"x": 1.0, "y": 2.0, "room": "kitchen", "has_audio": True},
                    "U02": {"x": 3.0, "y": 4.0, "room": "living", "has_audio": False},
                }
            }
        }
    }

    views = chime6_preprocess.build_views(positions, "dev", "S02")

    assert list(views) == ["U01"]
    assert views["U01"] == {
        "audio_path": "CHiME6/audio/dev/S02_U01.CH1.wav",
        "position_m": [1.0, 2.0],
        "room": "kitchen",
    }


def test_build_views_rejects_a_session_without_positions():
    with pytest.raises(KeyError):
        chime6_preprocess.build_views({"sessions": {}}, "dev", "S99")
