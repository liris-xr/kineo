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

from kineo.datasets.aistpp import aistpp_dataset
from kineo.datasets.aistpp.aistpp_dataset import VIDEO_FPS


def test_annotated_window_starts_where_the_reference_recording_cuts_it():
    annotation = aistpp_dataset.build_global_time_reference(
        annotated_start_frames={"c01": 775, "c02": 784},
        n_frames=575,
        reference_view="c01",
    ).first_or_default()

    torch.testing.assert_close(
        annotation.timestamps, (775 + torch.arange(575)) / VIDEO_FPS
    )


def test_reference_view_alone_sets_the_clock():
    annotated_start_frames = {"c01": 775, "c02": 784}

    on_c01 = aistpp_dataset.build_global_time_reference(
        annotated_start_frames, n_frames=575, reference_view="c01"
    ).first_or_default()
    on_c02 = aistpp_dataset.build_global_time_reference(
        annotated_start_frames, n_frames=575, reference_view="c02"
    ).first_or_default()

    torch.testing.assert_close(
        on_c02.timestamps - on_c01.timestamps,
        torch.full((575,), 9 / VIDEO_FPS, dtype=torch.float32),
    )


def test_refined_recordings_are_their_own_annotated_window():
    annotation = aistpp_dataset.build_global_time_reference(
        annotated_start_frames={"c01": 0, "c02": 0},
        n_frames=575,
        reference_view="c01",
    ).first_or_default()

    torch.testing.assert_close(annotation.timestamps, torch.arange(575) / VIDEO_FPS)


def test_annotated_frames_are_placed_in_every_recording():
    annotation = aistpp_dataset.build_global_time_reference(
        annotated_start_frames={"c01": 775, "c02": 784},
        n_frames=575,
        reference_view="c01",
    ).first_or_default()

    torch.testing.assert_close(
        annotation.closest_local_frame_idx["c01"], 775 + torch.arange(575)
    )
    torch.testing.assert_close(
        annotation.closest_local_frame_idx["c02"], 784 + torch.arange(575)
    )
