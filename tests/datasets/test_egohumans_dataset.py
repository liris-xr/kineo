# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import os

import orjson
import pytest
import torch
from PIL import Image

from kineo.datasets.egohumans.egohumans_dataset import EgoHumansSequenceDataset

FPS = 20


def write_sequences_file(dataset_dir, views_frames: dict[str, int]) -> str:
    """Writes a one-sequence dataset with the given views and frame counts."""
    for view_id, n_frames in views_frames.items():
        images_dir = os.path.join(dataset_dir, "001_tagging", "exo", view_id, "images")
        os.makedirs(images_dir)

        for frame_idx in range(n_frames):
            Image.new("RGB", (4, 4)).save(
                os.path.join(images_dir, f"{frame_idx:05d}.jpg")
            )

    sequences_file = os.path.join(dataset_dir, "egohumans_sequences.json")

    with open(sequences_file, "wb") as f:
        f.write(
            orjson.dumps(
                [
                    {
                        "sequence_name": "tagging_001",
                        "views": {
                            view_id: {
                                "images_dir": f"001_tagging/exo/{view_id}/images",
                                "fps": FPS,
                            }
                            for view_id in views_frames
                        },
                    }
                ]
            )
        )

    return sequences_file


def test_views_are_read_from_their_image_directories(tmp_path):
    sequences_file = write_sequences_file(str(tmp_path), {"cam01": 3, "cam02": 3})

    sequence = EgoHumansSequenceDataset(sequences_file)[0]

    assert sequence["sequence_name"] == "tagging_001"
    assert [view["view_id"] for view in sequence["views_inputs"]] == [
        "cam01",
        "cam02",
    ]
    assert [len(view["frame_loader"]) for view in sequence["views_inputs"]] == [3, 3]


def test_frames_are_ordered_and_timed_by_the_declared_fps(tmp_path):
    sequences_file = write_sequences_file(str(tmp_path), {"cam01": 3})

    frame_loader = EgoHumansSequenceDataset(sequences_file)[0]["views_inputs"][0][
        "frame_loader"
    ]

    assert [os.path.basename(path) for path in frame_loader.img_paths] == [
        "00000.jpg",
        "00001.jpg",
        "00002.jpg",
    ]
    torch.testing.assert_close(
        frame_loader.frame_timestamps_local, torch.arange(3) / FPS
    )


def test_a_view_whose_images_are_missing_is_reported(tmp_path):
    sequences_file = write_sequences_file(str(tmp_path), {"cam01": 0})

    with pytest.raises(FileNotFoundError, match="No images found"):
        EgoHumansSequenceDataset(sequences_file)[0]


def test_an_unknown_dataset_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        EgoHumansSequenceDataset(str(tmp_path / "missing.json"))
