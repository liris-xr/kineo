# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import glob
import os

import orjson
import torch

from kineo.datasets import annotations_io
from kineo.datasets.keypoints_sequence_dataset import (
    KeypointsSequence,
    KeypointsSequenceDataset,
    ViewInput,
)
from kineo.io.frame_sequence_loader import ImagesLoader


class EgoHumansSequenceDataset(KeypointsSequenceDataset):
    """EgoHumans subsequences read from their extracted exocentric frames.

    Views are genlocked, so a frame index means the same instant in all of
    them and no time offsets are annotated. Frames are JPEG files rather than
    a video, listed per view when the sequence is indexed.
    """

    def __init__(
        self,
        dataset_filepath: str,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            dataset_filepath: Path to the `egohumans_sequences.json` file
                written by the preprocessing.
            device: Device the frames are loaded onto.

        Raises:
            FileNotFoundError: If `dataset_filepath` does not exist.
        """
        if not os.path.exists(dataset_filepath):
            raise FileNotFoundError(f"Dataset file {dataset_filepath} does not exist")

        self.device = device
        self.dataset_filepath = dataset_filepath
        self.dataset_dirpath = os.path.dirname(dataset_filepath)

        with open(dataset_filepath, "rb") as f:
            self.sequences_data = orjson.loads(f.read())

    def __len__(self) -> int:
        return len(self.sequences_data)

    def __getitem__(self, index: int) -> KeypointsSequence:
        sequence_data = self.sequences_data[index]

        views_inputs: list[ViewInput] = []

        for view_id, view_info in sequence_data["views"].items():
            images_dir = os.path.join(self.dataset_dirpath, view_info["images_dir"])
            img_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

            if not img_paths:
                raise FileNotFoundError(f"No images found in {images_dir}")

            views_inputs.append(
                ViewInput(
                    view_id=view_id,
                    frame_loader=ImagesLoader(
                        img_paths=img_paths,
                        frame_timestamps_local=torch.arange(len(img_paths))
                        / view_info["fps"],
                        device=self.device,
                    ),
                    audio_loader=None,
                )
            )

        return KeypointsSequence(
            sequence_name=sequence_data["sequence_name"],
            views_inputs=views_inputs,
            annotations=annotations_io.load_sequence_annotations(
                self.dataset_dirpath, sequence_data
            ),
        )
