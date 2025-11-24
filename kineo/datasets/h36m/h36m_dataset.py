# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from typing import Literal
import orjson
import os
import torch

from kineo.io.frame_sequence_loader import VideoLoader
from kineo.datasets.keypoints_sequence_dataset import (
    KeypointsSequenceDataset,
    KeypointsSequence,
    KeypointsSequenceAnnotations,
    ViewInput,
)


class H36MSequenceDataset(KeypointsSequenceDataset):
    def __init__(
        self,
        dataset_filepath: str,
        split: Literal["train", "val"],
        device: torch.device = torch.device("cpu"),
    ):
        if not os.path.exists(dataset_filepath):
            raise FileNotFoundError(f"Dataset file {dataset_filepath} does not exist")

        self.device = device
        self.dataset_filepath = dataset_filepath
        self.dataset_dirpath = os.path.dirname(dataset_filepath)
        self.split = split
        self.sequences = self._load_sequences(dataset_filepath)

    def _load_sequences(self, dataset_filepath: str) -> list[KeypointsSequence]:
        with open(dataset_filepath, "rb") as f:
            sequences_data = orjson.loads(f.read())

        # Keep only the sequences with the correct split
        sequences_data = [
            sequence for sequence in sequences_data if sequence["split"] == self.split
        ]

        sequences = []

        for sequence_data in sequences_data:
            subaction_name = sequence_data["subaction_name"]
            subject_name = sequence_data["subject_name"]
            action_name = sequence_data["action_name"]

            views_inputs: list[ViewInput] = []

            # Parse the sequence views selected frames
            for view_name, view_info in sequence_data["views"].items():
                video_path = os.path.join(self.dataset_dirpath, view_info["video_path"])
                selected_frames = view_info["selected_frames"]
                if selected_frames["type"] == "range":
                    selected_frames = torch.arange(
                        selected_frames["start"],
                        selected_frames["stop"],
                        selected_frames["step"],
                    )
                else:
                    raise NotImplementedError

                views_inputs.append(
                    {
                        "name": view_name,
                        "frame_loader": VideoLoader(
                            video_path=video_path,
                            selected_frames=selected_frames,
                            device=self.device,
                        ),
                    }
                )

            if "annotations" in sequence_data:
                annotations_filepath = os.path.join(
                    self.dataset_dirpath, sequence_data["annotations"]
                )
                annotations = KeypointsSequenceAnnotations.load_from_file(
                    annotations_filepath
                )
            else:
                annotations = None

            sequence = KeypointsSequence(
                sequence_name=f"{subject_name}_{subaction_name}",
                views_inputs=views_inputs,
                annotations=annotations,
                action_name=action_name,
                subject_name=subject_name,
                subaction_name=subaction_name,
            )

            sequences.append(sequence)

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> KeypointsSequence:
        return self.sequences[index]


if __name__ == "__main__":
    dataset = H36MSequenceDataset(
        dataset_filepath="/mnt/hdd_storage/Charles_JAVERLIAT/Datasets/H3.6M/raw/h36m_protocol1_sequences.json",
        split="val",
        device=torch.device("cuda"),
    )

    print(dataset[0])
