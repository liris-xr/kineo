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
import torch

from kineo.annotations.global_time_reference import (
    GlobalTimeReferenceAnnotation,
    GlobalTimeReferenceAnnotations,
    GlobalTimeReferenceAnnotationsMetadata,
)
from kineo.datasets import annotations_io
from kineo.datasets.keypoints_sequence_dataset import (
    KeypointsSequence,
    KeypointsSequenceDataset,
    ViewInput,
)
from kineo.io.audio_loader import VideoAudioLoader
from kineo.io.frame_sequence_loader import VideoLoader

VIDEO_FPS = 60000 / 1001


def build_global_time_reference(
    annotated_start_frames: dict[str, int],
    n_frames: int,
    reference_view: str,
) -> GlobalTimeReferenceAnnotations:
    annotated_frames = torch.arange(n_frames)

    return GlobalTimeReferenceAnnotations(
        metadata=GlobalTimeReferenceAnnotationsMetadata(),
        annotations=[
            GlobalTimeReferenceAnnotation(
                timestamps=(annotated_start_frames[reference_view] + annotated_frames)
                / VIDEO_FPS,
                closest_local_frame_idx={
                    view_id: start_frame + annotated_frames
                    for view_id, start_frame in annotated_start_frames.items()
                },
            )
        ],
    )


class AISTPPSequenceDataset(KeypointsSequenceDataset):
    """AIST++ sequences read from the raw, unsynchronized AIST videos."""

    def __init__(
        self,
        dataset_filepath: str,
        device: torch.device = torch.device("cpu"),
    ):
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
            video_path = os.path.join(self.dataset_dirpath, view_info["video_path"])
            views_inputs.append(
                ViewInput(
                    view_id=view_id,
                    frame_loader=VideoLoader(
                        video_path=video_path,
                        device=self.device,
                    ),
                    audio_loader=VideoAudioLoader(
                        video_path=video_path,
                        device=self.device,
                    ),
                )
            )

        annotations = annotations_io.load_sequence_annotations(
            self.dataset_dirpath, sequence_data
        )
        annotations["global_time_reference"] = build_global_time_reference(
            annotated_start_frames={
                view_id: view_info["annotated_start_frame"]
                for view_id, view_info in sequence_data["views"].items()
            },
            n_frames=sequence_data["n_frames"],
            reference_view=next(iter(sequence_data["views"])),
        )

        return KeypointsSequence(
            sequence_name=sequence_data["sequence_name"],
            views_inputs=views_inputs,
            annotations=annotations,
        )
