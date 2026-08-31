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

# Every AIST video, raw and refined alike, is constant frame rate at 60000/1001.
VIDEO_FPS = 60000 / 1001


def build_global_time_reference(
    annotated_start_frames: dict[str, int],
    n_frames: int,
    reference_view: str,
) -> GlobalTimeReferenceAnnotations:
    """Instants the annotated frames happen at, on the pipeline's own clock.

    Annotations are numbered from the start of the annotated window, recordings
    from the start of the recording, and a pipeline resamples onto the clock of
    the view it pins to a zero offset. So `reference_view` must be that view.

    Args:
        annotated_start_frames: Frame of each recording the annotated window
            starts at. Zero for a recording that is the annotated window.
        n_frames: Length of the annotated window, in frames.
        reference_view: View whose clock the timestamps are expressed in.

    Returns:
        A single time reference, holding one timestamp per annotated frame.
    """
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
    """AIST++ sequences read from the raw, unsynchronized AIST videos.

    Views are whole raw recordings, each starting at its own moment, and their
    `cameras_temporal` annotations carry the ground-truth offsets between them.
    The annotations cover a window of those recordings rather than all of them,
    so the sequence also carries the `global_time_reference` saying where that
    window sits. Every view exposes an audio loader: the raw recordings kept
    the camera microphone, which is what a temporal calibration stage listens
    to.

    Sequences are built when indexed rather than upfront. A split holds a few
    thousand videos, and opening every reader at once would both exhaust file
    handles and pay for probing frame timestamps of videos that are never read.
    """

    def __init__(
        self,
        dataset_filepath: str,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            dataset_filepath: Path to an `aistpp_<split>_sequences.json` file
                written by the preprocessing.
            device: Device the frames and audio are loaded onto.

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
