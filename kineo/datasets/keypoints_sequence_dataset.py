# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, TypedDict, Iterator, Iterable

from kineo.io.frame_sequence_loader import FrameSequenceLoader
from kineo.io.audio_loader import AudioLoader


class ViewInput(TypedDict):
    view_id: str
    frame_loader: FrameSequenceLoader
    audio_loader: AudioLoader


class KeypointsSequence(TypedDict):
    sequence_name: str
    views_inputs: list[ViewInput]
    # Keyed by annotation kind, as returned by
    # annotations_io.load_sequence_annotations.
    annotations: dict[str, Any] | None


class KeypointsSequenceDataset(ABC, Iterable[KeypointsSequence]):
    @abstractmethod
    def __getitem__(self, index: int) -> KeypointsSequence:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[KeypointsSequence]:
        def iterator():
            for i in range(len(self)):
                yield self[i]

        return iterator()
