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
from typing import Sequence

from kineo.io.frame_sequence_loader import FrameSequenceLoader


class FrameRangeLoader(FrameSequenceLoader):
    def __init__(
        self,
        loader: FrameSequenceLoader,
        start_frame: int,
        n_frames: int | None = None,
    ):
        super().__init__(loader.device)

        if start_frame < 0:
            raise ValueError(f"start_frame must be positive, got {start_frame}")
        if start_frame >= loader.n_frames:
            raise ValueError(
                f"start_frame {start_frame} is past the end of a sequence of "
                f"{loader.n_frames} frames"
            )

        available = loader.n_frames - start_frame

        self._loader = loader
        self._start_frame = start_frame
        self._n_frames = (
            available if n_frames is None else min(n_frames, available)
        )

    @staticmethod
    def frames_for_duration(
        loader: FrameSequenceLoader, duration_seconds: float
    ) -> int:
        timestamps = loader.frame_timestamps_local
        return int((timestamps - timestamps[0] < duration_seconds).sum().item())

    def _source_index(self, frame_index: int) -> int:
        if frame_index < 0:
            frame_index += self._n_frames
        if not 0 <= frame_index < self._n_frames:
            raise IndexError(
                f"Frame {frame_index} is outside a window of "
                f"{self._n_frames} frames"
            )
        return self._start_frame + frame_index

    def load_frame_at(self, frame_index: int) -> torch.Tensor:
        return self._loader.load_frame_at(self._source_index(frame_index))

    def load_frames_at(
        self, frame_indices: Sequence | range | slice | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(frame_indices, slice):
            frame_indices = range(*frame_indices.indices(self._n_frames))
        elif isinstance(frame_indices, torch.Tensor):
            frame_indices = frame_indices.tolist()

        return self._loader.load_frames_at(
            [self._source_index(int(i)) for i in frame_indices]
        )

    @property
    def frame_timestamps_local(self) -> torch.Tensor:
        end = self._start_frame + self._n_frames
        return self._loader.frame_timestamps_local[self._start_frame : end]

    @property
    def resolution_hw(self) -> tuple[int, int]:
        return self._loader.resolution_hw

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def close(self):
        pass
