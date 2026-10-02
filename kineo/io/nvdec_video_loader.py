# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from typing import Sequence

import PyNvVideoCodec as nvc
import torch


class NvdecVideoLoader:
    """Loader replaying a video as a live camera, decoded with NVDEC."""

    def __init__(
        self,
        video_path: str,
        device: torch.device,
        frame_indices: Sequence[int] | None = None,
        buffer_size: int = 8,
    ):
        if device.type != "cuda":
            raise ValueError(
                f"The video engine needs a CUDA device, got {device}"
            )

        self.device = device
        self._decoder = nvc.CreateThreadedDecoder(
            encSource=video_path,
            bufferSize=buffer_size,
            gpuid=device.index or 0,
            useDeviceMemory=True,
            outputColorType=nvc.OutputColorType.RGBP,
        )
        metadata = self._decoder.get_stream_metadata()
        self._resolution_hw = (int(metadata.height), int(metadata.width))
        self._fps = float(metadata.average_fps)

        if frame_indices is None:
            frame_indices = range(int(metadata.num_frames))
        self._frame_indices = [int(i) for i in frame_indices]

        if any(
            b <= a for a, b in zip(self._frame_indices, self._frame_indices[1:])
        ):
            raise ValueError("frame_indices must be strictly increasing")

        self._served = 0
        self._decoded = 0
        self._batch: list = []
        self._batch_size = buffer_size

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def resolution_hw(self) -> tuple[int, int]:
        return self._resolution_hw

    @property
    def n_frames(self) -> int:
        return len(self._frame_indices)

    def is_opened(self) -> bool:
        return self._served < len(self._frame_indices)

    def _next_decoded(self) -> torch.Tensor | None:
        if not self._batch:
            self._batch = list(self._decoder.get_batch_frames(self._batch_size))
            if not self._batch:
                return None
        frame = self._batch.pop(0)
        self._decoded += 1
        # The decoder recycles its buffers
        return torch.from_dlpack(frame).clone()

    def read_frame(self) -> torch.Tensor | None:
        if not self.is_opened():
            return None

        wanted = self._frame_indices[self._served]
        while self._decoded <= wanted:
            frame = self._next_decoded()
            if frame is None:
                return None

        self._served += 1
        return frame

    def close(self):
        self._batch = []

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        frame = self.read_frame()
        if frame is None:
            raise StopIteration
        return frame
