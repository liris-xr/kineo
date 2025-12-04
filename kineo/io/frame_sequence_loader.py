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
from abc import ABC, abstractmethod
from typing import Sequence
from kineo.io.ffmpeg import get_frames_timing_info
from torchvision.io import read_file, decode_image, ImageReadMode
from concurrent.futures import ThreadPoolExecutor
import cv2
import os

from PIL import Image


class FrameSequenceLoader(ABC):
    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def load_frame_at(self, frame_index: int) -> torch.Tensor:
        pass

    @abstractmethod
    def load_frames_at(
        self, frame_indices: Sequence | range | slice | torch.Tensor
    ) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def frame_timestamps_local(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def resolution_hw(self) -> tuple[int, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_frames(self) -> int:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(
        self, index: int | Sequence | slice | range | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(index, int):
            return self.load_frame_at(index)
        elif isinstance(index, (Sequence | slice | range | torch.Tensor)):
            return self.load_frames_at(index)
        else:
            raise ValueError(
                f"Invalid index type: {type(index)}. Must be an int, a sequence of ints, a slice, a range, or a torch.Tensor."
            )

    def __del__(self):
        self.close()

    def close(self):
        pass


class ImagesLoader(FrameSequenceLoader):
    def __init__(
        self,
        img_paths: list[str],
        frame_timestamps_local: list[float],
        device: torch.device,
        image_read_mode: ImageReadMode = ImageReadMode.RGB,
        apply_exif_orientation: bool = True,
        num_workers: int = 4,
    ):
        super().__init__(device)
        self.img_paths = img_paths
        self._frame_timestamps_local = torch.as_tensor(frame_timestamps_local)
        self.image_read_mode = image_read_mode
        self.apply_exif_orientation = apply_exif_orientation
        self.num_workers = num_workers

        self._resolution_hw = Image.open(self.img_paths[0]).size[::-1]

        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    @property
    def resolution_hw(self) -> tuple[int, int]:
        return self._resolution_hw

    @property
    def n_frames(self) -> int:
        return len(self.img_paths)

    @property
    def frame_timestamps_local(self) -> torch.Tensor:
        return self._frame_timestamps_local

    def _load_single_frame(self, frame_idx: int) -> torch.Tensor:
        img_bytes: torch.Tensor = read_file(self.img_paths[frame_idx])
        img = decode_image(
            img_bytes,
            mode=self.image_read_mode,
            apply_exif_orientation=self.apply_exif_orientation,
        ).to(self.device)
        return img

    def load_frame_at(self, frame_idx: int) -> torch.Tensor:
        return self._load_single_frame(frame_idx)

    def load_frames_at(
        self, frame_indices: Sequence | range | slice | torch.Tensor
    ) -> torch.Tensor:
        frame_indices = torch.as_tensor(frame_indices)
        if len(frame_indices) == 1:
            # Fast path for single frame
            return self.load_frame_at(frame_indices[0]).unsqueeze(0)

        futures = [
            self._executor.submit(self._load_single_frame, idx) for idx in frame_indices
        ]
        frames = [f.result() for f in futures]

        if not all(frame.shape == frames[0].shape for frame in frames):
            raise ValueError(
                f"All frames must have the same shape. Got shapes: {[frame.shape for frame in frames]}"
            )

        return torch.stack(frames)

    def close(self):
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)


class VideoLoader(FrameSequenceLoader):
    def __init__(
        self,
        video_path: str,
        device: torch.device,
        selected_frames: torch.Tensor | Sequence | range | slice | None = None,
        api_preference: int = cv2.CAP_ANY,
    ):
        super().__init__(device)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found")

        self.video_path = video_path
        self.selected_frames = selected_frames
        self.device = device

        if self.selected_frames is not None:
            self.selected_frames = torch.as_tensor(self.selected_frames)

        self._video_reader = cv2.VideoCapture(self.video_path, api_preference)
        self._resolution_hw = (
            int(self._video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )

        self._n_frames = int(self._video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        self._frame_timestamps_local = get_frames_timing_info(self.video_path)[0] / 1e9

        if self.selected_frames is not None:
            self._n_frames = len(self.selected_frames)
            self._frame_timestamps_local = self._frame_timestamps_local[
                self.selected_frames
            ]

        self._current_frame_idx = -1
        self._current_frame: torch.Tensor | None = None

    def close(self):
        # Force delete the video reader to free the file
        if hasattr(self, "_video_reader"):
            self._video_reader.release()

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def resolution_hw(self) -> tuple[int, int]:
        return self._resolution_hw

    @property
    def frame_timestamps_local(self) -> torch.Tensor:
        return self._frame_timestamps_local

    def load_frame_at(self, frame_index: int) -> torch.Tensor:
        if frame_index < 0 or frame_index >= self.n_frames:
            raise IndexError(
                f"Frame index {frame_index} out of range [0, {self.n_frames}["
            )

        if self.selected_frames is not None:
            frame_index = self.selected_frames[frame_index].item()

        if isinstance(frame_index, torch.Tensor):
            frame_index = frame_index.item()

        if self._current_frame_idx == frame_index:
            return self._current_frame.clone()

        if self._current_frame_idx != -1 and frame_index > self._current_frame_idx:
            distance = frame_index - self._current_frame_idx
            if distance > 50: # If many frames ahead, jump
                self._video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = self._video_reader.read()
                if not ret:
                    raise ValueError(
                        f"Failed to read frame at CAP_PROP_POS_FRAMES={int(self._video_reader.get(cv2.CAP_PROP_POS_FRAMES))}"
                    )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame).permute(2, 0, 1)
                frame = frame.to(device=self.device, dtype=torch.uint8)
                self._current_frame = frame
                self._current_frame_idx = frame_index
                return self._current_frame.clone()
            else:
                while self._current_frame_idx < frame_index:
                    ret, frame = self._video_reader.read()
                    if not ret:
                        raise ValueError(
                            f"Failed to read frame at CAP_PROP_POS_FRAMES={int(self._video_reader.get(cv2.CAP_PROP_POS_FRAMES))}"
                        )
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame).permute(2, 0, 1)
                    frame = frame.to(device=self.device, dtype=torch.uint8)
                    self._current_frame = frame
                    self._current_frame_idx += 1
                return self._current_frame.clone()

        # Else, the frame is before the current frame
        self._video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self._video_reader.read()
        if not ret:
            raise ValueError(
                f"Failed to read frame at CAP_PROP_POS_FRAMES={int(self._video_reader.get(cv2.CAP_PROP_POS_FRAMES))}"
            )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        frame = frame.to(device=self.device, dtype=torch.uint8)
        self._current_frame = frame
        self._current_frame_idx = frame_index
        return self._current_frame.clone()

    def load_frames_at(
        self, frame_indices: Sequence | range | slice | torch.Tensor
    ) -> torch.Tensor:
        if isinstance(frame_indices, (range, slice)):
            frame_indices = torch.arange(
                frame_indices.start, frame_indices.stop, frame_indices.step
            )
        elif isinstance(frame_indices, Sequence):
            frame_indices = torch.as_tensor(frame_indices)
        elif not isinstance(frame_indices, torch.Tensor) or frame_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                f"Invalid frame indices type: {type(frame_indices)}. Must be a list, tuple, range, or int32/int64 torch.Tensor."
            )

        if torch.any(frame_indices < 0) or torch.any(frame_indices >= self.n_frames):
            raise IndexError(
                f"Frame indices {frame_indices} out of range [0, {self.n_frames}["
            )

        frames = [self.load_frame_at(frame_idx) for frame_idx in frame_indices]
        frames = torch.stack(frames)
        return frames
