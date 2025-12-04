from typing import Optional

import torch
import cv2


class LiveVideoLoader:
    def __init__(
            self,
            camera_idx: int,
            device: torch.device,
            api_preference: int = cv2.CAP_ANY,
    ):
        self.device = device
        self.camera_idx = camera_idx

        self._video_reader = cv2.VideoCapture(self.camera_idx, api_preference)

        if not self._video_reader.isOpened():
            raise Exception(f"Failed to open video capture for camera {camera_idx}")

        self._resolution_hw = (
            int(self._video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(self._video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        self.fps = self._video_reader.get(cv2.CAP_PROP_FPS)

    def is_opened(self):
        return self._video_reader.isOpened()

    def close(self):
        # Force delete the video reader to free the file
        if hasattr(self, "_video_reader"):
            self._video_reader.release()

    def read_frame(self) -> Optional[torch.Tensor]:
        ret, frame = self._video_reader.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        frame = frame.to(device=self.device, dtype=torch.uint8)
        return frame

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.read_frame()
        if frame is None:
            raise StopIteration
        return frame