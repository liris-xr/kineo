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
from kineo.io.ffmpeg import extract_audio


class AudioLoader(ABC):
    def __init__(self, device: torch.device):
        self.device = device

    @abstractmethod
    def load_audio(self) -> tuple[torch.Tensor, int]:
        pass


class VideoAudioLoader(AudioLoader):
    def __init__(
        self,
        video_path: str,
        device: torch.device,
    ):
        super().__init__(device)
        self.video_path = video_path

    def load_audio(self) -> tuple[torch.Tensor, int]:
        audio, sample_rate = extract_audio(self.video_path, show_progress=True)
        if audio is not None:
            audio = audio.to(self.device)
        return audio, sample_rate