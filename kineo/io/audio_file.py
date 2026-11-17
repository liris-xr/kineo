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

import dataclasses

# Not torchaudio.load, which needs FFmpeg matching torch
import soundfile
import torch


@dataclasses.dataclass(frozen=True)
class WaveformInfo:
    sample_rate: int
    n_frames: int
    n_channels: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.sample_rate


def get_waveform_info(filepath: str) -> WaveformInfo:
    info = soundfile.info(filepath)
    return WaveformInfo(
        sample_rate=info.samplerate,
        n_frames=info.frames,
        n_channels=info.channels,
    )


def load_waveform(
    filepath: str,
    start_frame: int = 0,
    n_frames: int = -1,
) -> tuple[torch.Tensor, int]:
    samples, sample_rate = soundfile.read(
        filepath,
        start=start_frame,
        frames=n_frames,
        dtype="float32",
        always_2d=True,
    )
    return torch.from_numpy(samples.T).contiguous(), sample_rate
