# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Reading of audio files into waveforms.

`torchaudio.load` delegates decoding to TorchCodec, which needs FFmpeg's shared
libraries resolvable at import time and a build pinned to the torch release.
Everything here is read from files ffmpeg already wrote, so libsndfile handles
it with no native library to line up.
"""

from __future__ import annotations

import dataclasses

import soundfile
import torch


@dataclasses.dataclass(frozen=True)
class WaveformInfo:
    """Shape of an audio file, read from its header alone."""

    sample_rate: int
    n_frames: int
    n_channels: int

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.n_frames / self.sample_rate


def get_waveform_info(filepath: str) -> WaveformInfo:
    """Reads an audio file's header, without decoding it.

    Args:
        filepath: Path to the audio file.

    Returns:
        The file's sample rate, length in frames and channel count.
    """
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
    """Reads an audio file, or a window of it, into a waveform.

    Args:
        filepath: Path to the audio file.
        start_frame: First frame to read.
        n_frames: Number of frames to read, or -1 to read to the end.

    Returns:
        The waveform, of shape (n_channels, n_frames) and dtype float32, and
        its sample rate.
    """
    samples, sample_rate = soundfile.read(
        filepath,
        start=start_frame,
        frames=n_frames,
        dtype="float32",
        always_2d=True,
    )
    return torch.from_numpy(samples.T).contiguous(), sample_rate
