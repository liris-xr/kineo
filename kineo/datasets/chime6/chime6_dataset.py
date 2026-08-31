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

import os
from typing import Iterator, TypedDict

import orjson
import torch

from kineo.io.audio_file import get_waveform_info
from kineo.io.audio_loader import AudioLoader, WaveformAudioLoader


class AudioView(TypedDict):
    view_id: str
    audio_path: str
    audio_loader: AudioLoader
    position_m: tuple[float, float]
    room: str


class AudioWindow(TypedDict):
    window_id: str
    session_id: str
    start_time_s: float
    duration_s: float
    content_class: str
    composition: dict[str, float]
    n_distinct_speakers: int
    max_concurrent_speakers: int
    speakers: list[str]
    selection: dict
    views: list[AudioView]


class CHiME6AudioDataset:
    """CHiME-6 evaluation windows, one per acoustic content cell.

    CHiME-6 ships no video, so this is not a `KeypointsSequenceDataset`: a
    window exposes microphone arrays rather than cameras. Its value is that the
    arrays are already synchronized and their positions known, which makes it
    ground truth for temporal calibration.

    Each view's loader is scoped to the window. Reading a device at any other
    position, as an injected de-sync does, means building a loader from the
    view's `audio_path`.
    """

    def __init__(
        self,
        windows_filepath: str,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            windows_filepath: Path to a `chime6_windows.json` file written by
                the preprocessing.
            device: Device the audio is loaded onto.

        Raises:
            FileNotFoundError: If `windows_filepath` does not exist.
        """
        if not os.path.exists(windows_filepath):
            raise FileNotFoundError(
                f"Windows file {windows_filepath} does not exist"
            )

        self.device = device
        self.windows_filepath = windows_filepath
        self.dataset_dirpath = os.path.dirname(windows_filepath)

        with open(windows_filepath, "rb") as f:
            content = orjson.loads(f.read())
        self.meta = content["meta"]
        self.windows_data = content["windows"]

    def __len__(self) -> int:
        return len(self.windows_data)

    def __getitem__(self, index: int) -> AudioWindow:
        window_data = self.windows_data[index]
        start_time = window_data["start_time_s"]
        duration = window_data["duration_s"]

        views: list[AudioView] = []
        for view_id, view_info in window_data["views"].items():
            audio_path = os.path.join(
                self.dataset_dirpath, view_info["audio_path"]
            )
            sample_rate = get_waveform_info(audio_path).sample_rate
            views.append(
                AudioView(
                    view_id=view_id,
                    audio_path=audio_path,
                    audio_loader=WaveformAudioLoader(
                        audio_path=audio_path,
                        device=self.device,
                        start_frame=round(start_time * sample_rate),
                        n_frames=round(duration * sample_rate),
                    ),
                    position_m=tuple(view_info["position_m"]),
                    room=view_info["room"],
                )
            )

        return AudioWindow(views=views, **{
            key: value
            for key, value in window_data.items()
            if key != "views"
        })

    def __iter__(self) -> Iterator[AudioWindow]:
        for i in range(len(self)):
            yield self[i]
