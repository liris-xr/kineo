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

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    Annotations,
    CameraTemporalAnnotations,
    CameraTemporalAnnotationsMetadata,
    CameraTemporalAnnotation,
)
from kineo.pipeline.pipeline import Pipeline

from torchaudio.transforms import MFCC
from torchaudio.functional import resample
from typing import Literal, Sequence
from dataclasses import dataclass
from tqdm import tqdm


@dataclass(frozen=True)
class TemporalCalibrationRuntimeConfig:
    ref_idx: int = 0
    n_mfcc: int = 13
    n_fft: int = 2048
    hop_duration: float = 0.005
    win_duration: float = 0.04
    n_mels: int = 128
    mel_scale: Literal["htk", "slaney"] = "htk"


class MFCCTemporalCalibrationStage(PipelineStage[TemporalCalibrationRuntimeConfig]):
    """
    Temporal calibration stage that uses MFCC cross-correlation to estimate the time offsets between views.

    Produces :class:`CameraTemporalAnnotations` with the time offsets for each view with key "camera_temporal".
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: TemporalCalibrationRuntimeConfig,
        dynamic_runtime_cfg: dict[str, TemporalCalibrationRuntimeConfig] | None = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: TemporalCalibrationRuntimeConfig,
    ):
        audio_waveforms = []
        audio_sample_rates = []

        for view in tqdm(views, desc="Loading audio waveforms", leave=False):
            audio_loader = view.get("audio_loader", None)
            if audio_loader is None:
                audio_waveforms.append(None)
                audio_sample_rates.append(None)
                continue

            audio, sample_rate = audio_loader.load_audio()
            audio_waveforms.append(audio)
            audio_sample_rates.append(sample_rate)

        if not all(audio_sample_rates):
            print(
                "At least one view has no audio, assuming time offset to be 0 (synchronized cameras)"
            )
            time_offsets = torch.zeros(len(views))
        else:
            time_offsets = estimate_time_offsets(
                audio_waveforms=audio_waveforms,
                audio_sample_rates=audio_sample_rates,
                ref_idx=runtime_cfg.ref_idx,
                n_mfcc=runtime_cfg.n_mfcc,
                n_fft=runtime_cfg.n_fft,
                hop_duration=runtime_cfg.hop_duration,
                win_duration=runtime_cfg.win_duration,
                n_mels=runtime_cfg.n_mels,
                mel_scale=runtime_cfg.mel_scale,
                shift_offsets=True,
            )

        annotations["camera_temporal"] = CameraTemporalAnnotations(
            metadata=CameraTemporalAnnotationsMetadata(),
            annotations=[
                CameraTemporalAnnotation(
                    view_id=view["view_id"],
                    frame_idx=0,
                    time_offset=time_offsets[i].item(),
                )
                for i, view in enumerate(views)
            ],
        )


def estimate_time_offsets(
    audio_waveforms: Sequence[torch.Tensor],
    audio_sample_rates: Sequence[int],
    ref_idx: int = 0,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_duration: float = 0.005,
    win_duration: float = 0.04,
    n_mels: int = 128,
    mel_scale: Literal["htk", "slaney"] = "htk",
    compute_device: torch.device = torch.device(
        "cpu" if torch.cuda.is_available() else "cuda"
    ),
    shift_offsets: bool = False,
) -> torch.Tensor:
    assert len(audio_waveforms) == len(audio_sample_rates), (
        "Number of waveforms and sample rates must match"
    )

    time_offsets = torch.zeros(len(audio_waveforms))
    ref_waveform = audio_waveforms[ref_idx]
    ref_sample_rate = audio_sample_rates[ref_idx]

    for i, (waveform, sample_rate) in tqdm(enumerate(
        zip(audio_waveforms, audio_sample_rates)
    ), desc="Estimating pairwise time offsets", leave=False, total=len(audio_waveforms)):
        if i == ref_idx:
            time_offsets[i] = 0.0
            continue

        time_offsets[i] = _estimate_pairwise_time_offset(
            waveform,
            sample_rate,
            ref_waveform,
            ref_sample_rate,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_duration=hop_duration,
            win_duration=win_duration,
            n_mels=n_mels,
            mel_scale=mel_scale,
            compute_device=compute_device,
        )

    min_time_offset = time_offsets.min()

    # Shift the offsets so that the minimum offset is 0
    if shift_offsets:
        time_offsets -= min_time_offset

    return time_offsets


def _estimate_pairwise_time_offset(
    audio_waveform: torch.Tensor,
    audio_sample_rate: int,
    ref_waveform: torch.Tensor,
    ref_sample_rate: int,
    n_mfcc: int = 13,
    n_fft: int = 2048,
    hop_duration: float = 0.005,
    win_duration: float = 0.04,
    n_mels: int = 128,
    mel_scale: Literal["htk", "slaney"] = "htk",
    compute_device: torch.device = torch.device(
        "cpu" if torch.cuda.is_available() else "cuda"
    ),
) -> torch.Tensor:
    """
    Estimates the time offset between two audio waveforms using MFCC cross-correlation.
    """
    min_sample_rate = min(audio_sample_rate, ref_sample_rate)

    audio_waveform = resample(audio_waveform, audio_sample_rate, min_sample_rate)
    ref_waveform = resample(ref_waveform, ref_sample_rate, min_sample_rate)

    audio_waveform = audio_waveform.to(compute_device)
    ref_waveform = ref_waveform.to(compute_device)

    hop_length = int(min_sample_rate * hop_duration)
    win_length = int(min_sample_rate * win_duration)

    mfcc_fn = MFCC(
        n_mfcc=n_mfcc,
        sample_rate=min_sample_rate,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "n_mels": n_mels,
            "mel_scale": mel_scale,
        },
    ).to(compute_device)

    mfcc_ref = mfcc_fn(ref_waveform).mean(dim=0)
    mfcc_target = mfcc_fn(audio_waveform).mean(dim=0)

    padding_size = mfcc_ref.shape[1]
    cross_correlation = torch.nn.functional.conv1d(
        mfcc_target.unsqueeze(0),
        mfcc_ref.unsqueeze(0),
        padding=padding_size,
        stride=1,
    ).squeeze()

    max_corr_idx = torch.argmax(cross_correlation, dim=0)

    hops_offset = max_corr_idx - padding_size
    time_offset = hops_offset * hop_length / min_sample_rate
    return time_offset
