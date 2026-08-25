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

            print(f"Loading audio for {view['view_id']}...")
            audio, sample_rate = audio_loader.load_audio()
            audio_waveforms.append(audio)
            audio_sample_rates.append(sample_rate)

        print("Computing time offsets...")

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
                shift_offsets=False,
            )

        print("--------------------------------")
        print("Time offsets:")
        for i, view in enumerate(views):
            is_ref = i == runtime_cfg.ref_idx
            print(f"{view['view_id']}: {time_offsets[i].item()}{' (ref)' if is_ref else ''}")
        print("--------------------------------")

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
        "cuda" if torch.cuda.is_available() else "cpu"
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


def _to_mono_f64(waveform: torch.Tensor) -> torch.Tensor:
    """
    Force mono and cast to float64.

    The float64 cast happens before any arithmetic to avoid overflow on
    integer-typed inputs (e.g. int16 where abs(-32768) overflows back to -32768).
    """
    waveform = waveform.to(torch.float64)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
    return waveform


def _mfcc_cross_correlate(ref: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
    """
    FFT-based cross-correlation summed across MFCC coefficients.

    Each coefficient is centered and normalized to unit variance before
    correlation so that no single cepstral band dominates the alignment signal
    (low-order MFCC coefficients carry much more energy than high-order ones
    without this step).

    Centering is what makes the peak meaningful, not a refinement of it: c0
    (log energy) has a mean on the order of 60x its standard deviation, and an
    uncentered correlation is therefore dominated by the product of the two
    means times the overlap length. That term is a triangular envelope peaking
    at full overlap, which pins ``argmax`` at zero lag for equal-length clips
    whatever the true offset is.

    Uses the time-reversal trick: convolving ``ref`` with ``sub[::-1]`` is
    equivalent to cross-correlating ``ref`` with ``sub``, without needing
    conjugate multiplication in the frequency domain.

    Args:
        ref: (n_mfcc, T_ref) MFCC tensor for the reference clip.
        sub: (n_mfcc, T_sub) MFCC tensor for the query clip.

    Returns:
        1D float64 tensor of length ``T_ref + T_sub - 1``. Zero-lag is at index
        ``T_sub - 1``; a peak at index k gives ``lag = k - (T_sub - 1)`` hops.
    """
    ref_c = ref - ref.mean(dim=1, keepdim=True)
    sub_c = sub - sub.mean(dim=1, keepdim=True)
    ref_n = ref_c / ref_c.std(dim=1, keepdim=True).clamp(min=1e-8)
    sub_rev = sub_c.flip(dims=[1]) / sub_c.std(dim=1, keepdim=True).clamp(min=1e-8)

    # Pad >= linear-conv length so the FFT's circular convolution equals the linear one.
    valid_len = ref.shape[1] + sub.shape[1] - 1
    fft_size = 1
    while fft_size < valid_len:
        fft_size <<= 1

    ref_f = torch.fft.rfft(ref_n, n=fft_size, dim=1)
    sub_f = torch.fft.rfft(sub_rev, n=fft_size, dim=1)

    corr_f = (ref_f * sub_f).sum(dim=0)
    corr = torch.fft.irfft(corr_f, n=fft_size)[:valid_len]
    return corr


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
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
) -> torch.Tensor:
    """
    Estimates the time offset between two audio waveforms using MFCC cross-correlation.

    Returns the signed offset in seconds following the convention
    ``time_offset = t_ref - t_target`` for the same physical event, so that
    ``local_timestamps + time_offset`` maps a view onto the reference clock.
    """
    # Resample target onto the reference grid so both MFCC transforms share hop/win lengths.
    if audio_sample_rate != ref_sample_rate:
        audio_waveform = resample(audio_waveform, audio_sample_rate, ref_sample_rate)

    ref_mono = _to_mono_f64(ref_waveform).to(compute_device)
    target_mono = _to_mono_f64(audio_waveform).to(compute_device)

    hop_length = int(ref_sample_rate * hop_duration)
    win_length = int(ref_sample_rate * win_duration)

    mfcc_fn = MFCC(
        n_mfcc=n_mfcc,
        sample_rate=ref_sample_rate,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "n_mels": n_mels,
            "mel_scale": mel_scale,
        },
    ).to(compute_device)

    mfcc_ref = mfcc_fn(ref_mono.float().unsqueeze(0)).squeeze(0)
    mfcc_target = mfcc_fn(target_mono.float().unsqueeze(0)).squeeze(0)

    corr = _mfcc_cross_correlate(mfcc_ref, mfcc_target)
    peak_idx = torch.argmax(corr)

    # Zero-lag is at index T_target - 1 (see _mfcc_cross_correlate docstring).
    lag_hops = peak_idx - (mfcc_target.shape[1] - 1)
    time_offset = lag_hops * hop_length / ref_sample_rate
    return time_offset.cpu()
