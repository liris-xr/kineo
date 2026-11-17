# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import os
import zlib

import numpy as np
import torch

from kineo.audio.degradations import (
    add_tone_bursts,
    add_white_noise,
    mix_at_snr,
)
from kineo.io.audio_file import get_waveform_info, load_waveform
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
    # Seconds around the peak ignored by the confidence
    confidence_exclusion_duration: float = 0.1
    correlation_output_path_template: str | None = None
    # Cut rather than muted to avoid a silence cue
    drop_audio_before_gt_timeline_jitter_s: float | None = None
    drop_audio_before_gt_timeline_lead_s: float = 0.0
    drop_audio_from_gt_timeline: bool = False
    noise_snr_db: float | None = None
    noise_source_dir: str | None = None
    noise_segment_spacing_s: float = 33.0
    beep_period_s: float | None = None
    beep_frequency_hz: float = 1000.0
    beep_duration_s: float = 0.05
    beep_level_db: float = 0.0


@dataclass(frozen=True)
class CorrelationPeakStats:
    peak_ratio: float
    peak_to_sidelobe: float
    runner_up_lag_hops: int | None


@dataclass(frozen=True)
class PairwiseOffsetEstimate:
    """Offset of one view to the reference, `t_ref - t_target` in seconds."""

    time_offset: float
    correlation: torch.Tensor
    zero_lag_idx: int
    peak_idx: int
    hop_duration: float
    peak_stats: CorrelationPeakStats

    @property
    def confidence(self) -> float:
        return 1.0 - self.peak_stats.peak_ratio


class MFCCTemporalCalibrationStage(PipelineStage[TemporalCalibrationRuntimeConfig]):
    """
    Temporal calibration stage that uses MFCC cross-correlation to estimate the time offsets between views.

    Produces :class:`CameraTemporalAnnotations` with the time offsets for each view with key "cameras_temporal".
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

        if runtime_cfg.drop_audio_from_gt_timeline:
            ends_s = audio_cut_starts(
                views, gt_annotations["global_time_reference"], jitter_s=0.0
            )
            audio_waveforms = [
                None
                if waveform is None
                else waveform[..., : int(round(end_s * sample_rate))]
                for waveform, sample_rate, end_s in zip(
                    audio_waveforms, audio_sample_rates, ends_s
                )
            ]

        cut_starts_s = [0.0] * len(views)
        if runtime_cfg.drop_audio_before_gt_timeline_jitter_s is not None:
            cut_starts_s = audio_cut_starts(
                views,
                gt_annotations["global_time_reference"],
                runtime_cfg.drop_audio_before_gt_timeline_jitter_s,
                runtime_cfg.drop_audio_before_gt_timeline_lead_s,
            )
            audio_waveforms = [
                None
                if waveform is None
                else waveform[..., int(round(start_s * sample_rate)) :]
                for waveform, sample_rate, start_s in zip(
                    audio_waveforms, audio_sample_rates, cut_starts_s
                )
            ]

        if runtime_cfg.beep_period_s is not None:
            gt_offsets = {
                a.view_id: a.time_offset
                for a in gt_annotations["cameras_temporal"].annotations
            }
            ref_id = views[runtime_cfg.ref_idx]["view_id"]
            for i, view in enumerate(views):
                if audio_waveforms[i] is None:
                    continue
                shift_s = (
                    gt_offsets[view["view_id"]] - gt_offsets[ref_id] + cut_starts_s[i]
                )
                duration_s = audio_waveforms[i].shape[-1] / audio_sample_rates[i]
                n_beeps = int((duration_s + abs(shift_s)) / runtime_cfg.beep_period_s) + 2
                audio_waveforms[i] = add_tone_bursts(
                    audio_waveforms[i],
                    audio_sample_rates[i],
                    [k * runtime_cfg.beep_period_s - shift_s for k in range(-n_beeps, n_beeps)],
                    runtime_cfg.beep_frequency_hz,
                    runtime_cfg.beep_duration_s,
                    runtime_cfg.beep_level_db,
                )

        if runtime_cfg.noise_snr_db is not None and runtime_cfg.noise_source_dir:
            for i in range(len(views)):
                if audio_waveforms[i] is None:
                    continue
                noise = load_noise_segment(
                    runtime_cfg.noise_source_dir,
                    channel=i % 16 + 1,
                    start_s=i * runtime_cfg.noise_segment_spacing_s,
                    n_samples=audio_waveforms[i].shape[-1],
                    sample_rate=audio_sample_rates[i],
                )
                audio_waveforms[i] = mix_at_snr(
                    audio_waveforms[i], noise, runtime_cfg.noise_snr_db
                )
        elif runtime_cfg.noise_snr_db is not None:
            for i, view in enumerate(views):
                if i == runtime_cfg.ref_idx or audio_waveforms[i] is None:
                    continue
                generator = torch.Generator().manual_seed(
                    zlib.crc32(view["view_id"].encode())
                )
                audio_waveforms[i] = add_white_noise(
                    audio_waveforms[i], runtime_cfg.noise_snr_db, generator
                )

        print("Computing time offsets...")

        if not all(audio_sample_rates):
            print(
                "At least one view has no audio, assuming time offset to be 0 (synchronized cameras)"
            )
            time_offsets = torch.zeros(len(views))
            confidences = torch.full((len(views),), float("nan"))
            estimates = [None] * len(views)
        else:
            time_offsets, confidences, estimates = (
                estimate_time_offsets_with_confidence(
                    audio_waveforms=audio_waveforms,
                    audio_sample_rates=audio_sample_rates,
                    ref_idx=runtime_cfg.ref_idx,
                    n_mfcc=runtime_cfg.n_mfcc,
                    n_fft=runtime_cfg.n_fft,
                    hop_duration=runtime_cfg.hop_duration,
                    win_duration=runtime_cfg.win_duration,
                    n_mels=runtime_cfg.n_mels,
                    mel_scale=runtime_cfg.mel_scale,
                    confidence_exclusion_duration=(
                        runtime_cfg.confidence_exclusion_duration
                    ),
                )
            )

        # Report the offsets on the uncut clocks.
        ref_cut_s = cut_starts_s[runtime_cfg.ref_idx]
        for i in range(len(views)):
            time_offsets[i] -= cut_starts_s[i] - ref_cut_s

        print("--------------------------------")
        print("Time offsets:")
        for i, view in enumerate(views):
            if i == runtime_cfg.ref_idx:
                print(f"{view['view_id']}: {time_offsets[i].item()} (ref)")
            else:
                print(
                    f"{view['view_id']}: {time_offsets[i].item()} "
                    f"(confidence {confidences[i].item():.3f})"
                )
        print("--------------------------------")

        annotations["cameras_temporal"] = CameraTemporalAnnotations(
            metadata=CameraTemporalAnnotationsMetadata(),
            annotations=[
                CameraTemporalAnnotation(
                    view_id=view["view_id"],
                    frame_idx=0,
                    time_offset=time_offsets[i].item(),
                    confidence=(
                        None
                        if torch.isnan(confidences[i])
                        else confidences[i].item()
                    ),
                )
                for i, view in enumerate(views)
            ],
        )

        if runtime_cfg.correlation_output_path_template is None:
            return

        for view, estimate in zip(views, estimates):
            if estimate is None:
                continue
            path = runtime_cfg.correlation_output_path_template.format(
                sequence_name=sequence_name, view_id=view["view_id"]
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(
                path,
                correlation=estimate.correlation.numpy().astype(np.float32),
                zero_lag_idx=estimate.zero_lag_idx,
                peak_idx=estimate.peak_idx,
                hop_duration=estimate.hop_duration,
                time_offset=estimate.time_offset,
                peak_ratio=estimate.peak_stats.peak_ratio,
                peak_to_sidelobe=estimate.peak_stats.peak_to_sidelobe,
            )


def load_noise_segment(
    source_dir: str,
    channel: int,
    start_s: float,
    n_samples: int,
    sample_rate: int,
) -> torch.Tensor:
    path = os.path.join(source_dir, f"ch{channel:02d}.wav")
    info = get_waveform_info(path)
    start = int(round(start_s * info.sample_rate)) % info.n_frames
    n_source = int(round(n_samples * info.sample_rate / sample_rate))
    noise, _ = load_waveform(path, start_frame=start, n_frames=n_source)
    noise = noise.mean(dim=0)
    if noise.shape[0] < n_source:
        head, _ = load_waveform(path, 0, n_source - noise.shape[0])
        noise = torch.cat([noise, head.mean(dim=0)])
    if info.sample_rate != sample_rate:
        noise = resample(noise, info.sample_rate, sample_rate)
    return noise[:n_samples]


def audio_cut_starts(
    views: list[ViewInput],
    gt_global_time_reference: Annotations,
    jitter_s: float,
    lead_s: float = 0.0,
) -> list[float]:
    timeline = gt_global_time_reference.first_or_default()
    frame_period_s = float(timeline.timestamps[1] - timeline.timestamps[0])
    starts_s = []
    for view in views:
        first_frame = int(timeline.closest_local_frame_idx[view["view_id"]].min())
        seed = zlib.crc32(view["view_id"].encode())
        jitter = torch.rand(1, generator=torch.Generator().manual_seed(seed))
        starts_s.append(
            max(0.0, first_frame * frame_period_s - lead_s + jitter.item() * jitter_s)
        )
    return starts_s


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
    time_offsets, _, _ = estimate_time_offsets_with_confidence(
        audio_waveforms=audio_waveforms,
        audio_sample_rates=audio_sample_rates,
        ref_idx=ref_idx,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_duration=hop_duration,
        win_duration=win_duration,
        n_mels=n_mels,
        mel_scale=mel_scale,
        compute_device=compute_device,
    )

    # Shift the offsets so that the minimum offset is 0
    if shift_offsets:
        time_offsets -= time_offsets.min()

    return time_offsets


def estimate_time_offsets_with_confidence(
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
    confidence_exclusion_duration: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, list[PairwiseOffsetEstimate | None]]:
    assert len(audio_waveforms) == len(audio_sample_rates), (
        "Number of waveforms and sample rates must match"
    )

    time_offsets = torch.zeros(len(audio_waveforms))
    confidences = torch.full((len(audio_waveforms),), float("nan"))
    estimates: list[PairwiseOffsetEstimate | None] = [None] * len(audio_waveforms)
    ref_waveform = audio_waveforms[ref_idx]
    ref_sample_rate = audio_sample_rates[ref_idx]

    for i, (waveform, sample_rate) in tqdm(enumerate(
        zip(audio_waveforms, audio_sample_rates)
    ), desc="Estimating pairwise time offsets", leave=False, total=len(audio_waveforms)):
        if i == ref_idx:
            continue

        estimate = _estimate_pairwise_time_offset(
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
            confidence_exclusion_duration=confidence_exclusion_duration,
        )
        time_offsets[i] = estimate.time_offset
        confidences[i] = estimate.confidence
        estimates[i] = estimate

    return time_offsets, confidences, estimates


def correlation_peak_confidence(
    corr: torch.Tensor, peak_idx: int, exclusion_hops: int
) -> CorrelationPeakStats:
    peak = corr[peak_idx]
    outside = torch.ones_like(corr, dtype=torch.bool)
    outside[max(0, peak_idx - exclusion_hops) : peak_idx + exclusion_hops + 1] = (
        False
    )
    if not outside.any():
        return CorrelationPeakStats(0.0, float("inf"), None)

    sidelobe = corr[outside]
    peak_to_sidelobe = (
        (peak - sidelobe.mean()) / sidelobe.std().clamp(min=1e-12)
    ).item()

    is_local_max = torch.zeros_like(outside)
    is_local_max[1:-1] = (corr[1:-1] > corr[:-2]) & (corr[1:-1] >= corr[2:])
    rivals = corr.masked_fill(~(outside & is_local_max), -torch.inf)

    # Negative correlation is not a rival peak
    peak_ratio = 0.0
    runner_up_lag_hops = None
    if rivals.max() > 0:
        peak_ratio = (rivals.max() / peak).clamp(max=1.0).item()
        runner_up_lag_hops = int(rivals.argmax().item()) - peak_idx

    return CorrelationPeakStats(
        peak_ratio=peak_ratio,
        peak_to_sidelobe=peak_to_sidelobe,
        runner_up_lag_hops=runner_up_lag_hops,
    )


def _to_mono_f64(waveform: torch.Tensor) -> torch.Tensor:
    waveform = waveform.to(torch.float64)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
    return waveform


def _mfcc_cross_correlate(ref: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
    # Standardize each coefficient, or c0 dominates the peak
    ref_c = ref - ref.mean(dim=1, keepdim=True)
    sub_c = sub - sub.mean(dim=1, keepdim=True)
    ref_n = ref_c / ref_c.std(dim=1, keepdim=True).clamp(min=1e-8)
    sub_rev = sub_c.flip(dims=[1]) / sub_c.std(dim=1, keepdim=True).clamp(min=1e-8)

    valid_len = ref.shape[1] + sub.shape[1] - 1
    fft_size = 1
    while fft_size < valid_len:
        fft_size <<= 1

    ref_f = torch.fft.rfft(ref_n, n=fft_size, dim=1)
    sub_f = torch.fft.rfft(sub_rev, n=fft_size, dim=1)

    corr_f = (ref_f * sub_f).sum(dim=0)
    corr = torch.fft.irfft(corr_f, n=fft_size)[:valid_len]
    return corr


def _mfcc_in_chunks(
    mfcc_fn: MFCC,
    waveform: torch.Tensor,
    hop_length: int,
    n_fft: int,
    chunk_frames: int = 30000,
) -> torch.Tensor:
    n_samples = waveform.shape[-1]
    chunk = chunk_frames * hop_length
    if n_samples <= chunk:
        return mfcc_fn(waveform).squeeze(0)

    # Pad in whole hops to stay on the frame grid
    pad = -(-(n_fft // 2) // hop_length) * hop_length
    pieces = []
    for start in range(0, n_samples, chunk):
        stop = min(start + chunk, n_samples)
        lo, hi = max(0, start - pad), min(n_samples, stop + pad)
        out = mfcc_fn(waveform[..., lo:hi]).squeeze(0)
        skip = (start - lo) // hop_length
        if stop == n_samples:
            pieces.append(out[:, skip:])
        else:
            pieces.append(out[:, skip : skip + (stop - start) // hop_length])

    return torch.cat(pieces, dim=1)[:, : n_samples // hop_length + 1]


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
    confidence_exclusion_duration: float = 0.1,
) -> PairwiseOffsetEstimate:
    """
    Estimates the time offset between two audio waveforms using MFCC cross-correlation.
    """
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

    mfcc_ref = _mfcc_in_chunks(
        mfcc_fn, ref_mono.float().unsqueeze(0), hop_length, n_fft
    )
    mfcc_target = _mfcc_in_chunks(
        mfcc_fn, target_mono.float().unsqueeze(0), hop_length, n_fft
    )

    corr = _mfcc_cross_correlate(mfcc_ref, mfcc_target).cpu()
    peak_idx = int(torch.argmax(corr).item())

    zero_lag_idx = mfcc_target.shape[1] - 1
    hop_duration_s = hop_length / ref_sample_rate
    time_offset = (peak_idx - zero_lag_idx) * hop_duration_s

    return PairwiseOffsetEstimate(
        time_offset=time_offset,
        correlation=corr,
        zero_lag_idx=zero_lag_idx,
        peak_idx=peak_idx,
        hop_duration=hop_duration_s,
        peak_stats=correlation_peak_confidence(
            corr,
            peak_idx,
            exclusion_hops=int(round(confidence_exclusion_duration / hop_duration_s)),
        ),
    )
