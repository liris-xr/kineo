# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import math

import torch


def add_white_noise(
    waveform: torch.Tensor, snr_db: float, generator: torch.Generator
) -> torch.Tensor:
    waveform = waveform.to(torch.float64).cpu()
    signal_power = waveform.pow(2).mean()
    noise = torch.randn(waveform.shape, generator=generator, dtype=torch.float64)
    return waveform + noise * torch.sqrt(signal_power / 10 ** (snr_db / 10))


def mix_at_snr(
    waveform: torch.Tensor, noise: torch.Tensor, snr_db: float
) -> torch.Tensor:
    waveform = waveform.to(torch.float64).cpu()
    noise = noise.to(torch.float64).cpu().reshape(-1)
    n = waveform.shape[-1]
    repeats = -(-n // noise.shape[0])
    noise = noise.repeat(repeats)[:n]
    gain = torch.sqrt(
        waveform.pow(2).mean() / noise.pow(2).mean().clamp(min=1e-20)
        / 10 ** (snr_db / 10)
    )
    return waveform + gain * noise


def add_tone_bursts(
    waveform: torch.Tensor,
    sample_rate: int,
    burst_starts_s: list[float],
    frequency_hz: float,
    duration_s: float,
    level_db: float,
) -> torch.Tensor:
    """Add sine bursts whose RMS is `level_db` above the waveform RMS."""
    waveform = waveform.to(torch.float64).cpu()
    n_samples = waveform.shape[-1]
    burst_len = int(round(duration_s * sample_rate))
    t = torch.arange(burst_len, dtype=torch.float64) / sample_rate
    # Raised-cosine envelope to avoid sharp onsets
    amplitude = waveform.pow(2).mean().sqrt() * 10 ** (level_db / 20) * math.sqrt(2)
    envelope = 0.5 - 0.5 * torch.cos(2 * math.pi * torch.arange(burst_len) / burst_len)
    burst = amplitude * envelope * torch.sin(2 * math.pi * frequency_hz * t)

    out = waveform.clone()
    for start_s in burst_starts_s:
        start = int(round(start_s * sample_rate))
        if start < 0 or start + burst_len > n_samples:
            continue
        out[..., start : start + burst_len] += burst
    return out
