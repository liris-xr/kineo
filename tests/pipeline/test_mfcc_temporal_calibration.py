import torch

from kineo.pipeline.stages.mfcc_temporal_calibration import estimate_time_offsets

SAMPLE_RATE = 16000
HOP_DURATION = 0.005


def crop(signal: torch.Tensor, start_s: float, duration_s: float) -> torch.Tensor:
    start = int(round(start_s * SAMPLE_RATE))
    return signal[:, start : start + int(round(duration_s * SAMPLE_RATE))]


def source_signal(duration_s: float = 12.0) -> torch.Tensor:
    # Noise gives every frame a distinct cepstral fingerprint, so the
    # cross-correlation has one unambiguous peak.
    torch.manual_seed(0)
    return torch.randn(1, int(duration_s * SAMPLE_RATE))


def test_offset_is_negative_when_target_is_read_earlier():
    # This pins the convention the CHiME-6 experiment reports its error
    # against: reading the target at `t0 - offset` places its content `offset`
    # later on the reference clock, so t_ref - t_target is -offset.
    signal = source_signal()
    offset = 0.5
    ref = crop(signal, 4.0, 3.0)
    target = crop(signal, 4.0 - offset, 3.0)

    offsets = estimate_time_offsets(
        audio_waveforms=[ref, target],
        audio_sample_rates=[SAMPLE_RATE, SAMPLE_RATE],
        ref_idx=0,
        hop_duration=HOP_DURATION,
    )

    assert offsets[1].item() == -offset


def test_offsets_are_recovered_across_magnitudes_and_both_signs():
    signal = source_signal()

    for offset in [0.0, 0.05, -0.05, 0.5, -0.5, 2.0, -2.0]:
        ref = crop(signal, 5.0, 3.0)
        target = crop(signal, 5.0 - offset, 3.0)

        offsets = estimate_time_offsets(
            audio_waveforms=[ref, target],
            audio_sample_rates=[SAMPLE_RATE, SAMPLE_RATE],
            ref_idx=0,
            hop_duration=HOP_DURATION,
        )

        # Estimates are quantized to whole hops.
        assert abs(offsets[1].item() + offset) <= HOP_DURATION


def test_offset_survives_a_sample_rate_mismatch():
    signal = source_signal()
    offset = 0.5
    ref = crop(signal, 4.0, 3.0)
    target = crop(signal, 4.0 - offset, 3.0)
    resampled = target[:, ::2]

    offsets = estimate_time_offsets(
        audio_waveforms=[ref, resampled],
        audio_sample_rates=[SAMPLE_RATE, SAMPLE_RATE // 2],
        ref_idx=0,
        hop_duration=HOP_DURATION,
    )

    assert abs(offsets[1].item() + offset) <= HOP_DURATION
