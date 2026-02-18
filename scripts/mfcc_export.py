import sys
from typing import Literal

import plotly.graph_objects as go
import torch
from torchaudio.transforms import MFCC

from kineo.io.ffmpeg import extract_audio


def compute_stft(audio_waveform: torch.Tensor, n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048):
    """
    Compute magnitude STFT of an audio waveform.

    Returns:
        stft_magnitude: Tensor of shape (freq_bins, time_frames)
    """
    # Create Hann window to reduce spectral leakage
    window = torch.hann_window(win_length, device=audio_waveform.device)

    stft_complex = torch.stft(
        audio_waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True
    )
    stft_magnitude = stft_complex.abs()
    if stft_magnitude.dim() > 2:
        stft_magnitude = stft_magnitude.mean(dim=0)  # average channels
    return stft_magnitude

def compute_mfcc(
        audio_waveform: torch.Tensor,
        sample_rate: int,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_duration: float = 0.005,
        win_duration: float = 0.04,
        n_mels: int = 128,
        mel_scale: Literal["htk", "slaney"] = "htk",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.Tensor:
    """
    Compute MFCC features from an audio waveform.

    Args:
        audio_waveform: Audio tensor of shape (channels, samples)
        sample_rate: Sample rate of the audio
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_duration: Hop duration in seconds
        win_duration: Window duration in seconds
        n_mels: Number of mel filterbanks
        mel_scale: Mel scale type
        device: Computation device

    Returns:
        MFCC features of shape (n_mfcc, time_frames)
    """
    audio_waveform = audio_waveform.to(device)

    hop_length = int(sample_rate * hop_duration)
    win_length = int(sample_rate * win_duration)

    mfcc_transform = MFCC(
        n_mfcc=n_mfcc,
        sample_rate=sample_rate,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "win_length": win_length,
            "n_mels": n_mels,
            "mel_scale": mel_scale,
        },
    ).to(device)

    mfcc_features = mfcc_transform(audio_waveform)

    # Average across channels if multi-channel
    if mfcc_features.dim() > 2:
        mfcc_features = mfcc_features.mean(dim=0)

    return mfcc_features


if __name__ == "__main__":
    # Video paths
    video_path1 = "D:/Charles_JAVERLIAT/Captations Guedelon/raw_videos/09_09_2025_luc/gopro1_linear_carriere1.MP4"
    output_path_mfcc = "../outputs/gopro1_mfcc.png"
    output_path_stft = "../outputs/gopro1_stft.png"

    # Process first video
    print("Processing video 1...")
    print("Extracting audio from video 1...")
    waveform1, sample_rate1 = extract_audio(video_path1, show_progress=True)

    if waveform1 is None or sample_rate1 is None:
        print("No audio stream found in video 1.")
        sys.exit(1)

    print(f"Audio 1 loaded: {waveform1.shape}, sample rate: {sample_rate1}")
    print("Computing MFCC features for video 1...")
    mfcc1 = compute_mfcc(waveform1, sample_rate1)
    print(f"MFCC 1 shape: {mfcc1.shape}")

    print("\nComputing STFT for Video 1...")
    stft1 = compute_stft(waveform1, n_fft=2048, hop_length=int(sample_rate1 * 0.005),
                         win_length=int(sample_rate1 * 0.04))
    print(f"STFT 1 shape: {stft1.shape}")

    hop_duration = 0.005  # Same as in compute_mfcc

    start_time = 48.0  # seconds
    end_time = 72.0    # seconds

    STFT_DOWNSAMPLE_FACTOR = 4

    start_frame = int(start_time / hop_duration)
    end_frame = int(end_time / hop_duration)

    mfcc1_subset = mfcc1[:3, start_frame:end_frame]
    stft1_subset = stft1[::STFT_DOWNSAMPLE_FACTOR, start_frame:end_frame]

    fig3 = go.Figure()
    stft1_db = 20 * torch.log10(stft1 + 1e-10)  # Convert to dB scale

    fig3.add_trace(
        go.Heatmap(
            z=stft1_db.cpu().numpy(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Magnitude (dB)'),
            hovertemplate='Frame: %{x}<br>Frequency Bin: %{y}<br>Magnitude: %{z:.2f} dB<extra></extra>'
        )
    )
    fig3.update_layout(
        title='STFT Spectrogram - Video 1',
        xaxis_title='Frames',
        yaxis_title='Frequency Bins',
        height=500,
        width=1400,
        font=dict(size=12)
    )
    fig3.write_image(output_path_stft)
    print(f"Saved: {output_path_stft}")


    # Create time axis in seconds
    num_frames = mfcc1_subset.shape[1]
    time_axis = [start_time + i * hop_duration for i in range(num_frames)]

    # Create Plotly figure
    fig = go.Figure(data=go.Heatmap(
        z=mfcc1_subset.cpu().numpy(),
        x=time_axis,
        y=[0, 1, 2],  # MFCC coefficient indices
        colorscale='Viridis',
        colorbar=dict(title='MFCC Value'),
        hovertemplate='Time Frame: %{x}<br>MFCC Coefficient: %{y}<br>Value: %{z}<extra></extra>'
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(
                text='Time (s)',
                font=dict(size=20)
            ),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title=dict(
                text='MFCC Coefficients',
                font=dict(size=20)
            ),
            tickfont=dict(size=16),
            tickmode='array',
            tickvals=[0, 1, 2],
            ticktext=['0', '1', '2']
        ),
        width=1920,
        height=400
    )

    fig.write_image(output_path_mfcc)
    fig.show()
