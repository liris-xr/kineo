import torch
from torchaudio.transforms import MFCC
from typing import Literal
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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


def compute_cross_correlation(mfcc1: torch.Tensor, mfcc2: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-correlation between two MFCC features.

    Args:
        mfcc1: First MFCC tensor of shape (n_mfcc, time_frames)
        mfcc2: Second MFCC tensor of shape (n_mfcc, time_frames)

    Returns:
        Cross-correlation tensor
    """
    padding_size = mfcc1.shape[1]
    cross_correlation = torch.nn.functional.conv1d(
        mfcc2.unsqueeze(0),
        mfcc1.unsqueeze(0),
        padding=padding_size,
        stride=1,
    ).squeeze()

    return cross_correlation


def merge_interleave_mfcc(mfcc1: torch.Tensor, mfcc2: torch.Tensor) -> torch.Tensor:
    """
    Merge and interleave MFCC coefficients from two audio sources.

    Args:
        mfcc1: First MFCC tensor of shape (n_mfcc, time_frames)
        mfcc2: Second MFCC tensor of shape (n_mfcc, time_frames)

    Returns:
        Merged MFCC tensor of shape (2 * n_mfcc, min_time_frames) where
        coefficients are interleaved: [mfcc1[0], mfcc2[0], mfcc1[1], mfcc2[1], ...]
    """
    # Ensure both MFCCs have the same time dimension
    min_time_frames = min(mfcc1.shape[1], mfcc2.shape[1])
    mfcc1_trimmed = mfcc1[:, :min_time_frames]
    mfcc2_trimmed = mfcc2[:, :min_time_frames]

    # Stack the two MFCCs and interleave
    # Shape: (n_mfcc, 2, time_frames)
    stacked = torch.stack([mfcc1_trimmed, mfcc2_trimmed], dim=1)

    # Reshape to interleave: (2 * n_mfcc, time_frames)
    merged = stacked.reshape(-1, min_time_frames)

    return merged


if __name__ == "__main__":
    # Video paths
    video_path1 = "D:/Charles_JAVERLIAT/Captations Guedelon/raw_videos/09_09_2025_luc/gopro1_linear_carriere1.MP4"
    video_path2 = "D:/Charles_JAVERLIAT/Captations Guedelon/raw_videos/09_09_2025_luc/gopro2_linear_carriere1.MP4"

    # Output directory
    output_dir = "../outputs/mfcc_output"

    # Plotting options
    PLOT_STFT = True  # Set to False to skip STFT plotting (saves time)
    STFT_DOWNSAMPLE_FACTOR = 4  # Downsample STFT for faster plotting (1 = no downsampling)

    # Create output directory if it doesn't exist
    import os

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Process first video
    print("Processing video 1...")
    print("Extracting audio from video 1...")
    waveform1, sample_rate1 = extract_audio(video_path1, show_progress=True)

    # Process second video
    print("\nProcessing video 2...")
    print("Extracting audio from video 2...")
    waveform2, sample_rate2 = extract_audio(video_path2, show_progress=True)

    if waveform1 is None or sample_rate1 is None:
        print("No audio stream found in video 1.")
        sys.exit(1)

    if waveform2 is None or sample_rate2 is None:
        print("No audio stream found in video 2.")
        sys.exit(1)

    print("\nComputing STFT for Video 1...")
    stft1 = compute_stft(waveform1, n_fft=2048, hop_length=int(sample_rate1 * 0.005),
                         win_length=int(sample_rate1 * 0.04))
    print(f"STFT 1 shape: {stft1.shape}")
    stft1_size_mb = stft1.element_size() * stft1.nelement() / (1024 ** 2)
    print(f"STFT 1 size: {stft1_size_mb:.2f} MB")

    print("\nComputing STFT for Video 2...")
    stft2 = compute_stft(waveform2, n_fft=2048, hop_length=int(sample_rate2 * 0.005),
                         win_length=int(sample_rate2 * 0.04))
    print(f"STFT 2 shape: {stft2.shape}")
    stft2_size_mb = stft2.element_size() * stft2.nelement() / (1024 ** 2)
    print(f"STFT 2 size: {stft2_size_mb:.2f} MB")

    print(f"Audio 1 loaded: {waveform1.shape}, sample rate: {sample_rate1}")
    print("Computing MFCC features for video 1...")
    mfcc1 = compute_mfcc(waveform1, sample_rate1)
    print(f"MFCC 1 shape: {mfcc1.shape}")
    mfcc1_size_mb = mfcc1.element_size() * mfcc1.nelement() / (1024 ** 2)
    print(f"MFCC 1 size: {mfcc1_size_mb:.2f} MB")

    print(f"Audio 2 loaded: {waveform2.shape}, sample rate: {sample_rate2}")
    print("Computing MFCC features for video 2...")
    mfcc2 = compute_mfcc(waveform2, sample_rate2)
    print(f"MFCC 2 shape: {mfcc2.shape}")
    mfcc2_size_mb = mfcc2.element_size() * mfcc2.nelement() / (1024 ** 2)
    print(f"MFCC 2 size: {mfcc2_size_mb:.2f} MB")

    # Print compression ratio
    print(f"\nCompression ratio (STFT to MFCC):")
    print(f"Video 1: {stft1_size_mb / mfcc1_size_mb:.2f}x reduction")
    print(f"Video 2: {stft2_size_mb / mfcc2_size_mb:.2f}x reduction")

    # Downsample STFT for plotting if requested
    if STFT_DOWNSAMPLE_FACTOR > 1:
        print(f"\nDownsampling STFT by factor of {STFT_DOWNSAMPLE_FACTOR} for faster plotting...")
        stft1_plot = stft1[::STFT_DOWNSAMPLE_FACTOR, ::STFT_DOWNSAMPLE_FACTOR]
        stft2_plot = stft2[::STFT_DOWNSAMPLE_FACTOR, ::STFT_DOWNSAMPLE_FACTOR]
        print(f"STFT 1 plot shape: {stft1_plot.shape} (reduced from {stft1.shape})")
        print(f"STFT 2 plot shape: {stft2_plot.shape} (reduced from {stft2.shape})")
    else:
        stft1_plot = stft1
        stft2_plot = stft2

    # Compute cross-correlation
    print("\nComputing cross-correlation...")
    cross_corr = compute_cross_correlation(mfcc1, mfcc2)
    print(f"Cross-correlation shape: {cross_corr.shape}")

    # Find the peak of cross-correlation
    max_corr_idx = torch.argmax(cross_corr, dim=0)
    padding_size = mfcc1.shape[1]
    time_offset_frames = max_corr_idx - padding_size

    hop_duration = 0.005  # seconds
    time_offset_seconds = time_offset_frames.item() * hop_duration

    print(f"Time offset: {time_offset_frames.item()} frames ({time_offset_seconds:.3f} seconds)")

    # Merge and interleave MFCCs (without alignment)
    print("\nMerging and interleaving MFCC coefficients (no alignment)...")
    merged_mfcc = merge_interleave_mfcc(mfcc1, mfcc2)
    print(f"Merged MFCC shape: {merged_mfcc.shape}")
    print(f"Original MFCC1 had {mfcc1.shape[0]} coefficients, MFCC2 had {mfcc2.shape[0]} coefficients")
    print(f"Merged MFCC has {merged_mfcc.shape[0]} coefficients (interleaved)")

    # Apply time offset alignment to MFCC2 and create aligned merged MFCC
    print("\nCreating time-aligned merged MFCC...")
    offset_frames = -time_offset_frames.item()

    if offset_frames > 0:
        # Video 2 starts AFTER Video 1 → pad mfcc2 at beginning
        pad_start = torch.zeros(
            (mfcc2.shape[0], offset_frames),
            dtype=mfcc2.dtype,
            device=mfcc2.device
        )
        mfcc2_aligned = torch.cat((pad_start, mfcc2), dim=1)

    elif offset_frames < 0:
        # Video 2 starts BEFORE Video 1 → remove extra beginning by padding at end
        mfcc2_aligned = mfcc2[:, -offset_frames:]  # keep frames after offset
    else:
        mfcc2_aligned = mfcc2

    # Pad mfcc2 at the end if shorter than mfcc1
    if mfcc2_aligned.shape[1] < mfcc1.shape[1]:
        mfcc2_aligned = torch.nn.functional.pad(
            mfcc2_aligned,
            (0, mfcc1.shape[1] - mfcc2_aligned.shape[1]),  # pad at end
            mode="constant",
            value=0.0
        )

    mfcc1_aligned = mfcc1  # mfcc1 stays fixed

    # Create aligned merged MFCC
    merged_mfcc_aligned = merge_interleave_mfcc(mfcc1_aligned, mfcc2_aligned)

    print(f"Aligned merged MFCC shape: {merged_mfcc_aligned.shape}")

    # Create individual plots and save as PNG files

    # Plot 1: MFCC - Video 1
    print("\nCreating individual plots...")
    fig1 = go.Figure()
    fig1.add_trace(
        go.Heatmap(
            z=mfcc1.cpu().numpy(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='MFCC Value'),
            hovertemplate='Frame: %{x}<br>Coefficient: %{y}<br>Value: %{z}<extra></extra>'
        )
    )
    fig1.update_layout(
        title='MFCC - Video 1',
        xaxis_title='Frames',
        yaxis_title='MFCC Coefficients',
        height=500,
        width=1400,
        font=dict(size=12)
    )
    output_path = os.path.join(output_dir, "mfcc_video1.png")
    fig1.write_image(output_path)
    print(f"Saved: {output_path}")

    # Plot 2: MFCC - Video 2
    fig2 = go.Figure()
    fig2.add_trace(
        go.Heatmap(
            z=mfcc2.cpu().numpy(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='MFCC Value'),
            hovertemplate='Frame: %{x}<br>Coefficient: %{y}<br>Value: %{z}<extra></extra>'
        )
    )
    fig2.update_layout(
        title='MFCC - Video 2',
        xaxis_title='Frames',
        yaxis_title='MFCC Coefficients',
        height=500,
        width=1400,
        font=dict(size=12)
    )
    output_path = os.path.join(output_dir, "mfcc_video2.png")
    fig2.write_image(output_path)
    print(f"Saved: {output_path}")

    if PLOT_STFT:
        # Plot 3: STFT - Video 1
        print("\nPlotting STFT for Video 1...")
        fig3 = go.Figure()
        stft1_db = 20 * torch.log10(stft1_plot + 1e-10)  # Convert to dB scale

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
        output_path = os.path.join(output_dir, "stft_video1.png")
        fig3.write_image(output_path)
        print(f"Saved: {output_path}")

        # Plot 4: STFT - Video 2
        print("Plotting STFT for Video 2...")
        fig4 = go.Figure()
        stft2_db = 20 * torch.log10(stft2_plot + 1e-10)  # Convert to dB scale

        fig4.add_trace(
            go.Heatmap(
                z=stft2_db.cpu().numpy(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Magnitude (dB)'),
                hovertemplate='Frame: %{x}<br>Frequency Bin: %{y}<br>Magnitude: %{z:.2f} dB<extra></extra>'
            )
        )
        fig4.update_layout(
            title='STFT Spectrogram - Video 2',
            xaxis_title='Frames',
            yaxis_title='Frequency Bins',
            height=500,
            width=1400,
            font=dict(size=12)
        )
        output_path = os.path.join(output_dir, "stft_video2.png")
        fig4.write_image(output_path)
        print(f"Saved: {output_path}")
    else:
        print("\nSkipping STFT plotting (PLOT_STFT=False)")
        stft1_db = 20 * torch.log10(stft1_plot + 1e-10)
        stft2_db = 20 * torch.log10(stft2_plot + 1e-10)

    # Plot 5: Merged & Interleaved MFCC (Unaligned)
    fig5 = go.Figure()
    fig5.add_trace(
        go.Heatmap(
            z=merged_mfcc.cpu().numpy(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='MFCC Value'),
            hovertemplate='Frame: %{x}<br>Interleaved Coeff: %{y}<br>Value: %{z}<extra></extra>'
        )
    )
    fig5.update_layout(
        title='Merged & Interleaved MFCC (Unaligned)',
        xaxis_title='Frames',
        yaxis_title='Interleaved Coefficients',
        height=500,
        width=1400,
        font=dict(size=12)
    )
    output_path = os.path.join(output_dir, "mfcc_merged_unaligned.png")
    fig5.write_image(output_path)
    print(f"Saved: {output_path}")

    # Plot 6: Cross-Correlation
    time_lags = np.arange(len(cross_corr)) - padding_size
    fig6 = go.Figure()
    fig6.add_trace(
        go.Scatter(
            x=time_lags,
            y=cross_corr.cpu().numpy(),
            mode='lines',
            name='Cross-correlation',
            line=dict(color='blue')
        )
    )
    fig6.add_vline(
        x=time_offset_frames.item(),
        line_dash="dash",
        line_color="red",
        line_width=2,
        annotation_text=f'Max at {time_offset_frames.item()} frames'
    )
    fig6.update_layout(
        title='Cross-Correlation',
        xaxis_title='Time Lag (frames)',
        yaxis_title='Correlation',
        height=400,
        width=1400,
        font=dict(size=12),
        showlegend=False
    )
    output_path = os.path.join(output_dir, "cross_correlation.png")
    fig6.write_image(output_path)
    print(f"Saved: {output_path}")

    # Plot 7: Merged & Interleaved MFCC (Aligned)
    fig7 = go.Figure()
    fig7.add_trace(
        go.Heatmap(
            z=merged_mfcc_aligned.cpu().numpy(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='MFCC Value'),
            hovertemplate='Frame: %{x}<br>Interleaved Coeff: %{y}<br>Value: %{z}<extra></extra>'
        )
    )
    fig7.update_layout(
        title=f'Merged & Interleaved MFCC (Aligned - Offset: {time_offset_seconds:.3f}s)',
        xaxis_title='Frames',
        yaxis_title='Interleaved Coefficients',
        height=500,
        width=1400,
        font=dict(size=12)
    )
    output_path = os.path.join(output_dir, "mfcc_merged_aligned.png")
    fig7.write_image(output_path)
    print(f"Saved: {output_path}")

    # Create combined subplots for interactive viewing
    if PLOT_STFT:
        num_rows = 7
        subplot_titles = (
            'MFCC - Video 1',
            'MFCC - Video 2',
            'STFT Spectrogram - Video 1',
            'STFT Spectrogram - Video 2',
            'Merged & Interleaved MFCC (Unaligned)',
            'Cross-Correlation',
            f'Merged & Interleaved MFCC (Aligned - Offset: {time_offset_seconds:.3f}s)'
        )
        row_heights = [0.14, 0.14, 0.14, 0.14, 0.14, 0.12, 0.14]
        total_height = 1800
    else:
        num_rows = 5
        subplot_titles = (
            'MFCC - Video 1',
            'MFCC - Video 2',
            'Merged & Interleaved MFCC (Unaligned)',
            'Cross-Correlation',
            f'Merged & Interleaved MFCC (Aligned - Offset: {time_offset_seconds:.3f}s)'
        )
        row_heights = [0.2, 0.2, 0.2, 0.15, 0.2]
        total_height = 1400

    fig = make_subplots(
        rows=num_rows, cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
        row_heights=row_heights
    )

    # Plot MFCC 1
    fig.add_trace(
        go.Heatmap(
            z=mfcc1.cpu().numpy(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                len=0.13,
                y=0.945,
                title='MFCC'
            ),
            hovertemplate='Frame: %{x}<br>Coefficient: %{y}<br>Value: %{z}<extra></extra>'
        ),
        row=1, col=1
    )

    # Plot MFCC 2
    fig.add_trace(
        go.Heatmap(
            z=mfcc2.cpu().numpy(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                len=0.13 if PLOT_STFT else 0.18,
                y=0.81 if PLOT_STFT else 0.73,
                title='MFCC'
            ),
            hovertemplate='Frame: %{x}<br>Coefficient: %{y}<br>Value: %{z}<extra></extra>'
        ),
        row=2, col=1
    )

    current_row = 3

    if PLOT_STFT:
        # Plot STFT 1
        fig.add_trace(
            go.Heatmap(
                z=stft1_db.cpu().numpy(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    len=0.13,
                    y=0.675,
                    title='Magnitude (dB)'
                ),
                hovertemplate='Frame: %{x}<br>Frequency Bin: %{y}<br>Magnitude: %{z:.2f} dB<extra></extra>'
            ),
            row=current_row, col=1
        )
        current_row += 1

        # Plot STFT 2
        fig.add_trace(
            go.Heatmap(
                z=stft2_db.cpu().numpy(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    len=0.13,
                    y=0.54,
                    title='Magnitude (dB)'
                ),
                hovertemplate='Frame: %{x}<br>Frequency Bin: %{y}<br>Magnitude: %{z:.2f} dB<extra></extra>'
            ),
            row=current_row, col=1
        )
        current_row += 1

    # Plot Unaligned Merged MFCC
    fig.add_trace(
        go.Heatmap(
            z=merged_mfcc.cpu().numpy(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                len=0.13 if PLOT_STFT else 0.18,
                y=0.405 if PLOT_STFT else 0.54,
                title='MFCC'
            ),
            hovertemplate='Frame: %{x}<br>Interleaved Coeff: %{y}<br>Value: %{z}<extra></extra>'
        ),
        row=current_row, col=1
    )
    current_row += 1

    # Plot cross-correlation
    fig.add_trace(
        go.Scatter(
            x=time_lags,
            y=cross_corr.cpu().numpy(),
            mode='lines',
            name='Cross-correlation',
            showlegend=False
        ),
        row=current_row, col=1
    )

    # Add max correlation line
    fig.add_vline(
        x=time_offset_frames.item(), line_dash="dash", line_color="red", line_width=2,
        annotation_text=f'Max at {time_offset_frames.item()} frames',
        row=current_row, col=1
    )
    current_row += 1

    # Plot Aligned Merged MFCC
    fig.add_trace(
        go.Heatmap(
            z=merged_mfcc_aligned.cpu().numpy(),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                len=0.13 if PLOT_STFT else 0.18,
                y=0.07 if PLOT_STFT else 0.1,
                title='MFCC'
            ),
            hovertemplate='Frame: %{x}<br>Interleaved Coeff: %{y}<br>Value: %{z}<extra></extra>'
        ),
        row=current_row, col=1
    )

    # Update axes
    fig.update_xaxes(title_text="Frames", row=1, col=1)
    fig.update_yaxes(title_text="MFCC Coefficients", row=1, col=1)

    fig.update_xaxes(title_text="Frames", row=2, col=1)
    fig.update_yaxes(title_text="MFCC Coefficients", row=2, col=1)

    current_row = 3
    if PLOT_STFT:
        fig.update_xaxes(title_text="Frames", row=current_row, col=1)
        fig.update_yaxes(title_text="Frequency Bins", row=current_row, col=1)
        current_row += 1

        fig.update_xaxes(title_text="Frames", row=current_row, col=1)
        fig.update_yaxes(title_text="Frequency Bins", row=current_row, col=1)
        current_row += 1

    fig.update_xaxes(title_text="Frames", row=current_row, col=1)
    fig.update_yaxes(title_text="Interleaved Coefficients", row=current_row, col=1)
    current_row += 1

    fig.update_xaxes(title_text="Time Lag (frames)", row=current_row, col=1)
    fig.update_yaxes(title_text="Correlation", row=current_row, col=1)
    current_row += 1

    fig.update_xaxes(title_text="Frames", row=current_row, col=1)
    fig.update_yaxes(title_text="Interleaved Coefficients", row=current_row, col=1)

    # Update layout
    fig.update_layout(
        height=total_height,
        width=1400,
        showlegend=False,
        font=dict(size=12)
    )

    fig.show()

    print(f"\nSynchronization result:")
    print(f"Video 2 is offset by {time_offset_seconds:.3f} seconds relative to Video 1")
    if time_offset_seconds > 0:
        print(f"Video 2 starts {time_offset_seconds:.3f} seconds AFTER Video 1")
    else:
        print(f"Video 2 starts {abs(time_offset_seconds):.3f} seconds BEFORE Video 1")

    print(f"\nAll individual plots saved to: {output_dir}")