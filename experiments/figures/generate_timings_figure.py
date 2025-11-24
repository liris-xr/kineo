import os
import argparse
import pickle
from kineo.annotations.stage_timing import StageTimingsAnnotations
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotations
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.ticker import FuncFormatter
matplotlib.use("TkAgg")

def compute_sequence_timings(annotations_dir: str) -> dict:

    pred_stage_timings_file = os.path.join(
        annotations_dir, "stage_timings.pkl"
    )
    global_time_reference_file = os.path.join(
        annotations_dir, "global_time_reference.pkl"
    )
    if not os.path.exists(pred_stage_timings_file) or not os.path.exists(global_time_reference_file):
        raise FileNotFoundError(f"Annotations file {pred_stage_timings_file} or {global_time_reference_file} not found")
    with open(pred_stage_timings_file, "rb") as f:
        pred_stage_timings = StageTimingsAnnotations.from_dict(pickle.load(f))
    with open(global_time_reference_file, "rb") as f:
        global_time_reference = GlobalTimeReferenceAnnotations.from_dict(pickle.load(f)).first_or_default()

    sampling_total_duration = 0.0
    intrinsics_init_total_duration = 0.0
    extrinsics_init_total_duration = 0.0
    bundle_adjustment_total_duration = 0.0
    scaling_total_duration = 0.0
    detection_total_duration = 0.0

    detection_stages_names = [
        "MMLab Bbox Keypoints Detection",
        "MMDet Bbox Detection",
        "NLF Keypoints Detection",
    ]

    sampling_stage_name = "Keypoints Pairs Sampling"
    intrinsics_init_stage_name = "MoGe Intrinsics Estimation"
    extrinsics_init_stage_name = "SfM Camera Extrinsics Initialization"
    bundle_adjustment_stages_names = [
        "Bundle Adjustment Sampling",
        "Bundle Adjustment (First Pass)",
        "Bundle Adjustment (Second Pass)",
        "Bundle Adjustment (Third Pass)",
    ]
    scaling_stages_names = [
        "SMPL Global Scale Estimation",
        "Global Scale Application",
    ]

    for stage_timing in pred_stage_timings.annotations:
        if stage_timing.stage_name in detection_stages_names:
            detection_total_duration += stage_timing.duration_seconds
        if stage_timing.stage_name == sampling_stage_name:
            sampling_total_duration += stage_timing.duration_seconds
            detection_total_duration += stage_timing.duration_seconds
        if stage_timing.stage_name == intrinsics_init_stage_name:
            intrinsics_init_total_duration += stage_timing.duration_seconds
        if stage_timing.stage_name == extrinsics_init_stage_name:
            extrinsics_init_total_duration += stage_timing.duration_seconds
        if stage_timing.stage_name in bundle_adjustment_stages_names:
            bundle_adjustment_total_duration += stage_timing.duration_seconds
        if stage_timing.stage_name in scaling_stages_names:
            scaling_total_duration += stage_timing.duration_seconds

    sequence_duration = global_time_reference.timestamps.max().item()

    return {
        "sequence_duration": sequence_duration,
        "intrinsics_init_total_duration": intrinsics_init_total_duration,
        "extrinsics_init_total_duration": extrinsics_init_total_duration,
        "bundle_adjustment_total_duration": bundle_adjustment_total_duration,
        "scaling_total_duration": scaling_total_duration,
        "detection_total_duration": detection_total_duration,
        "sampling_total_duration": sampling_total_duration,
    }

def format_duration(duration: float) -> str:
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    if hours == 0:
        if minutes == 0:
            return f"{seconds:02d}s"
        return f"{minutes:02d}min{seconds:02d}s"
    return f"{hours:02d}h{minutes:02d}min"

def generate_h36m_calibration_runtime_rows(kineo_eval_data_dir: str, output_path: str = "timings_figure.pdf", dpi: int = 300, show_cumulative_seq_duration: bool = True):
    
    kineo_h36m_runtime_rtmpose_short_video_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_runtime_rtmpose_short_video", "annotations", "runtime_eval"
    )
    kineo_h36m_runtime_rtmpose_medium_video_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_runtime_rtmpose_medium_video", "annotations", "runtime_eval"
    )
    kineo_h36m_runtime_rtmpose_long_video_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_runtime_rtmpose_long_video", "annotations", "runtime_eval"
    )

    kineo_h36m_runtime_rtmpose_short_video_b32_half_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_runtime_rtmpose_short_video_b32_half", "annotations", "runtime_eval"
    )
    kineo_h36m_runtime_rtmpose_medium_video_b32_half_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_runtime_rtmpose_medium_video_b32_half", "annotations", "runtime_eval"
    )
    kineo_h36m_runtime_rtmpose_long_video_b32_half_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_runtime_rtmpose_long_video_b32_half", "annotations", "runtime_eval"
    )

    kineo_h36m_runtime_nlf_short_video_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_runtime_nlf_short_video", "annotations", "runtime_eval"
    )
    kineo_h36m_runtime_nlf_medium_video_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_runtime_nlf_medium_video", "annotations", "runtime_eval"
    )
    kineo_h36m_runtime_nlf_long_video_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_runtime_nlf_long_video", "annotations", "runtime_eval"
    )

    short_seq_rtmpose_timings = compute_sequence_timings(kineo_h36m_runtime_rtmpose_short_video_annotations_path)
    medium_seq_rtmpose_timings = compute_sequence_timings(kineo_h36m_runtime_rtmpose_medium_video_annotations_path)
    long_seq_rtmpose_timings = compute_sequence_timings(kineo_h36m_runtime_rtmpose_long_video_annotations_path)
    short_seq_rtmpose_b32_half_timings = compute_sequence_timings(kineo_h36m_runtime_rtmpose_short_video_b32_half_annotations_path)
    medium_seq_rtmpose_b32_half_timings = compute_sequence_timings(kineo_h36m_runtime_rtmpose_medium_video_b32_half_annotations_path)
    long_seq_rtmpose_b32_half_timings = compute_sequence_timings(kineo_h36m_runtime_rtmpose_long_video_b32_half_annotations_path)

    # TODO: update
    # short_seq_nlf_timings = short_seq_rtmpose_timings
    # medium_seq_nlf_timings = medium_seq_rtmpose_timings
    # long_seq_nlf_timings = long_seq_rtmpose_timings
    short_seq_nlf_timings = compute_sequence_timings(kineo_h36m_runtime_nlf_short_video_annotations_path)
    medium_seq_nlf_timings = compute_sequence_timings(kineo_h36m_runtime_nlf_medium_video_annotations_path)
    long_seq_nlf_timings = compute_sequence_timings(kineo_h36m_runtime_nlf_long_video_annotations_path)

    short_seq_duration = short_seq_rtmpose_timings['sequence_duration']
    short_seq_duration_cumulative = short_seq_duration * 4

    medium_seq_duration = medium_seq_rtmpose_timings['sequence_duration']
    medium_seq_duration_cumulative = medium_seq_duration * 4
    
    long_seq_duration = long_seq_rtmpose_timings['sequence_duration']
    long_seq_duration_cumulative = long_seq_duration * 4

    data = {
        "NLF": {
            "S": short_seq_nlf_timings,
            "M": medium_seq_nlf_timings,
            "L": long_seq_nlf_timings,
        },
        "RTMPose": {
            "S": short_seq_rtmpose_timings,
            "M": medium_seq_rtmpose_timings,
            "L": long_seq_rtmpose_timings,
        },
        "RTMPose (B32 Half)": {
            "S": short_seq_rtmpose_b32_half_timings,
            "M": medium_seq_rtmpose_b32_half_timings,
            "L": long_seq_rtmpose_b32_half_timings,
        },
    }

    cumulative_seq_durations = {
        "S": short_seq_duration_cumulative,
        "M": medium_seq_duration_cumulative,
        "L": long_seq_duration_cumulative,
    }

    bar_width = 0.35
    group_margin = 0.2
    subgroup_margin = 0.02

    font_size_subgroups = 42
    font_size_groups = 44
    font_size_total_duration = 20
    font_size_axis_labels = 44
    font_size_axis_ticks = 36
    font_size_legend = 44

    fig_size = (30, 20)

    groups = list(data.keys())
    subgroups = list(data[groups[0]].keys())
    categories_timings_keys = ["intrinsics_init_total_duration", "extrinsics_init_total_duration", "bundle_adjustment_total_duration", "scaling_total_duration", "sampling_total_duration", "detection_total_duration"]
    categories_names = ["Intrinsics Initialization", "Extrinsics Initialization", "Bundle Adjustment", "Global Scaling", "Keypoints Pairs Sampling", "Keypoints Detection"]
    categories_colors = ["#FFB000", "#FE6100", "#DC267F", "#785EF0", "#648FFF", "#C9CDD8"]

    values = np.array([
        data[group][subgroup][category_timings_key] 
        for group in groups 
        for subgroup in subgroups 
        for category_timings_key in categories_timings_keys
    ])
    values = values.reshape(len(groups), len(subgroups), len(categories_timings_keys))

    n_groups = len(groups)
    n_subgroups = len(subgroups)
    x_group_positions = np.arange(n_groups) * (n_subgroups * (bar_width + subgroup_margin) + group_margin)

    fig, (ax_high, ax_low) = plt.subplots(
        2, 1, sharex=True, figsize=fig_size,
        gridspec_kw={'height_ratios':[1,1]}
    )

    low_limit = 0
    high_limit = values.sum(axis=2).max() * 1.1
    split_value = 38 # in seconds

    ax_low.set_ylim(low_limit, split_value)
    ax_high.set_ylim(split_value, high_limit)

    ax_low.spines['top'].set_visible(False)
    ax_high.spines['bottom'].set_visible(False)
    ax_low.tick_params(labeltop=False)
    ax_high.tick_params(labelbottom=False)

    # Diagonal break lines
    d = 0.015
    kwargs = dict(transform=ax_high.transAxes, color='k', clip_on=False, lw=1)
    ax_high.plot((-d, +d), (-d, +d), **kwargs)
    ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs = dict(transform=ax_low.transAxes, color='k', clip_on=False, lw=1)
    ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    for ax in [ax_low, ax_high]:
        for i, group in enumerate(groups):
            for j, subgroup in enumerate(subgroups):
                x = x_group_positions[i] + j * (bar_width + subgroup_margin)
                bottom = 0
                for k, category in enumerate(categories_names):
                    label = category if i == 0 and j == 0 else None
                    color = categories_colors[k]
                    value = values[i, j, k]
                    ax.bar(x, value, width=bar_width, label=label, color=color, bottom=bottom)
                    bottom += value

                if show_cumulative_seq_duration:
                    cumulative_seq_duration = cumulative_seq_durations[subgroup]
                    rect = plt.Rectangle(
                        (x - bar_width / 2, 0),
                        bar_width,
                        cumulative_seq_duration,
                        fill=False,
                        linestyle=":",
                        linewidth=1.2,
                        edgecolor='#34495e',
                        alpha=0.9,
                        zorder=3
                    )
                    ax.add_patch(rect)

                print(group, subgroup, format_duration(cumulative_seq_durations[subgroup]))

                total_estimation_duration = values[i, j, :].sum()
                total_estimation_duration_label_text = format_duration(total_estimation_duration)
                total_time_target_ax = ax_high if total_estimation_duration > split_value else ax_low

                if ax is total_time_target_ax:
                    total_time_target_ax.text(
                        x,
                        total_estimation_duration + 20,
                        total_estimation_duration_label_text,
                        ha='center',
                        va='bottom',
                        fontsize=font_size_total_duration,
                        fontweight='bold',
                        color='#2c3e50'
                    )

    # Plot group and subgroup labels
    for i, group in enumerate(groups):
        group_center = x_group_positions[i] + (n_subgroups * (bar_width + subgroup_margin) - subgroup_margin) / 2 - bar_width / 2

        for j, subgroup in enumerate(subgroups):
            x = x_group_positions[i] + j * (bar_width + subgroup_margin)
            ax_low.annotate(
                subgroup,
                xy=(x, 0),
                xytext=(0, -10),
                textcoords='offset points',
                ha='center',
                va='top',
                fontsize=font_size_subgroups
            )

        ax_low.annotate(
            group,
            xy=(group_center, 0),
            xytext=(0, -15 - font_size_subgroups),  # offset in points
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=font_size_groups,
            fontweight='bold'
        )

    handles, labels = ax_high.get_legend_handles_labels()
    if show_cumulative_seq_duration:
        cumulative_seq_duration_handle = plt.Line2D(
            [0], [0], color='k', linestyle=':', linewidth=1.2, alpha=0.9, zorder=3
        )
        handles.append(cumulative_seq_duration_handle)
        labels.append('Cumulative Sequence Duration (Duration x 4 views)')

    ax_low.set_xticks([])
    ax_high.set_xticks([])

    ax_low.tick_params(axis='y', labelsize=font_size_axis_ticks)
    ax_high.tick_params(axis='y', labelsize=font_size_axis_ticks)
    ax_low.yaxis.set_major_formatter(lambda x, pos: format_duration(x))
    ax_high.yaxis.set_major_formatter(lambda x, pos: format_duration(x))

    fig.text(0.01, 0.5, 'Processing Time', va='center', rotation='vertical', fontsize=font_size_axis_labels, fontweight='bold')

    # ax_high.legend(handles=handles, labels=labels, fontsize=font_size_legend)
    ax_low.grid(True)
    ax_high.grid(True)
    ax_low.set_axisbelow(True)
    ax_high.set_axisbelow(True)
    
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=font_size_legend,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=dpi)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate human metrics benchmark rows for Kineo on EgoHumans and Human3.6M and for HSfM on Human3.6M"
    )
    parser.add_argument(
        "kineo_eval_data_dir", type=str, help="Path to Kineo evaluation data directory"
    )
    parser.add_argument(
        "--output-path", type=str, help="Path to save the figure", default="outputs/runtime/kineo_timings.pdf"
    )
    parser.add_argument(
        "--dpi", type=int, help="DPI of the figure", default=300
    )
    parser.add_argument(
        "--show-cumulative-seq-duration", action="store_true", help="Show cumulative sequence duration"
    )
    args = parser.parse_args()

    generate_h36m_calibration_runtime_rows(args.kineo_eval_data_dir, args.output_path, args.dpi, args.show_cumulative_seq_duration)