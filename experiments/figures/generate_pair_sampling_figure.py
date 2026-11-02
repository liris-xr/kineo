# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Generate the EgoHumans pair sampling figure for one camera pair."""

import argparse
import os
import pickle
import statistics
from types import SimpleNamespace

import matplotlib
import numpy as np
import torch
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from omegaconf import OmegaConf

from kineo.annotations import Keypoints2DAnnotations
from kineo.annotations.calibration_points_pairs import (
    CalibrationPointsAnnotation,
    CalibrationPointsAnnotations,
)
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotations
from kineo.datasets.egohumans.egohumans_dataset import EgoHumansSequenceDataset
from kineo.pipeline.stages.keypoints_pairs_sampling import (
    KeypointsPairsSamplingRuntimeConfig,
    KeypointsPairsSamplingStage,
)

ARMS = {
    "Random": "configs/experiments/benchmarks/egohumans_benchmark_nlf_estRt_estK_estD.yaml",
    "FPS": "configs/experiments/ablation_study/sampling/egohumans_sampling_fps.yaml",
}
SAVED_ANNOTATIONS = {
    "calibration_points": CalibrationPointsAnnotations,
    "keypoints_2d": Keypoints2DAnnotations,
    "global_time_reference": GlobalTimeReferenceAnnotations,
    "cameras_intrinsics": CameraIntrinsicsAnnotations,
}
ROW_ORDER = ("FPS", "Top-P", "Random")
ROW_NAMES = {
    "FPS": "Farthest-point (FPS)",
    "Top-P": "Confidence-only (Top-P)",
    "Random": "Random (ours)",
}
CANDIDATE_COLOR = "#D0D0D0"
PICK_COLOR = plt.get_cmap("viridis")(1.0)
HULL_COLOR = "white"
OUTSIDE_BRIGHTNESS = 0.28
INSIDE_BRIGHTNESS = 0.95
MISSED_COLOR = "#CC79A7"
MISSED_ALPHA = 0.65
# Marker areas in points squared
CANDIDATE_SIZE = 0.5
PICK_SIZE = 0.7
FONT_SIZE = 8
ANNOTATION_FONT_SIZE = 7.5
# Figure layout in inches
FIGURE_WIDTH = 7.0
ROW_LABEL_WIDTH = 0.17
COLUMN_HEADER_HEIGHT = 0.17
LEGEND_HEIGHT = 0.22
GAP = 0.04


def sampler_settings() -> tuple[float, float, float]:
    stages = {
        arm: OmegaConf.load(config_file).pipeline.stages.keypoints_pairs_sampling.runtime_cfg
        for arm, config_file in ARMS.items()
    }
    thresholds = {cfg.pair_avg_conf_score_thr for cfg in stages.values()}
    if len(thresholds) != 1:
        raise ValueError("Both arms must filter with the same threshold")
    fps = stages["FPS"]
    return thresholds.pop(), fps.image_plane_weight, fps.time_weight


def load_saved(run_dir: str, sequence_name: str, name: str):
    path = os.path.join(run_dir, "annotations", sequence_name, f"{name}.pkl")
    with open(path, "rb") as f:
        return SAVED_ANNOTATIONS[name].from_dict(pickle.load(f))


def candidate_pool(
    sequence_name: str, annotations: dict, view_ids: list[str], threshold: float
) -> dict:
    stage = KeypointsPairsSamplingStage(
        name="Candidate Pool",
        order=0,
        runtime_cfg=KeypointsPairsSamplingRuntimeConfig(
            max_points_pairs=-1,
            pair_avg_conf_score_thr=threshold,
        ),
    )
    pool = dict(annotations)
    stage.forward(
        sequence_name=sequence_name,
        # The stage only reads the pipeline device
        pipeline=SimpleNamespace(device=torch.device("cpu")),
        views=[{"view_id": view_id} for view_id in view_ids],
        annotations=pool,
        gt_annotations={},
        runtime_cfg=stage.runtime_cfg,
    )
    return pool


def pair_annotation(
    annotations: dict, view1_id: str, view2_id: str
) -> CalibrationPointsAnnotation:
    return next(
        a
        for a in annotations["calibration_points"]
        if a.view1_id == view1_id and a.view2_id == view2_id
    )


def background_frame(view, global_time_reference, view_id: str) -> np.ndarray:
    local_indices = global_time_reference.closest_local_frame_idx[view_id]
    local_index = int(local_indices[len(local_indices) // 2])
    frame = view["frame_loader"].load_frame_at(local_index)
    return frame.permute(1, 2, 0).cpu().numpy()


def hull_share(picks: np.ndarray, candidates: np.ndarray) -> float:
    if len(picks) < 3:
        return 0.0
    return float(ConvexHull(picks).volume / ConvexHull(candidates).volume)


def full_frame(resolution_hw: tuple[int, int]) -> tuple:
    h, w = resolution_hw
    return 0, w, 0, h


def hull_loop(points: np.ndarray) -> np.ndarray:
    vertices = ConvexHull(points).vertices
    return points[np.append(vertices, vertices[0])]


def points_per_image(
    run_dirs: dict[str, str], sequence_name: str
) -> tuple[dict, dict[str, dict], dict]:
    reference = {
        name: load_saved(run_dirs["Random"], sequence_name, name)
        for name in SAVED_ANNOTATIONS
    }
    view_ids = [a.view_id for a in reference["cameras_intrinsics"]]
    threshold, _, _ = sampler_settings()
    pool = candidate_pool(sequence_name, reference, view_ids, threshold)
    saved = {
        arm: {"calibration_points": load_saved(run_dir, sequence_name, "calibration_points")}
        for arm, run_dir in run_dirs.items()
    }

    candidates, picks = {}, {arm: {} for arm in run_dirs}
    for annot in pool["calibration_points"]:
        if annot.view1_id >= annot.view2_id or len(annot.points1) < 3:
            continue
        pair = f"{annot.view1_id}-{annot.view2_id}"
        candidates[f"{pair}/{annot.view1_id}"] = annot.points1.numpy()
        candidates[f"{pair}/{annot.view2_id}"] = annot.points2.numpy()
        for arm, annotations in saved.items():
            picked = pair_annotation(annotations, annot.view1_id, annot.view2_id)
            picks[arm][f"{pair}/{annot.view1_id}"] = picked.points1.numpy()
            picks[arm][f"{pair}/{annot.view2_id}"] = picked.points2.numpy()
    return candidates, picks, reference


def image_share(points: np.ndarray, resolution_hw: tuple[int, int]) -> float:
    h, w = resolution_hw
    return float(ConvexHull(points).volume / (w * h))


def missed_region(candidates: np.ndarray, picks: np.ndarray) -> PathPatch:
    outer, inner = hull_loop(candidates), hull_loop(picks)[::-1]
    vertices = np.concatenate([outer, inner])
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[len(outer)] = Path.MOVETO
    codes[len(outer) - 1] = codes[-1] = Path.CLOSEPOLY
    return PathPatch(
        Path(vertices, codes), facecolor=(*matplotlib.colors.to_rgb(MISSED_COLOR), MISSED_ALPHA),
        edgecolor=MISSED_COLOR, linewidth=0.5,
    )


def scaled(image: np.ndarray, brightness: float) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image * brightness, 0.0, 1.0)


def draw_panel(ax, image, resolution_hw, candidates, picks):
    ax.imshow(scaled(image, OUTSIDE_BRIGHTNESS))
    inside = ax.imshow(scaled(image, INSIDE_BRIGHTNESS))
    loop = hull_loop(candidates)
    inside.set_clip_path(Path(loop), transform=ax.transData)
    ax.add_patch(missed_region(candidates, picks))
    ax.scatter(
        candidates[:, 0], candidates[:, 1], s=CANDIDATE_SIZE, c=CANDIDATE_COLOR,
        alpha=0.5, linewidths=0, rasterized=True,
    )
    ax.scatter(
        picks[:, 0], picks[:, 1], s=PICK_SIZE, color=PICK_COLOR, alpha=0.9,
        linewidths=0, rasterized=True,
    )
    loop = hull_loop(picks)
    ax.plot(
        loop[:, 0], loop[:, 1], color=PICK_COLOR, linewidth=1.2,
        path_effects=[path_effects.withStroke(linewidth=2.0, foreground="black")],
    )
    # Drawn last to stay above the selection hull
    loop = hull_loop(candidates)
    ax.plot(
        loop[:, 0], loop[:, 1], color=HULL_COLOR, linewidth=0.8,
        linestyle=(0, (3, 2)),
        path_effects=[path_effects.withStroke(linewidth=1.6, foreground="black")],
    )

    x0, x1, y0, y1 = full_frame(resolution_hw)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_axis_off()
    ax.text(
        0.985, 0.03,
        f"Selection covers {100 * hull_share(picks, candidates):.0f}% of candidate hull\n"
        f"Candidates cover {100 * image_share(candidates, resolution_hw):.0f}% of image",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=ANNOTATION_FONT_SIZE,
        multialignment="right", linespacing=1.3,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55,
                  edgecolor="none"),
    )


def draw_figure(
    view_ids: list[str],
    picks: dict[str, dict[str, np.ndarray]],
    candidates: dict[str, np.ndarray],
    images: dict[str, np.ndarray],
    resolutions: dict[str, tuple[int, int]],
    output_file: str,
):
    aspects = [resolutions[v][1] / resolutions[v][0] for v in view_ids]
    view_h = (FIGURE_WIDTH - ROW_LABEL_WIDTH - GAP) / sum(aspects)
    rows_h = len(picks) * view_h + (len(picks) - 1) * GAP
    fig_h = COLUMN_HEADER_HEIGHT + rows_h + LEGEND_HEIGHT
    fig = plt.figure(figsize=(FIGURE_WIDTH, fig_h))
    to_fig = lambda x, y, w, h: (x / FIGURE_WIDTH, y / fig_h, w / FIGURE_WIDTH, h / fig_h)

    top = fig_h - COLUMN_HEADER_HEIGHT
    arms = sorted(picks, key=ROW_ORDER.index)
    for row, arm in enumerate(arms):
        arm_picks = picks[arm]
        bottom = top - (row + 1) * view_h - row * GAP
        fig.text(
            *to_fig(ROW_LABEL_WIDTH / 2, bottom + view_h / 2, 0, 0)[:2],
            f"({chr(ord('a') + row)}) {ROW_NAMES[arm]}", rotation=90, ha="center", va="center",
            fontsize=FONT_SIZE,
        )
        left = ROW_LABEL_WIDTH
        for column, (view_id, aspect) in enumerate(zip(view_ids, aspects)):
            width = view_h * aspect
            ax = fig.add_axes(to_fig(left, bottom, width, view_h))
            draw_panel(
                ax, images[view_id], resolutions[view_id], candidates[view_id],
                arm_picks[view_id],
            )
            if row == 0:
                fig.text(
                    *to_fig(left + width / 2, top + COLUMN_HEADER_HEIGHT / 2, 0, 0)[:2],
                    f"View {column + 1}",
                    ha="center", va="center", fontsize=FONT_SIZE,
                )
            left += width + GAP

    n_picks = len(next(iter(picks["Random"].values())))
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=4,
               markerfacecolor=CANDIDATE_COLOR, markeredgecolor="0.5",
               markeredgewidth=0.3, label="Candidates"),
        Line2D([], [], color=HULL_COLOR, linewidth=0.8, linestyle=(0, (3, 2)),
               path_effects=[path_effects.withStroke(linewidth=1.6, foreground="black")],
               label="Candidate hull"),
        Line2D([], [], marker="o", linestyle="none", markersize=4,
               markerfacecolor=PICK_COLOR, markeredgecolor="none",
               label=f"Selected ($P={n_picks}$)"),
        Line2D([], [], color=PICK_COLOR, linewidth=1.2,
               path_effects=[path_effects.withStroke(linewidth=2.0, foreground="black")],
               label="Selection hull"),
        Patch(facecolor=(*matplotlib.colors.to_rgb(MISSED_COLOR), MISSED_ALPHA),
              edgecolor=MISSED_COLOR, linewidth=0.5,
              label="Missed by the selection"),
    ]
    fig.legend(
        handles=handles, loc="center", ncol=len(handles), frameon=False,
        fontsize=FONT_SIZE, handlelength=1.8, handletextpad=0.5, columnspacing=1.6,
        bbox_to_anchor=to_fig(ROW_LABEL_WIDTH, 0, FIGURE_WIDTH - ROW_LABEL_WIDTH, LEGEND_HEIGHT),
    )

    fig.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def load_pair(
    run_dirs: dict[str, str], dataset_dir: str, sequence_name: str, pair: str | None
) -> dict:
    candidates, picks, reference = points_per_image(run_dirs, sequence_name)
    gains = {}
    for key, pool in candidates.items():
        gain = hull_share(picks["FPS"][key], pool) - hull_share(picks["Random"][key], pool)
        gains.setdefault(key.split("/")[0], []).append(gain)
    if pair is None:
        pair = max(gains, key=lambda p: statistics.mean(gains[p]))
    view_ids = pair.split("-")
    keys = {v: f"{pair}/{v}" for v in view_ids}

    dataset = EgoHumansSequenceDataset(
        os.path.join(dataset_dir, "egohumans_sequences.json"), device=torch.device("cpu")
    )
    index = next(
        i
        for i, data in enumerate(dataset.sequences_data)
        if data["sequence_name"] == sequence_name
    )
    views_by_id = {view["view_id"]: view for view in dataset[index]["views_inputs"]}
    global_time_reference = reference["global_time_reference"].first_or_default()
    return {
        "pair": pair,
        "view_ids": view_ids,
        "picks": {arm: {v: p[keys[v]] for v in view_ids} for arm, p in picks.items()},
        "candidates": {v: candidates[keys[v]] for v in view_ids},
        "images": {
            v: background_frame(views_by_id[v], global_time_reference, v)
            for v in view_ids
        },
        "resolutions": {
            a.view_id: tuple(a.resolution_hw) for a in reference["cameras_intrinsics"]
        },
    }


def main(
    run_dirs: dict[str, str],
    dataset_dir: str,
    output_dir: str,
    sequence_name: str,
    pair: str | None,
):
    os.makedirs(output_dir, exist_ok=True)
    # Delete this cache to reload the pair
    cache_file = os.path.join(
        output_dir,
        f"pair_sampling_{sequence_name}_{pair or 'largest'}_{len(run_dirs)}arms.pkl",
    )
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
    else:
        data = load_pair(run_dirs, dataset_dir, sequence_name, pair)
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)

    for v in data["view_ids"]:
        pool = data["candidates"][v]
        print(
            f"{v}: candidates cover "
            f"{100 * image_share(pool, data['resolutions'][v]):.1f}% of the image, "
            + "   ".join(
                f"{arm} {100 * hull_share(p[v], pool):.1f}%"
                for arm, p in data["picks"].items()
            )
        )

    draw_figure(
        data["view_ids"],
        data["picks"],
        data["candidates"],
        data["images"],
        data["resolutions"],
        os.path.join(
            output_dir, f"pair_sampling_comparison_{sequence_name}_{data['pair']}.pdf"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("random_run_dir", help="Output directory of the random run")
    parser.add_argument("fps_run_dir", help="Output directory of the FPS run")
    parser.add_argument("dataset_dir", help="EgoHumans dataset directory")
    parser.add_argument("output_dir")
    parser.add_argument("--sequence", default="tennis_005")
    parser.add_argument(
        "--pair",
        help="View pair to plot, as <view1>-<view2>, instead of the one with the "
        "largest difference",
    )
    parser.add_argument(
        "--topp-run-dir", help="Output directory of the confidence-only (top-P) run"
    )
    args = parser.parse_args()

    run_dirs = {"Random": args.random_run_dir, "FPS": args.fps_run_dir}
    if args.topp_run_dir:
        run_dirs["Top-P"] = args.topp_run_dir
    main(
        run_dirs,
        args.dataset_dir,
        args.output_dir,
        args.sequence,
        args.pair,
    )
