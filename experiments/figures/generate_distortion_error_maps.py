"""
Task 1 & 2: per-view spatial distribution of the distortion error.

For a selection of (sequence, view) pairs, produce:
  - <seq>_<view>_mask_overlay.png   error colormap on the frame + mask contour
  - colorbar.png                    single shared colorbar (px), fixed vmax

This lets us visually correlate where people are observed (keypoints mask) with
where the estimated distortion is accurate.

Usage:
    pixi run python experiments/figures/generate_distortion_error_maps.py \
        <egohumans_dataset_dir> <pred_annotations_dir> <output_dir>
"""

import argparse
import os

import cv2
import numpy as np
import orjson
import torch
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from distortion_error_utils import (
    build_ba_keypoints_mask,
    build_keypoints_mask,
    colorize_error,
    compute_distortion_error_map,
    draw_mask_contour,
    load_ba_keypoints,
    load_gt_camera_intrinsics,
    load_middle_frame_rgb,
    load_pred_camera_intrinsics,
    load_pred_keypoints_2d,
    overlay_rgb,
)

# (sequence_name, view_id) pairs to render. Matches generate_distortions_figure.py.
SUBSETS = [
    ("tagging_001", "cam03"),
    ("legoassemble_001", "cam01"),
    ("fencing_002", "cam02"),
    ("basketball_001", "cam01"),
    ("volleyball_001", "cam05"),
    ("badminton_001", "cam15"),
    ("tennis_001", "cam01"),
]


def save_colorbar(path: str, vmax: float):
    fig, ax = plt.subplots(figsize=(1.2, 4))
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap="turbo"), cax=ax)
    cb.set_label("Distortion error (px)")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_rgb(path: str, rgb: np.ndarray):
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("egohumans_dataset_dir")
    parser.add_argument("pred_annotations_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--kp-radius", type=int, default=25)
    parser.add_argument("--kp-score-thr", type=float, default=0.3)
    parser.add_argument(
        "--mask-source",
        choices=["ba", "detections"],
        default="ba",
        help="ba: bundle-adjustment keypoints (constrained calibration). "
        "detections: all NLF 2D detections.",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Restrict to these sequence names (default: all in SUBSETS).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export every view of every sequence (ignores SUBSETS).",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="Downscale each saved overlay to this width (keeps aspect). "
        "Recommended with --all to limit disk usage.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=30.0,
        help="Fixed color scale max (px), shared across views for comparability. "
        "Pass 0 for per-image 99th percentile.",
    )
    args = parser.parse_args()

    vmax = None if args.vmax == 0 else args.vmax
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    sequences_file = os.path.join(
        args.egohumans_dataset_dir, "egohumans_sequences.json"
    )
    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())

    subsets = SUBSETS
    if args.sequences:
        subsets = [p for p in SUBSETS if p[0] in set(args.sequences)]
    wanted = set(subsets)
    seq_names = {s for s, _ in subsets}

    for sequence in tqdm(sequences, desc="Sequences"):
        seq_name = sequence["sequence_name"]
        if not args.all and seq_name not in seq_names:
            continue

        try:
            gt_ci_all = load_gt_camera_intrinsics(args.egohumans_dataset_dir, sequence)
            pred_ci_all = load_pred_camera_intrinsics(args.pred_annotations_dir, sequence)
            if args.mask_source == "ba":
                ba_kp = load_ba_keypoints(args.pred_annotations_dir, sequence)
            else:
                pred_kp = load_pred_keypoints_2d(args.pred_annotations_dir, sequence)
        except FileNotFoundError as e:
            tqdm.write(f"skip {seq_name}: {e}")
            continue

        for view_id in sequence["views"]:
            if not args.all and (seq_name, view_id) not in wanted:
                continue

            gt_ci = gt_ci_all.filter_by_view_id(view_id).first_or_default()
            pred_ci = pred_ci_all.filter_by_view_id(view_id).first_or_default()
            if gt_ci is None or pred_ci is None:
                continue
            img_h, img_w = gt_ci.resolution_hw

            err = compute_distortion_error_map(
                gt_ci, pred_ci, img_w, img_h, device=device
            )
            if args.mask_source == "ba":
                mask = build_ba_keypoints_mask(
                    ba_kp, view_id, img_w, img_h, radius=args.kp_radius,
                )
            else:
                mask = build_keypoints_mask(
                    pred_kp, view_id, img_w, img_h,
                    radius=args.kp_radius, score_thr=args.kp_score_thr,
                )
            try:
                frame = load_middle_frame_rgb(
                    args.egohumans_dataset_dir, sequence, view_id
                )
            except FileNotFoundError as e:
                tqdm.write(f"skip {seq_name}/{view_id}: {e}")
                continue
            if frame.shape[:2] != (img_h, img_w):
                frame = cv2.resize(frame, (img_w, img_h))

            color, vmax_used = colorize_error(err, vmax=vmax)
            overlay = overlay_rgb(frame, color, alpha=args.overlay_alpha)
            mask_overlay = draw_mask_contour(overlay, mask)

            if args.output_width and mask_overlay.shape[1] > args.output_width:
                oh = round(args.output_width * mask_overlay.shape[0] / mask_overlay.shape[1])
                mask_overlay = cv2.resize(mask_overlay, (args.output_width, oh))

            base = os.path.join(args.output_dir, f"{seq_name}_{view_id}")
            save_rgb(f"{base}_mask_overlay.png", mask_overlay)

            tqdm.write(
                f"{seq_name}/{view_id}: vmax={vmax_used:.2f}px "
                f"mask_cov={mask.mean() * 100:.1f}%"
            )

    # Single shared colorbar (scale is fixed across all views).
    if vmax is not None:
        save_colorbar(os.path.join(args.output_dir, "colorbar.png"), vmax)
    else:
        print("vmax not fixed (--vmax 0): per-image scale, shared colorbar skipped.")


if __name__ == "__main__":
    main()
