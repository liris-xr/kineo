"""
Task 3: distortion error inside the observed (people) mask vs over the whole
image, aggregated across ALL sequences/views.

Quantifies whether the estimated distortion is accurate where people are
observed (keypoints-union mask) but degrades in unobserved regions.

Outputs:
  - distortion_error_stats.csv    per (sequence, view) row of stats
  - distortion_error_masked_vs_full.png  summary bar chart

Usage:
    pixi run python experiments/figures/generate_distortion_error_stats.py \
        <egohumans_dataset_dir> <pred_annotations_dir> <output_dir> [--limit N]
"""

import argparse
import csv
import os

import numpy as np
import orjson
import torch
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from distortion_error_utils import (
    build_ba_keypoints_mask,
    build_keypoints_mask,
    compute_distortion_error_map,
    error_stats,
    load_ba_keypoints,
    load_gt_camera_intrinsics,
    load_pred_camera_intrinsics,
    load_pred_keypoints_2d,
)

CSV_FIELDS = [
    "sequence", "view", "mask_coverage",
    "full_mean", "full_median", "full_p95", "full_max",
    "masked_mean", "masked_median", "masked_p95", "masked_max",
    "unmasked_mean", "unmasked_median", "unmasked_p95", "unmasked_max",
]


def summary_figure(rows: list[dict], path: str):
    """Grouped bar chart: masked vs unmasked vs full median error, per sequence.

    Uses the *median* stat (robust): a few sequences have divergent undistort at
    the periphery (failed calibration) that make the mean meaningless.
    """
    # Aggregate per sequence (median over its views' medians).
    seqs = sorted({r["sequence"] for r in rows})
    masked = [np.median([r["masked_median"] for r in rows if r["sequence"] == s]) for s in seqs]
    unmasked = [np.median([r["unmasked_median"] for r in rows if r["sequence"] == s]) for s in seqs]
    full = [np.median([r["full_median"] for r in rows if r["sequence"] == s]) for s in seqs]

    x = np.arange(len(seqs))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(seqs) * 0.35), 5))
    ax.bar(x - w, masked, w, label="People (BA mask)", color="#2ca02c")
    ax.bar(x, unmasked, w, label="Background", color="#d62728")
    ax.bar(x + w, full, w, label="Whole image", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(seqs, rotation=90, fontsize=6)
    ax.set_ylabel("Median distortion error (px)")
    ax.set_title("Distortion error: observed (BA) vs background vs whole image")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("egohumans_dataset_dir")
    parser.add_argument("pred_annotations_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--kp-radius", type=int, default=25)
    parser.add_argument("--kp-score-thr", type=float, default=0.3)
    parser.add_argument(
        "--mask-source", choices=["ba", "detections"], default="ba",
        help="ba: bundle-adjustment keypoints (constrained calibration). "
        "detections: all NLF 2D detections.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N sequences (debug).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    sequences_file = os.path.join(
        args.egohumans_dataset_dir, "egohumans_sequences.json"
    )
    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())
    if args.limit:
        sequences = sequences[: args.limit]

    rows = []
    for sequence in tqdm(sequences, desc="Sequences"):
        seq_name = sequence["sequence_name"]
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
            st = error_stats(err, mask)
            st.update({"sequence": seq_name, "view": view_id})
            rows.append(st)

    if not rows:
        print("No rows computed.")
        return

    csv_path = os.path.join(args.output_dir, "distortion_error_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in CSV_FIELDS})

    fig_path = os.path.join(args.output_dir, "distortion_error_masked_vs_full.png")
    summary_figure(rows, fig_path)

    # Global summary. Use median across views (robust to divergent failures).
    def agg(key):
        return float(np.median([r[key] for r in rows]))

    print("\n=== Distortion error (px), median over all views ===")
    print(f"  People (BA):  median={agg('masked_median'):.3f}  p95={agg('masked_p95'):.3f}")
    print(f"  Background:   median={agg('unmasked_median'):.3f}  p95={agg('unmasked_p95'):.3f}")
    print(f"  Whole image:  median={agg('full_median'):.3f}  p95={agg('full_p95'):.3f}")
    print(f"  Mask coverage: {agg('mask_coverage') * 100:.1f}%")

    # Flag views whose calibration diverges (undistort blows up at periphery).
    broken = [r for r in rows if float(r["full_median"]) > 50.0]
    if broken:
        print(f"\n  {len(broken)} views with full_median > 50px (likely failed calibration):")
        for r in sorted(broken, key=lambda r: -float(r["full_median"]))[:15]:
            print(f"    {r['sequence']}/{r['view']}: median={float(r['full_median']):.0f}px "
                  f"p95={float(r['full_p95']):.0f}px")
    print(f"\nWrote {csv_path}\nWrote {fig_path}")


if __name__ == "__main__":
    main()
