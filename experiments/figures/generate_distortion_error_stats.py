"""Reports the distortion error where people were observed and where they were not.

Writes distortion_error_stats.csv, one row per view, and a bar chart of the
median error per sequence. Quantifies whether the estimated distortion is
accurate only where the calibration was constrained.

Usage:
    python experiments/figures/generate_distortion_error_stats.py \
        <egohumans_dataset_dir> <pred_annotations_dir> <output_dir>
"""

import argparse
import csv
import os

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from distortion_error import (
    REGIONS,
    STATS,
    error_stats,
    iter_view_errors,
    load_sequences,
)

CSV_FIELDS = ["sequence", "view", "observed_coverage", "valid_fraction"] + [
    f"{region}_{stat}" for region in REGIONS for stat in STATS
]

# A view whose median error is this large has a diverging undistortion at the
# periphery rather than a slightly wrong distortion.
FAILED_CALIBRATION_PX = 50.0

REGION_COLORS = {
    "observed": "#2ca02c",
    "background": "#d62728",
    "full": "#1f77b4",
}


def save_summary_figure(filepath: str, rows: list[dict]):
    """Bar chart of each sequence's median error, per region.

    Medians rather than means throughout: a handful of views diverge at the
    periphery and would carry the average on their own.
    """
    sequences = sorted({row["sequence"] for row in rows})
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(8, len(sequences) * 0.35), 5))
    for offset, region in zip((-width, 0.0, width), REGIONS):
        medians = [
            np.median(
                [
                    row[f"{region}_median"]
                    for row in rows
                    if row["sequence"] == sequence
                ]
            )
            for sequence in sequences
        ]
        ax.bar(
            np.arange(len(sequences)) + offset,
            medians,
            width,
            label=region,
            color=REGION_COLORS[region],
        )

    ax.set_xticks(np.arange(len(sequences)))
    ax.set_xticklabels(sequences, rotation=90, fontsize=6)
    ax.set_ylabel("Median distortion error (px)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("egohumans_dataset_dir")
    parser.add_argument("pred_annotations_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--kp-radius", type=int, default=25)
    parser.add_argument("--kp-score-thr", type=float, default=0.3)
    parser.add_argument(
        "--mask-source",
        choices=["bundle_adjustment", "detections"],
        default="bundle_adjustment",
        help="Keypoints standing for the observed region: the ones that "
        "constrained the calibration, or every 2D detection.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only the first N sequences."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sequences = load_sequences(args.egohumans_dataset_dir)
    if args.limit:
        sequences = sequences[: args.limit]

    view_errors = iter_view_errors(
        dataset_dir=args.egohumans_dataset_dir,
        pred_annotations_dir=args.pred_annotations_dir,
        sequences=sequences,
        mask_source=args.mask_source,
        radius=args.kp_radius,
        score_threshold=args.kp_score_thr,
    )

    rows = [
        {
            "sequence": view_error.sequence_name,
            "view": view_error.view_id,
            **error_stats(view_error.error, view_error.observed),
        }
        for view_error in tqdm(view_errors, desc="Views")
    ]

    if not rows:
        print("No view produced an error map.")
        return

    csv_filepath = os.path.join(args.output_dir, "distortion_error_stats.csv")
    with open(csv_filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    figure_filepath = os.path.join(args.output_dir, "distortion_error.png")
    save_summary_figure(figure_filepath, rows)

    # AnyCalib reports the median over the dataset of each image's mean error;
    # the per-view median is kept beside it because a fold or a failed
    # calibration reaches the mean and not the median.
    print(f"\nIntrinsics error (px) over {len(rows)} views")
    print(f"  {'region':<12}{'RE':>10}{'median':>10}{'p95':>10}")
    for region in REGIONS:
        print(
            f"  {region:<12}"
            + "".join(
                f"{np.median([row[f'{region}_{stat}'] for row in rows]):>10.3f}"
                for stat in ("mean", "median", "p95")
            )
        )
    coverage = np.median([row["observed_coverage"] for row in rows])
    valid = np.median([row["valid_fraction"] for row in rows])
    print(f"  observed region covers {coverage * 100:.1f}% of the frame")
    print(f"  both cameras called {valid * 100:.1f}% of the samples valid")

    failed = [
        row for row in rows if row["full_median"] > FAILED_CALIBRATION_PX
    ]
    if failed:
        print(f"\n{len(failed)} views above {FAILED_CALIBRATION_PX:.0f}px median:")
        for row in sorted(failed, key=lambda row: -row["full_median"])[:15]:
            print(
                f"  {row['sequence']}/{row['view']}: "
                f"median={row['full_median']:.0f}px p95={row['full_p95']:.0f}px"
            )

    print(f"\nWrote {csv_filepath}\nWrote {figure_filepath}")


if __name__ == "__main__":
    main()
