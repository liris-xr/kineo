"""Renders where the estimated distortion is wrong, view by view.

Writes one <sequence>_<view>_error.png per view and a shared colorbar, so that
the regions people were observed in can be read against the regions the
distortion is accurate in.

Usage:
    python experiments/figures/generate_distortion_error_maps.py \
        <egohumans_dataset_dir> <pred_annotations_dir> <output_dir>
"""

import argparse
import os

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from distortion_error import (
    iter_view_errors,
    load_middle_frame,
    load_sequences,
    overlay_error,
)

# Views rendered by default, matching generate_distortions_figure.py.
VIEWS = {
    ("tagging_001", "cam03"),
    ("legoassemble_001", "cam01"),
    ("fencing_002", "cam02"),
    ("basketball_001", "cam01"),
    ("volleyball_001", "cam05"),
    ("badminton_001", "cam15"),
    ("tennis_001", "cam01"),
}


def save_colorbar(filepath: str, vmax: float):
    """Writes the colorbar the fixed scale corresponds to."""
    fig, ax = plt.subplots(figsize=(1.2, 4))
    colorbar = plt.colorbar(
        cm.ScalarMappable(norm=colors.Normalize(0.0, vmax), cmap="turbo"), cax=ax
    )
    colorbar.set_label("Distortion error (px)")
    fig.savefig(filepath, bbox_inches="tight", dpi=150)
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
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    parser.add_argument(
        "--all", action="store_true", help="Render every view of every sequence."
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=30.0,
        help="Error the color scale saturates at, in pixels, shared across "
        "views. Pass 0 to scale each view on its own 99th percentile.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    view_errors = iter_view_errors(
        dataset_dir=args.egohumans_dataset_dir,
        pred_annotations_dir=args.pred_annotations_dir,
        sequences=load_sequences(args.egohumans_dataset_dir),
        mask_source=args.mask_source,
        views=None if args.all else VIEWS,
        radius=args.kp_radius,
        score_threshold=args.kp_score_thr,
    )

    for view_error in tqdm(view_errors, desc="Views"):
        try:
            frame = load_middle_frame(
                args.egohumans_dataset_dir, view_error.images_dir
            )
        except FileNotFoundError as error:
            tqdm.write(f"skip {view_error.sequence_name}: {error}")
            continue

        finite = view_error.error[np.isfinite(view_error.error)]
        vmax = args.vmax or float(np.percentile(finite, 99))
        overlay = overlay_error(
            frame,
            view_error.error,
            view_error.observed,
            vmax=vmax,
            alpha=args.overlay_alpha,
        )

        filename = f"{view_error.sequence_name}_{view_error.view_id}_error.png"
        cv2.imwrite(
            os.path.join(args.output_dir, filename),
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        tqdm.write(
            f"{view_error.sequence_name}/{view_error.view_id}: vmax={vmax:.2f}px "
            f"observed={view_error.observed.mean() * 100:.1f}%"
        )

    if args.vmax:
        save_colorbar(os.path.join(args.output_dir, "colorbar.png"), args.vmax)


if __name__ == "__main__":
    main()
