# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib import patheffects
import matplotlib.pyplot as plt
import numpy as np
import orjson

ALL_VIEWS = "All"
# HSfM Table S.1 (arXiv 2412.17806), rounded to cm
HSFM_PA_MPJPE_MM = {2: 50.0, 4: 60.0, 8: 50.0}

EST_COLOR, GT_COLOR, HSFM_COLOR = "#0072B2", "#D55E00", "#595959"
SURFACE, INK, MUTED, GRID = "white", "black", "#7a7a7a", "#e0e0e0"
# Figure size in inches
FIGURE_SIZE = (3.33, 2.6)
FONT_SIZE = 7
N_BOOTSTRAP = 10000
SEED = 0


def n_views_of(run_dir: str) -> int | str:
    match = re.search(r"_(\d+)views$", os.path.basename(os.path.normpath(run_dir)))
    return int(match.group(1)) if match else ALL_VIEWS


def load_pa_mpjpe_mm(run_dir: str) -> dict[str, float]:
    out = {}
    for path in glob.glob(os.path.join(run_dir, "metrics", "*.json")):
        with open(path, "rb") as f:
            stats = orjson.loads(f.read())["human_stats"]
        out[os.path.splitext(os.path.basename(path))[0]] = (
            stats["pa-mpjpe"]["mean"] * 1000)
    return out


def bootstrap_mean_ci(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    means = values[..., indices].mean(axis=-1)
    return np.percentile(means, [2.5, 97.5], axis=-1)


def main(output_dir: str, est_runs: list[str], gt_runs: list[str],
         dpi: int) -> None:
    settings = {
        "Ours, estimated intrinsics": (est_runs, EST_COLOR, "o"),
        "Ours, GT intrinsics": (gt_runs, GT_COLOR, "s"),
    }
    results = {name: {n_views_of(d): load_pa_mpjpe_mm(d) for d in runs}
               for name, (runs, _, _) in settings.items()}
    views = sorted(next(iter(results.values())),
                   key=lambda n: (n == ALL_VIEWS, n if n != ALL_VIEWS else 0))
    for name, runs in results.items():
        if sorted(runs, key=str) != sorted(views, key=str):
            raise ValueError(f"{name} does not cover the views {views}")
    common = sorted(set.intersection(
        *(set(r) for runs in results.values() for r in runs.values())))
    values = {name: np.array([[runs[n][s] for s in common] for n in views])
              for name, runs in results.items()}
    indices = np.random.default_rng(SEED).integers(
        0, len(common), size=(N_BOOTSTRAP, len(common)))
    means = {name: v.mean(axis=1) for name, v in values.items()}
    cis = {name: bootstrap_mean_ci(v, indices) for name, v in values.items()}
    print(f"{len(common)} sequences scored by every run")
    for i, n in enumerate(views):
        print(f"{n!s:>3} views  " + "  ".join(
            f"{name}: {means[name][i]:.2f} mm "
            f"[{cis[name][0, i]:.2f}, {cis[name][1, i]:.2f}]"
            for name in settings))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT_SIZE,
        "text.color": INK, "axes.labelcolor": INK,
        # TrueType instead of Type 3, required by ACM
        "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(length=0, colors=INK, labelsize=FONT_SIZE)
    halo = [patheffects.withStroke(linewidth=3, foreground=SURFACE)]

    xs = np.arange(len(views))
    has_all = ALL_VIEWS in views
    n_fixed = len(views) - has_all
    for name, (_, color, marker) in settings.items():
        mean, ci = means[name], cis[name]
        ax.fill_between(xs[:n_fixed], ci[0, :n_fixed], ci[1, :n_fixed],
                        color=color, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(xs[:n_fixed], mean[:n_fixed], color=color, linewidth=2,
                zorder=3)
        if has_all:
            ax.vlines(xs[-1], ci[0, -1], ci[1, -1], color=color, alpha=0.35,
                      linewidth=5, zorder=1)
            ax.plot(xs[-2:], mean[-2:], color=color, linewidth=2,
                    linestyle=(0, (2, 1.5)), zorder=3)
        ax.plot(xs, mean, linestyle="none", marker=marker, markersize=7,
                color=color, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=4, label=name)

    hsfm_x = [views.index(n) for n in HSFM_PA_MPJPE_MM if n in views]
    hsfm_y = [HSFM_PA_MPJPE_MM[views[x]] for x in hsfm_x]
    ax.plot(hsfm_x, hsfm_y, color=HSFM_COLOR, linewidth=1.2,
            linestyle=(0, (1, 1.5)), zorder=2)
    ax.plot(hsfm_x, hsfm_y, linestyle="none", marker="D", markersize=6,
            color=HSFM_COLOR, markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=4, label="HSfM")
    for x, y in zip(hsfm_x, hsfm_y):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", va="center", fontsize=FONT_SIZE,
                    color=HSFM_COLOR, zorder=5, path_effects=halo)

    for i, n in enumerate(views):
        higher = max(settings, key=lambda name: means[name][i])
        for name, (_, color, _) in settings.items():
            is_higher = name == higher
            if n == 2:
                offset, ha = ((8, 3), "left") if is_higher else ((-7, 0), "right")
            elif n == 3:
                offset, ha = ((7, 5), "left") if is_higher else ((-7, -5), "right")
            else:
                offset, ha = (0, 7 if is_higher else -11), "center"
            ax.annotate(f"{means[name][i]:.1f}", (xs[i], means[name][i]),
                        textcoords="offset points", xytext=offset, ha=ha,
                        va="center", fontsize=FONT_SIZE, color=color, zorder=5,
                        path_effects=halo)

    top = 25 * np.ceil(max(m.max() for m in means.values()) / 25 + 0.2)
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, top + 1, 25))
    ax.set_xticks(xs, [str(n) for n in views])
    ax.set_xlim(-1.4, xs[-1] + 0.4)  # room for the lower 2-view label
    ax.set_xlabel("Number of views")
    ax.set_ylabel("PA-MPJPE (mm)")
    ax.legend(loc="upper right", frameon=False, fontsize=FONT_SIZE,
              handlelength=1.2, borderaxespad=0.1)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(output_dir, f"n_views_pa_mpjpe_egohumans.{ext}")
        fig.savefig(path, dpi=dpi, facecolor=SURFACE)
        print(f"Figure saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PA-MPJPE against the number of views on EgoHumans")
    parser.add_argument("output_dir", help="Directory to save the figure in")
    parser.add_argument("--est-runs", nargs="+", required=True,
                        help="Benchmark run directories, estimated intrinsics")
    parser.add_argument("--gt-runs", nargs="+", required=True,
                        help="Benchmark run directories, GT intrinsics")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    main(args.output_dir, args.est_runs, args.gt_runs, args.dpi)
