import argparse
import glob
import os
import pickle

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from distortion_error import (
    GRID_HW,
    iter_view_errors,
    load_gt_intrinsics,
    load_sequences,
    prediction_intrinsics_path,
)
from kineo.annotations.camera_intrinsics import (
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotations,
)
from kineo.geometry.transformations import distort_image, undistort_image

SCENES = {
    "tagging": "Tagging",
    "legoassemble": "Lego assembly",
    "fencing": "Fencing",
    "basketball": "Basketball",
    "volleyball": "Volleyball",
    "badminton": "Badminton",
    "tennis": "Tennis",
}
SUBCAPTIONS = (
    "(a) Original distorted image\nwith the ground-truth grid",
    "(b) Undistorted image (estimated)\nwith both grids",
    "(c) Error map, focal length\naligned to the ground truth",
)

ERROR_MAX_PX = 50.0
ERROR_TICKS = (0, 10, 20, 30, 40, 50)
COLORMAP = "viridis"
EXCLUDED_COLOR = "#bdbdbd"
EXCLUDED_HATCH_COLOR = "#7a7a7a"
OUTLINE_COLOR = "#CC79A7"
OUTLINE_HALO_COLOR = "black"
GT_GRID_COLOR = "black"
ESTIMATED_GRID_COLOR = "red"
GRID_LINES = 20
GRID_PADDING = 3000
GT_GRID_LINE_COLOR = (0, 0, 0)
ESTIMATED_GRID_LINE_COLOR = (0, 0, 255)

ARROW_GRID_HW = (7, 12)
ARROW_MIN_PX = 1.0
ARROW_COLOR = "white"
ARROW_EDGE_COLOR = "black"

FONT_SIZE = 7
IMAGE_SIZE = (1280, 720)
# Layout in inches
ROW_LABEL_WIDTH = 0.16
PANEL_GAP = 0.04
TOP_MARGIN = 0.02
SUBCAPTION_GAP = 0.03
SUBCAPTION_HEIGHT = 0.26
LEGEND_GAP = 0.06
LEGEND_LINE_HEIGHT = 0.16
LEGEND_ONE_LINE_WIDTH = 6.5
COLORBAR_GAP = 0.08
COLORBAR_WIDTH = 0.09
COLORBAR_LABEL_WIDTH = 0.45
COLORBAR_TIP = 0.12


def load_pred_camera_intrinsics(
    pred_annotations_dir: str,
    sequence: dict,
) -> CameraIntrinsicsAnnotations:
    camera_intrinsics_file = prediction_intrinsics_path(
        os.path.join(pred_annotations_dir, sequence["sequence_name"])
    )
    with open(camera_intrinsics_file, "rb") as f:
        pred_camera_intrinsics = CameraIntrinsicsAnnotations.from_dict(pickle.load(f))
    return pred_camera_intrinsics


def load_middle_frame(dataset_dir: str, images_dir: str) -> torch.Tensor:
    paths = sorted(glob.glob(os.path.join(dataset_dir, images_dir, "*.jpg")))
    frame = cv2.cvtColor(cv2.imread(paths[len(paths) // 2]), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).permute(2, 0, 1)


def create_grid(
    width: int,
    height: int,
    n_hlines: int,
    n_vlines: int,
    line_width: int = 2,
    color: tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """
    Create a transparent RGBA grid image.

    Args:
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
        n_hlines (int): Number of horizontal grid lines.
        n_vlines (int): Number of vertical grid lines.
        line_width (int): Thickness of the grid lines in pixels.
        color (tuple): BGR color of the grid lines.

    Returns:
        torch.Tensor: RGBA image of shape (4, height, width), dtype=torch.uint8
    """
    # Create a fully transparent image
    grid_img = np.zeros((height, width, 4), dtype=np.uint8)

    hlines_spacing = height / n_hlines
    vlines_spacing = width / n_vlines

    for vline_idx in range(n_vlines):
        x = min(int(vline_idx * vlines_spacing), width - 1)
        cv2.line(
            grid_img,
            (x, 0),
            (x, height),
            color=color + (255,),
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )

    for hline_idx in range(n_hlines):
        y = min(int(hline_idx * hlines_spacing), height - 1)
        cv2.line(
            grid_img,
            (0, y),
            (width, y),
            color=color + (255,),
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )

    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGRA2RGBA)
    return torch.from_numpy(grid_img).permute(2, 0, 1).to(dtype=torch.uint8)


def overlay_grid(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    assert img.shape[0] == 3 and grid.shape[0] == 4

    dtype = img.dtype
    if dtype == torch.uint8:
        img = img.float() / 255.0

    if grid.dtype == torch.uint8:
        grid = grid.float() / 255.0

    grid_alpha = grid[3, :, :]
    grid_rgb = grid[:3, :, :]
    new_img = grid_alpha * grid_rgb + (1 - grid_alpha) * img

    if dtype == torch.uint8:
        new_img = (new_img * 255.0).to(dtype=torch.uint8)

    return new_img


def create_distorted_grid(
    img_w: int,
    img_h: int,
    grid_padding: int,
    K: torch.Tensor,
    D: torch.Tensor,
    distortion_model: str,
    n_hlines: int = 20,
    n_vlines: int = 20,
    color: tuple[int, int, int] = (0, 0, 0),
    device: torch.device = torch.device("cpu"),
):
    grid = create_grid(
        width=img_w + grid_padding,
        height=img_h + grid_padding,
        n_hlines=n_hlines,
        n_vlines=n_vlines,
        line_width=3,
        color=color,
    ).to(device)

    K_new = K.clone()
    K_new[0, 2] += grid_padding / 2
    K_new[1, 2] += grid_padding / 2

    distorted_grid = distort_image(
        img=grid,
        K=K_new.to(device),
        D=D.to(device),
        distortion_model=distortion_model,
    )

    distorted_grid = distorted_grid[
        ...,
        grid_padding // 2 :,
        grid_padding // 2 :,
    ][
        ...,
        :img_h,
        :img_w,
    ]

    return distorted_grid


def undistorted_grid(
    distorted_grid: torch.Tensor,
    K: torch.Tensor,
    D: torch.Tensor,
    distortion_model: str,
    device: torch.device = torch.device("cpu"),
):
    undistorted_grid = undistort_image(
        img=distorted_grid.to(device),
        K=K.to(device),
        D=D.to(device),
        distortion_model=distortion_model,
    )

    return undistorted_grid


def render_grids(
    frame_rgb: torch.Tensor,
    gt_intrinsics: CameraIntrinsicsAnnotation,
    pred_intrinsics: CameraIntrinsicsAnnotation,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    img_h, img_w = frame_rgb.shape[1:]
    frame_rgb = frame_rgb.to(device)

    def distorted(color):
        return create_distorted_grid(
            img_w=img_w,
            img_h=img_h,
            grid_padding=GRID_PADDING,
            K=gt_intrinsics.K,
            D=gt_intrinsics.distortion_coefficients,
            distortion_model=gt_intrinsics.distortion_model.value,
            n_hlines=GRID_LINES,
            n_vlines=GRID_LINES,
            color=color,
            device=device,
        )

    def undistorted(img, intrinsics):
        return undistorted_grid(
            distorted_grid=img,
            K=intrinsics.K,
            D=intrinsics.distortion_coefficients,
            distortion_model=intrinsics.distortion_model.value,
            device=device,
        )

    distorted_grid_gt = distorted(GT_GRID_LINE_COLOR)
    undistorted_grid_gt = undistorted(distorted_grid_gt, gt_intrinsics)
    undistorted_grid_pred = undistorted(
        distorted(ESTIMATED_GRID_LINE_COLOR), pred_intrinsics
    )

    original = overlay_grid(frame_rgb, distorted_grid_gt)
    undistorted_gt = overlay_grid(
        undistorted(frame_rgb, gt_intrinsics), undistorted_grid_gt
    )
    undistorted_pred = overlay_grid(
        overlay_grid(undistorted(frame_rgb, pred_intrinsics), undistorted_grid_gt),
        undistorted_grid_pred,
    )
    return original, undistorted_gt, undistorted_pred


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE,
            "hatch.linewidth": 0.35,
            "hatch.color": EXCLUDED_HATCH_COLOR,
            # TrueType instead of Type 3, required by ACM
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def error_norm() -> colors.Normalize:
    return colors.Normalize(0.0, ERROR_MAX_PX, clip=False)


def error_cmap() -> colors.Colormap:
    cmap = plt.get_cmap(COLORMAP).copy()
    cmap.set_bad(EXCLUDED_COLOR)
    return cmap


def hide_frame(ax, height: int, width: int):
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_image(ax, image: torch.Tensor):
    image = cv2.resize(image.permute(1, 2, 0).cpu().numpy(), IMAGE_SIZE,
                       interpolation=cv2.INTER_AREA)
    ax.imshow(image, extent=(0, GRID_HW[1], GRID_HW[0], 0))
    hide_frame(ax, *GRID_HW)


def draw_arrows(ax, vectors: np.ndarray):
    h, w = vectors.shape[:2]
    ys = ((np.arange(ARROW_GRID_HW[0]) + 0.5) * h / ARROW_GRID_HW[0]).astype(int)
    xs = ((np.arange(ARROW_GRID_HW[1]) + 0.5) * w / ARROW_GRID_HW[1]).astype(int)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    u, v = vectors[yy, xx, 0], vectors[yy, xx, 1]
    norm = np.hypot(u, v)
    keep = np.isfinite(norm) & (norm >= ARROW_MIN_PX)
    ax.quiver(
        xx[keep] + 0.5, yy[keep] + 0.5, u[keep] / norm[keep], v[keep] / norm[keep],
        angles="xy", pivot="middle", scale_units="xy",
        scale=1 / (0.55 * w / ARROW_GRID_HW[1]),
        width=0.006, headwidth=3.5, headlength=3.5, headaxislength=3.2,
        color=ARROW_COLOR, edgecolor=ARROW_EDGE_COLOR, linewidth=0.3,
    )
    dots = np.isfinite(norm) & (norm < ARROW_MIN_PX)
    ax.scatter(xx[dots] + 0.5, yy[dots] + 0.5, s=2.5, color=ARROW_COLOR,
               edgecolors=ARROW_EDGE_COLOR, linewidths=0.3)


def draw_map(ax, error: np.ndarray, vectors: np.ndarray, observed: np.ndarray):
    ax.imshow(np.ma.masked_invalid(error), cmap=error_cmap(), norm=error_norm(),
              interpolation="none", extent=(0, error.shape[1], error.shape[0], 0))

    # An extent would flip the contours vertically
    xs = np.arange(error.shape[1]) + 0.5
    ys = np.arange(error.shape[0]) + 0.5
    excluded = ~np.isfinite(error)
    if excluded.any():
        ax.contourf(xs, ys, excluded.astype(float), levels=[0.5, 1.5],
                    colors="none", hatches=["//////"])

    draw_arrows(ax, vectors)

    outline = ax.contour(xs, ys, observed.astype(float), levels=[0.5],
                         colors=OUTLINE_COLOR, linewidths=0.6)
    outline.set_path_effects(
        [path_effects.Stroke(linewidth=1.2, foreground=OUTLINE_HALO_COLOR),
         path_effects.Normal()]
    )
    hide_frame(ax, *error.shape)


def draw_colorbar(cax, length: float, **kwargs):
    mappable = plt.cm.ScalarMappable(norm=error_norm(), cmap=error_cmap())
    colorbar = plt.colorbar(mappable, cax=cax, extend="max",
                            extendfrac=COLORBAR_TIP / length, **kwargs)
    colorbar.set_ticks(ERROR_TICKS)
    labels = [f"{tick:g}" for tick in ERROR_TICKS]
    labels[-1] = f"≥{labels[-1]}"
    colorbar.set_ticklabels(labels)
    colorbar.ax.minorticks_off()
    colorbar.outline.set_linewidth(0.4)
    colorbar.ax.tick_params(width=0.4, length=2, pad=1.5)
    return colorbar


def draw_legend(ax, ncol: int):
    ax.axis("off")
    halo = [path_effects.Stroke(linewidth=1.8, foreground=OUTLINE_HALO_COLOR),
            path_effects.Normal()]
    handles = [
        Line2D([], [], color=GT_GRID_COLOR, linewidth=1.0),
        Line2D([], [], color=ESTIMATED_GRID_COLOR, linewidth=1.0),
        Line2D([], [], color=OUTLINE_COLOR, linewidth=0.8, path_effects=halo),
        Line2D([], [], linestyle="none", marker=r"$\rightarrow$", markersize=9,
               markerfacecolor=ARROW_COLOR, markeredgecolor=ARROW_EDGE_COLOR,
               markeredgewidth=0.3),
        Patch(facecolor=EXCLUDED_COLOR, hatch="//////",
              edgecolor=EXCLUDED_HATCH_COLOR, linewidth=0),
    ]
    labels = ["Ground-truth grid", "Estimated grid",
              "Region observed during calibration", "Error direction",
              "Excluded (no valid reprojection)"]
    ax.legend(handles, labels, loc="center", ncol=ncol, frameon=False,
              handlelength=1.2, handletextpad=0.4, columnspacing=0.7,
              borderaxespad=0, borderpad=0)


def draw_figure(output_base: str, rows: list[dict], width: float):
    grid_width = (width - ROW_LABEL_WIDTH - COLORBAR_GAP - COLORBAR_WIDTH
                  - COLORBAR_LABEL_WIDTH)
    panel_w = (grid_width - 2 * PANEL_GAP) / 3
    panel_h = panel_w * GRID_HW[0] / GRID_HW[1]
    rows_h = len(rows) * panel_h + (len(rows) - 1) * PANEL_GAP
    sub_top = TOP_MARGIN + rows_h + SUBCAPTION_GAP
    legend_top = sub_top + SUBCAPTION_HEIGHT + LEGEND_GAP
    legend_lines = 1 if width >= LEGEND_ONE_LINE_WIDTH else 2
    height = legend_top + legend_lines * LEGEND_LINE_HEIGHT

    fig = plt.figure(figsize=(width, height))

    def axes(left, top, w, h):
        return fig.add_axes([left / width, (height - top - h) / height,
                             w / width, h / height])

    def column_left(col):
        return ROW_LABEL_WIDTH + col * (panel_w + PANEL_GAP)

    for i, row in enumerate(rows):
        top = TOP_MARGIN + i * (panel_h + PANEL_GAP)
        ax = axes(0, top, ROW_LABEL_WIDTH, panel_h)
        ax.axis("off")
        ax.text(0.5, 0.5, row["label"], transform=ax.transAxes, ha="center",
                va="center", rotation=90)
        draw_image(axes(column_left(0), top, panel_w, panel_h), row["original"])
        draw_image(axes(column_left(1), top, panel_w, panel_h),
                   row["undistorted"])
        draw_map(axes(column_left(2), top, panel_w, panel_h), row["error"],
                 row["vectors"], row["observed"])

    for col, subcaption in enumerate(SUBCAPTIONS):
        ax = axes(column_left(col), sub_top, panel_w, SUBCAPTION_HEIGHT)
        ax.axis("off")
        ax.text(0.5, 0.5, subcaption, transform=ax.transAxes, ha="center",
                va="center", linespacing=1.3)

    colorbar = draw_colorbar(axes(column_left(2) + panel_w + COLORBAR_GAP,
                                  TOP_MARGIN, COLORBAR_WIDTH, rows_h), rows_h)
    colorbar.set_label("Error (px)", labelpad=2)
    draw_legend(axes(column_left(0), legend_top, 3 * panel_w + 2 * PANEL_GAP,
                     legend_lines * LEGEND_LINE_HEIGHT),
                ncol=5 if legend_lines == 1 else 3)

    fig.savefig(f"{output_base}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_panels(output_dir: str, rows: list[dict], panel_width: float, legend_width: float):
    panel_height = panel_width * GRID_HW[0] / GRID_HW[1]
    for row in rows:
        for kind in ("original", "undistorted", "error"):
            fig = plt.figure(figsize=(panel_width, panel_height))
            ax = fig.add_axes([0, 0, 1, 1])
            if kind == "error":
                draw_map(ax, row["error"], row["vectors"], row["observed"])
            else:
                draw_image(ax, row[kind])
            fig.savefig(os.path.join(output_dir, f"{row['scene']}_{kind}.pdf"), dpi=300)
            plt.close(fig)

    colorbar_width = 1.6
    height = 2 * LEGEND_LINE_HEIGHT + 0.06
    fig = plt.figure(figsize=(legend_width, height))
    legend_ax = fig.add_axes([0, 0, 1 - (colorbar_width + 0.2) / legend_width, 1])
    draw_legend(legend_ax, ncol=3)
    cax = fig.add_axes([1 - colorbar_width / legend_width, 0.25, colorbar_width / legend_width,
                        0.07 / height])
    colorbar = draw_colorbar(cax, colorbar_width, orientation="horizontal")
    colorbar.ax.set_title("Error (px)", fontsize=FONT_SIZE, pad=2)
    fig.savefig(os.path.join(output_dir, "legend.pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("egohumans_dataset_dir")
    parser.add_argument("pred_annotations_dir")
    parser.add_argument("output_base", help="Output path, without extension.")
    parser.add_argument("--views", required=True,
                        help="Comma-separated sequence/view pairs, in row order.")
    parser.add_argument("--width", type=float, default=6.96,
                        help="Figure width in inches (7 in once saved).")
    parser.add_argument("--panels", action="store_true",
                        help="Write one PDF per panel and a legend strip into "
                             "output_base, a directory, instead of one figure.")
    parser.add_argument("--kp-radius", type=int, default=25)
    parser.add_argument(
        "--kp-score-thr",
        type=float,
        default=0.0,
        help="Keeps the keypoints scoring above this; 0 keeps every keypoint "
        "that has weight in bundle adjustment.",
    )
    args = parser.parse_args()

    views = [tuple(view.split("/")) for view in args.views.split(",")]
    sequences = {s["sequence_name"]: s
                 for s in load_sequences(args.egohumans_dataset_dir)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_style()

    errors = {
        (view_error.sequence_name, view_error.view_id): view_error
        for view_error in iter_view_errors(
            dataset_dir=args.egohumans_dataset_dir,
            pred_annotations_dir=args.pred_annotations_dir,
            sequences=[sequences[name] for name, _ in views],
            views=set(views),
            radius=args.kp_radius,
            score_threshold=args.kp_score_thr,
        )
    }

    rows = []
    for name, view_id in views:
        sequence = sequences[name]
        gt = load_gt_intrinsics(args.egohumans_dataset_dir, sequence)
        pred = load_pred_camera_intrinsics(args.pred_annotations_dir, sequence)
        frame = load_middle_frame(args.egohumans_dataset_dir,
                                  sequence["views"][view_id]["images_dir"])
        original, _, undistorted = render_grids(
            frame,
            gt.filter_by_view_id(view_id).first_or_default(),
            pred.filter_by_view_id(view_id).first_or_default(),
            device,
        )
        view_error = errors[(name, view_id)]
        rows.append({
            "scene": name.rsplit("_", 1)[0],
            "label": SCENES[name.rsplit("_", 1)[0]],
            "original": original,
            "undistorted": undistorted,
            "error": view_error.aligned_error,
            "vectors": view_error.aligned_vectors,
            "observed": view_error.observed,
        })

    if args.panels:
        os.makedirs(args.output_base, exist_ok=True)
        save_panels(args.output_base, rows, panel_width=args.width / 3,
                    legend_width=args.width)
        print(f"Wrote panels to {args.output_base}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.output_base)), exist_ok=True)
    draw_figure(args.output_base, rows, args.width)
    print(f"Wrote {args.output_base}.pdf/.png")


if __name__ == "__main__":
    main()
