import torch
import orjson
import os
import cv2
from tqdm import tqdm
import glob
import pickle

from kineo.geometry.transformations import undistort_image, distort_image
from kineo.annotations.camera_intrinsics import (
    CameraIntrinsicsAnnotation,
    CameraIntrinsicsAnnotations,
)

from distortion_error import prediction_intrinsics_path

import numpy as np

GRID_LINES = 20
GRID_PADDING = 3000
GT_GRID_COLOR = (0, 0, 0)
ESTIMATED_GRID_COLOR = (0, 0, 255)


def load_gt_camera_intrinsics(
    dataset_dir: str,
    sequence: dict,
) -> CameraIntrinsicsAnnotations:
    camera_intrinsics_file = sequence["annotations"]["cameras_intrinsics"]
    camera_intrinsics_file = os.path.join(dataset_dir, camera_intrinsics_file)

    with open(camera_intrinsics_file, "rb") as f:
        gt_camera_intrinsics = CameraIntrinsicsAnnotations.from_dict(
            orjson.loads(f.read())
        )

    return gt_camera_intrinsics


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
    """RGB frame in the middle of the view's sequence, (3, H, W) uint8."""
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
    """The frame with the ground-truth grid, then undistorted with the
    ground-truth camera, then with the estimated camera under both grids."""
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

    distorted_grid_gt = distorted(GT_GRID_COLOR)
    undistorted_grid_gt = undistorted(distorted_grid_gt, gt_intrinsics)
    undistorted_grid_pred = undistorted(
        distorted(ESTIMATED_GRID_COLOR), pred_intrinsics
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "egohumans_dataset_dir", type=str, help="Path to dataset directory"
    )
    parser.add_argument(
        "egohumans_pred_annotations_dir",
        type=str,
        help="Path to predicted annotations directory",
    )
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    parser.add_argument(
        "--views",
        type=str,
        required=True,
        help="Comma-separated sequence/view pairs, e.g. tennis_008/cam17",
    )
    parser.add_argument(
        "--output_size",
        type=str,
        default="1280x720",
        help="Output image size in format WxH (default: 1280x720)",
    )
    args = parser.parse_args()

    output_width, output_height = map(int, args.output_size.split("x"))
    views = [tuple(view.split("/")) for view in args.views.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(
        os.path.join(args.egohumans_dataset_dir, "egohumans_sequences.json"), "rb"
    ) as f:
        sequences = {s["sequence_name"]: s for s in orjson.loads(f.read())}

    for sequence_name, view_id in tqdm(views, desc="Processing views"):
        sequence = sequences[sequence_name]
        gt_cam_intrinsics = load_gt_camera_intrinsics(
            args.egohumans_dataset_dir, sequence
        ).filter_by_view_id(view_id).first_or_default()
        pred_cam_intrinsics = load_pred_camera_intrinsics(
            args.egohumans_pred_annotations_dir, sequence
        ).filter_by_view_id(view_id).first_or_default()

        frame_rgb = load_middle_frame(
            args.egohumans_dataset_dir, sequence["views"][view_id]["images_dir"]
        )
        images = render_grids(frame_rgb, gt_cam_intrinsics, pred_cam_intrinsics, device)

        prefix = os.path.join(args.output_dir, f"{sequence_name}_{view_id}")
        for suffix, image in zip(
            ("original", "undistorted_gt", "undistorted_pred"), images
        ):
            image = image.flip(0).permute(1, 2, 0).cpu().numpy()
            cv2.imwrite(
                f"{prefix}_{suffix}.jpg",
                cv2.resize(image, (output_width, output_height)),
            )
        tqdm.write(f"Saved distortions figure for {sequence_name} view {view_id}")
