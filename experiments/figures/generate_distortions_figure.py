import torch
import orjson
import os
import cv2
from tqdm import tqdm
import glob
import pickle

from kineo.geometry.transformations import undistort_image, distort_image
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.io.frame_sequence_loader import ImagesLoader

import numpy as np


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
    camera_intrinsics_file = os.path.join(
        pred_annotations_dir,
        sequence["sequence_name"],
        "camera_intrinsics.pkl",
    )
    with open(camera_intrinsics_file, "rb") as f:
        pred_camera_intrinsics = CameraIntrinsicsAnnotations.from_dict(pickle.load(f))
    return pred_camera_intrinsics


def create_view_inputs(
    dataset_dir: str,
    sequence: dict,
    device: torch.device,
) -> list[ViewInput]:
    cameras = list(sequence["views"].keys())
    views = []
    for camera in cameras:
        images_dir = sequence["views"][camera]["images_dir"]
        fps = sequence["views"][camera]["fps"]
        imgs_paths = sorted(
            glob.glob(
                os.path.join(
                    dataset_dir,
                    images_dir,
                    "*.jpg",
                )
            )
        )
        n_imgs = len(imgs_paths)

        frame_timestamps_local = (torch.arange(n_imgs) / fps).tolist()

        views.append(
            ViewInput(
                view_id=camera,
                frame_loader=ImagesLoader(
                    img_paths=imgs_paths,
                    frame_timestamps_local=frame_timestamps_local,
                    device=device,
                ),
                audio_loader=None,
            )
        )

    return views


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
        grid_hspacing (int): Horizontal spacing between grid lines in pixels.
        grid_vspacing (int): Vertical spacing between grid lines in pixels.
        line_width (int): Thickness of the grid lines in pixels.
        color (tuple): RGBA color of the grid lines.

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
        K=K_new.to(original_frame_rgb.device),
        D=D.to(original_frame_rgb.device),
        distortion_model=distortion_model,
    ).to(original_frame_rgb.device)

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
        "--output_size",
        type=str,
        default="1280x720",
        help="Output image size in format WxH (default: 1280x720)",
    )
    args = parser.parse_args()
    dataset_dir = args.egohumans_dataset_dir
    pred_annotations_dir = args.egohumans_pred_annotations_dir
    output_dir = args.output_dir
    output_size = args.output_size

    output_width, output_height = map(int, output_size.split("x"))

    sequences_file = os.path.join(dataset_dir, "egohumans_sequences.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())

    subsets = [
        ("tagging_001", "cam03"),
        ("legoassemble_001", "cam01"),
        ("fencing_002", "cam02"),
        ("basketball_001", "cam01"),
        ("volleyball_001", "cam05"),
        ("badminton_001", "cam15"),
        ("tennis_001", "cam01"),
    ]

    for sequence in tqdm(sequences, desc="Processing sequences"):
        sequence_name = sequence["sequence_name"]

        if sequence_name not in [s[0] for s in subsets]:
            continue

        sequence_basename = sequence_name.split("_")[0]

        views_inputs = create_view_inputs(dataset_dir, sequence, device)

        gt_cameras_intrinsics = load_gt_camera_intrinsics(dataset_dir, sequence)
        pred_cameras_intrinsics = load_pred_camera_intrinsics(
            pred_annotations_dir, sequence
        )

        for view_input in views_inputs:
            view_id = view_input["view_id"]
            frame_loader = view_input["frame_loader"]
            n_frames = frame_loader.n_frames

            if (sequence_name, view_id) not in subsets:
                continue

            gt_cam_intrinsics = gt_cameras_intrinsics.filter_by_view_id(
                view_id
            ).first_or_default()
            pred_cam_intrinsics = pred_cameras_intrinsics.filter_by_view_id(
                view_id
            ).first_or_default()

            # Take a frame roughly in the middle of the sequence
            frame_idx = n_frames // 2

            original_frame_rgb = frame_loader.load_frame_at(frame_idx)

            img_h, img_w = original_frame_rgb.shape[1:]

            distorted_grid_gt = create_distorted_grid(
                img_w=img_w,
                img_h=img_h,
                grid_padding=3000,
                K=gt_cam_intrinsics.K,
                D=gt_cam_intrinsics.distortion_coefficients,
                distortion_model=gt_cam_intrinsics.distortion_model.value,
                n_hlines=20,
                n_vlines=20,
                color=(0, 0, 0),
                device=device,
            )

            undistorted_grid_gt = undistorted_grid(
                distorted_grid=distorted_grid_gt,
                K=gt_cam_intrinsics.K,
                D=gt_cam_intrinsics.distortion_coefficients,
                distortion_model=gt_cam_intrinsics.distortion_model.value,
                device=device,
            )

            undistorted_grid_pred = undistorted_grid(
                distorted_grid=create_distorted_grid(
                    img_w=img_w,
                    img_h=img_h,
                    grid_padding=3000,
                    K=gt_cam_intrinsics.K,
                    D=gt_cam_intrinsics.distortion_coefficients,
                    distortion_model=gt_cam_intrinsics.distortion_model.value,
                    n_hlines=20,
                    n_vlines=20,
                    color=(0, 0, 255),
                    device=device,
                ),
                K=pred_cam_intrinsics.K,
                D=pred_cam_intrinsics.distortion_coefficients,
                distortion_model=pred_cam_intrinsics.distortion_model.value,
                device=device,
            )

            undistorted_gt_frame_rgb = undistort_image(
                img=original_frame_rgb.to(device),
                K=gt_cam_intrinsics.K.to(device),
                D=gt_cam_intrinsics.distortion_coefficients.to(device),
                distortion_model=gt_cam_intrinsics.distortion_model.value,
            )

            undistorted_pred_frame_rgb = undistort_image(
                img=original_frame_rgb.to(device),
                K=pred_cam_intrinsics.K.to(device),
                D=pred_cam_intrinsics.distortion_coefficients.to(device),
                distortion_model=pred_cam_intrinsics.distortion_model.value,
            )

            original_frame_rgb = overlay_grid(
                original_frame_rgb,
                distorted_grid_gt,
            )
            undistorted_gt_frame_rgb = overlay_grid(
                undistorted_gt_frame_rgb,
                undistorted_grid_gt,
            )
            undistorted_pred_frame_rgb = overlay_grid(
                undistorted_pred_frame_rgb,
                undistorted_grid_gt,
            )
            undistorted_pred_frame_rgb = overlay_grid(
                undistorted_pred_frame_rgb,
                undistorted_grid_pred,
            )

            original_frame_rgb = original_frame_rgb.flip(0).permute(1, 2, 0).cpu().numpy()
            original_frame_rgb = cv2.resize(original_frame_rgb, (output_width, output_height))

            undistorted_gt_frame_rgb = undistorted_gt_frame_rgb.flip(0).permute(1, 2, 0).cpu().numpy()
            undistorted_gt_frame_rgb = cv2.resize(undistorted_gt_frame_rgb, (output_width, output_height))

            undistorted_pred_frame_rgb = undistorted_pred_frame_rgb.flip(0).permute(1, 2, 0).cpu().numpy()
            undistorted_pred_frame_rgb = cv2.resize(undistorted_pred_frame_rgb, (output_width, output_height))

            cv2.imwrite(
                os.path.join(output_dir, f"{sequence_basename}_original.jpg"),
                original_frame_rgb,
            )
            cv2.imwrite(
                os.path.join(
                    output_dir, f"{sequence_basename}_undistorted_gt.jpg"
                ),
                undistorted_gt_frame_rgb,
            )
            cv2.imwrite(
                os.path.join(
                    output_dir, f"{sequence_basename}_undistorted_pred.jpg"
                ),
                undistorted_pred_frame_rgb,
            )
            tqdm.write(
                f"Saved distortions figure for sequence {sequence_name} view {view_id}"
            )
