import torch
import pickle
import orjson
import os
import numpy as np
from tqdm import tqdm
from kineo.geometry.camera import inverse_Rt
from kineo.geometry.transformations import compute_similarity_transform
from kineo.annotations.camera_extrinsics import (
    CameraExtrinsicsAnnotation,
    CameraExtrinsicsAnnotations,
)
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.keypoints_3d import (
    Keypoints3DAnnotation,
    Keypoints3DAnnotations,
)
from aitviewer.scene.node import Node
from aitviewer.scene.camera import PinholeCamera
from aitviewer.headless import HeadlessRenderer
from aitviewer.configuration import CONFIG as C
from kineo.eval.human_metrics import compute_human_metrics, get_min_median_max_frames

import json
import traceback
import shutil
from kineo.visualization.viz_3d import add_keypoints_3d_and_cameras
import cv2
import re

C.window_type = "pyglet"

def sanitize_sequence_name(sequence_name: str) -> str:
    allowed_pattern = r'[^a-zA-Z0-9_\+\-\.,=]'
    sanitized = re.sub(allowed_pattern, '_', sequence_name)
    return sanitized

def load_hsfm_predictions_in_folder(
    folder_path: str,
) -> tuple[
    CameraIntrinsicsAnnotations, CameraExtrinsicsAnnotations, Keypoints3DAnnotations
]:
    camera_intrinsics_file = os.path.join(folder_path, "camera_intrinsics.pkl")
    camera_extrinsics_file = os.path.join(folder_path, "camera_extrinsics.pkl")
    keypoints_3d_file = os.path.join(folder_path, "keypoints_3d.pkl")

    if not os.path.exists(camera_intrinsics_file):
        raise FileNotFoundError(
            f"Camera intrinsics file not found: {camera_intrinsics_file}"
        )

    if not os.path.exists(camera_extrinsics_file):
        raise FileNotFoundError(
            f"Camera extrinsics file not found: {camera_extrinsics_file}"
        )

    if not os.path.exists(keypoints_3d_file):
        raise FileNotFoundError(f"Keypoints 3D file not found: {keypoints_3d_file}")

    with open(camera_intrinsics_file, "rb") as f:
        hsfm_camera_intrinsics = CameraIntrinsicsAnnotations.from_dict(pickle.load(f))

    with open(camera_extrinsics_file, "rb") as f:
        hsfm_camera_extrinsics = CameraExtrinsicsAnnotations.from_dict(pickle.load(f))

    with open(keypoints_3d_file, "rb") as f:
        hsfm_keypoints_3d = Keypoints3DAnnotations.from_dict(pickle.load(f))

    return hsfm_camera_intrinsics, hsfm_camera_extrinsics, hsfm_keypoints_3d


def load_gt_keypoints_3d(
    dataset_dir: str,
    sequence: dict,
) -> Keypoints3DAnnotations:
    keypoints_3d_file = sequence["annotations"]["keypoints_3d"]
    keypoints_3d_file = os.path.join(dataset_dir, keypoints_3d_file)

    with open(keypoints_3d_file, "rb") as f:
        gt_keypoints_3d = Keypoints3DAnnotations.from_dict(orjson.loads(f.read()))

    return gt_keypoints_3d


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


def load_gt_camera_extrinsics(
    dataset_dir: str,
    sequence: dict,
) -> CameraExtrinsicsAnnotations:
    camera_extrinsics_file = sequence["annotations"]["cameras_extrinsics"]
    camera_extrinsics_file = os.path.join(dataset_dir, camera_extrinsics_file)

    with open(camera_extrinsics_file, "rb") as f:
        gt_camera_extrinsics = CameraExtrinsicsAnnotations.from_dict(
            orjson.loads(f.read())
        )

    return gt_camera_extrinsics


def load_kineo_predictions_in_folder(
    folder_path: str,
) -> tuple[
    CameraIntrinsicsAnnotations,
    CameraExtrinsicsAnnotations,
    Keypoints3DAnnotations,
]:
    camera_intrinsics_file = os.path.join(folder_path, "camera_intrinsics.pkl")
    camera_extrinsics_file = os.path.join(folder_path, "camera_extrinsics.pkl")
    keypoints_3d_file = os.path.join(folder_path, "keypoints_3d.pkl")

    if not os.path.exists(camera_intrinsics_file):
        raise FileNotFoundError(
            f"Camera intrinsics file not found: {camera_intrinsics_file}"
        )

    if not os.path.exists(camera_extrinsics_file):
        raise FileNotFoundError(
            f"Camera extrinsics file not found: {camera_extrinsics_file}"
        )

    if not os.path.exists(keypoints_3d_file):
        raise FileNotFoundError(f"Keypoints 3D file not found: {keypoints_3d_file}")

    with open(camera_intrinsics_file, "rb") as f:
        kineo_camera_intrinsics = CameraIntrinsicsAnnotations.from_dict(pickle.load(f))

    with open(camera_extrinsics_file, "rb") as f:
        kineo_camera_extrinsics = CameraExtrinsicsAnnotations.from_dict(pickle.load(f))

    with open(keypoints_3d_file, "rb") as f:
        kineo_keypoints_3d = Keypoints3DAnnotations.from_dict(pickle.load(f))

    return kineo_camera_intrinsics, kineo_camera_extrinsics, kineo_keypoints_3d


def align_hsfm_pred_to_gt(
    pred_camera_extrinsics: CameraExtrinsicsAnnotations,
    pred_keypoints_3d: Keypoints3DAnnotations,
    gt_camera_extrinsics: CameraExtrinsicsAnnotations,
    estimate_scale: bool = True,
) -> tuple[CameraExtrinsicsAnnotations, Keypoints3DAnnotations]:
    views_ids = list(gt_camera_extrinsics.views_ids)

    pred_cam_frames_indices = pred_camera_extrinsics.frames
    pred_keypoints_3d_frames_indices = pred_keypoints_3d.frames
    assert pred_cam_frames_indices == pred_keypoints_3d_frames_indices, (
        "Predicted camera extrinsics and keypoints 3D must have the same frames indices."
    )

    n_views = len(views_ids)
    views_ids = gt_camera_extrinsics.views_ids

    assert len(gt_camera_extrinsics) == n_views, (
        "Expected one gt extrinsics for each view (static cameras)."
    )

    gt_cam_pos = torch.stack(
        [
            inverse_Rt(a.Rt)[..., :3, 3]
            for a in sorted(gt_camera_extrinsics._annotations, key=lambda x: x.view_id)
        ]
    )

    aligned_pred_camera_extrinsics: list[CameraExtrinsicsAnnotation] = []
    aligned_pred_keypoints_3d: list[Keypoints3DAnnotation] = []

    for frame_idx in pred_cam_frames_indices:
        frame_pred_cam_extrinsics = pred_camera_extrinsics.filter_by_frame_idx(
            frame_idx
        )
        frame_pred_keypoints_3d = pred_keypoints_3d.filter_by_frame_idx(frame_idx)

        assert frame_pred_cam_extrinsics.views_ids == views_ids, (
            "Predicted camera extrinsics and GT camera extrinsics must have the same views IDs."
        )

        pred_cam_pos = torch.stack(
            [
                inverse_Rt(a.Rt)[..., :3, 3]
                for a in sorted(
                    frame_pred_cam_extrinsics._annotations, key=lambda x: x.view_id
                )
            ]
        )

        R, t, s = compute_similarity_transform(
            X=pred_cam_pos,
            Y=gt_cam_pos,
            estimate_scale=estimate_scale,
        )

        frame_pred_cam_extrinsics = (
            frame_pred_cam_extrinsics.apply_similarity_transform(R=R, t=t, s=s)
        )
        frame_pred_keypoints_3d = frame_pred_keypoints_3d.apply_similarity_transform(
            R=R, t=t, s=s
        )

        aligned_pred_camera_extrinsics.extend(frame_pred_cam_extrinsics._annotations)
        aligned_pred_keypoints_3d.extend(frame_pred_keypoints_3d._annotations)

    aligned_pred_camera_extrinsics = CameraExtrinsicsAnnotations(
        metadata=pred_camera_extrinsics.metadata,
        annotations=aligned_pred_camera_extrinsics,
    )
    aligned_pred_keypoints_3d = Keypoints3DAnnotations(
        metadata=pred_keypoints_3d.metadata,
        annotations=aligned_pred_keypoints_3d,
    )

    return aligned_pred_camera_extrinsics, aligned_pred_keypoints_3d


def align_kineo_pred_to_gt(
    pred_camera_extrinsics: CameraExtrinsicsAnnotations,
    pred_keypoints_3d: Keypoints3DAnnotations,
    gt_camera_extrinsics: CameraExtrinsicsAnnotations,
    estimate_scale: bool = True,
) -> tuple[CameraExtrinsicsAnnotations, Keypoints3DAnnotations]:
    views_ids = list(gt_camera_extrinsics.views_ids)

    n_views = len(views_ids)
    views_ids = gt_camera_extrinsics.views_ids

    assert len(gt_camera_extrinsics) == n_views, (
        "Expected one gt extrinsics for each view (static cameras)."
    )
    assert len(pred_camera_extrinsics) == n_views, (
        "Expected one predicted camera extrinsics for each view (static cameras)."
    )

    gt_cam_pos = torch.stack(
        [
            inverse_Rt(a.Rt)[..., :3, 3]
            for a in sorted(gt_camera_extrinsics._annotations, key=lambda x: x.view_id)
        ]
    )
    pred_cam_pos = torch.stack(
        [
            inverse_Rt(a.Rt)[..., :3, 3]
            for a in sorted(
                pred_camera_extrinsics._annotations, key=lambda x: x.view_id
            )
        ]
    )

    R, t, s = compute_similarity_transform(
        X=pred_cam_pos,
        Y=gt_cam_pos,
        estimate_scale=estimate_scale,
    )

    aligned_pred_camera_extrinsics = pred_camera_extrinsics.apply_similarity_transform(
        R=R, t=t, s=s
    )
    aligned_pred_keypoints_3d = pred_keypoints_3d.apply_similarity_transform(
        R=R, t=t, s=s
    )

    return aligned_pred_camera_extrinsics, aligned_pred_keypoints_3d


def load_predictions_in_folder(
    folder_path: str,
) -> tuple[
    CameraIntrinsicsAnnotations,
    CameraExtrinsicsAnnotations,
    Keypoints3DAnnotations,
]:
    camera_intrinsics_file = os.path.join(folder_path, "camera_intrinsics.pkl")
    camera_extrinsics_file = os.path.join(folder_path, "camera_extrinsics.pkl")
    keypoints_3d_file = os.path.join(folder_path, "keypoints_3d.pkl")

    if not os.path.exists(camera_intrinsics_file):
        raise FileNotFoundError(
            f"Camera intrinsics file not found: {camera_intrinsics_file}"
        )

    if not os.path.exists(camera_extrinsics_file):
        raise FileNotFoundError(
            f"Camera extrinsics file not found: {camera_extrinsics_file}"
        )

    if not os.path.exists(keypoints_3d_file):
        raise FileNotFoundError(f"Keypoints 3D file not found: {keypoints_3d_file}")

    with open(camera_intrinsics_file, "rb") as f:
        kineo_camera_intrinsics = CameraIntrinsicsAnnotations.from_dict(pickle.load(f))

    with open(camera_extrinsics_file, "rb") as f:
        kineo_camera_extrinsics = CameraExtrinsicsAnnotations.from_dict(pickle.load(f))

    with open(keypoints_3d_file, "rb") as f:
        kineo_keypoints_3d = Keypoints3DAnnotations.from_dict(pickle.load(f))

    return kineo_camera_intrinsics, kineo_camera_extrinsics, kineo_keypoints_3d

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument(
        "hsfm_eval_data_dir", type=str, help="Path to HSFM evaluation data directory"
    )
    parser.add_argument(
        "kineo_eval_data_dir", type=str, help="Path to Kineo evaluation data directory"
    )
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    parser.add_argument(
        "--skip-videos", action="store_true", help="Skip videos generation"
    )
    parser.add_argument(
        "--downscale-factor", type=float, help="Downscale factor for the images", default=None,
    )
    parser.add_argument(
        "--n-decimals", type=int, help="Number of decimals for the metrics", default=3,
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    downscale_factor = args.downscale_factor
    n_decimals = args.n_decimals
    sequences_file = os.path.join(dataset_dir, "h36m_protocol1_sequences.json")

    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())

    window_height = 600
    window_width = 1000
    viewer = HeadlessRenderer(size=(window_width, window_height))
    hsfm_node = None
    kineo_node = None

    viewer.playback_fps = 10

    skeleton_radius = 0.03
    camera_scale = 2

    camera = PinholeCamera(
        position=np.array([0.514, 3.698, 7.201]),
        target=np.array([0.502, 3.217, 6.335]),
        cols=window_width,
        rows=window_height,
    )
    viewer.scene.origin.enabled = False
    viewer.scene.add(camera)
    viewer.scene.camera = camera

    latex_table_rows = []

    latex_comparative_figs_dir = os.path.join(args.output_dir, "latex_comparative_figs")
    if os.path.exists(latex_comparative_figs_dir):
        shutil.rmtree(latex_comparative_figs_dir, ignore_errors=True)
    os.makedirs(latex_comparative_figs_dir, exist_ok=True)

    with open(os.path.join(os.path.dirname(__file__), "h36m_comparison_table_row_template.tex"), "r") as f:
        latex_table_row_template = f.read()

    all_joints_w_mpjpe_kineo = []
    all_joints_w_mpjpe_hsfm = []

    for sequence in tqdm(sequences, desc="Processing sequences"):
        if sequence["split"] != "val":
            continue

        sanitized_seq_name = sanitize_sequence_name(sequence["sequence_name"])

        try:
            gt_keypoints_3d = load_gt_keypoints_3d(dataset_dir, sequence)
            gt_camera_intrinsics = load_gt_camera_intrinsics(dataset_dir, sequence)
            gt_camera_extrinsics = load_gt_camera_extrinsics(dataset_dir, sequence)

            views_ids = gt_camera_intrinsics.views_ids
            views_resolution_hw = [
                gt_camera_intrinsics.filter_by_view_id(view_id)
                .first_or_default()
                .resolution_hw
                for view_id in views_ids
            ]
            subject_id = gt_keypoints_3d.subjects_ids[0]

            kineo_camera_intrinsics, kineo_camera_extrinsics, kineo_keypoints_3d = (
                load_kineo_predictions_in_folder(
                    os.path.join(args.kineo_eval_data_dir, sequence["sequence_name"]),
                )
            )

            hsfm_camera_intrinsics, hsfm_camera_extrinsics, hsfm_keypoints_3d = (
                load_hsfm_predictions_in_folder(
                    folder_path=os.path.join(
                        args.hsfm_eval_data_dir, sequence["sequence_name"]
                    ),
                )
            )

            kineo_human_metrics = compute_human_metrics(
                gt_keypoints_3d_annotations=gt_keypoints_3d,
                gt_cam_extrinsics_annotations=gt_camera_extrinsics,
                pred_keypoints_3d_annotations=kineo_keypoints_3d,
                pred_cam_extrinsics_annotations=kineo_camera_extrinsics,
            )
            hsfm_human_metrics = compute_human_metrics(
                gt_keypoints_3d_annotations=gt_keypoints_3d,
                gt_cam_extrinsics_annotations=gt_camera_extrinsics,
                pred_keypoints_3d_annotations=hsfm_keypoints_3d,
                pred_cam_extrinsics_annotations=hsfm_camera_extrinsics,
            )

            kineo_min_median_max = get_min_median_max_frames(human_metrics=kineo_human_metrics)
            hsfm_min_median_max = get_min_median_max_frames(human_metrics=hsfm_human_metrics)

            if kineo_min_median_max is None:
                print(f"No valid Kineo metrics for sequence {sequence['sequence_name']}")
                continue
            if hsfm_min_median_max is None:
                print(f"No valid HSfM metrics for sequence {sequence['sequence_name']}")
                continue

            estimate_scale = True

            hsfm_camera_extrinsics, hsfm_keypoints_3d = align_hsfm_pred_to_gt(
                pred_camera_extrinsics=hsfm_camera_extrinsics,
                pred_keypoints_3d=hsfm_keypoints_3d,
                gt_camera_extrinsics=gt_camera_extrinsics,
                estimate_scale=estimate_scale,
            )

            kineo_camera_extrinsics, kineo_keypoints_3d = align_kineo_pred_to_gt(
                pred_camera_extrinsics=kineo_camera_extrinsics,
                pred_keypoints_3d=kineo_keypoints_3d,
                gt_camera_extrinsics=gt_camera_extrinsics,
                estimate_scale=estimate_scale,
            )

            if hsfm_node is not None:
                viewer.scene.remove(hsfm_node)
            if kineo_node is not None:
                viewer.scene.remove(kineo_node)

            hsfm_color = (1, 0, 0, 1)
            kineo_color = (41 / 255, 128 / 255, 185 / 255, 1)
            gt_color = (0, 0, 0, 1)
            skeleton_radius = 0.03
            camera_scale = 2

            hsfm_node = Node("HSfM")
            kineo_node = Node("Kineo")
            viewer.scene.add(hsfm_node)
            viewer.scene.add(kineo_node)

            add_keypoints_3d_and_cameras(
                node=hsfm_node,
                camera_extrinsics=hsfm_camera_extrinsics,
                camera_intrinsics=hsfm_camera_intrinsics,
                keypoints_3d=hsfm_keypoints_3d,
                camera_scale=camera_scale,
                skeleton_radius=skeleton_radius,
                cameras_colors=hsfm_color,
                subjects_colors=hsfm_color,
            )
            add_keypoints_3d_and_cameras(
                node=hsfm_node,
                camera_extrinsics=kineo_camera_extrinsics,
                camera_intrinsics=kineo_camera_intrinsics,
                keypoints_3d=kineo_keypoints_3d,
                cameras_colors=kineo_color,
                subjects_colors=kineo_color,
                camera_scale=camera_scale,
                skeleton_radius=skeleton_radius,
            )
            add_keypoints_3d_and_cameras(
                node=hsfm_node,
                camera_extrinsics=gt_camera_extrinsics,
                camera_intrinsics=gt_camera_intrinsics,
                keypoints_3d=gt_keypoints_3d,
                cameras_colors=gt_color,
                subjects_colors=gt_color,
                camera_scale=camera_scale,
                skeleton_radius=skeleton_radius,
            )

            add_keypoints_3d_and_cameras(
                node=kineo_node,
                camera_extrinsics=kineo_camera_extrinsics,
                camera_intrinsics=kineo_camera_intrinsics,
                keypoints_3d=kineo_keypoints_3d,
                cameras_colors=kineo_color,
                subjects_colors=kineo_color,
                camera_scale=camera_scale,
                skeleton_radius=skeleton_radius,
            )
            add_keypoints_3d_and_cameras(
                node=kineo_node,
                camera_extrinsics=hsfm_camera_extrinsics,
                camera_intrinsics=hsfm_camera_intrinsics,
                keypoints_3d=hsfm_keypoints_3d,
                cameras_colors=hsfm_color,
                subjects_colors=hsfm_color,
                camera_scale=camera_scale,
                skeleton_radius=skeleton_radius,
            )
            add_keypoints_3d_and_cameras(
                node=kineo_node,
                camera_extrinsics=gt_camera_extrinsics,
                camera_intrinsics=gt_camera_intrinsics,
                keypoints_3d=gt_keypoints_3d,
                cameras_colors=gt_color,
                subjects_colors=gt_color,
                camera_scale=camera_scale,
                skeleton_radius=skeleton_radius,
            )

            kineo_frames_dir = os.path.join(
                args.output_dir, f"{sequence['sequence_name']}_kineo"
            )
            hsfm_frames_dir = os.path.join(
                args.output_dir, f"{sequence['sequence_name']}_hsfm"
            )

            # viewer.run()

            if not args.skip_videos:
                kineo_node.enabled = True
                hsfm_node.enabled = False
                video_path = os.path.join(
                    args.output_dir, f"{sequence['sequence_name']}_kineo.mp4"
                )
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(kineo_frames_dir):
                    shutil.rmtree(kineo_frames_dir, ignore_errors=True)
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                os.makedirs(kineo_frames_dir, exist_ok=True)
                viewer.save_video(
                    frame_dir=kineo_frames_dir,
                    video_dir=video_path,
                    output_fps=10,
                    ensure_no_overwrite=False,
                )

                hsfm_node.enabled = True
                kineo_node.enabled = False
                video_path = os.path.join(
                    args.output_dir, f"{sequence['sequence_name']}_hsfm.mp4"
                )
                if os.path.exists(video_path):
                    os.remove(video_path)
                if os.path.exists(hsfm_frames_dir):
                    shutil.rmtree(hsfm_frames_dir, ignore_errors=True)
                os.makedirs(os.path.dirname(video_path), exist_ok=True)
                os.makedirs(hsfm_frames_dir, exist_ok=True)
                viewer.save_video(
                    frame_dir=hsfm_frames_dir,
                    video_dir=video_path,
                    output_fps=10,
                    ensure_no_overwrite=False,
                )

            seq_name = sequence["sequence_name"]

            latex_row = latex_table_row_template.replace("{seq_name_escaped}", seq_name.replace("_", "\_"))
            latex_row = latex_row.replace("{seq_name}", seq_name)
            latex_row = latex_row.replace("{sanitized_seq_name}", sanitized_seq_name)
            latex_row = latex_row.replace("{hsfm_min}", f"{hsfm_min_median_max['min_w_mpjpe_value']:.{n_decimals}f}")
            latex_row = latex_row.replace("{hsfm_median}", f"{hsfm_min_median_max['median_w_mpjpe_value']:.{n_decimals}f}")
            latex_row = latex_row.replace("{hsfm_max}", f"{hsfm_min_median_max['max_w_mpjpe_value']:.{n_decimals}f}")
            latex_row = latex_row.replace("{hsfm_mean}", f"{hsfm_min_median_max['mean_w_mpjpe_value']:.{n_decimals}f}")
            latex_row = latex_row.replace("{hsfm_std}", f"{hsfm_min_median_max['std_w_mpjpe_value']:.{n_decimals}f}")
            latex_row = latex_row.replace("{kineo_min}", f"{kineo_min_median_max['min_w_mpjpe_value']:.{n_decimals}f}")
            latex_row = latex_row.replace("{kineo_median}", f"{kineo_min_median_max['median_w_mpjpe_value']:.{n_decimals}f}")
            latex_row = latex_row.replace("{kineo_max}", f"{kineo_min_median_max['max_w_mpjpe_value']:.{n_decimals}f}")
            latex_row = latex_row.replace("{kineo_mean}", f"{kineo_min_median_max['mean_w_mpjpe_value']:.{n_decimals}f}")
            latex_row = latex_row.replace("{kineo_std}", f"{kineo_min_median_max['std_w_mpjpe_value']:.{n_decimals}f}")
            
            latex_table_rows.append(latex_row)

            def copy_frame(frame_path, output_path):
                frame = cv2.imread(frame_path)
                if downscale_factor is not None:
                    frame = cv2.resize(frame, (int(frame.shape[1] / downscale_factor), int(frame.shape[0] / downscale_factor)))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, frame)

            copy_frame(
                os.path.join(
                    hsfm_frames_dir,
                    "0000",
                    f"frame_{hsfm_min_median_max['median_w_mpjpe_frame']:06d}.png",
                ),
                os.path.join(
                    latex_comparative_figs_dir, f"{sanitized_seq_name}_hsfm_median.jpg"
                ),
            )

            copy_frame(
                os.path.join(
                    hsfm_frames_dir,
                    "0000",
                    f"frame_{hsfm_min_median_max['min_w_mpjpe_frame']:06d}.png",
                ),
                os.path.join(
                    latex_comparative_figs_dir, f"{sanitized_seq_name}_hsfm_min.jpg"
                ),
            )

            copy_frame(
                os.path.join(
                    hsfm_frames_dir,
                    "0000",
                    f"frame_{hsfm_min_median_max['max_w_mpjpe_frame']:06d}.png",
                ),
                os.path.join(
                    latex_comparative_figs_dir, f"{sanitized_seq_name}_hsfm_max.jpg"
                ),
            )

            copy_frame(
                os.path.join(
                    kineo_frames_dir,
                    "0000",
                    f"frame_{kineo_min_median_max['median_w_mpjpe_frame']:06d}.png",
                ),
                os.path.join(
                    latex_comparative_figs_dir, f"{sanitized_seq_name}_kineo_median.jpg"
                ),
            )

            copy_frame(
                os.path.join(
                    kineo_frames_dir,
                    "0000",
                    f"frame_{kineo_min_median_max['min_w_mpjpe_frame']:06d}.png",
                ),
                os.path.join(
                    latex_comparative_figs_dir, f"{sanitized_seq_name}_kineo_min.jpg"
                ),
            )

            copy_frame(
                os.path.join(
                    kineo_frames_dir,
                    "0000",
                    f"frame_{kineo_min_median_max['max_w_mpjpe_frame']:06d}.png",
                ),
                os.path.join(
                    latex_comparative_figs_dir, f"{sanitized_seq_name}_kineo_max.jpg"
                ),
            )
        except Exception:
            print(
                f"Error processing sequence {sequence['sequence_name']}: {traceback.format_exc()}"
            )
            continue

    latex_table_rows_str = "\n\hline\n".join(latex_table_rows)

    with open(os.path.join(os.path.dirname(__file__), "h36m_comparison_table_template.tex"), "r") as f:
        latex_table_template = f.read()
    latex_table = latex_table_template.replace("{latex_table_rows_str}", latex_table_rows_str)

    with open(os.path.join(latex_comparative_figs_dir, "h36m_comparison_table.tex"), "w") as f:
        f.write(latex_table)
    print(
        f"Saved latex table to {os.path.join(latex_comparative_figs_dir, 'h36m_comparison_table.tex')}"
    )
