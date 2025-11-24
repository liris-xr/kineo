import torch
import pickle
import orjson
import os
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
from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import PinholeCamera
from aitviewer.configuration import CONFIG as C
from aitviewer.scene.node import Node

from kineo.geometry.conversions import convert_world_points_from_opencv_to_opengl
from kineo.visualization.viz_3d import add_keypoints_3d

import numpy as np
from kineo.eval.human_metrics import compute_human_metrics, get_min_median_max_frames
import traceback

C.window_type = "pyglet"


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

def render_gt_pelvis_focused_frame(
    viewer: HeadlessRenderer,
    frame_idx: int,
    hsfm_pred_keypoints_3d: Keypoints3DAnnotations,
    kineo_pred_keypoints_3d: Keypoints3DAnnotations,
    gt_keypoints_3d: Keypoints3DAnnotations,
    output_path: str,
    camera_distance: float = 3.0,
    gt_color: tuple[float, float, float, float] = (0, 0, 0, 1),
    hsfm_pred_color: tuple[float, float, float, float] = (1, 0, 0, 1),
    kineo_pred_color: tuple[float, float, float, float] = (0, 0, 1, 1),
    skeleton_radius: float = 0.03,
    downscale_factor: float | None = None,
):
    gt_node = Node("GT")
    pred_node = Node("Pred")
    viewer.scene.add(gt_node)
    viewer.scene.add(pred_node)

    add_keypoints_3d(
        node=gt_node,
        keypoints_3d=gt_keypoints_3d,
        subjects_colors=gt_color,
        skeleton_radius=skeleton_radius,
    )
    add_keypoints_3d(
        node=pred_node,
        keypoints_3d=hsfm_pred_keypoints_3d,
        subjects_colors=hsfm_pred_color,
        skeleton_radius=skeleton_radius,
    )
    add_keypoints_3d(
        node=pred_node,
        keypoints_3d=kineo_pred_keypoints_3d,
        subjects_colors=kineo_pred_color,
        skeleton_radius=skeleton_radius,
    )

    kps = gt_keypoints_3d.filter_by_frame_idx(frame_idx).first_or_default()
    kps = convert_world_points_from_opencv_to_opengl(kps.xyz)
    pelvis_pose = kps[0].cpu().numpy()
    left_hip_pose = kps[4].cpu().numpy()
    right_vec = (left_hip_pose - pelvis_pose) / np.linalg.norm(
        left_hip_pose - pelvis_pose
    )
    up_vec = np.array([0, 1, 0])
    forward_vec = np.cross(right_vec, up_vec)

    camera_pos = pelvis_pose + forward_vec * camera_distance
    camera_target = pelvis_pose

    camera = PinholeCamera(
        position=camera_pos,
        target=camera_target,
        cols=viewer.window_size[0],
        rows=viewer.window_size[1],
    )
    viewer.scene.add(camera)
    viewer.scene.camera = camera
    viewer.scene.origin.enabled = False
    viewer.scene.floor.enabled = False
    viewer.run_animations = False
    viewer.scene.current_frame_id = frame_idx

    viewer.save_frame(output_path, scale_factor=1 / downscale_factor if downscale_factor is not None else None)

    viewer.scene.remove(camera)
    viewer.scene.remove(gt_node)
    viewer.scene.remove(pred_node)


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
        "--downscale-factor", type=float, help="Downscale factor for the images", default=None,
    )
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    downscale_factor = args.downscale_factor

    sequences_file = os.path.join(dataset_dir, "h36m_protocol1_sequences.json")

    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())

    hsfm_best_seq_name = None
    hsfm_best_seq_mean = None
    hsfm_best_seq_hsfm_metrics = None
    hsfm_best_seq_kineo_metrics = None

    hsfm_best_seq_hsfm_pred_camera_extrinsics = None
    hsfm_best_seq_hsfm_pred_camera_intrinsics = None
    hsfm_best_seq_hsfm_pred_keypoints_3d = None

    hsfm_best_seq_kineo_pred_camera_extrinsics = None
    hsfm_best_seq_kineo_pred_camera_intrinsics = None
    hsfm_best_seq_kineo_pred_keypoints_3d = None

    hsfm_best_seq_gt_camera_extrinsics = None
    hsfm_best_seq_gt_camera_intrinsics = None
    hsfm_best_seq_gt_keypoints_3d = None

    kineo_best_seq_name = None
    kineo_best_seq_mean = None
    kineo_best_seq_kineo_metrics = None
    kineo_best_seq_hsfm_metrics = None

    kineo_best_seq_kineo_pred_camera_extrinsics = None
    kineo_best_seq_kineo_pred_camera_intrinsics = None
    kineo_best_seq_kineo_pred_keypoints_3d = None

    kineo_best_seq_hsfm_pred_camera_extrinsics = None
    kineo_best_seq_hsfm_pred_camera_intrinsics = None
    kineo_best_seq_hsfm_pred_keypoints_3d = None

    kineo_best_seq_gt_camera_extrinsics = None
    kineo_best_seq_gt_camera_intrinsics = None
    kineo_best_seq_gt_keypoints_3d = None

    for sequence in tqdm(sequences, desc="Processing sequences"):
        if sequence["split"] != "val":
            continue

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

            hsfm_human_metrics = compute_human_metrics(
                gt_keypoints_3d_annotations=gt_keypoints_3d,
                gt_cam_extrinsics_annotations=gt_camera_extrinsics,
                pred_keypoints_3d_annotations=hsfm_keypoints_3d,
                pred_cam_extrinsics_annotations=hsfm_camera_extrinsics,
            )
            kineo_human_metrics = compute_human_metrics(
                gt_keypoints_3d_annotations=gt_keypoints_3d,
                gt_cam_extrinsics_annotations=gt_camera_extrinsics,
                pred_keypoints_3d_annotations=kineo_keypoints_3d,
                pred_cam_extrinsics_annotations=kineo_camera_extrinsics,
            )

            hsfm_min_median_max = get_min_median_max_frames(human_metrics=hsfm_human_metrics)
            kineo_min_median_max = get_min_median_max_frames(human_metrics=kineo_human_metrics)

            if kineo_min_median_max is None:
                print(f"No valid Kineo metrics for sequence {sequence['sequence_name']}")
                continue
            if hsfm_min_median_max is None:
                print(f"No valid HSfM metrics for sequence {sequence['sequence_name']}")
                continue

            seq_name = sequence["sequence_name"]

            # Select the best sequence for each method (HSfM and Kineo) based on the median W-MPJPE
            if (
                hsfm_best_seq_mean is None
                or hsfm_min_median_max["mean_w_mpjpe_value"] < hsfm_best_seq_mean
            ):
                hsfm_best_seq_name = seq_name
                hsfm_best_seq_mean = hsfm_min_median_max["mean_w_mpjpe_value"]
                hsfm_best_seq_hsfm_metrics = hsfm_min_median_max
                hsfm_best_seq_kineo_metrics = kineo_min_median_max

                hsfm_best_seq_hsfm_pred_camera_extrinsics = hsfm_camera_extrinsics
                hsfm_best_seq_hsfm_pred_camera_intrinsics = hsfm_camera_intrinsics
                hsfm_best_seq_hsfm_pred_keypoints_3d = hsfm_keypoints_3d

                hsfm_best_seq_kineo_pred_camera_extrinsics = kineo_camera_extrinsics
                hsfm_best_seq_kineo_pred_camera_intrinsics = kineo_camera_intrinsics
                hsfm_best_seq_kineo_pred_keypoints_3d = kineo_keypoints_3d

                hsfm_best_seq_gt_camera_extrinsics = gt_camera_extrinsics
                hsfm_best_seq_gt_camera_intrinsics = gt_camera_intrinsics
                hsfm_best_seq_gt_keypoints_3d = gt_keypoints_3d

            if (
                kineo_best_seq_mean is None
                or kineo_min_median_max["mean_w_mpjpe_value"] < kineo_best_seq_mean
            ):
                kineo_best_seq_name = seq_name
                kineo_best_seq_mean = kineo_min_median_max["mean_w_mpjpe_value"]
                kineo_best_seq_kineo_metrics = kineo_min_median_max
                kineo_best_seq_hsfm_metrics = hsfm_min_median_max

                kineo_best_seq_kineo_pred_camera_extrinsics = (
                    kineo_camera_extrinsics
                )
                kineo_best_seq_kineo_pred_camera_intrinsics = (
                    kineo_camera_intrinsics
                )
                kineo_best_seq_kineo_pred_keypoints_3d = kineo_keypoints_3d

                kineo_best_seq_hsfm_pred_camera_extrinsics = hsfm_camera_extrinsics
                kineo_best_seq_hsfm_pred_camera_intrinsics = hsfm_camera_intrinsics
                kineo_best_seq_hsfm_pred_keypoints_3d = hsfm_keypoints_3d

                kineo_best_seq_gt_camera_extrinsics = gt_camera_extrinsics
                kineo_best_seq_gt_camera_intrinsics = gt_camera_intrinsics
                kineo_best_seq_gt_keypoints_3d = gt_keypoints_3d
        except Exception:
            print(
                f"Error processing sequence {sequence['sequence_name']}: {traceback.format_exc()}"
            )
            continue

    latex_median_frame_figs_dir = os.path.join(
        args.output_dir, "latex_median_frame_figs"
    )
    os.makedirs(latex_median_frame_figs_dir, exist_ok=True)

    estimate_scale = False

    hsfm_best_seq_hsfm_pred_camera_extrinsics, hsfm_best_seq_hsfm_pred_keypoints_3d = (
        align_hsfm_pred_to_gt(
            pred_camera_extrinsics=hsfm_best_seq_hsfm_pred_camera_extrinsics,
            pred_keypoints_3d=hsfm_best_seq_hsfm_pred_keypoints_3d,
            gt_camera_extrinsics=hsfm_best_seq_gt_camera_extrinsics,
            estimate_scale=estimate_scale,
        )
    )
    (
        hsfm_best_seq_kineo_pred_camera_extrinsics,
        hsfm_best_seq_kineo_pred_keypoints_3d,
    ) = align_kineo_pred_to_gt(
        pred_keypoints_3d=hsfm_best_seq_kineo_pred_keypoints_3d,
        pred_camera_extrinsics=hsfm_best_seq_kineo_pred_camera_extrinsics,
        gt_camera_extrinsics=hsfm_best_seq_gt_camera_extrinsics,
        estimate_scale=estimate_scale,
    )

    (
        kineo_best_seq_kineo_pred_camera_extrinsics,
        kineo_best_seq_kineo_pred_keypoints_3d,
    ) = align_kineo_pred_to_gt(
        pred_camera_extrinsics=kineo_best_seq_kineo_pred_camera_extrinsics,
        pred_keypoints_3d=kineo_best_seq_kineo_pred_keypoints_3d,
        gt_camera_extrinsics=kineo_best_seq_gt_camera_extrinsics,
        estimate_scale=estimate_scale,
    )

    (
        kineo_best_seq_hsfm_pred_camera_extrinsics,
        kineo_best_seq_hsfm_pred_keypoints_3d,
    ) = align_hsfm_pred_to_gt(
        pred_keypoints_3d=kineo_best_seq_hsfm_pred_keypoints_3d,
        pred_camera_extrinsics=kineo_best_seq_hsfm_pred_camera_extrinsics,
        gt_camera_extrinsics=kineo_best_seq_gt_camera_extrinsics,
        estimate_scale=estimate_scale,
    )

    window_width = 500
    window_height = 800
    viewer = HeadlessRenderer(size=(window_width, window_height))
    # viewer = Viewer(size=(window_width, window_height))

    gt_color = (0, 0, 0, 1)
    kineo_color = (41 / 255, 128 / 255, 185 / 255, 1)
    hsfm_color = (1, 0, 0, 1)
    skeleton_radius = 0.02

    camera_distance = 3.0

    print(hsfm_best_seq_name)
    print(kineo_best_seq_name)

    for key in ["min", "max", "median"]:
        render_gt_pelvis_focused_frame(
            viewer=viewer,
            frame_idx=hsfm_best_seq_hsfm_metrics[f"{key}_w_mpjpe_frame"],
            hsfm_pred_keypoints_3d=hsfm_best_seq_hsfm_pred_keypoints_3d,
            kineo_pred_keypoints_3d=hsfm_best_seq_kineo_pred_keypoints_3d,
            gt_keypoints_3d=hsfm_best_seq_gt_keypoints_3d,
            output_path=os.path.join(
                latex_median_frame_figs_dir, f"hsfm_best_seq_{key}_frame.png"
            ),
            camera_distance=camera_distance,
            gt_color=gt_color,
            hsfm_pred_color=hsfm_color,
            kineo_pred_color=kineo_color,
            skeleton_radius=skeleton_radius,
            downscale_factor=downscale_factor,
        )

        render_gt_pelvis_focused_frame(
            viewer=viewer,
            frame_idx=kineo_best_seq_hsfm_metrics[f"{key}_w_mpjpe_frame"],
            hsfm_pred_keypoints_3d=kineo_best_seq_hsfm_pred_keypoints_3d,
            kineo_pred_keypoints_3d=kineo_best_seq_kineo_pred_keypoints_3d,
            gt_keypoints_3d=kineo_best_seq_gt_keypoints_3d,
            output_path=os.path.join(
                latex_median_frame_figs_dir, f"kineo_best_seq_{key}_frame.png"
            ),
            camera_distance=camera_distance,
            gt_color=gt_color,
            hsfm_pred_color=hsfm_color,
            kineo_pred_color=kineo_color,
            skeleton_radius=skeleton_radius,
            downscale_factor=downscale_factor,
        )

    