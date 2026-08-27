import torch
import orjson
import os
import numpy as np

from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.reconstructed_scene import WorldReconstructedSceneAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.bboxes_2d import BBox2DAnnotations

import pickle
from kineo.geometry.camera import inverse_Rt
from kineo.geometry.transformations import compute_similarity_transform
from kineo.geometry.conversions import convert_world_points_from_opencv_to_opengl

from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import PinholeCamera
from aitviewer.configuration import CONFIG as C
from aitviewer.scene.node import Node
from aitviewer.viewer import Viewer

from kineo.visualization.viz_3d import add_keypoints_3d
from kineo.visualization.viz_3d import add_cameras
from kineo.visualization.viz_3d import add_world_reconstruction

C.window_type = "pyglet"

def load_gt_annotations(
    dataset_dir: str,
    sequence: dict,
) -> tuple[Keypoints3DAnnotations, CameraIntrinsicsAnnotations, CameraExtrinsicsAnnotations]:
    camera_intrinsics_file = os.path.join(
        dataset_dir, sequence["annotations"]["cameras_intrinsics"]
    )
    keypoints_3d_file = os.path.join(
        dataset_dir, sequence["annotations"]["keypoints_3d"]
    )
    camera_extrinsics_file = os.path.join(
        dataset_dir, sequence["annotations"]["cameras_extrinsics"]
    )

    with open(keypoints_3d_file, "rb") as f:
        gt_keypoints_3d = Keypoints3DAnnotations.from_dict(
            orjson.loads(f.read())
        )

    with open(camera_intrinsics_file, "rb") as f:
        gt_camera_intrinsics = (
            CameraIntrinsicsAnnotations.from_dict(orjson.loads(f.read()))
        )

    with open(camera_extrinsics_file, "rb") as f:
        gt_camera_extrinsics = (
            CameraExtrinsicsAnnotations.from_dict(orjson.loads(f.read()))
        )

    return (
        gt_keypoints_3d,
        gt_camera_intrinsics,
        gt_camera_extrinsics,
    )


def load_predicted_annotations(
    predictions_dir: str,
    sequence: dict,
) -> tuple[Keypoints3DAnnotations, CameraIntrinsicsAnnotations, CameraExtrinsicsAnnotations, WorldReconstructedSceneAnnotations]:
    seq_name = sequence["sequence_name"]

    pred_keypoints_3d_file = os.path.join(
        predictions_dir, seq_name, "keypoints_3d.pkl"
    )
    pred_camera_intrinsics_file = os.path.join(
        predictions_dir, seq_name, "cameras_intrinsics.pkl"
    )
    pred_camera_extrinsics_file = os.path.join(
        predictions_dir, seq_name, "cameras_extrinsics.pkl"
    )
    pred_world_reconstructed_scene_file = os.path.join(
        predictions_dir, seq_name, "world_reconstructed_scene.pkl"
    )
    with open(pred_keypoints_3d_file, "rb") as f:
        pred_keypoints_3d = Keypoints3DAnnotations.from_dict(
            pickle.load(f)
        )
    with open(pred_camera_intrinsics_file, "rb") as f:
        pred_camera_intrinsics = (
            CameraIntrinsicsAnnotations.from_dict(pickle.load(f))
        )
    with open(pred_camera_extrinsics_file, "rb") as f:
        pred_camera_extrinsics = (
            CameraExtrinsicsAnnotations.from_dict(pickle.load(f))
        )
    with open(pred_world_reconstructed_scene_file, "rb") as f:
        pred_world_reconstructed_scene = (
            WorldReconstructedSceneAnnotations.from_dict(pickle.load(f))
        )

    return (
        pred_keypoints_3d,
        pred_camera_intrinsics,
        pred_camera_extrinsics,
        pred_world_reconstructed_scene,
    )


def load_gt_keypoints_and_bboxes_2d(
    dataset_dir: str,
    sequence: dict,
) -> tuple[Keypoints2DAnnotations, BBox2DAnnotations]:
    keypoints_2d_file = sequence["annotations"]["keypoints_2d"]
    keypoints_2d_file = os.path.join(dataset_dir, keypoints_2d_file)

    with open(keypoints_2d_file, "rb") as f:
        gt_keypoints_2d = Keypoints2DAnnotations.from_dict(orjson.loads(f.read()))

    bboxes_2d_file = sequence["annotations"]["bboxes_2d"]
    bboxes_2d_file = os.path.join(dataset_dir, bboxes_2d_file)

    with open(bboxes_2d_file, "rb") as f:
        gt_bboxes_2d = BBox2DAnnotations.from_dict(orjson.loads(f.read()))

    return gt_keypoints_2d, gt_bboxes_2d

def align_kineo_pred_to_gt(
    pred_camera_extrinsics: CameraExtrinsicsAnnotations,
    pred_keypoints_3d: Keypoints3DAnnotations,
    pred_world_reconstructed_scene: WorldReconstructedSceneAnnotations,
    gt_camera_extrinsics: CameraExtrinsicsAnnotations,
    estimate_scale: bool = True,
) -> tuple[CameraExtrinsicsAnnotations, Keypoints3DAnnotations, WorldReconstructedSceneAnnotations]:
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
    aligned_pred_world_reconstructed_scene = pred_world_reconstructed_scene.apply_similarity_transform(
        R=R, t=t, s=s
    )

    return aligned_pred_keypoints_3d, aligned_pred_camera_extrinsics, aligned_pred_world_reconstructed_scene


def render_frame(
    viewer: HeadlessRenderer,
    frame_idx: int,
    camera_position: np.ndarray,
    camera_target: np.ndarray,
    gt_camera_extrinsics: CameraExtrinsicsAnnotations,
    gt_camera_intrinsics: CameraIntrinsicsAnnotations,
    pred_camera_extrinsics: CameraExtrinsicsAnnotations,
    pred_camera_intrinsics: CameraIntrinsicsAnnotations,
    pred_keypoints_3d: Keypoints3DAnnotations,
    pred_world_reconstructed_scene: WorldReconstructedSceneAnnotations,
    output_path: str,
    gt_color: tuple[float, float, float, float] = (0, 0, 0, 1),
    pred_color: tuple[float, float, float, float] = (0, 0, 1, 1),
    skeleton_radius: float = 0.03,
    camera_scale: float = 1,
    world_points_size: float = 0.7,
    world_points_confidence_threshold: float = 0.2,
    world_z_clipping_threshold: float = None,
    downscale_factor: float | None = None,
    max_world_points_to_show: int = 1_000_000,
):
    gt_node = Node("GT")
    pred_node = Node("Pred")
    viewer.scene.add(gt_node)
    viewer.scene.add(pred_node)

    add_cameras(
        node=gt_node,
        camera_extrinsics=gt_camera_extrinsics,
        camera_intrinsics=gt_camera_intrinsics,
        cameras_colors=gt_color,
        camera_scale=camera_scale,
        convert_coordinates_to_opengl=True,
    )

    add_cameras(
        node=pred_node,
        camera_extrinsics=pred_camera_extrinsics,
        camera_intrinsics=pred_camera_intrinsics,
        cameras_colors=pred_color,
        camera_scale=camera_scale,
        convert_coordinates_to_opengl=True,
    )
    add_keypoints_3d(
        node=pred_node,
        keypoints_3d=pred_keypoints_3d,
        subjects_colors=pred_color,
        skeleton_radius=skeleton_radius,
        convert_coordinates_to_opengl=True,
    )
    add_world_reconstruction(
        node=pred_node,
        world_reconstruction=pred_world_reconstructed_scene,
        world_points_size=world_points_size,
        world_z_clipping_threshold=world_z_clipping_threshold,
        max_world_points_to_show=max_world_points_to_show,
        world_points_confidence_threshold=world_points_confidence_threshold,
        convert_coordinates_to_opengl=True,
    )

    camera = PinholeCamera(
        position=camera_position,
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
    print(f"Exported frame to {output_path}")

    viewer.scene.remove(camera)
    viewer.scene.remove(gt_node)
    viewer.scene.remove(pred_node)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "h36m_dataset_dir",
        type=str,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "h36m_pred_annotations_dir",
        type=str,
        help="Path to predicted annotations directory",
    )
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()
    dataset_dir = args.h36m_dataset_dir
    pred_annotations_dir = args.h36m_pred_annotations_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    sequences_file = os.path.join(dataset_dir, "h36m_protocol1_sequences.json")
    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())
    sequence = next(s for s in sequences if s["sequence_name"] == "S9_Directions 1")

    moge_pred_annotations_dir = os.path.join(pred_annotations_dir, "h36m_scene_reconstruction_moge", "annotations")
    vggt_pred_annotations_dir = os.path.join(pred_annotations_dir, "h36m_scene_reconstruction_vggt", "annotations")

    gt_keypoints_3d_annotations, gt_camera_intrinsics_annotations, gt_camera_extrinsics_annotations = load_gt_annotations(dataset_dir, sequence)
    moge_pred_keypoints_3d_annotations, moge_pred_camera_intrinsics_annotations, moge_pred_camera_extrinsics_annotations, moge_pred_world_reconstructed_scene_annotations = load_predicted_annotations(moge_pred_annotations_dir, sequence)
    vggt_pred_keypoints_3d_annotations, vggt_pred_camera_intrinsics_annotations, vggt_pred_camera_extrinsics_annotations, vggt_pred_world_reconstructed_scene_annotations = load_predicted_annotations(vggt_pred_annotations_dir, sequence)

    moge_pred_keypoints_3d_annotations, moge_pred_camera_extrinsics_annotations, moge_pred_world_reconstructed_scene_annotations = align_kineo_pred_to_gt(
        pred_camera_extrinsics=moge_pred_camera_extrinsics_annotations,
        pred_keypoints_3d=moge_pred_keypoints_3d_annotations,
        pred_world_reconstructed_scene=moge_pred_world_reconstructed_scene_annotations,
        gt_camera_extrinsics=gt_camera_extrinsics_annotations,
    )
    vggt_pred_keypoints_3d_annotations, vggt_pred_camera_extrinsics_annotations, vggt_pred_world_reconstructed_scene_annotations = align_kineo_pred_to_gt(
        pred_camera_extrinsics=vggt_pred_camera_extrinsics_annotations,
        pred_keypoints_3d=vggt_pred_keypoints_3d_annotations,
        pred_world_reconstructed_scene=vggt_pred_world_reconstructed_scene_annotations,
        gt_camera_extrinsics=gt_camera_extrinsics_annotations,
    )

    cam_position=np.array([7.496, 7.5, 13.502])
    cam_target=np.array([0.2, 0.068, -0.2])

    kineo_color = (41 / 255, 128 / 255, 185 / 255, 1)
    skeleton_radius = 0.03
    camera_scale = 2

    viewer = HeadlessRenderer(size=(1000, 1000))
    render_frame(
        viewer=viewer,
        frame_idx=0,
        camera_position=cam_position,
        camera_target=cam_target,
        gt_camera_extrinsics=gt_camera_extrinsics_annotations,
        gt_camera_intrinsics=gt_camera_intrinsics_annotations,
        pred_camera_extrinsics=moge_pred_camera_extrinsics_annotations,
        pred_camera_intrinsics=moge_pred_camera_intrinsics_annotations,
        pred_keypoints_3d=moge_pred_keypoints_3d_annotations,
        pred_world_reconstructed_scene=moge_pred_world_reconstructed_scene_annotations,
        world_z_clipping_threshold=1.8,
        world_points_size=1.0,
        pred_color=kineo_color,
        camera_scale=camera_scale,
        skeleton_radius=skeleton_radius,
        output_path=os.path.join(output_dir, "moge_scene_reconstruction.png"),
    )

    render_frame(
        viewer=viewer,
        frame_idx=0,
        camera_position=cam_position,
        camera_target=cam_target,
        gt_camera_extrinsics=gt_camera_extrinsics_annotations,
        gt_camera_intrinsics=gt_camera_intrinsics_annotations,
        pred_camera_extrinsics=vggt_pred_camera_extrinsics_annotations,
        pred_camera_intrinsics=vggt_pred_camera_intrinsics_annotations,
        pred_keypoints_3d=vggt_pred_keypoints_3d_annotations,
        pred_world_reconstructed_scene=vggt_pred_world_reconstructed_scene_annotations,
        world_z_clipping_threshold=1.8,
        world_points_size=1.0,
        pred_color=kineo_color,
        camera_scale=camera_scale,
        skeleton_radius=skeleton_radius,
        output_path=os.path.join(output_dir, "vggt_scene_reconstruction.png"),
    )