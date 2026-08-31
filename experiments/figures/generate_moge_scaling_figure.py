import torch
import orjson
import os
import numpy as np
import cv2
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.keypoints_format import H36M_17_KEYPOINTS_FORMAT
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from aitviewer.scene.camera import PinholeCamera, OpenCVCamera
from aitviewer.scene.node import Node
from aitviewer.headless import HeadlessRenderer
from aitviewer.viewer import Viewer
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.lines import Lines
from aitviewer.configuration import CONFIG as C

from moge.model.v2 import MoGeModel

from kineo.geometry.conversions import (
    convert_world_points_from_opencv_to_opengl,
    convert_Rt_from_opencv_to_opengl,
)
from kineo.geometry.transformations import inverse_Rt

H36M_KPS_FORMAT = H36M_17_KEYPOINTS_FORMAT

C.window_type = "pyglet"

# Fix for camera frustum rendering
np.float = float


def load_gt_keypoints_3d(
    dataset_dir: str,
    sequence: dict,
) -> torch.Tensor:
    keypoints_3d_file = sequence["annotations"]["keypoints_3d"]
    keypoints_3d_file = os.path.join(dataset_dir, keypoints_3d_file)

    with open(keypoints_3d_file, "rb") as f:
        gt_keypoints_3d = Keypoints3DAnnotations.from_dict(orjson.loads(f.read()))

    gt_kps_3d = torch.stack(
        [a.xyz for a in sorted(gt_keypoints_3d.annotations, key=lambda x: x.frame_idx)]
    )
    gt_kps_3d = convert_world_points_from_opencv_to_opengl(gt_kps_3d)

    return gt_kps_3d


def load_gt_camera_params(
    dataset_dir: str,
    sequence: dict,
    camera_id: str,
) -> torch.Tensor:
    camera_extrinsics_file = sequence["annotations"]["cameras_extrinsics"]
    camera_extrinsics_file = os.path.join(dataset_dir, camera_extrinsics_file)

    camera_intrinsics_file = sequence["annotations"]["cameras_intrinsics"]
    camera_intrinsics_file = os.path.join(dataset_dir, camera_intrinsics_file)

    with open(camera_extrinsics_file, "rb") as f:
        camera_extrinsics = CameraExtrinsicsAnnotations.from_dict(
            orjson.loads(f.read())
        )

    with open(camera_intrinsics_file, "rb") as f:
        camera_intrinsics = CameraIntrinsicsAnnotations.from_dict(
            orjson.loads(f.read())
        )

    gt_Rt = camera_extrinsics.filter_by_view_id(camera_id).first_or_default().Rt
    gt_Rt = convert_Rt_from_opencv_to_opengl(gt_Rt)

    gt_K = camera_intrinsics.filter_by_view_id(camera_id).first_or_default().K
    return gt_Rt, gt_K


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal")
    moge_model.to(device)

    sequences_file = os.path.join(dataset_dir, "h36m_protocol1_sequences.json")
    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())
    sequence = next(s for s in sequences if s["sequence_name"] == "S1_Directions 1")

    frame_idx = 0
    kps_3d = load_gt_keypoints_3d(dataset_dir, sequence).to(device)
    target_kps_3d = kps_3d[frame_idx]
    kps_connectivity = torch.tensor(
        H36M_KPS_FORMAT.keypoints_connectivity, dtype=torch.int32, device=device
    )

    views_ids = list(sequence["views"].keys())
    view_id = views_ids[0]
    view = sequence["views"][view_id]
    view_video_path = os.path.join(args.dataset_dir, view["video_path"])
    view_selected_frames = range(
        view["selected_frames"]["start"],
        view["selected_frames"]["stop"],
        view["selected_frames"]["step"],
    )
    video_frame_idx = view_selected_frames[frame_idx]

    gt_Rt, gt_K = load_gt_camera_params(dataset_dir, sequence, view_id)

    cap = cv2.VideoCapture(view_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)

    ret, img = cap.read()
    if not ret:
        raise ValueError("Failed to read video")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    img_h, img_w = img.shape[1:]

    fov_x = (2 * torch.atan(img_w / (2 * gt_K[0, 0]))).rad2deg()

    moge_output = moge_model.infer(img, fov_x=fov_x)

    points = moge_output["points"].reshape(-1, 3)
    depth = moge_output["depth"]
    colors = img.permute(1, 2, 0).reshape(-1, 3)
    colors = torch.cat([colors, torch.ones_like(colors[:, 0:1])], dim=1)

    res = 4096
    viewer = HeadlessRenderer(size=(res, res))
    # viewer = Viewer(size=(800, 800))

    camera = PinholeCamera(
        position=np.array([4.859, 1.945, -5.461]),
        target=np.array([1.417, 1.193, -3.587]),
        cols=res,
        rows=res,
    )
    viewer.scene.add(camera)
    viewer.scene.camera = camera

    viewer.scene.origin.enabled = False
    viewer.scene.floor.c1 = (208 / 255, 206 / 255, 219 / 255, 1.0)
    viewer.scene.floor.c2 = (201 / 255, 199 / 255, 211 / 255, 1.0)

    h36m_skeleton = Skeletons(
        joint_positions=target_kps_3d.unsqueeze(0).cpu().numpy(),
        joint_connections=kps_connectivity.cpu().numpy(),
        name="H3.6M Skeleton",
        color=(0.0, 0.0, 0.0, 1.0),
    )

    moge_node = Node(name="MoGe Output")
    pcd_cam = OpenCVCamera(
        K=gt_K.cpu().numpy(), Rt=np.eye(4)[:3, :], cols=img_w, rows=img_h
    )
    pcd_cam.show_frustum(width=img_w, height=img_h, distance=10.0)

    # Create a copy to control the frustum lines width
    cam_color = (83 / 255, 92 / 255, 104 / 255, 1.0)
    frustum_lines_width = 0.001
    frustum_copy = Lines(
        pcd_cam.frustum.lines,
        r_base=frustum_lines_width,
        r_tip=frustum_lines_width,
        color=cam_color,
        mode="lines",
        cast_shadow=False,
        name="Frustum",
    )
    pcd_cam.color = cam_color
    pcd_cam.active_color = cam_color
    pcd_cam.inactive_color = cam_color
    pcd_cam.material.color = cam_color
    pcd_cam.mesh.color = cam_color
    frustum_copy.material = pcd_cam.material
    pcd_cam.add(frustum_copy)
    pcd_cam.hide_frustum()

    pcd = PointClouds(
        points=points.unsqueeze(0).cpu().numpy(),
        colors=colors.unsqueeze(0).cpu().numpy(),
        name="MoGe Points",
    )
    moge_node.add(pcd)
    moge_node.add(pcd_cam)
    pcd.enabled = False

    if pcd_cam.origin is not None:
        pcd_cam.origin.enabled = False

    depth_np = depth.cpu().numpy()
    depth_np = 255 - cv2.normalize(
        depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    depth_np = cv2.applyColorMap(depth_np, cv2.COLORMAP_MAGMA)
    depth_np = cv2.cvtColor(depth_np, cv2.COLOR_BGR2RGB)

    cam_billboard = Billboard.from_camera_and_distance(
        camera=pcd_cam,
        distance=1.5,
        cols=img_w,
        rows=img_h,
        textures=[depth_np],
    )
    cam_billboard.texture_alpha = 0.9
    moge_node.add(cam_billboard)

    moge_node.position = inverse_Rt(gt_Rt)[:3, 3].cpu().numpy()
    moge_node.rotation = inverse_Rt(gt_Rt)[:3, :3].cpu().numpy()

    viewer.scene.add(moge_node)
    viewer.scene.add(h36m_skeleton)

    kps_indices = torch.tensor([0, 11, 12, 14, 5])
    n_lines = kps_indices.shape[0]

    line_starts = moge_node.position[None, ...].repeat(n_lines, axis=0)
    line_ends = target_kps_3d[kps_indices].cpu().numpy()

    line_r_base = 0.001
    line_r_tip = 0.01

    line_strip = np.zeros((2 * n_lines, 3))
    line_strip[::2] = line_starts
    line_strip[1::2] = line_ends
    lines = Lines(
        line_strip,
        r_base=line_r_base,
        r_tip=line_r_tip,
        mode="lines",
        cast_shadow=False,
    )
    lines.color = (104 / 255, 109 / 255, 224 / 255, 1.0)
    lines.material.ambient = 0.5
    viewer.scene.add(lines)

    # viewer.run()

    os.makedirs(args.output_dir, exist_ok=True)

    viewer._init_scene()
    viewer.export_frame(
        os.path.join(args.output_dir, "moge_scaling_figure.png"),
        transparent_background=True,
    )
    print(f"Exported figure to {args.output_dir}/moge_scaling_figure.png")
