import torch
import orjson
import os
import numpy as np
from smplx import SMPL, SMPLLayer
from smplx.lbs import vertices2joints
from tqdm import tqdm

from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.keypoints_format import H36M_17_KEYPOINTS_FORMAT
from aitviewer.scene.camera import WeakPerspectiveCamera

from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.smpl import SMPLSequence, SMPLLayer as AitSMPLLayer
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.configuration import CONFIG as C

from kineo.geometry.conversions import convert_world_points_from_opencv_to_opengl

H36M_KPS_FORMAT = H36M_17_KEYPOINTS_FORMAT

C.window_type = "pyglet"

# Fix to properly load smpl model without errors
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.long = np.int_
np.complex = np.complex_
np.object = np.object_
np.str = np.str_
np.unicode = np.unicode_


def _compute_bone_lengths(
    joints: torch.Tensor, joints_connectivity: torch.Tensor
) -> torch.Tensor:
    bone_lengths = torch.norm(
        joints[..., joints_connectivity[:, 0], :]
        - joints[..., joints_connectivity[:, 1], :],
        dim=-1,
    )
    return bone_lengths


def _compute_smpl_joints(
    smpl_layer: SMPLLayer,
    smpl_joints_regressor: torch.Tensor,
    betas: torch.Tensor,
) -> torch.Tensor:
    smpl_output = smpl_layer.forward_shape(betas)
    vertices = smpl_output.vertices
    joints = vertices2joints(smpl_joints_regressor, vertices)
    return joints


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


def load_h36m_joints_regressor(body_model_dir: str) -> torch.Tensor:
    # From https://github.com/ubc-vision/joint-regressor-refinement/tree/master
    J_regressor = (
        torch.load(os.path.join(body_model_dir, "h36m_joints_regressor.pt"))
        .float()
        .cpu()
    )
    return J_regressor


def optimize_smpl_betas(
    smpl_model: SMPL,
    J_regressor: torch.Tensor,
    target_kps_3d: torch.Tensor,
    kps_connectivity: torch.Tensor,
    lower_leg_scale: float = 1,
    upper_leg_scale: float = 1,
    lower_arm_scale: float = 1,
    upper_arm_scale: float = 1,
    pelvis_scale: float = 1,
    shoulder_scale: float = 1,
    spine_scale: float = 1,
) -> torch.Tensor:
    betas = torch.zeros(10, device=target_kps_3d.device, requires_grad=True)

    target_bone_lengths = _compute_bone_lengths(target_kps_3d, kps_connectivity)
    target_bone_lengths[4] *= upper_leg_scale  # Right upper leg (1, 2)
    target_bone_lengths[5] *= lower_leg_scale  # Right lower leg (2, 3)
    target_bone_lengths[1] *= upper_leg_scale  # Left upper leg (4, 5)
    target_bone_lengths[2] *= lower_leg_scale  # Left lower leg (5, 6)
    target_bone_lengths[14] *= upper_arm_scale  # Right upper arm (14, 15)
    target_bone_lengths[15] *= lower_arm_scale  # Right lower arm (15, 16)
    target_bone_lengths[11] *= upper_arm_scale  # Left upper arm (11, 12)
    target_bone_lengths[12] *= lower_arm_scale  # Left lower arm (12, 13)

    target_bone_lengths[3] *= pelvis_scale  # Pelvis (0, 1)
    target_bone_lengths[0] *= pelvis_scale  # Pelvis (0, 4)

    target_bone_lengths[13] *= shoulder_scale  # Right shoulder (8, 14)
    target_bone_lengths[10] *= shoulder_scale  # Left shoulder (8, 11)

    target_bone_lengths[6] *= spine_scale  # Spine (0, 7)
    target_bone_lengths[7] *= spine_scale  # Spine (7, 8)

    def closure():
        optimizer.zero_grad()

        smpl_joints = _compute_smpl_joints(smpl_model, J_regressor, betas.unsqueeze(0))
        smpl_bone_lengths = _compute_bone_lengths(smpl_joints, kps_connectivity)

        bone_lengths_loss = (target_bone_lengths - smpl_bone_lengths).abs().mean()
        betas_prior_loss = torch.norm(betas, p=2, dim=-1).mean()

        loss = 1000 * bone_lengths_loss + betas_prior_loss
        loss.backward()
        return loss

    optimizer = torch.optim.LBFGS([betas], lr=0.1, line_search_fn="strong_wolfe")

    n_max_iters = 100
    pbar = tqdm(range(n_max_iters), desc="Optimizing SMPL", leave=True)

    for _ in pbar:
        loss = optimizer.step(closure)
        pbar.set_postfix(loss=loss.item())

    return betas.detach()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("smpl_model_dir", type=str, help="Path to SMPL model directory")
    parser.add_argument("output_dir", type=str, help="Path to output directory")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir

    C.smplx_models = os.path.dirname(args.smpl_model_dir)

    sequences_file = os.path.join(dataset_dir, "h36m_protocol1_sequences.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smpl_model = SMPL(
        model_path=args.smpl_model_dir,
        gender="neutral",
        num_betas=10,
        batch_size=1,
    ).to(device)
    J_regressor = load_h36m_joints_regressor(args.smpl_model_dir).to(device)
    mean_betas = torch.zeros(10, device=device)

    joints = _compute_smpl_joints(
        smpl_model, J_regressor, mean_betas.unsqueeze(0)
    ).squeeze(0)

    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())
    sequence = next(s for s in sequences if s["sequence_name"] == "S1_Directions 1")

    kps_3d = load_gt_keypoints_3d(dataset_dir, sequence).to(device)
    target_kps_3d = kps_3d[0]  # take the first frame only for the example
    kps_connectivity = torch.tensor(
        H36M_KPS_FORMAT.keypoints_connectivity, dtype=torch.int32, device=device
    )

    optimized_betas = optimize_smpl_betas(
        smpl_model,
        J_regressor,
        target_kps_3d,
        kps_connectivity,
        lower_leg_scale=1.2,  # Emphasize the difference in shape
        upper_leg_scale=1.2,
        lower_arm_scale=0.8,
        upper_arm_scale=0.8,
        pelvis_scale=1.4,
        shoulder_scale=1.2,
        spine_scale=1.2,
    )
    optimized_joints = _compute_smpl_joints(
        smpl_model, J_regressor, optimized_betas.unsqueeze(0)
    ).squeeze(0)

    res = 4096
    viewer = HeadlessRenderer(size=(res, res))
    # viewer = Viewer()

    scale = 0.8

    camera = WeakPerspectiveCamera(
        scale=np.array([scale, scale]),
        translation=np.array([0, -0.3]),
        cols=res,
        rows=res,
    )
    viewer.scene.add(camera)
    viewer.scene.camera = camera

    viewer.scene.floor.enabled = False
    viewer.scene.background_color = (0.0, 0.0, 0.0, 0.0)

    skeleton_radius = 0.02

    gray_color = (0.5, 0.5, 0.5, 1.0)
    blue_color = (104 / 255, 109 / 255, 224 / 255, 1.0)
    black_color = (0.0, 0.0, 0.0, 1.0)
    red_color = (1.0, 0.0, 0.0, 1.0)

    ait_smpl_layer = AitSMPLLayer(model_type="smpl", gender="neutral", num_betas=10)

    mean_smpl_seq = SMPLSequence.t_pose(ait_smpl_layer)
    mean_smpl_seq.color = blue_color
    mean_smpl_h36m_skeleton = Skeletons(
        joint_positions=joints.unsqueeze(0).cpu().numpy(),
        joint_connections=kps_connectivity.cpu().numpy(),
        name="H3.6M Skeleton",
        color=black_color,
        radius=skeleton_radius,
    )
    mean_smpl_seq.skeleton_seq.enabled = False
    mean_smpl_seq.add(mean_smpl_h36m_skeleton)
    mean_smpl_seq.name = "Mean SMPL"

    optimized_smpl_seq = SMPLSequence.t_pose(ait_smpl_layer, betas=optimized_betas)
    optimized_smpl_seq.color = blue_color
    optimized_smpl_h36m_skeleton = Skeletons(
        joint_positions=optimized_joints.unsqueeze(0).cpu().numpy(),
        joint_connections=kps_connectivity.cpu().numpy(),
        name="H3.6M Skeleton",
        color=black_color,
        radius=skeleton_radius,
    )
    optimized_smpl_seq.skeleton_seq.enabled = False
    optimized_smpl_seq.add(optimized_smpl_h36m_skeleton)
    optimized_smpl_seq.name = "Target SMPL"

    viewer.scene.add(mean_smpl_seq)
    viewer.scene.add(optimized_smpl_seq)

    os.makedirs(args.output_dir, exist_ok=True)

    viewer._init_scene()
    viewer.scene.origin.enabled = False

    def render(
        show_mean_smpl_mesh: bool,
        show_optimized_smpl_mesh: bool,
        show_mean_smpl_h36m_skeleton: bool,
        show_optimized_smpl_h36m_skeleton: bool,
        name: str,
    ):
        mean_smpl_seq.mesh_seq.enabled = show_mean_smpl_mesh
        optimized_smpl_seq.mesh_seq.enabled = show_optimized_smpl_mesh
        mean_smpl_h36m_skeleton.enabled = show_mean_smpl_h36m_skeleton
        optimized_smpl_h36m_skeleton.enabled = show_optimized_smpl_h36m_skeleton
        viewer.export_frame(
            os.path.join(args.output_dir, f"{name}.png"),
            transparent_background=True,
        )

    render(
        show_mean_smpl_mesh=True,
        show_optimized_smpl_mesh=False,
        show_mean_smpl_h36m_skeleton=False,
        show_optimized_smpl_h36m_skeleton=False,
        name="mean_smpl_mesh",
    )

    render(
        show_mean_smpl_mesh=False,
        show_optimized_smpl_mesh=True,
        show_mean_smpl_h36m_skeleton=False,
        show_optimized_smpl_h36m_skeleton=False,
        name="optimized_smpl_mesh",
    )

    mean_smpl_h36m_skeleton.color = red_color
    optimized_smpl_h36m_skeleton.color = red_color
    render(
        show_mean_smpl_mesh=False,
        show_optimized_smpl_mesh=False,
        show_mean_smpl_h36m_skeleton=True,
        show_optimized_smpl_h36m_skeleton=False,
        name="mean_smpl_skeleton_red",
    )

    render(
        show_mean_smpl_mesh=False,
        show_optimized_smpl_mesh=False,
        show_mean_smpl_h36m_skeleton=False,
        show_optimized_smpl_h36m_skeleton=True,
        name="optimized_smpl_skeleton_red",
    )

    mean_smpl_h36m_skeleton.color = black_color
    optimized_smpl_h36m_skeleton.color = black_color
    render(
        show_mean_smpl_mesh=False,
        show_optimized_smpl_mesh=False,
        show_mean_smpl_h36m_skeleton=True,
        show_optimized_smpl_h36m_skeleton=False,
        name="mean_smpl_skeleton_black",
    )

    render(
        show_mean_smpl_mesh=False,
        show_optimized_smpl_mesh=False,
        show_mean_smpl_h36m_skeleton=False,
        show_optimized_smpl_h36m_skeleton=True,
        name="optimized_smpl_skeleton_black",
    )
