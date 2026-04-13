from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.pipeline import ViewInput
from kineo.pipeline.pipeline import Annotations
from kineo.annotations.smpl_params import SMPLShapeAnnotations, SMPLPoseAnnotations
from dataclasses import dataclass
import os
import numpy as np
import warnings
import torch

from smplx import SMPLLayer
import kineo.pipeline.stages.bvh.quat as quat
import kineo.pipeline.stages.bvh.bvh as bvh
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation

from smplx import build_layer
from smplx.joint_names import SMPLH_JOINT_NAMES, JOINT_NAMES

# Fix to properly load smpl model without errors
np.bool = np.bool_
np.int = np.int_
np.long = np.int_
np.object = np.object_
np.str = np.str_

try:
    np.float = np.float_
except AttributeError:
    np.float = np.float64
try:
    np.complex = np.complex_
except AttributeError:
    np.complex = np.complex128
try:
    np.unicode = np.unicode_
except AttributeError:
    np.unicode = np.str_

@dataclass(frozen=True)
class ExportBvhRuntimeConfig:
    output_path_template: str = "./outputs/bvh/{sequence_name}_{subject_id}.bvh"


class ExportBvhStage(PipelineStage[ExportBvhRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: ExportBvhRuntimeConfig,
        smpl_model_path: str = "./body_models/smplx/SMPLX_NEUTRAL.npz",
        smpl_model_type: str = "smplx",
        smpl_num_betas: int = 10,
        smpl_gender: str = "neutral",
        smpl_use_face_contour: bool = False,
        smpl_use_pca: bool = True,
        dynamic_runtime_cfg: dict[str, ExportBvhRuntimeConfig] | None = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )

        self.smpl_joint_names = JOINT_NAMES if smpl_model_type == "smpl" else SMPLH_JOINT_NAMES

        self.smpl_layer = build_layer(
            model_path=smpl_model_path,
            model_type=smpl_model_type,
            num_betas=smpl_num_betas,
            gender=smpl_gender,
            use_face_contour=smpl_use_face_contour,
            use_pca=smpl_use_pca,
        )
        self.smpl_layer = self.smpl_layer.cpu()
        self.smpl_layer = self.smpl_layer.eval()


    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: ExportBvhRuntimeConfig,
    ):
        device = pipeline.device
        self.smpl_layer = self.smpl_layer.to(device)

        pred_smpl_shape: SMPLShapeAnnotations = annotations["smpl_shape"].to(device)
        pred_smpl_pose: SMPLPoseAnnotations = annotations["smpl_pose"].to(device)

        global_time_reference: GlobalTimeReferenceAnnotation = annotations.get("global_time_reference", None)

        if global_time_reference is not None:
            global_time_reference = global_time_reference.first_or_default()

            dt = torch.diff(global_time_reference.timestamps).mean().item()

            if dt > 0:
                target_fps = 1 / dt
            else:
                target_fps = 25
                warnings.warn(
                    f"Sequence {sequence_name} has a dt of {dt} seconds, which is less than 0.01 seconds. Setting target fps to 25."
                )
        else:
            target_fps = 25

        subjects_ids = pred_smpl_shape.subjects_ids
        for subject_id in subjects_ids:
            formatted_output_path = runtime_cfg.output_path_template.format(
                sequence_name=sequence_name, 
                subject_id=subject_id,
            )
            formatted_output_path = os.path.abspath(formatted_output_path)
            os.makedirs(os.path.dirname(formatted_output_path), exist_ok=True)

            subject_smpl_shape = pred_smpl_shape.filter_by_subject_id(subject_id)
            subject_smpl_pose = pred_smpl_pose.filter_by_subject_id(subject_id)

            export_bvh(
                output_path=formatted_output_path,
                smpl_shape=subject_smpl_shape,
                smpl_pose=subject_smpl_pose,
                smpl_layer=self.smpl_layer,
                fps=target_fps,
            )

        self.smpl_layer = self.smpl_layer.cpu()
        annotations["smpl_shape"] = pred_smpl_shape.cpu()
        annotations["smpl_pose"] = pred_smpl_pose.cpu()

def export_bvh(
    output_path: str,
    smpl_shape: SMPLShapeAnnotations,
    smpl_pose: SMPLPoseAnnotations,
    smpl_layer: SMPLLayer,
    fps: float,
):
    names = [
        "Pelvis",
        "Left_hip",
        "Right_hip",
        "Spine1",
        "Left_knee",
        "Right_knee",
        "Spine2",
        "Left_ankle",
        "Right_ankle",
        "Spine3",
        "Left_foot",
        "Right_foot",
        "Neck",
        "Left_collar",
        "Right_collar",
        "Head",
        "Left_shoulder",
        "Right_shoulder",
        "Left_elbow",
        "Right_elbow",
        "Left_wrist",
        "Right_wrist",
        "Left_palm",
        "Right_palm",
    ]

    parents = smpl_layer.parents.detach().cpu().numpy()
    parents = parents[:24]

    rest = smpl_layer.forward(
        betas=smpl_shape.first_or_default().betas.reshape(1, -1).detach(),
    )
    rest_joints = rest.joints.detach().cpu().numpy().squeeze()
    rest_pose = rest_joints[:24, :]

    root_offset = rest_pose[0]

    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset

    n_frames = smpl_pose.n_frames
    rots = np.zeros((n_frames, 24, 3))
    trans = np.zeros((n_frames, 3))

    for annotation in smpl_pose:
        frame_idx = annotation.frame_idx
        trans[frame_idx] = annotation.trans.detach().cpu().numpy()
        rots[frame_idx] = annotation.pose[:24].detach().cpu().numpy()

    # --- Prepend rest frame: zero pose (identity rotation), root at origin ---
    rest_rots = np.zeros((1, 24, 3))   # all axis-angles = 0 → identity
    rest_trans = np.zeros((1, 3))       # root at origin
    rots = np.concatenate([rest_rots, rots], axis=0)
    trans = np.concatenate([rest_trans, trans], axis=0)
    # -------------------------------------------------------------------------

    rots = quat.from_axis_angle(rots)

    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:,0] += trans
    rotations = np.degrees(quat.to_euler(rots, order=order))

    bvh_data ={
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / fps,
    }

    if not output_path.endswith(".bvh"):
        output_path = output_path + ".bvh"

    bvh.save(output_path, bvh_data)
    print(f"BVH file saved to {output_path}")