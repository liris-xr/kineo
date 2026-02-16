# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.pipeline import ViewInput
from kineo.pipeline.pipeline import Annotations
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotation
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.global_scale import (
    GlobalScaleAnnotations,
    GlobalScaleAnnotationsMetadata,
    GlobalScaleAnnotation,
)
import torch
from pathlib import Path

from smplx import SMPLLayer
from smplx.lbs import vertices2joints

from tqdm import tqdm

from kineo.maths import weighted_mean, weighted_median
from kineo.optimization.utils import optimizer_should_stop

import numpy as np
from dataclasses import dataclass

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
class SMPLGlobalScaleEstimationRuntimeConfig:
    n_iters: int = 100
    keypoints_indices: list[int] | None = None
    bones_lengths_loss_weight: float = 1.0
    bones_lengths_huber_loss_delta: float = 0.5
    betas_prior_loss_weight: float = 1.0
    tolerance_grad: float = 1e-05
    tolerance_change: float = 1e-09
    use_lbfgs: bool = True


class SMPLGlobalScaleEstimationStage(
    PipelineStage[SMPLGlobalScaleEstimationRuntimeConfig]
):
    def __init__(
            self,
            name: str,
            order: int,
            smpl_model_path: str,
            joint_regressor_path: str,
            runtime_cfg: SMPLGlobalScaleEstimationRuntimeConfig,
            dynamic_runtime_cfg: (
                    dict[str, SMPLGlobalScaleEstimationRuntimeConfig] | None
            ) = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )
        self.smpl_layer = SMPLLayer(
            model_path=smpl_model_path, gender="neutral", num_betas=10
        )
        self.smpl_layer.cpu()

        extension = Path(joint_regressor_path).suffix

        if extension == ".pt":
            self.smpl_joints_regressor = torch.load(joint_regressor_path)
        elif extension == ".npy":
            self.smpl_joints_regressor = torch.from_numpy(np.load(joint_regressor_path))
        elif extension == ".npz":
            self.smpl_joints_regressor = torch.from_numpy(np.load(joint_regressor_path))
        else:
            raise ValueError(f"Unsupported joint regressor file extension: {extension}")

        self.smpl_joints_regressor = self.smpl_joints_regressor.cpu()

    def forward(
            self,
            sequence_name: str,
            pipeline: Pipeline,
            views: list[ViewInput],
            annotations: dict[str, Annotations],
            gt_annotations: dict[str, Annotations],
            runtime_cfg: SMPLGlobalScaleEstimationRuntimeConfig,
    ):
        device = pipeline.device

        self.smpl_layer = self.smpl_layer.to(device)
        self.smpl_joints_regressor = self.smpl_joints_regressor.to(device)

        global_time_reference: GlobalTimeReferenceAnnotation = annotations[
            "global_time_reference"
        ].first_or_default()

        if global_time_reference is None:
            raise ValueError("Global time reference is not available")

        keypoints_3d: Keypoints3DAnnotations = annotations["keypoints_3d"]
        # Assuming format is the same for all subjects
        kps_format_name = keypoints_3d.first_or_default().format
        kps_format = next(
            keypoints_format
            for keypoints_format in keypoints_3d.metadata.formats
            if keypoints_format.name == kps_format_name
        )

        world_timestamps = global_time_reference.timestamps

        n_frames = len(world_timestamps)
        n_keypoints = kps_format.n_keypoints
        subjects_ids = list(keypoints_3d.subjects_ids)
        n_subjects = len(subjects_ids)

        if runtime_cfg.keypoints_indices is not None:
            keypoints_indices = torch.as_tensor(
                runtime_cfg.keypoints_indices, device=device
            )
        else:
            keypoints_indices = torch.arange(n_keypoints, device=device)

        if len(self.smpl_joints_regressor) != len(keypoints_indices):
            raise ValueError(
                f"The number of keypoints in the SMPL joints regressor "
                f"({len(self.smpl_joints_regressor)}) does not match the number "
                f"of keypoints used from the annotations ({len(keypoints_indices)}). "
                f"Either specify the keypoints_indices in config to match the "
                f"ones from the joint regressor, or make sure your keypoints detector "
                f"outputs the same format of keypoints as the SMPL joints regressor."
            )

        joints_connectivity = torch.as_tensor(
            kps_format.keypoints_connectivity, device=device
        )

        # Bones to actually use for the optimization based on the selected keypoints (if any)
        joints_connectivity_mask = torch.ones_like(
            joints_connectivity[:, 0], dtype=torch.bool
        )

        if runtime_cfg.keypoints_indices is not None:
            keypoints_indices = torch.as_tensor(
                runtime_cfg.keypoints_indices, device=device
            )

            joints_connectivity_mask = torch.isin(
                joints_connectivity[:, 0], keypoints_indices
            ) & torch.isin(joints_connectivity[:, 1], keypoints_indices)

        joints_connectivity = joints_connectivity[joints_connectivity_mask]

        subjects_joints = torch.zeros(
            (n_frames, n_subjects, n_keypoints, 3), device=device
        )
        subjects_joints_scores = torch.zeros(
            (n_frames, n_subjects, n_keypoints), device=device
        )

        subject_id_to_idx = {subject_id: i for i, subject_id in enumerate(subjects_ids)}

        for ann in keypoints_3d:
            subject_idx = subject_id_to_idx[ann.subject_id]
            subjects_joints[ann.frame_idx, subject_idx] = ann.xyz
            subjects_joints_scores[ann.frame_idx, subject_idx] = ann.scores

        # Bones defined using the keypoints in arbitrary scale
        bone_lengths = _compute_bone_lengths(subjects_joints, joints_connectivity)
        bone_scores = _compute_bone_scores(subjects_joints_scores, joints_connectivity)
        bone_lengths = torch.nan_to_num(bone_lengths, 0)
        bone_scores = torch.nan_to_num(bone_scores, 0)

        # Compute the median over the frames (dim=0)
        bone_lengths = weighted_median(
            values=bone_lengths,
            weights=bone_scores,
            dim=0,
        )[0].reshape(-1)
        bone_scores = torch.mean(bone_scores, dim=0).reshape(-1)
        best_bone_idx = torch.argmax(bone_scores)

        # SMPL joints in metric space
        smpl_joints = _compute_smpl_joints(
            self.smpl_layer,
            self.smpl_joints_regressor,
            torch.zeros((n_subjects, self.smpl_layer.num_betas), device=device),
        )
        smpl_bone_lengths = _compute_bone_lengths(smpl_joints, joints_connectivity).reshape(-1)

        best_bone_arbitrary_length = bone_lengths[best_bone_idx]
        best_bone_smpl_length = smpl_bone_lengths[best_bone_idx]

        global_scale_init = best_bone_smpl_length / best_bone_arbitrary_length

        if not global_scale_init.isfinite() or global_scale_init.eq(0):
            raise ValueError(
                f"Expected global scale to be finite and non-zero, got {global_scale_init}"
            )

        # We initialize global scale with the ratios between SMPL bones (in metric space) and the subjects bones lengths
        log_global_scale = torch.tensor(
            global_scale_init.log().item(),
            device=device,
            requires_grad=True,
        )

        betas = torch.zeros(
            (n_subjects, self.smpl_layer.num_betas),
            device=device,
            requires_grad=True,
        )

        def closure():
            optimizer.zero_grad()

            scaled_bone_lengths = log_global_scale.exp() * bone_lengths

            smpl_joints = _compute_smpl_joints(
                self.smpl_layer, self.smpl_joints_regressor, betas
            )
            smpl_bone_lengths = _compute_bone_lengths(smpl_joints, joints_connectivity)
            smpl_bone_lengths = smpl_bone_lengths.reshape(-1)

            valid_bones_mask = scaled_bone_lengths.isfinite()
            valid_scaled_bone_lengths = scaled_bone_lengths[valid_bones_mask]
            valid_bone_scores = bone_scores[valid_bones_mask]
            valid_smpl_bone_lengths = smpl_bone_lengths[valid_bones_mask]

            bone_lengths_loss = torch.nn.functional.huber_loss(
                input=valid_smpl_bone_lengths,
                target=valid_scaled_bone_lengths,
                delta=runtime_cfg.bones_lengths_huber_loss_delta,
                reduction="none",
            )
            bone_lengths_loss = weighted_mean(
                values=bone_lengths_loss, weights=valid_bone_scores
            )

            betas_l2_loss = (betas ** 2).sum(dim=-1).mean()

            loss = (
                    runtime_cfg.bones_lengths_loss_weight * bone_lengths_loss
                    + runtime_cfg.betas_prior_loss_weight * betas_l2_loss
            )
            loss.backward()
            return loss

        if runtime_cfg.use_lbfgs:
            optimizer = torch.optim.LBFGS(
                [log_global_scale, betas], lr=1, line_search_fn="strong_wolfe"
            )
        else:
            optimizer = torch.optim.AdamW([log_global_scale, betas], lr=1e-3)

        n_max_iters = runtime_cfg.n_iters
        pbar = tqdm(range(n_max_iters), desc="Optimizing global scale", leave=False)

        prev_losses = []

        for _ in pbar:
            loss = optimizer.step(closure)
            pbar.set_postfix(loss=loss.item())

            if optimizer_should_stop(
                    optimizer,
                    loss,
                    prev_losses,
                    patience=5,
                    tolerance_grad=runtime_cfg.tolerance_grad,
                    tolerance_change=runtime_cfg.tolerance_change,
            ):
                break

            prev_losses.append(loss.detach())

        global_scale_annotations = GlobalScaleAnnotations(
            metadata=GlobalScaleAnnotationsMetadata(),
            annotations=[
                GlobalScaleAnnotation(
                    frame_idx=0, scale=log_global_scale.detach().exp().item()
                )
            ],
        )

        annotations["global_scale"] = global_scale_annotations

        self.smpl_layer = self.smpl_layer.cpu()
        self.smpl_joints_regressor = self.smpl_joints_regressor.cpu()


def _compute_bone_lengths(
        joints: torch.Tensor, joints_connectivity: torch.Tensor
) -> torch.Tensor:
    bone_lengths = torch.norm(
        joints[..., joints_connectivity[:, 0], :]
        - joints[..., joints_connectivity[:, 1], :],
        dim=-1,
    )
    return bone_lengths


def _compute_bone_scores(
        joints_scores: torch.Tensor,
        joints_connectivity: torch.Tensor,
) -> torch.Tensor:
    # The bone score is the geometric mean of the scores of the two joints
    bone_scores = torch.sqrt(
        joints_scores[..., joints_connectivity[:, 0]]
        * joints_scores[..., joints_connectivity[:, 1]]
    )
    return bone_scores


def _compute_smpl_joints(
        smpl_layer: SMPLLayer,
        smpl_joints_regressor: torch.Tensor,
        betas: torch.Tensor,
) -> torch.Tensor:
    smpl_output = smpl_layer.forward_shape(betas)
    vertices = smpl_output.vertices
    joints = vertices2joints(smpl_joints_regressor, vertices)
    return joints


def create_coordinate_grid(H, W, device="cpu", dtype=torch.float32):
    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")  # Shape: (H, W)
    grid = torch.stack([xx, yy], dim=-1)  # Shape: (H, W, 2)
    return grid
