from __future__ import annotations
import torch

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    Annotations,
    Keypoints3DAnnotations,
    Keypoints2DAnnotations,
    CameraIntrinsicsAnnotations,
    CameraExtrinsicsAnnotations,
)
from kineo.annotations.smpl_params import (
    SMPLShapeAnnotationsMetadata,
    SMPLShapeAnnotation,
    SMPLShapeAnnotations,
    SMPLShapeFormat,
    SMPLPoseAnnotations,
    SMPLPoseAnnotationsMetadata,
    SMPLPoseAnnotation,
    SMPLPoseFormat,
)
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.stages.nlf.smpl_keypoints_format import (
    NLF_SMPL_KEYPOINTS_FORMAT,
    NLF_SMPLX_KEYPOINTS_FORMAT,
)
from kineo.pipeline.stages.nlf.model_wrapper import NLFModelWrapper

import numpy as np
from dataclasses import dataclass

from kineo.geometry.transformations import (
    compute_similarity_transform,
    apply_similarity_transform_to_points,
)

from smplx import SMPLXLayer, build_layer

from kineo.geometry.camera import (
    transform_points_from_world_to_camera,
    project_points_from_camera_to_image,
)
import roma

from tqdm import tqdm

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


@dataclass
class NLFSMPLFittingRuntimeConfig:
    beta_regularizer: float = 10.0
    beta_regularizer2: float = 0.0
    num_fitting_iter: int = 30
    n_hands_fix_iters: int = 10
    huber_delta: float = 1.0


# TODO: in case of single view, export keypoints 3d as well
class NLFSMPLFittingStage(PipelineStage[NLFSMPLFittingRuntimeConfig]):
    """
    Stage for detecting keypoints from images by jointly using NLF.

    Produces :class:`Keypoints2DAnnotations` with the detected keypoints for each view with key "keypoints_2d".
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: NLFSMPLFittingRuntimeConfig,
        dynamic_runtime_cfg: dict[str, NLFSMPLFittingRuntimeConfig] | None = None,
        torchscript_model_path: str = "./checkpoints/nlf_l_multi_0.3.2.torchscript",
        smpl_model_path: str = "./body_models/smplx/SMPLX_NEUTRAL.npz",
        smpl_model_type: str = "smplx",
        smpl_num_betas: int = 10,
        smpl_gender: str = "neutral",
        smpl_use_face_contour: bool = False,
        smpl_use_pca: bool = True,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )
        self.model = NLFModelWrapper(torchscript_model_path)
        self.model = self.model.cpu()
        self.model = self.model.eval()

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
        runtime_cfg: NLFSMPLFittingRuntimeConfig,
    ):
        device = pipeline.device
        self.model = self.model.to(device)

        keypoints_3d_annotations: Keypoints3DAnnotations = annotations["keypoints_3d"]

        n_frames = keypoints_3d_annotations.n_frames
        subjects_ids = keypoints_3d_annotations.subjects_ids
        n_keypoints = keypoints_3d_annotations.metadata.formats[0].n_keypoints

        all_smpl_shape_annotations: list[SMPLShapeAnnotation] = []
        all_smpl_pose_annotations: list[SMPLPoseAnnotation] = []

        for subject_id in subjects_ids:
            subject_kps_3d_annotations = keypoints_3d_annotations.filter_by_subject_id(
                subject_id
            )

            subject_kps_format_name = (
                subject_kps_3d_annotations.first_or_default().format
            )
            subject_kps_format = next(
                format
                for format in subject_kps_3d_annotations.metadata.formats
                if format.name == subject_kps_format_name
            )

            if not (
                subject_kps_format_name == NLF_SMPL_KEYPOINTS_FORMAT.name
                or subject_kps_format_name == NLF_SMPLX_KEYPOINTS_FORMAT.name
            ):
                raise ValueError(
                    f"Invalid keypoints format: {subject_kps_format_name}. Expected {NLF_SMPL_KEYPOINTS_FORMAT.name} or {NLF_SMPLX_KEYPOINTS_FORMAT.name}."
                )

            model_name = (
                "smpl"
                if subject_kps_format_name == NLF_SMPL_KEYPOINTS_FORMAT.name
                else "smplx"
            )

            n_keypoints = subject_kps_format.n_keypoints
            kps_xyz = torch.zeros(
                n_frames, n_keypoints, 3, dtype=torch.float32, device=device
            )
            kps_weights = torch.zeros(
                n_frames, n_keypoints, dtype=torch.float32, device=device
            )

            for ann in tqdm(subject_kps_3d_annotations.annotations, desc="Collecting 3D keypoints", leave=False):
                kps_xyz[ann.frame_idx, :] = ann.xyz
                kps_weights[ann.frame_idx, :] = ann.scores

            fit_results = self.model.fit_smpl_model(
                model_name=model_name,
                poses3d_flat=kps_xyz,
                confidences_flat=kps_weights,
                num_iter=runtime_cfg.num_fitting_iter,
                beta_regularizer=runtime_cfg.beta_regularizer,
                beta_regularizer2=runtime_cfg.beta_regularizer2,
                final_adjust_rots=True,
                requested_keys=["pose_rotvecs", "shape_betas", "trans"],
            )
            poses = fit_results["pose"].reshape(n_frames, -1, 3).clone()
            trans = fit_results["trans"].clone()
            betas = fit_results["betas"].clone()

            # poses_root, poses_body, poses_left_hand, poses_right_hand, poses_face = (
            #     SMPLPoseFormat.SMPLX.split_poses(poses, dim=-2)
            # )

            # poses_left_hand, poses_right_hand = self._align_subject_hands(
            #     smpl_layer=self.smpl_layer,
            #     betas=betas,
            #     subject_kps_3d=subject_kps_3d_annotations,
            #     n_iters=runtime_cfg.n_hands_fix_iters,
            # )

            # poses = torch.cat(
            #     [
            #         poses_root.view(n_frames, -1, 3),
            #         poses_body.view(n_frames, -1, 3),
            #         poses_left_hand.view(n_frames, -1, 3),
            #         poses_right_hand.view(n_frames, -1, 3),
            #         poses_face.view(n_frames, -1, 3),
            #     ],
            #     dim=-2,
            # )

            betas = betas.permute(1, 0).mean(dim=-1).reshape(-1)

            shape_format = (
                SMPLShapeFormat.SMPL_10
                if model_name == "smpl"
                else SMPLShapeFormat.SMPLX_10
            )
            pose_format = (
                SMPLPoseFormat.SMPL if model_name == "smpl" else SMPLPoseFormat.SMPLX
            )

            body_model = (
                self.model.model.fitters.smplx.body_model
                if model_name == "smplx"
                else self.model.model.fitters.smpl.body_model
            )

            all_smpl_shape_annotations.append(
                SMPLShapeAnnotation(
                    frame_idx=0,
                    subject_id=subject_id,
                    betas=betas,
                    format=shape_format,
                )
            )
            all_smpl_pose_annotations.extend(
                [
                    SMPLPoseAnnotation(
                        frame_idx=frame_idx,
                        subject_id=subject_id,
                        pose=poses[frame_idx],
                        trans=trans[frame_idx],
                        format=pose_format,
                    )
                    for frame_idx in range(n_frames)
                ]
            )

        self.model = self.model.cpu()

        smpl_shape_annotations = SMPLShapeAnnotations(
            metadata=SMPLShapeAnnotationsMetadata(),
            annotations=all_smpl_shape_annotations,
        )
        smpl_pose_annotations = SMPLPoseAnnotations(
            metadata=SMPLPoseAnnotationsMetadata(),
            annotations=all_smpl_pose_annotations,
        )

        annotations["smpl_shape"] = smpl_shape_annotations
        annotations["smpl_pose"] = smpl_pose_annotations

    def _align_subject_hands(
        self,
        smpl_layer: SMPLXLayer,
        betas: torch.Tensor,
        subject_kps_3d: Keypoints3DAnnotations,
        n_iters: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        device = betas.device
        smpl_layer = smpl_layer.to(device)

        n_frames = betas.shape[0]
        mean_betas = betas.permute(1, 0).mean(dim=-1).reshape(-1)

        # Targets
        target_left_fingers = torch.zeros(
            (n_frames, smpl_layer.NUM_HAND_JOINTS, 3), device=device
        )
        target_right_fingers = torch.zeros(
            (n_frames, smpl_layer.NUM_HAND_JOINTS, 3), device=device
        )

        for annot in tqdm(
            subject_kps_3d.annotations, desc="Collecting 3D keypoints", leave=False
        ):
            target_left_fingers[annot.frame_idx] = annot.xyz[
                [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39], :
            ]
            target_right_fingers[annot.frame_idx] = annot.xyz[
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54], :
            ]

        # left_hand_pose = torch.zeros((n_frames, smpl_layer.NUM_HAND_JOINTS, 3), device=device, requires_grad=True)
        # right_hand_pose = torch.zeros((n_frames, smpl_layer.NUM_HAND_JOINTS, 3), device=device, requires_grad=True)

        left_hand_pose_pca = torch.zeros(
            (n_frames, smpl_layer.num_pca_comps), device=device, requires_grad=True
        )
        right_hand_pose_pca = torch.zeros(
            (n_frames, smpl_layer.num_pca_comps), device=device, requires_grad=True
        )

        optimizer = torch.optim.LBFGS(
            [left_hand_pose_pca, right_hand_pose_pca],
            lr=1,
            line_search_fn="strong_wolfe",
        )

        pbar = tqdm(range(n_iters), desc="Optimizing hands poses")

        loss_per_frame = torch.zeros((n_frames,), device=device)

        left_hand_components = torch.from_numpy(smpl_layer.np_left_hand_components).to(
            device
        )
        right_hand_components = torch.from_numpy(
            smpl_layer.np_right_hand_components
        ).to(device)

        def opt_closure():
            nonlocal loss_per_frame
            optimizer.zero_grad()

            left_hand_pose = torch.einsum(
                "bi,ij->bj", [left_hand_pose_pca, left_hand_components]
            ).view(n_frames, -1, 3)
            right_hand_pose = torch.einsum(
                "bi,ij->bj", [right_hand_pose_pca, right_hand_components]
            ).view(n_frames, -1, 3)

            result = smpl_layer.forward(
                betas=mean_betas.view(1, -1).expand(n_frames, -1),
                left_hand_pose=roma.rotvec_to_rotmat(left_hand_pose).view(
                    n_frames, -1, 3, 3
                ),
                right_hand_pose=roma.rotvec_to_rotmat(right_hand_pose).view(
                    n_frames, -1, 3, 3
                ),
            )

            current_left = result.joints[
                ..., [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39], :
            ]
            current_right = result.joints[
                ..., [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54], :
            ]

            Rl, Tl, sl = compute_similarity_transform(
                current_left, target_left_fingers, estimate_scale=True
            )
            current_left = apply_similarity_transform_to_points(
                current_left, Rl, Tl, sl
            )

            Rr, Tr, sr = compute_similarity_transform(
                current_right, target_right_fingers, estimate_scale=True
            )
            current_right = apply_similarity_transform_to_points(
                current_right, Rr, Tr, sr
            )

            loss_per_frame = ((current_left - target_left_fingers) ** 2).sum(
                dim=(1, 2)
            ) + ((current_right - target_right_fingers) ** 2).sum(dim=(1, 2))

            loss = loss_per_frame.mean()

            loss.backward()
            return loss

        for _ in pbar:
            loss = optimizer.step(opt_closure)

            pbar.set_postfix(
                {
                    "mean": f"{loss_per_frame.mean().detach().item():.4f}",
                    "max": f"{loss_per_frame.max().detach().item():.4f}",
                }
            )

        left_hand_pose = torch.einsum(
            "bi,ij->bj", [left_hand_pose_pca, left_hand_components]
        ).view(n_frames, -1, 3)
        right_hand_pose = torch.einsum(
            "bi,ij->bj", [right_hand_pose_pca, right_hand_components]
        ).view(n_frames, -1, 3)

        return left_hand_pose.detach(), right_hand_pose.detach()

    def _refine_subject_model(
        self,
        smpl_layer: SMPLXLayer,
        betas: torch.Tensor,
        pose: torch.Tensor,
        trans: torch.Tensor,
        subject_kps_2d: Keypoints2DAnnotations,
        camera_intrinsics_annotations: CameraIntrinsicsAnnotations,
        camera_extrinsics_annotations: CameraExtrinsicsAnnotations,
        n_iters: int,
        huber_delta: float = 1.0,
    ):
        frame_idx = 120

        views_ids = camera_intrinsics_annotations.views_ids
        view_id_to_idx = {view_id: idx for idx, view_id in enumerate(views_ids)}

        device = pose.device

        n_frames = pose.shape[0]
        n_joints = 55
        n_vertices = 1024
        # n_joints = body_model.num_joints
        # n_vertices = body_model.num_vertices
        n_views = len(views_ids)

        joints_2d = torch.zeros((n_frames, n_views, n_joints, 2), device=device)
        joints_2d_scores = torch.zeros((n_frames, n_views, n_joints), device=device)
        vertices_2d = torch.zeros((n_frames, n_views, n_vertices, 2), device=device)
        vertices_2d_scores = torch.zeros((n_frames, n_views, n_vertices), device=device)

        import cv2
        import numpy as np

        for annot in tqdm(
            subject_kps_2d.annotations, desc="Collecting 2D keypoints", leave=False
        ):
            view_idx = view_id_to_idx[annot.view_id]
            frame_idx = annot.frame_idx

            joints_2d_xy, vertices_2d_xy = torch.split(
                annot.xy, [n_joints, n_vertices], dim=0
            )
            joints_2d_scores_xy, vertices_2d_scores_xy = torch.split(
                annot.scores, [n_joints, n_vertices], dim=0
            )

            joints_2d[frame_idx, view_idx] = joints_2d_xy
            joints_2d_scores[frame_idx, view_idx] = joints_2d_scores_xy
            vertices_2d[frame_idx, view_idx] = vertices_2d_xy
            vertices_2d_scores[frame_idx, view_idx] = vertices_2d_scores_xy

        joints_2d = joints_2d[frame_idx]
        joints_2d_scores = joints_2d_scores[frame_idx]
        vertices_2d = vertices_2d[frame_idx]
        vertices_2d_scores = vertices_2d_scores[frame_idx]

        Rts = torch.zeros((n_views, 3, 4), device=device)
        Ks = torch.zeros((n_views, 3, 3), device=device)
        Ds = torch.zeros((n_views, 5), device=device)

        for view_idx, view_id in enumerate(views_ids):
            view_camera_extrinsics = camera_extrinsics_annotations.filter_by_view_id(
                view_id
            ).first_or_default()
            view_camera_intrinsics = camera_intrinsics_annotations.filter_by_view_id(
                view_id
            ).first_or_default()
            Rts[view_idx] = view_camera_extrinsics.Rt
            Ks[view_idx] = view_camera_intrinsics.K
            Ds[view_idx] = view_camera_intrinsics.distortion_coefficients

        distortion_model = (
            camera_intrinsics_annotations.first_or_default().distortion_model
        )

        smpl_layer = smpl_layer.to(device)

        def opt_closure():
            optimizer.zero_grad()

            result = smpl_layer.forward(
                betas=betas_opt.view(1, -1),
                global_orient=roma.rotvec_to_rotmat(pose_opt[0]).view(1, 3, 3),
                body_pose=roma.rotvec_to_rotmat(
                    pose_opt[1 : smpl_layer.NUM_BODY_JOINTS + 1]
                ).view(1, -1, 3, 3),
                transl=trans[frame_idx].view(1, 3),
            )
            joints_world = result.joints[0][:n_joints]

            joints_cam = transform_points_from_world_to_camera(
                joints_world.reshape(-1, 3), Rts
            )

            # Project to cameras
            joints_cam_proj, _ = project_points_from_camera_to_image(
                joints_cam, Ks, Ds, distortion_model.value
            )

            joints_residuals = ((joints_cam_proj - joints_2d) ** 2).sum(dim=-1)
            joints_reproj_loss = (joints_residuals * joints_2d_scores**2).mean()

            loss = joints_reproj_loss
            loss.backward()

            return loss

        betas_opt = betas.clone().detach()
        betas_opt = betas_opt.requires_grad_(True)
        pose_opt = pose[frame_idx].clone().detach()
        pose_opt = pose_opt.requires_grad_(True)

        optimizer = torch.optim.Adam([pose_opt], lr=1e-3)
        pbar = tqdm(range(n_iters), desc="Optimizing SMPL model", leave=False)

        for _ in pbar:
            loss = optimizer.step(opt_closure)
            pbar.set_postfix(loss=loss.detach().item())

        pose_out = pose.clone().detach()
        pose_out[frame_idx] = pose_opt.reshape(1, -1, 3).detach()

        return {
            "betas": betas_opt.detach(),
            "pose": pose_out.detach(),
            "trans": trans.reshape(n_frames, 3).detach(),
        }
