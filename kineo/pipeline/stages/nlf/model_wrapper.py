import torch

from typing import Dict, List, Literal
import math

import kineo.pipeline.stages.nlf.ptu3d as ptu3d
import kineo.pipeline.stages.nlf.ptu as ptu
import kineo.pipeline.stages.nlf.warping as warping


class NLFModelWrapper(torch.nn.Module):
    def __init__(self, torchscript_model_path: str):
        super().__init__()
        self.model = torch.jit.load(torchscript_model_path)
        self.skeleton_names = list(self.model.per_skeleton_indices.keys())

    def get_skeleton_weights(self, skeleton_name: str) -> Dict[str, torch.Tensor]:
        if skeleton_name not in self.skeleton_names:
            raise ValueError(
                f"Unknown skeleton name {skeleton_name}, use one of {self.skeleton_names}"
            )
        inds = self.model.per_skeleton_indices[skeleton_name]
        cano_points = self.model.crop_model.canonical_locs()[inds]
        return self.model.get_weights_for_canonical_points(cano_points)

    @torch.inference_mode()
    def infer_skeleton_keypoints(
        self,
        images: torch.Tensor,
        boxes: list[torch.Tensor],
        intrinsic_matrix: list[torch.Tensor] | torch.Tensor | None = None,
        extrinsic_matrix: list[torch.Tensor] | torch.Tensor | None = None,
        distortion_coeffs: list[torch.Tensor] | torch.Tensor | None = None,
        world_up_vector: list[torch.Tensor] | torch.Tensor | None = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 1,
        rot_aug_max_degrees: float = 25.0,
        skeleton_name: str = "smpl_24",
        use_half_precision: bool = True,
    ) -> dict[str, torch.Tensor]:
        if skeleton_name not in self.skeleton_names:
            raise ValueError(
                f"Unknown skeleton name {skeleton_name}, use one of {self.skeleton_names}"
            )
        weights = self.get_skeleton_weights(skeleton_name)

        num_joints = weights["w_tensor"].shape[0]
        num_vertices = 0

        with torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=use_half_precision
        ):
            return self.forward(
                images=images,
                boxes=boxes,
                weights=weights,
                num_vertices=num_vertices,
                num_joints=num_joints,
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=extrinsic_matrix,
                distortion_coeffs=distortion_coeffs,
                world_up_vector=world_up_vector,
                default_fov_degrees=default_fov_degrees,
                internal_batch_size=internal_batch_size,
                antialias_factor=antialias_factor,
                num_aug=num_aug,
                rot_aug_max_degrees=rot_aug_max_degrees,
            )

    @torch.inference_mode()
    def infer_smpl_keypoints(
        self,
        images: torch.Tensor,
        boxes: list[torch.Tensor],
        intrinsic_matrix: list[torch.Tensor] | torch.Tensor | None = None,
        extrinsic_matrix: list[torch.Tensor] | torch.Tensor | None = None,
        distortion_coeffs: list[torch.Tensor] | torch.Tensor | None = None,
        world_up_vector: list[torch.Tensor] | torch.Tensor | None = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 1,
        rot_aug_max_degrees: float = 25.0,
        model_name: Literal["smpl", "smplx"] = "smpl",
        use_half_precision: bool = True,
    ) -> dict[str, torch.Tensor]:
        if model_name not in ["smpl", "smplx"]:
            raise ValueError(
                f"Invalid model name: {model_name}. Expected 'smpl' or 'smplx'."
            )

        fitter = getattr(self.model.fitters, model_name)
        num_vertices = fitter.body_model.num_vertices
        num_joints = fitter.body_model.num_joints

        weights = self.model.weights[model_name]
        if weights["w_tensor"].ndim == 0:
            weights = self.model.get_weights_for_canonical_points(
                self.model.cano_all[model_name]
            )

        with torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=use_half_precision
        ):
            return self.forward(
                images=images,
                boxes=boxes,
                weights=weights,
                num_vertices=num_vertices,
                num_joints=num_joints,
                intrinsic_matrix=intrinsic_matrix,
                extrinsic_matrix=extrinsic_matrix,
                distortion_coeffs=distortion_coeffs,
                world_up_vector=world_up_vector,
                default_fov_degrees=default_fov_degrees,
                internal_batch_size=internal_batch_size,
                antialias_factor=antialias_factor,
                num_aug=num_aug,
                rot_aug_max_degrees=rot_aug_max_degrees,
            )

    @torch.inference_mode()
    def fit_smpl_model(
        self,
        model_name: Literal["smpl", "smplx"],
        poses3d_flat: torch.Tensor,
        confidences_flat: torch.Tensor,
        beta_regularizer: float,
        beta_regularizer2: float,
        final_adjust_rots: bool,
        num_iter: int,
        requested_keys: list[str],
    ):
        if model_name not in ["smpl", "smplx"]:
            raise ValueError(
                f"Invalid model name: {model_name}. Expected 'smpl' or 'smplx'."
            )

        # Shape should be (batch, n_keypoints, 3)
        assert poses3d_flat.ndim == 3 and poses3d_flat.shape[-1] == 3

        fitter = getattr(self.model.fitters, model_name)
        num_vertices = fitter.body_model.num_vertices
        num_joints = fitter.body_model.num_joints

        mean_poses = torch.mean(poses3d_flat, dim=-2, keepdim=True)
        poses3d_flat = poses3d_flat - mean_poses

        joints_flat, vertices_flat = torch.split(
            poses3d_flat, [num_joints, num_vertices], dim=-2
        )
        joints_weights, vertices_weights = torch.split(
            confidences_flat, [num_joints, num_vertices], dim=-1
        )

        fit = fitter.fit(
            vertices_flat,
            joints_flat,
            vertex_weights=vertices_weights,
            joint_weights=joints_weights,
            num_iter=num_iter,
            beta_regularizer=beta_regularizer,
            beta_regularizer2=beta_regularizer2,
            final_adjust_rots=final_adjust_rots,
            requested_keys=requested_keys,
        )

        pose = fit["pose_rotvecs"]
        betas = fit["shape_betas"]
        trans = fit["trans"] + mean_poses.squeeze(-2)
        return {
            "pose": pose,
            "betas": betas,
            "trans": trans,
        }

    def forward(
        self,
        images: torch.Tensor,
        boxes: list[torch.Tensor],
        weights: Dict[str, torch.Tensor],
        num_vertices: int,
        num_joints: int,
        intrinsic_matrix: list[torch.Tensor] | torch.Tensor | None = None,
        extrinsic_matrix: list[torch.Tensor] | torch.Tensor | None = None,
        distortion_coeffs: list[torch.Tensor] | torch.Tensor | None = None,
        world_up_vector: list[torch.Tensor] | torch.Tensor | None = None,
        default_fov_degrees: float = 55.0,
        internal_batch_size: int = 64,
        antialias_factor: int = 1,
        num_aug: int = 1,
        rot_aug_max_degrees: float = 25.0,
    ):
        if images.ndim == 3:
            images = images.unsqueeze(0)

        if images.dtype == torch.uint8:
            images = images / 255.0

        images = im_to_linear(images)
        n_images = len(images)
        device = images.device

        if sum(len(b) for b in boxes) == 0:
            return self._predict_empty(images, num_vertices, num_joints)

        if intrinsic_matrix is None:
            intrinsic_matrix = ptu3d.intrinsic_matrix_from_field_of_view(
                default_fov_degrees, images.shape[2:4], device=device
            )

        if len(intrinsic_matrix) == 1:
            # If intrinsic_matrix is not given, fill it in based on field of view
            intrinsic_matrix = torch.repeat_interleave(
                intrinsic_matrix, n_images, dim=0
            )

        if distortion_coeffs is None:
            distortion_coeffs = torch.zeros((n_images, 5), device=device)
        # If one distortion coeff/extrinsic matrix is given, repeat it for all images
        if len(distortion_coeffs) == 1:
            distortion_coeffs = torch.repeat_interleave(
                distortion_coeffs, n_images, dim=0
            )

        if extrinsic_matrix is None:
            extrinsic_matrix = torch.eye(4, device=device).unsqueeze(0)
        if len(extrinsic_matrix) == 1:
            extrinsic_matrix = torch.repeat_interleave(
                extrinsic_matrix, n_images, dim=0
            )

        # Now repeat these camera params for each box
        n_box_per_image_list = [len(b) for b in boxes]
        n_box_per_image = torch.tensor([len(b) for b in boxes], device=device)

        intrinsic_matrix = torch.repeat_interleave(
            intrinsic_matrix, n_box_per_image, dim=0
        )
        distortion_coeffs = torch.repeat_interleave(
            distortion_coeffs, n_box_per_image, dim=0
        )

        # Up-vector in camera-space
        if world_up_vector is None:
            world_up_vector = torch.tensor(
                [0, -1, 0], device=device, dtype=torch.float32
            )

        camspace_up = torch.einsum(
            "c,bCc->bC", world_up_vector, extrinsic_matrix[..., :3, :3]
        )
        camspace_up = torch.repeat_interleave(camspace_up, n_box_per_image, dim=0)

        # Set up the test-time augmentation parameters
        aug_gammas = ptu.linspace(0.6, 1.0, num_aug, dtype=torch.float32, device=device)

        aug_angle_range = rot_aug_max_degrees * (torch.pi / 180.0)
        aug_angles = ptu.linspace(
            -aug_angle_range,
            aug_angle_range,
            num_aug,
            dtype=torch.float32,
            device=device,
        )

        if num_aug == 1:
            aug_scales = torch.tensor([1.0], device=device, dtype=torch.float32)
        else:
            aug_scales = torch.cat(
                [
                    ptu.linspace(
                        0.8,
                        1.0,
                        num_aug // 2,
                        endpoint=False,
                        dtype=torch.float32,
                        device=device,
                    ),
                    torch.linspace(
                        1.0,
                        1.1,
                        num_aug - num_aug // 2,
                        dtype=torch.float32,
                        device=device,
                    ),
                ],
                dim=0,
            )
        aug_should_flip = (
            torch.arange(0, num_aug, device=device) - num_aug // 2
        ) % 2 != 0
        aug_flipmat = torch.tensor(
            [[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32, device=device
        )
        aug_maybe_flipmat = torch.where(
            aug_should_flip[:, torch.newaxis, torch.newaxis],
            aug_flipmat,
            torch.eye(3, device=device),
        )
        aug_rotmat = ptu3d.rotation_mat(-aug_angles, rot_axis="z")
        aug_rotflipmat = aug_maybe_flipmat @ aug_rotmat

        poses3d_flat, uncert_flat = self._predict_in_batches(
            images,
            weights,
            intrinsic_matrix,
            distortion_coeffs,
            camspace_up,
            boxes,
            internal_batch_size,
            aug_should_flip,
            aug_rotflipmat,
            aug_gammas,
            aug_scales,
            antialias_factor,
        )

        poses3d_flat = ptu3d.scale_align(poses3d_flat)
        mean = torch.mean(poses3d_flat, dim=(-3, -2), keepdim=True)
        poses3d_flat_submean = (poses3d_flat - mean).float()
        poses3d_flat_submean, final_weights = ptu.weighted_geometric_median(
            poses3d_flat_submean, uncert_flat**-1.5, dim=-3, n_iter=10, eps=50.0
        )
        poses3d_flat = poses3d_flat_submean.double() + mean.squeeze(1)

        uncert_flat = ptu.weighted_mean(uncert_flat, final_weights, dim=-2)

        # Project the 3D poses to get the 2D poses
        poses2d_flat_normalized = ptu3d.to_homogeneous(
            warping.distort_points(
                ptu3d.project(poses3d_flat.float()), distortion_coeffs
            )
        )
        poses2d_flat = torch.einsum(
            "bnk,bjk->bnj", poses2d_flat_normalized, intrinsic_matrix[:, :2, :]
        )

        n_box_per_image_list = [len(b) for b in boxes]

        if sum(n_box_per_image_list) == 0:
            return self._predict_empty(images, num_vertices, num_joints)

        n_box_per_image = torch.tensor(n_box_per_image_list, device=device)
        # Convert to world coordinates
        inv_extrinsic_matrix = torch.repeat_interleave(
            torch.linalg.inv(extrinsic_matrix.double()), n_box_per_image, dim=0
        )
        poses3d_flat = torch.einsum(
            "bnk,bjk->bnj",
            ptu3d.to_homogeneous(poses3d_flat),
            inv_extrinsic_matrix[:, :3, :],
        )

        vertices3d_flat, joints3d_flat = torch.split(
            poses3d_flat,
            [num_vertices, num_joints],
            dim=-2,
        )

        vertices2d_flat, joints2d_flat = torch.split(
            poses2d_flat,
            [num_vertices, num_joints],
            dim=-2,
        )

        vertices_uncertainties_flat, joints_uncertainties_flat = torch.split(
            uncert_flat,
            [num_vertices, num_joints],
            dim=-1,
        )

        vertices3d = list(torch.split(vertices3d_flat.float(), n_box_per_image_list))
        vertices2d = list(torch.split(vertices2d_flat.float(), n_box_per_image_list))
        joints3d = list(torch.split(joints3d_flat.float(), n_box_per_image_list))
        joints2d = list(torch.split(joints2d_flat.float(), n_box_per_image_list))
        vertices_uncertainties = list(
            torch.split(vertices_uncertainties_flat.float(), n_box_per_image_list)
        )
        joints_uncertainties = list(
            torch.split(joints_uncertainties_flat.float(), n_box_per_image_list)
        )

        batch_size = len(vertices3d)

        vertices_confidences = [
            torch.zeros_like(vertices2d[i][..., 0:1]) for i in range(batch_size)
        ]
        joints_confidences = [
            torch.zeros_like(joints2d[i][..., 0:1]) for i in range(batch_size)
        ]

        for i in range(batch_size):
            valid_vertices_mask = (
                torch.isfinite(vertices_uncertainties[i])
                & torch.isfinite(vertices3d[i]).all(dim=-1)
                & torch.isfinite(vertices2d[i]).all(dim=-1)
            )
            valid_joints_mask = (
                torch.isfinite(joints_uncertainties[i])
                & torch.isfinite(joints3d[i]).all(dim=-1)
                & torch.isfinite(joints2d[i]).all(dim=-1)
            )

            vertices3d[i] = torch.where(
                valid_vertices_mask.unsqueeze(-1),
                vertices3d[i],
                torch.zeros_like(vertices3d[i]),
            )
            joints3d[i] = torch.where(
                valid_joints_mask.unsqueeze(-1),
                joints3d[i],
                torch.zeros_like(joints3d[i]),
            )
            vertices2d[i] = torch.where(
                valid_vertices_mask.unsqueeze(-1),
                vertices2d[i],
                torch.zeros_like(vertices2d[i]),
            )
            joints2d[i] = torch.where(
                valid_joints_mask.unsqueeze(-1),
                joints2d[i],
                torch.zeros_like(joints2d[i]),
            )
            vertices_uncertainties[i] = torch.where(
                valid_vertices_mask,
                vertices_uncertainties[i],
                torch.zeros_like(vertices_uncertainties[i]),
            )
            joints_uncertainties[i] = torch.where(
                valid_joints_mask,
                joints_uncertainties[i],
                torch.zeros_like(joints_uncertainties[i]),
            )
            vertices_confidences[i] = torch.where(
                valid_vertices_mask,
                confidence_from_uncertainty(vertices_uncertainties[i]),
                torch.zeros_like(vertices_uncertainties[i]),
            )
            joints_confidences[i] = torch.where(
                valid_joints_mask,
                confidence_from_uncertainty(joints_uncertainties[i]),
                torch.zeros_like(joints_uncertainties[i]),
            )

        result = dict(
            vertices3d=vertices3d,
            joints3d=joints3d,
            vertices2d=vertices2d,
            joints2d=joints2d,
            vertices_uncertainties=vertices_uncertainties,
            joints_uncertainties=joints_uncertainties,
            vertices_confidences=vertices_confidences,
            joints_confidences=joints_confidences,
        )
        return result

    def _predict_empty(
        self,
        image: torch.Tensor,
        num_vertices: int,
        num_joints: int,
    ):
        device = image.device

        vertices3d = torch.zeros(
            (0, num_vertices, 3), dtype=torch.float32, device=device
        )
        vertices2d = torch.zeros(
            (0, num_vertices, 2), dtype=torch.float32, device=device
        )

        joints3d = torch.zeros((0, num_joints, 3), dtype=torch.float32, device=device)
        joints2d = torch.zeros((0, num_joints, 2), dtype=torch.float32, device=device)

        vertices_uncertainties = torch.zeros(
            (0, num_vertices), dtype=torch.float32, device=device
        )
        vertices_confidences = torch.zeros(
            (0, num_vertices), dtype=torch.float32, device=device
        )

        joints_uncertainties = torch.zeros(
            (0, num_joints), dtype=torch.float32, device=device
        )
        joints_confidences = torch.zeros(
            (0, num_joints), dtype=torch.float32, device=device
        )

        n_images = image.shape[0]

        result = dict(
            vertices3d=[vertices3d] * n_images,
            joints3d=joints3d,
            vertices2d=[vertices2d] * n_images,
            joints2d=[joints2d] * n_images,
            vertices_uncertainties=[vertices_uncertainties] * n_images,
            joints_uncertainties=[joints_uncertainties] * n_images,
            vertices_confidences=[vertices_confidences] * n_images,
            joints_confidences=[joints_confidences] * n_images,
        )
        return result

    def _predict_in_batches(
        self,
        images: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        camspace_up: torch.Tensor,
        boxes: List[torch.Tensor],
        internal_batch_size: int,
        aug_should_flip: torch.Tensor,
        aug_rotflipmat: torch.Tensor,
        aug_gammas: torch.Tensor,
        aug_scales: torch.Tensor,
        antialias_factor: int,
    ):
        num_aug = len(aug_gammas)
        boxes_per_batch = internal_batch_size // num_aug
        boxes_flat = torch.cat(boxes, dim=0)
        image_id_per_box = torch.repeat_interleave(
            torch.arange(len(boxes)), torch.tensor([len(b) for b in boxes])
        )

        if boxes_per_batch == 0:
            # Run all as a single batch
            return self._predict_single_batch(
                images,
                weights,
                intrinsic_matrix,
                distortion_coeffs,
                camspace_up,
                boxes_flat,
                image_id_per_box,
                aug_rotflipmat,
                aug_should_flip,
                aug_scales,
                aug_gammas,
                antialias_factor,
            )
        else:
            # Chunk the image crops into batches and predict them one by one
            n_total_boxes = len(boxes_flat)
            n_batches = int(math.ceil(n_total_boxes / boxes_per_batch))
            poses3d_batches = []
            uncert_batches = []
            for i in range(n_batches):
                batch_slice = slice(i * boxes_per_batch, (i + 1) * boxes_per_batch)
                poses3d, uncert = self._predict_single_batch(
                    images,
                    weights,
                    intrinsic_matrix[batch_slice],
                    distortion_coeffs[batch_slice],
                    camspace_up[batch_slice],
                    boxes_flat[batch_slice],
                    image_id_per_box[batch_slice],
                    aug_rotflipmat,
                    aug_should_flip,
                    aug_scales,
                    aug_gammas,
                    antialias_factor,
                )
                poses3d_batches.append(poses3d)
                uncert_batches.append(uncert)
            return torch.cat(poses3d_batches, dim=0), torch.cat(uncert_batches, dim=0)

    def _predict_single_batch(
        self,
        images: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        intrinsic_matrix: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        camspace_up: torch.Tensor,
        boxes: torch.Tensor,
        image_ids: torch.Tensor,
        aug_rotflipmat: torch.Tensor,
        aug_should_flip: torch.Tensor,
        aug_scales: torch.Tensor,
        aug_gammas: torch.Tensor,
        antialias_factor: int,
    ):
        # Get crops and info about the transformation used to create them
        # Each has shape [num_aug, n_boxes, ...]
        crops, new_intrinsic_matrix, R = self._get_crops(
            images,
            intrinsic_matrix,
            distortion_coeffs,
            camspace_up,
            boxes,
            image_ids,
            aug_rotflipmat,
            aug_scales,
            aug_gammas,
            antialias_factor,
        )

        # Flatten each and predict the pose with the crop model
        new_intrinsic_matrix_flat = torch.reshape(new_intrinsic_matrix, (-1, 3, 3))
        res = self.model.crop_model.input_resolution
        crops_flat = torch.reshape(crops, (-1, 3, res, res))

        n_cases = crops.shape[1]
        aug_should_flip_flat = torch.repeat_interleave(aug_should_flip, n_cases, dim=0)

        poses_flat, uncert_flat = self.predict_multi_same_weights(
            crops_flat, new_intrinsic_matrix_flat, weights, aug_should_flip_flat
        )
        poses_flat = poses_flat.double()
        n_joints = poses_flat.shape[-2]

        poses = torch.reshape(poses_flat, [-1, n_cases, n_joints, 3])
        uncert = torch.reshape(uncert_flat, [-1, n_cases, n_joints])
        poses_orig_camspace = poses @ R.double()

        # Transpose to [n_boxes, num_aug, ...]
        return poses_orig_camspace.transpose(0, 1), uncert.transpose(0, 1)

    def predict_multi_same_weights(
        self,
        image: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        flip_canonicals_per_image: torch.Tensor,
    ):
        features_processed = self.model.crop_model.get_features(image)
        coords2d, coords3d, uncertainties = (
            self.model.crop_model.heatmap_head.decode_features_multi_same_weights(
                features_processed, weights, flip_canonicals_per_image
            )
        )

        with torch.amp.autocast("cuda", enabled=False):
            return self.model.crop_model.heatmap_head.reconstruct_absolute(
                coords2d.float(),
                coords3d.float(),
                uncertainties.float(),
                intrinsic_matrix.float(),
            )

    def _get_crops(
        self,
        images: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        camspace_up: torch.Tensor,
        boxes: torch.Tensor,
        image_ids: torch.Tensor,
        aug_rotflipmat: torch.Tensor,
        aug_scales: torch.Tensor,
        aug_gammas: torch.Tensor,
        antialias_factor: int,
    ):
        with torch.amp.autocast("cuda", enabled=False):
            return self.model._get_crops(
                images,
                intrinsic_matrix.float(),
                distortion_coeffs.float(),
                camspace_up.float(),
                boxes.float(),
                image_ids,
                aug_rotflipmat.float(),
                aug_scales.float(),
                aug_gammas.float(),
                antialias_factor,
            )


def confidence_from_uncertainty(w: torch.Tensor):
    return torch.exp(-5 * w)


def im_to_linear(im: torch.Tensor):
    if im.dtype == torch.uint8:
        return im.to(dtype=torch.float16).mul_(1.0 / 255.0).pow_(2.2)
    elif im.dtype == torch.uint16:
        return (
            im.to(dtype=torch.float16)
            .mul_(1.0 / 65504.0)
            .nan_to_num_(posinf=1.0)
            .pow_(2.2)
        )
        # return im.to(dtype=torch.float16).nan_to_num_(posinf=65504.0).div_(65504.0).pow_(2.2)
    elif im.dtype == torch.float16:
        return im**2.2
    else:
        return im.to(dtype=torch.float16).pow_(2.2)
