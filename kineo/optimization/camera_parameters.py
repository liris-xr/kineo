# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import torch

from roma import special_gramschmidt
from kineo.geometry.camera import (
    inverse_K,
    inverse_Rt,
    compute_focal_length_rnorm_from_hfov,
)
from typing import Sequence, Literal
from kineo.torch_utils import check_shape


class CameraIntrinsicsParameters(torch.nn.Module):
    """
    Camera intrinsic parameters for optimization.

    This class contains camera intrinsic parameters that are either fixed (like image size) or optimizable parameters:

    Fixed parameters:
    - Image size (height, width) in pixels

    Optimizable parameters:
    - Radial distortion coefficients (k1, k2, k3)
    - Tangential distortion coefficients (p1, p2)
    - Focal length (normalized)
    - Principal point (normalized)
    """

    def __init__(
        self,
        image_size_hw_px: torch.Tensor | tuple[int, int] | Sequence[tuple[int, int]],
        distortion_model: Literal["brown_conrady", "opencv_fisheye"] = "brown_conrady",
        batch_size: int = 1,
        fx_and_fy: bool = False,
        shared_parameters: bool = False,
        requires_grad: bool = False,
        device: torch.device = "cpu",
    ):
        super(CameraIntrinsicsParameters, self).__init__()

        image_size_hw_px = torch.as_tensor(
            image_size_hw_px, device=device, dtype=torch.float32
        )

        if image_size_hw_px.ndim == 1:
            image_size_hw_px = image_size_hw_px.unsqueeze(0)

        check_shape(image_size_hw_px, [(1, 2), (batch_size, 2)])

        self._batch_size = batch_size
        self._fx_and_fy = fx_and_fy
        self._image_size_hw_px = image_size_hw_px.expand(batch_size, -1)
        self._shared_parameters = shared_parameters
        self._distortion_model = distortion_model

        if distortion_model == "brown_conrady":
            self._distortion_coefficients = torch.nn.Parameter(
                torch.zeros(
                    (1 if shared_parameters else batch_size, 5),
                    device=device,
                    dtype=torch.float32,
                ),
                requires_grad=requires_grad,
            )
        elif distortion_model == "opencv_fisheye":
            self._distortion_coefficients = torch.nn.Parameter(
                torch.zeros(
                    (1 if shared_parameters else batch_size, 4),
                    device=device,
                    dtype=torch.float32,
                ),
                requires_grad=requires_grad,
            )
        else:
            raise ValueError(f"Invalid distortion model: {distortion_model}")

        # Resolution-independent focal length (fx, fy)
        default_f = compute_focal_length_rnorm_from_hfov(60)

        if fx_and_fy:
            self._normalized_focal_length = torch.nn.Parameter(
                torch.tensor(
                    [default_f, default_f],
                    device=device,
                    dtype=torch.float32,
                ).repeat(1 if shared_parameters else batch_size, 1),
                requires_grad=requires_grad,
            )
        else:
            self._normalized_focal_length = torch.nn.Parameter(
                torch.tensor(
                    [default_f],
                    device=device,
                    dtype=torch.float32,
                ).repeat(1 if shared_parameters else batch_size, 1),
                requires_grad=requires_grad,
            )

        # Resolution-independent principal point (cx, cy)
        self._normalized_principal_point = torch.nn.Parameter(
            torch.tensor(
                [0.5, 0.5],
                device=device,
                dtype=torch.float32,
            ).repeat(1 if shared_parameters else batch_size, 1),
            requires_grad=requires_grad,
        )

    @property
    def shared_parameters(self) -> bool:
        return self._shared_parameters

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def image_size_hw_px(self) -> torch.Tensor:
        return self._image_size_hw_px.expand(self._batch_size, -1)

    @property
    def K(self) -> torch.Tensor:
        """
        Get the camera intrinsic matrix (camera to image plane).
        """
        K = torch.zeros(
            (self._batch_size, 3, 3), device=self._normalized_focal_length.device
        )

        if self._fx_and_fy:
            K[..., 0, 0] = (
                self._normalized_focal_length[..., 0] * self._image_size_hw_px[..., 1]
            )
            K[..., 1, 1] = (
                self._normalized_focal_length[..., 1] * self._image_size_hw_px[..., 0]
            )
        else:
            K[..., 0, 0] = (
                self._normalized_focal_length[..., 0] * self._image_size_hw_px[..., 1]
            )
            K[..., 1, 1] = K[..., 0, 0]

        K[..., 0, 2] = (
            self._normalized_principal_point[..., 0] * self._image_size_hw_px[..., 1]
        )
        K[..., 1, 2] = (
            self._normalized_principal_point[..., 1] * self._image_size_hw_px[..., 0]
        )
        K[..., 2, 2] = 1.0
        return K.expand(self._batch_size, -1, -1)

    @property
    def K_inv(self) -> torch.Tensor:
        """
        Get the inverse of the camera intrinsic matrix (image plane to camera).
        """
        return inverse_K(self.K)

    @K.setter
    def K(self, K: torch.Tensor) -> None:
        if K.ndim == 2:
            K = K.unsqueeze(0)

        check_shape(K, [(self._batch_size, 3, 3), (1, 3, 3)])
        K = K.expand(self._batch_size, -1, -1)

        focal_length_x_rnorm = K[..., 0, 0] / self._image_size_hw_px[..., 1]
        focal_length_y_rnorm = K[..., 1, 1] / self._image_size_hw_px[..., 0]
        focal_length_rnorm = torch.stack(
            [focal_length_x_rnorm, focal_length_y_rnorm], dim=-1
        )

        if not self._fx_and_fy:
            focal_length_rnorm = focal_length_rnorm.mean(dim=-1, keepdim=True)

        principal_point_x_rnorm = K[..., 0, 2] / self._image_size_hw_px[..., 1]
        principal_point_y_rnorm = K[..., 1, 2] / self._image_size_hw_px[..., 0]
        principal_point_rnorm = torch.stack(
            [principal_point_x_rnorm, principal_point_y_rnorm], dim=-1
        )

        if self._shared_parameters:
            self._normalized_focal_length.data = focal_length_rnorm.to(
                self._normalized_focal_length.device
            ).mean(dim=0, keepdim=True)
            self._normalized_principal_point.data = principal_point_rnorm.to(
                self._normalized_principal_point.device
            ).mean(dim=0, keepdim=True)
        else:
            self._normalized_focal_length.data = focal_length_rnorm.to(
                self._normalized_focal_length.device
            )
            self._normalized_principal_point.data = principal_point_rnorm.to(
                self._normalized_principal_point.device
            )

    @property
    def normalized_principal_point(self) -> torch.Tensor:
        """
        Get the principal point normalized by the camera resolution.
        """
        return self._normalized_principal_point

    @property
    def normalized_focal_length(self) -> torch.Tensor:
        """
        Get the focal length normalized by the camera resolution.
        """
        return self._normalized_focal_length.expand(self._batch_size, 2)

    @property
    def distortion_coefficients(self) -> torch.Tensor:
        return self._distortion_coefficients

    @distortion_coefficients.setter
    def distortion_coefficients(self, distortion_coefficients: torch.Tensor) -> None:
        if self._distortion_model == "brown_conrady":
            check_shape(distortion_coefficients, [(1, 5), (self._batch_size, 5)])

            if self._shared_parameters:
                self._distortion_coefficients.data = distortion_coefficients.to(
                    self._distortion_coefficients.device
                ).mean(dim=0, keepdim=True)
            else:
                self._distortion_coefficients.data = distortion_coefficients.to(
                    self._distortion_coefficients.device
                ).expand(self._batch_size, -1)

        elif self._distortion_model == "opencv_fisheye":
            check_shape(distortion_coefficients, [(1, 4), (self._batch_size, 4)])

            if self._shared_parameters:
                self._distortion_coefficients.data = distortion_coefficients.to(
                    self._distortion_coefficients.device
                ).mean(dim=0, keepdim=True)
            else:
                self._distortion_coefficients.data = distortion_coefficients.to(
                    self._distortion_coefficients.device
                ).expand(self._batch_size, -1)
        else:
            raise ValueError(f"Invalid distortion model: {self._distortion_model}")

    @property
    def h_fov(self) -> torch.Tensor:
        """
        Get the horizontal field of view in degrees.
        """
        return torch.rad2deg(2 * torch.atan(1 / self._normalized_focal_length[..., 0]))

    @property
    def v_fov(self) -> torch.Tensor:
        """
        Get the vertical field of view in degrees.
        """
        if self._fx_and_fy:
            return torch.rad2deg(
                2 * torch.atan(1 / self._normalized_focal_length[..., 1])
            )
        else:
            return torch.rad2deg(
                2 * torch.atan(1 / self._normalized_focal_length[..., 0])
            )

    def set_optimized_parameters(
        self,
        distortion_coefficients: bool | None = None,
        focal_length: bool | None = None,
        principal_point: bool | None = None,
    ) -> None:
        """
        Enable or disable optimization of camera parameters.

        Args:
            distortion_coefficients: Whether to optimize distortion coefficients.
            focal_length: Whether to optimize focal length.
            principal_point: Whether to optimize principal point.
        """
        if distortion_coefficients is not None:
            self._distortion_coefficients.requires_grad = distortion_coefficients
        if focal_length is not None:
            self._normalized_focal_length.requires_grad = focal_length
        if principal_point is not None:
            self._normalized_principal_point.requires_grad = principal_point

    @property
    def optimized_parameters(self) -> list[torch.Tensor]:
        """
        Get the optimized parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def named_optimized_parameters(self) -> dict[str, torch.Tensor]:
        """
        Get the optimized parameters with their names.
        """
        params = {}

        if self._distortion_coefficients.requires_grad:
            params["distortion_coefficients"] = self._distortion_coefficients
        if self._normalized_focal_length.requires_grad:
            params["focal_length"] = self._normalized_focal_length
        if self._normalized_principal_point.requires_grad:
            params["principal_point"] = self._normalized_principal_point
        return params


class CameraExtrinsicsParameters(torch.nn.Module):
    """
    Camera extrinsic parameters for optimization.

    This class contains camera extrinsic parameters that are optimizable:

    Optimizable parameters:
    - Camera rotation (6D representation)
    - Camera translation

    The camera rotation and translation are using the OpenCV convention (right-handed +Z-up, +X-right, +Y-forward)
    """

    def __init__(
        self,
        batch_size: int = 1,
        requires_grad: bool = False,
        device: torch.device = "cpu",
    ):
        super(CameraExtrinsicsParameters, self).__init__()

        self._batch_size = batch_size

        # 3D translation (world to camera)
        self._transl = torch.nn.Parameter(
            torch.zeros(
                (batch_size, 3),
                device=device,
                dtype=torch.float32,
            ),
            requires_grad=requires_grad,
        )

        # 6D rotation (world to camera)
        self._rot6d = torch.nn.Parameter(
            torch.eye(3, device=device, dtype=torch.float32)
            .repeat(batch_size, 1, 1)[..., :3, :2]
            .reshape(
                (
                    batch_size,
                    6,
                )
            ),
            requires_grad=requires_grad,
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def transl(self) -> torch.Tensor:
        return self._transl

    @transl.setter
    def transl(self, transl: torch.Tensor) -> None:
        check_shape(transl, [(self._batch_size, 3), (1, 3)])
        transl = transl.expand(self._batch_size, -1)
        self._transl.data = transl.to(self._transl.device)

    @property
    def rot6d(self) -> torch.Tensor:
        return self._rot6d

    @rot6d.setter
    def rot6d(self, rot6d: torch.Tensor) -> None:
        check_shape(rot6d, [(self._batch_size, 6), (1, 6)])
        rot6d = rot6d.expand(self._batch_size, -1)
        self._rot6d.data = rot6d.to(self._rot6d.device)

    @property
    def Rt(self) -> torch.Tensor:
        """
        Get the camera extrinsic matrix (world to camera).
        """
        R = special_gramschmidt(self._rot6d.view(self._batch_size, 3, 2))
        t = self._transl.view(self._batch_size, 3, 1)
        return torch.cat([R, t], dim=-1)

    @Rt.setter
    def Rt(self, Rt: torch.Tensor):
        """
        Set the camera extrinsic matrix (world to camera).
        Internally converts the 3x4 matrix to a 6D rotation and a translation vector.
        """
        check_shape(Rt, [(self._batch_size, 3, 4), (1, 3, 4)])
        Rt = Rt.expand(self._batch_size, -1, -1)

        self._rot6d.data = (
            Rt[..., :3, :2].reshape((self._batch_size, 6)).to(self._rot6d.device)
        )
        self._transl.data = (
            Rt[..., :3, 3].reshape((self._batch_size, 3)).to(self._transl.device)
        )

    @property
    def Rt_inv(self) -> torch.Tensor:
        """
        Get the inverse of the camera extrinsic matrix (camera to world).
            Rt_inverse = [R.T, -R.T @ t]
        """
        return inverse_Rt(self.Rt)

    @property
    def world_pos(self) -> torch.Tensor:
        """
        Get the position of the camera in the world frame (in OpenCV's convention).
        """
        R = special_gramschmidt(self._rot6d.view(self._batch_size, 3, 2))
        t = self._transl.view(self._batch_size, 3, 1)
        return (-R.transpose(-2, -1) @ t).squeeze(-1)

    @world_pos.setter
    def world_pos(self, world_pos: torch.Tensor) -> None:
        """
        Set the position of the camera in the world frame.
        """
        check_shape(world_pos, [(self._batch_size, 3), (1, 3)])
        world_pos = world_pos.expand(self._batch_size, -1)
        R = special_gramschmidt(self._rot6d.view(self._batch_size, 3, 2))
        self._transl.data = -R.transpose(-2, -1) @ world_pos.to(R.device)

    @property
    def world_rotmat(self) -> torch.Tensor:
        """
        Get the rotation matrix of the camera in the world frame.
        """
        R = special_gramschmidt(self._rot6d.view(self._batch_size, 3, 2))
        return R.transpose(-2, -1)

    @world_rotmat.setter
    def world_rotmat(self, world_rotmat: torch.Tensor) -> None:
        """
        Set the rotation matrix of the camera in the world frame.
        """
        check_shape(world_rotmat, [(self._batch_size, 3, 3), (1, 3, 3)])
        world_rotmat = world_rotmat.expand(self._batch_size, -1, -1)
        rot6d = world_rotmat.transpose(-2, -1)[..., :, :2].reshape(
            (self._batch_size, 6)
        )
        self._rot6d.data = rot6d.to(self._rot6d.device)

    def set_optimized_parameters(
        self,
        rotation: bool | None = None,
        translation: bool | None = None,
    ) -> None:
        """
        Enable or disable optimization of camera parameters.

        Args:
            rotation: Whether to optimize camera rotation.
            translation: Whether to optimize camera translation.
        """
        if rotation is not None:
            self._rot6d.requires_grad = rotation
        if translation is not None:
            self._transl.requires_grad = translation

    @property
    def optimized_parameters(self) -> list[torch.Tensor]:
        """
        Get the optimized parameters.
        """
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def named_optimized_parameters(self) -> dict[str, torch.Tensor]:
        """
        Get the optimized parameters with their names.
        """
        params = {}
        if self._rot6d.requires_grad:
            params["rot6d"] = self._rot6d
        if self._transl.requires_grad:
            params["transl"] = self._transl
        return params
