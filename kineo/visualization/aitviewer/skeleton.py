# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import numpy as np

from aitviewer.renderables.skeletons import Skeletons
import matplotlib.pyplot as plt


class SkeletonWithConfidence(Skeletons):

    def __init__(
        self,
        joints_positions: np.ndarray,
        joints_connections: list[tuple[int, int]],
        joints_confidences: np.ndarray | None = None,
        bones_confidences: np.ndarray | None = None,
        radius: float = 0.01,
        colormap: str = "inferno",
        normalize_skeleton_confidence_per_frame: bool = False,
        icon="\u0089",
        **kwargs,
    ):
        """
        Initializer.
        :param joints_positions: A np array of shape (F, J, 3) containing J joint positions over F many time steps.
        :param joints_connections: The definition of the skeleton as a numpy array of shape (N_LINES, 2) where each row
          defines one connection between joints. The max entry in this array must be < J.
        :param joints_confidences: A np array of shape (F, J) containing the confidence of each joint over F many time steps.
        :param bones_confidences: A np array of shape (F, N_LINES) containing the confidence of each bone over F many time steps.
        :param radius: Radius of the sphere located at each joint's position.
        :param colors: 4-tuple color, yellow by default.
        :param kwargs: Remaining render arguments.
        """
        if not joints_positions.ndim == 3:
            raise ValueError(
                f"Expected joint_positions to be of shape (F, J, 3), got {joints_positions.shape}"
            )

        if joints_confidences is None:
            joints_confidences = np.ones_like(joints_positions[:, :, [0]]).astype(
                np.float32
            )

        if bones_confidences is None:
            bones_confidences = np.ones(
                (joints_positions.shape[0], len(joints_connections), 1)
            ).astype(np.float32)

        if not isinstance(bones_confidences, np.ndarray):
            bones_confidences = np.array(bones_confidences).astype(np.float32)

        if bones_confidences.shape == (
            joints_positions.shape[0],
            len(joints_connections),
        ):
            bones_confidences = bones_confidences.reshape(
                joints_positions.shape[0], len(joints_connections), 1
            )

        if not bones_confidences.shape == (
            joints_positions.shape[0],
            len(joints_connections),
            1,
        ):
            raise ValueError(
                f"Expected bones_confidences to be of shape (F, N_LINES, 1), got {bones_confidences.shape}"
            )

        if not isinstance(joints_confidences, np.ndarray):
            joints_confidences = np.array(joints_confidences).astype(np.float32)

        if joints_confidences.shape == (
            joints_positions.shape[0],
            joints_positions.shape[1],
        ):
            joints_confidences = joints_confidences.reshape(
                joints_positions.shape[0], joints_positions.shape[1], 1
            )

        if not joints_confidences.shape == (
            joints_positions.shape[0],
            joints_positions.shape[1],
            1,
        ):
            raise ValueError(
                f"Expected joint_confidences to be of shape (F, J, 1), got {joints_confidences.shape}"
            )

        super(SkeletonWithConfidence, self).__init__(
            joints_positions,
            joints_connections,
            radius=radius,
            icon=icon,
            **kwargs,
        )

        self.colormap = plt.get_cmap(colormap)
        self.joints_confidences = joints_confidences
        self.bones_confidences = bones_confidences
        self.normalize_skeleton_confidence_per_frame = normalize_skeleton_confidence_per_frame
        self.update_colors()

    def _get_color(self, confidences: np.ndarray) -> np.ndarray:
        """
        Compute the color of the joints based on the confidences.
        :param confidences: A np array of shape (*, 1) containing the confidence of each joint.
        :return: A np array of shape (*, 4) containing the color of each joint.
        """
        return self.colormap(confidences.reshape(-1)).reshape(
            *confidences.shape[:-1], 4
        )

    def update_colors(self):
        frame_joints_confidences = self.joints_confidences[self.current_frame_id]
        frame_bones_confidences = self.bones_confidences[self.current_frame_id]

        if self.normalize_skeleton_confidence_per_frame:
            joints_max_confidence = np.max(frame_joints_confidences)
            joints_min_confidence = np.min(frame_joints_confidences)
            frame_joints_confidences = (frame_joints_confidences - joints_min_confidence) / (joints_max_confidence - joints_min_confidence)

        if self.normalize_skeleton_confidence_per_frame:
            bones_max_confidence = np.max(frame_bones_confidences)
            bones_min_confidence = np.min(frame_bones_confidences)
            frame_bones_confidences = (frame_bones_confidences - bones_min_confidence) / (bones_max_confidence - bones_min_confidence)

        joints_colors = self._get_color(frame_joints_confidences)
        bones_colors = self._get_color(frame_bones_confidences)
        self.spheres.sphere_colors = joints_colors
        self.lines.line_colors = bones_colors

    def on_frame_update(self):
        self.update_colors()


class SkeletonWithVisibility(Skeletons):
    """
    Render a skeleton as a set of spheres that are connected with cone-shaped lines.
    """

    def __init__(
        self,
        joint_positions: np.ndarray,
        joint_connections: list[tuple[int, int]],
        joint_visibility: np.ndarray | None = None,
        radius: float = 0.01,
        colors: np.ndarray = np.array((1.0, 177 / 255, 1 / 255, 1.0)).reshape(1, 4),
        icon="\u0089",
        **kwargs,
    ):
        """
        Initializer.
        :param joint_positions: A np array of shape (F, J, 3) containing J joint positions over F many time steps.
        :param joint_connections: The definition of the skeleton as a numpy array of shape (N_LINES, 2) where each row
          defines one connection between joints. The max entry in this array must be < J.
        :param joint_visibility: A np array of shape (F, J) containing the visibility of each joint over F many time steps.
        :param radius: Radius of the sphere located at each joint's position.
        :param colors: 4-tuple color, yellow by default.
        :param kwargs: Remaining render arguments.
        """
        if not joint_positions.ndim == 3:
            raise ValueError(
                f"Expected joint_positions to be of shape (F, J, 3), got {joint_positions.shape}"
            )

        if joint_visibility is None:
            joint_visibility = np.ones_like(joint_positions[:, :, 0]).astype(bool)

        if not isinstance(joint_visibility, np.ndarray):
            joint_visibility = np.array(joint_visibility).astype(bool)

        if not isinstance(colors, np.ndarray):
            colors = np.array(colors)

        if colors.ndim == 1:
            colors = colors.reshape(1, 4).repeat(joint_positions.shape[0], axis=0)

        super(SkeletonWithVisibility, self).__init__(
            joint_positions,
            joint_connections,
            radius=radius,
            color=colors[0],
            icon=icon,
            **kwargs,
        )

        self.colors = colors
        self.joint_visibility = joint_visibility

        self.bone_visibility = np.zeros(
            (joint_positions.shape[0], len(joint_connections)), dtype=bool
        )
        for i, (j1, j2) in enumerate(joint_connections):
            self.bone_visibility[:, i] = (
                joint_visibility[:, j1] & joint_visibility[:, j2]
            )

        self.update_colors()
        self.update_visibility()

    @property
    def joint_visibility(self):
        return self._joint_visibility

    @joint_visibility.setter
    def joint_visibility(self, joint_visibility):
        self._joint_visibility = joint_visibility

    def update_colors(self):
        self.material.color = self.colors[self.current_frame_id]
        self.spheres.color = self.colors[self.current_frame_id]
        self.lines.color = self.colors[self.current_frame_id]

    def update_visibility(self):
        non_visible_joints = np.where(self.joint_visibility[self.current_frame_id] == 0)
        new_colors = np.tile(np.array(self.material.color), (self.spheres.n_spheres, 1))
        new_colors[non_visible_joints] = (0, 0, 0, 0)
        self.spheres.sphere_colors = new_colors

        non_visible_bones = np.where(self.bone_visibility[self.current_frame_id] == 0)
        new_colors = np.tile(np.array(self.material.color), (self.lines.n_lines, 1))
        new_colors[non_visible_bones] = (0, 0, 0, 0)
        self.lines.line_colors = new_colors

    def on_frame_update(self):
        self.update_colors()
        self.update_visibility()
