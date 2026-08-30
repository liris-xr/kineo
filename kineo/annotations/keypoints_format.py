# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class KeypointsFormat:
    name: str  # format of the keypoints (e.g., "h36m", "coco")
    n_keypoints: int
    keypoints_names: list[str]
    keypoints_connectivity: list[tuple[int, int]]

    def _post_init__(self):
        assert isinstance(
            self.keypoints_names, list
        ), "keypoints_names should be a list"
        assert all(
            isinstance(name, str) for name in self.keypoints_names
        ), "keypoints_names should be a list of strings"
        assert (
            len(self.keypoints_names) == self.n_keypoints
        ), "n_keypoints should be equal to the length of names"
        assert isinstance(
            self.keypoints_connectivity, list
        ), "keypoints_connectivity should be a list"
        assert all(
            isinstance(conn, tuple) for conn in self.keypoints_connectivity
        ), "keypoints_connectivity should be a list of tuples"
        assert len(self.keypoints_connectivity) == len(
            set(self.keypoints_connectivity)
        ), "keypoints_connectivity should not contain duplicate connections"

    @staticmethod
    def from_mmpose_dataset(dataset_name: str) -> KeypointsFormat:
        """Reads a format from MMPose's dataset configs.

        Only for formats named by a model at runtime rather than known ahead of
        time: it locates the installed MMPose source tree, so it works solely
        where the MMLab stack is already a dependency. The formats this codebase
        names itself are the constants below.
        """
        from mmengine.infer.infer import BaseInferencer

        repo_or_mim_dir = BaseInferencer._get_repo_or_mim_dir("mmpose")
        dataset_filepath = os.path.join(
            repo_or_mim_dir, "configs", "_base_", "datasets", f"{dataset_name}.py"
        )

        if not os.path.exists(dataset_filepath):
            raise FileNotFoundError(
                f"Dataset {dataset_name} not found in {repo_or_mim_dir}"
            )

        with open(dataset_filepath, "r") as f:
            module = {}
            exec(f.read(), module)
            dataset_info = module["dataset_info"]

        dataset_name = dataset_info["dataset_name"]
        keypoint_info: dict[int, dict] = dataset_info["keypoint_info"]
        skeleton_info: dict[int, dict] = dataset_info["skeleton_info"]

        n_keypoints = len(keypoint_info)
        names = [info["name"] for info in keypoint_info.values()]

        connectivity = []
        for link_info in skeleton_info.values():
            keypoint1, keypoint2 = link_info["link"]
            connectivity.append((names.index(keypoint1), names.index(keypoint2)))

        return KeypointsFormat(
            name=dataset_name,
            n_keypoints=n_keypoints,
            keypoints_names=names,
            keypoints_connectivity=connectivity,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_keypoints": self.n_keypoints,
            "keypoints_names": self.keypoints_names,
            "keypoints_connectivity": self.keypoints_connectivity,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> KeypointsFormat:
        return KeypointsFormat(
            name=dict_data["name"],
            n_keypoints=dict_data["n_keypoints"],
            keypoints_names=dict_data["keypoints_names"],
            keypoints_connectivity=dict_data["keypoints_connectivity"],
        )


# Transcribed from MMPose's `configs/_base_/datasets/{coco,h36m}.py` so that
# reading a keypoints format does not require MMPose to be installed.
COCO_17_KEYPOINTS_FORMAT = KeypointsFormat(
    name="coco",
    n_keypoints=17,
    keypoints_names=[
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ],
    keypoints_connectivity=[
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 11),
        (6, 12),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
    ],
)

H36M_17_KEYPOINTS_FORMAT = KeypointsFormat(
    name="h36m",
    n_keypoints=17,
    keypoints_names=[
        "root",
        "right_hip",
        "right_knee",
        "right_foot",
        "left_hip",
        "left_knee",
        "left_foot",
        "spine",
        "thorax",
        "neck_base",
        "head",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ],
    keypoints_connectivity=[
        (0, 4),
        (4, 5),
        (5, 6),
        (0, 1),
        (1, 2),
        (2, 3),
        (0, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (8, 11),
        (11, 12),
        (12, 13),
        (8, 14),
        (14, 15),
        (15, 16),
    ],
)
