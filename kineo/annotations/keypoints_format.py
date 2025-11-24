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

try:
    from mmengine.infer.infer import BaseInferencer
except ImportError:
    pass


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
            keypoint1_idx = names.index(keypoint1)
            keypoint2_idx = names.index(keypoint2)
            connectivity.append((keypoint1_idx, keypoint2_idx))

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
