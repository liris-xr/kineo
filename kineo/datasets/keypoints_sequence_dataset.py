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
from abc import ABC, abstractmethod
from typing import TypedDict, Iterator, Iterable
import torch
from mmengine.infer.infer import BaseInferencer
import os
import orjson

from kineo.io.frame_sequence_loader import FrameSequenceLoader
from kineo.io.audio_loader import AudioLoader


class Keypoints2D(TypedDict):
    xy: torch.Tensor
    annotated: torch.Tensor


class Keypoints3D(TypedDict):
    xyz: torch.Tensor
    annotated: torch.Tensor


class Annotations2D(TypedDict):
    bboxes_xyxy: dict[int, dict[str, torch.Tensor]] | None
    keypoints: dict[int, dict[str, Keypoints2D]] | None

    @staticmethod
    def parse(dict_annotations: dict | None) -> Annotations2D | None:
        if dict_annotations is None:
            return None
        bboxes_xyxy = dict_annotations.pop("bboxes_xyxy")
        keypoints = dict_annotations.pop("keypoints")
        keypoints = {int(k): v for k, v in keypoints.items()}
        bboxes_xyxy = {int(k): v for k, v in bboxes_xyxy.items()}
        return Annotations2D(
            bboxes_xyxy=bboxes_xyxy, keypoints=keypoints, **dict_annotations
        )


class Annotations3D(TypedDict):
    keypoints: dict[int, Keypoints3D] | None

    @staticmethod
    def parse(dict_annotations: dict | None) -> Annotations3D | None:
        if dict_annotations is None:
            return None
        keypoints = dict_annotations.pop("keypoints")
        keypoints = {int(k): v for k, v in keypoints.items()}
        return Annotations3D(keypoints=keypoints, **dict_annotations)


class AnnotationsCamera(TypedDict):
    intrinsics: torch.Tensor | None
    extrinsics: torch.Tensor | None
    distortion_coefficients: torch.Tensor | None
    camera_model: str | None

    @staticmethod
    def parse(dict_annotations: dict | None) -> AnnotationsCamera | None:
        if dict_annotations is None:
            return None
        intrinsics = dict_annotations.pop("intrinsics")
        extrinsics = dict_annotations.pop("extrinsics")
        distortion_coefficients = dict_annotations.pop("distortion_coefficients")
        camera_model = dict_annotations.pop("camera_model")
        return AnnotationsCamera(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            distortion_coefficients=distortion_coefficients,
            camera_model=camera_model,
            **dict_annotations,
        )


class KeypointsMetadata(TypedDict):
    n_keypoints: int
    format: str
    names: list[str]
    connectivity: list[tuple[int, int]]

    def __init__(
        self,
        n_keypoints: int,
        format: str,
        names: list[str],
        connectivity: list[tuple[int, int]],
    ):
        self.n_keypoints = n_keypoints
        self.format = format
        self.names = names
        self.connectivity = connectivity

        assert len(self.names) == self.n_keypoints, (
            "n_keypoints should be equal to the length of names"
        )

    @staticmethod
    def from_mmpose_dataset(dataset_name: str) -> KeypointsMetadata:
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

        return KeypointsMetadata(
            format=dataset_name,
            n_keypoints=n_keypoints,
            names=names,
            connectivity=connectivity,
        )

    @staticmethod
    def parse(dict_metadata: dict | None) -> KeypointsMetadata | None:
        if dict_metadata is None:
            return None
        n_keypoints = dict_metadata.pop("n_keypoints")
        format = dict_metadata.pop("format")
        names = dict_metadata.pop("names")
        connectivity = dict_metadata.pop("connectivity")
        return KeypointsMetadata(
            n_keypoints=n_keypoints,
            format=format,
            names=names,
            connectivity=connectivity,
            **dict_metadata,
        )


class KeypointsSequenceMetadata(TypedDict):
    keypoints: KeypointsMetadata

    @staticmethod
    def parse(dict_metadata: dict | None) -> KeypointsSequenceMetadata | None:
        if dict_metadata is None:
            return None
        keypoints_metadata = dict_metadata.pop("keypoints")
        keypoints_metadata = KeypointsMetadata.parse(keypoints_metadata)
        return KeypointsSequenceMetadata(keypoints=keypoints_metadata, **dict_metadata)


class KeypointsSequenceAnnotations(TypedDict):
    metadata: KeypointsSequenceMetadata | None
    annotations_2d: Annotations2D | None
    annotations_3d: Annotations3D | None
    annotations_cameras: dict[str, AnnotationsCamera] | None

    @staticmethod
    def load_from_file(filepath: str) -> KeypointsSequenceAnnotations:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Sequence annotations file {filepath} does not exist"
            )

        with open(filepath, "rb") as f:
            raw_annotations = orjson.loads(f.read())

        return KeypointsSequenceAnnotations.parse(raw_annotations)

    @staticmethod
    def parse(dict_annotations: dict | None) -> KeypointsSequenceAnnotations | None:
        if dict_annotations is None:
            return None

        metadata_dict = dict_annotations.pop("metadata")
        annotations_2d_dict = dict_annotations.pop("annotations_2d")
        annotations_3d_dict = dict_annotations.pop("annotations_3d")
        annotations_cameras_dict = dict_annotations.pop("annotations_cameras")

        annotations_2d = Annotations2D.parse(annotations_2d_dict)
        annotations_3d = Annotations3D.parse(annotations_3d_dict)

        annotations_cameras = {}
        for camera_name, camera_dict in annotations_cameras_dict.items():
            annotations_cameras[camera_name] = AnnotationsCamera.parse(camera_dict)

        metadata = KeypointsSequenceMetadata.parse(metadata_dict)
        annotations = KeypointsSequenceAnnotations(
            metadata=metadata,
            annotations_2d=annotations_2d,
            annotations_3d=annotations_3d,
            annotations_cameras=annotations_cameras,
        )

        return annotations


class ViewInput(TypedDict):
    view_id: str
    frame_loader: FrameSequenceLoader
    audio_loader: AudioLoader


class KeypointsSequence(TypedDict):
    sequence_name: str
    views_inputs: list[ViewInput]
    annotations: KeypointsSequenceAnnotations | None


class KeypointsSequenceDataset(ABC, Iterable[KeypointsSequence]):
    @abstractmethod
    def __getitem__(self, index: int) -> KeypointsSequence:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __iter__(self) -> Iterator[KeypointsSequence]:
        def iterator():
            for i in range(len(self)):
                yield self[i]

        return iterator()
