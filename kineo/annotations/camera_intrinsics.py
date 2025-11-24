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
from typing import Any
import torch
from kineo.annotations.annotations import Annotations
from enum import Enum


class CameraDistortionModel(Enum):
    BROWN_CONRADY = "brown_conrady"
    OPENCV_FISHEYE = "opencv_fisheye"


@dataclass(frozen=True)
class CameraIntrinsicsAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CameraIntrinsicsAnnotationsMetadata:
        return CameraIntrinsicsAnnotationsMetadata(**dict_data)


@dataclass(frozen=True)
class CameraIntrinsicsAnnotation:
    view_id: str
    frame_idx: int  # frame index in the video/image sequence (local time reference)
    K: torch.Tensor  # (3, 3)
    distortion_coefficients: torch.Tensor  # (*,)
    distortion_model: CameraDistortionModel
    resolution_hw: tuple[int, int]

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if self.K.dtype != torch.float32:
            raise ValueError(
                f"Expected K to be of type torch.float32, got {self.K.dtype}"
            )
        if self.K.shape != (3, 3):
            raise ValueError(f"Expected K to be of shape (3, 3), got {self.K.shape}")
        if self.distortion_coefficients.dtype != torch.float32:
            raise ValueError(
                f"Expected distortion_coefficients to be of type torch.float32, got {self.distortion_coefficients.dtype}"
            )

        if self.distortion_model == CameraDistortionModel.BROWN_CONRADY:
            if self.distortion_coefficients.shape != (5,):
                raise ValueError(
                    f"Expected distortion_coefficients to be of shape (5,) for brown_conrady (k1, k2, p1, p2, k3), got {self.distortion_coefficients.shape}"
                )
        elif self.distortion_model == CameraDistortionModel.OPENCV_FISHEYE:
            if self.distortion_coefficients.shape != (4,):
                raise ValueError(
                    f"Expected distortion_coefficients to be of shape (4,) for opencv_fisheye (k1, k2, k3, k4), got {self.distortion_coefficients.shape}"
                )
        else:
            raise NotImplementedError(
                f"Distortion model {self.distortion_model} not implemented"
            )

        if (
            not isinstance(self.resolution_hw, (tuple, list))
            or len(self.resolution_hw) != 2
        ):
            raise ValueError(
                f"Expected resolution_hw to be a tuple of length 2, got {self.resolution_hw}"
            )

    def to_dict(self) -> dict:
        return {
            "view_id": self.view_id,
            "frame_idx": self.frame_idx,
            "K": self.K.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "distortion_model": self.distortion_model,
            "resolution_hw": self.resolution_hw,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CameraIntrinsicsAnnotation:
        dist_model = dict_data["distortion_model"]
        return CameraIntrinsicsAnnotation(
            view_id=dict_data["view_id"],
            frame_idx=dict_data["frame_idx"],
            K=torch.tensor(dict_data["K"]),
            distortion_coefficients=torch.tensor(dict_data["distortion_coefficients"]),
            distortion_model=CameraDistortionModel(dist_model if isinstance(dist_model, str) else dist_model.value),
            resolution_hw=dict_data["resolution_hw"],
        )

    def cpu(self) -> CameraIntrinsicsAnnotation:
        return CameraIntrinsicsAnnotation(
            view_id=self.view_id,
            frame_idx=self.frame_idx,
            K=self.K.cpu(),
            distortion_coefficients=self.distortion_coefficients.cpu(),
            distortion_model=self.distortion_model,
            resolution_hw=self.resolution_hw,
        )


class CameraIntrinsicsAnnotations(Annotations[CameraIntrinsicsAnnotation]):
    _metadata: CameraIntrinsicsAnnotationsMetadata

    def __init__(
        self,
        metadata: CameraIntrinsicsAnnotationsMetadata,
        annotations: list[CameraIntrinsicsAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CameraIntrinsicsAnnotations:
        return CameraIntrinsicsAnnotations(
            metadata=CameraIntrinsicsAnnotationsMetadata.from_dict(
                dict_data["metadata"]
            ),
            annotations=[
                CameraIntrinsicsAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    @property
    def metadata(self) -> CameraIntrinsicsAnnotationsMetadata:
        return self._metadata

    def get_closest_by_frame_idx(self, frame_idx: int) -> CameraIntrinsicsAnnotations:
        # Find the closest annotation to the given frame_idx
        frames = torch.tensor(self.frames, dtype=torch.int32)
        closest_frame_idx = frames[torch.argmin(torch.abs(frames - frame_idx))]
        return self.filter_by_frame_idx(closest_frame_idx.item())

    def filter_by_view_id(self, view_id: str) -> CameraIntrinsicsAnnotations:
        return CameraIntrinsicsAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.view_id == view_id
            ],
        )

    def filter_by_view_ids(self, view_ids: list[str]) -> CameraIntrinsicsAnnotations:
        return CameraIntrinsicsAnnotations(
            metadata=self._metadata,
            annotations=[a for a in self._annotations if a.view_id in view_ids],
        )

    def filter_by_frame_idx(self, frame_idx: int) -> CameraIntrinsicsAnnotations:
        return CameraIntrinsicsAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.frame_idx == frame_idx
            ],
        )

    def first_or_default(self, default: Any = None) -> CameraIntrinsicsAnnotation:
        if not self._annotations:
            return default
        return self._annotations[0]

    @property
    def views_ids(self) -> list[str]:
        # We sort the views ids to make the order deterministic
        return sorted(list(set([a.view_id for a in self._annotations])))

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def frames(self) -> list[int]:
        return sorted(list(set([a.frame_idx for a in self._annotations])))

    def cpu(self) -> CameraIntrinsicsAnnotations:
        return CameraIntrinsicsAnnotations(
            metadata=self._metadata,
            annotations=[annotation.cpu() for annotation in self._annotations],
        )
