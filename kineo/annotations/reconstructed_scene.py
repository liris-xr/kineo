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
from kineo.geometry.transformations import apply_similarity_transform_to_points


@dataclass(frozen=True)
class WorldReconstructedSceneAnnotationsMetadata:
    version: str = "1"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> WorldReconstructedSceneAnnotationsMetadata:
        return WorldReconstructedSceneAnnotationsMetadata(
            version=dict_data["version"],
        )


@dataclass(frozen=True)
class WorldReconstructedSceneAnnotation:
    points_xyz: torch.Tensor  # (N, 3)
    points_colors: torch.Tensor  # (N, 3)
    points_confidences: torch.Tensor  # (N,)

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if not isinstance(self.points_xyz, torch.Tensor):
            raise ValueError(
                f"Expected points_xyz to be of type torch.Tensor, got {type(self.points_xyz)}"
            )
        if not isinstance(self.points_colors, torch.Tensor):
            raise ValueError(
                f"Expected points_colors to be of type torch.Tensor, got {type(self.points_colors)}"
            )
        if self.points_xyz.ndim != 2 or self.points_xyz.shape[1] != 3:
            raise ValueError(
                f"Expected points_xyz to be of shape (N, 3), got {self.points_xyz.shape}"
            )
        if self.points_colors.ndim != 2 or self.points_colors.shape[1] != 3:
            raise ValueError(
                f"Expected points_colors to be of shape (N, 3), got {self.points_colors.shape}"
            )
        if self.points_confidences.ndim != 1:
            raise ValueError(
                f"Expected points_confidences to be of shape (N,), got {self.points_confidences.shape}"
            )
        if self.points_xyz.dtype != torch.float32:
            raise ValueError(
                f"Expected points_xyz to be of dtype torch.float32, got {self.points_xyz.dtype}"
            )
        if self.points_colors.dtype != torch.float32:
            raise ValueError(
                f"Expected points_colors to be of dtype torch.float32, got {self.points_colors.dtype}"
            )
        if self.points_confidences.dtype != torch.float32:
            raise ValueError(
                f"Expected points_confidences to be of dtype torch.float32, got {self.points_confidences.dtype}"
            )

    def to_dict(self) -> dict:
        return {
            "points_xyz": self.points_xyz.tolist(),
            "points_colors": self.points_colors.tolist(),
            "points_confidences": self.points_confidences.tolist(),
        }

    @staticmethod
    def from_dict(dict_data: dict) -> WorldReconstructedSceneAnnotation:
        return WorldReconstructedSceneAnnotation(
            points_xyz=torch.tensor(dict_data["points_xyz"], dtype=torch.float32),
            points_colors=torch.tensor(dict_data["points_colors"], dtype=torch.float32),
            points_confidences=torch.tensor(
                dict_data["points_confidences"], dtype=torch.float32
            ),
        )

    def to(self, device: torch.device) -> WorldReconstructedSceneAnnotation:
        return WorldReconstructedSceneAnnotation(
            points_xyz=self.points_xyz.to(device),
            points_colors=self.points_colors.to(device),
            points_confidences=self.points_confidences.to(device),
        )

    def cpu(self) -> WorldReconstructedSceneAnnotation:
        return self.to(torch.device("cpu"))

    def apply_similarity_transform(self, R: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> WorldReconstructedSceneAnnotation:
        device = self.points_xyz.device
        R = R.to(device)
        t = t.to(device)
        s = s.to(device)
        return WorldReconstructedSceneAnnotation(
            points_xyz=apply_similarity_transform_to_points(self.points_xyz, R, t, s),
            points_colors=self.points_colors,
            points_confidences=self.points_confidences,
        )

class WorldReconstructedSceneAnnotations(
    Annotations[WorldReconstructedSceneAnnotation]
):
    def __init__(
        self,
        annotations: list[WorldReconstructedSceneAnnotation],
        metadata: WorldReconstructedSceneAnnotationsMetadata = WorldReconstructedSceneAnnotationsMetadata(),
    ):
        super().__init__(annotations)
        self._metadata = metadata

    @property
    def metadata(self) -> WorldReconstructedSceneAnnotationsMetadata:
        return self._metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> WorldReconstructedSceneAnnotations:
        return WorldReconstructedSceneAnnotations(
            metadata=WorldReconstructedSceneAnnotationsMetadata.from_dict(
                dict_data["metadata"]
            ),
            annotations=[
                WorldReconstructedSceneAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def first_or_default(
        self, default: Any = None
    ) -> WorldReconstructedSceneAnnotation:
        return self._annotations[0] if self._annotations else default

    def apply_similarity_transform(self, R: torch.Tensor, t: torch.Tensor, s: torch.Tensor) -> WorldReconstructedSceneAnnotations:
        return WorldReconstructedSceneAnnotations(
            metadata=self._metadata,
            annotations=[annotation.apply_similarity_transform(R, t, s) for annotation in self._annotations],
        )

    def to(self, device: torch.device) -> WorldReconstructedSceneAnnotations:
        return WorldReconstructedSceneAnnotations(
            metadata=self._metadata,
            annotations=[annotation.to(device) for annotation in self._annotations],
        )

    def cpu(self) -> WorldReconstructedSceneAnnotations:
        return self.to(torch.device("cpu"))
