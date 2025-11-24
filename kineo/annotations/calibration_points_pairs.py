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
import warnings
from collections import defaultdict
from kineo.annotations.annotations import Annotations


@dataclass(frozen=True)
class CalibrationPointsAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CalibrationPointsAnnotationsMetadata:
        return CalibrationPointsAnnotationsMetadata(version=dict_data["version"])


@dataclass(frozen=True)
class CalibrationPointsAnnotation:
    view1_id: str
    view2_id: str
    points1: torch.Tensor  # (n_points, 2)
    points2: torch.Tensor  # (n_points, 2)
    confidence_scores: torch.Tensor  # (n_points,)

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if self.points1.dtype != torch.float32:
            raise ValueError(
                f"Expected xy to be of type torch.float32, got {self.points1.dtype}"
            )
        if self.points1.ndim != 2:
            raise ValueError(
                f"Expected xy to be of shape (n_points, 2), got {self.points1.shape}"
            )

    def to_dict(self) -> dict:
        return {
            "view_i_id": self.view1_id,
            "view_j_id": self.view2_id,
            "points_i": self.points1.tolist(),
            "points_j": self.points2.tolist(),
            "confidence_scores": self.confidence_scores.tolist(),
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CalibrationPointsAnnotation:
        return CalibrationPointsAnnotation(
            view1_id=dict_data["view_i_id"],
            view2_id=dict_data["view_j_id"],
            points1=torch.tensor(dict_data["points_i"]),
            points2=torch.tensor(dict_data["points_j"]),
            confidence_scores=torch.tensor(dict_data["confidence_scores"]),
        )
    
    def to(self, device: torch.device) -> CalibrationPointsAnnotation:
        return CalibrationPointsAnnotation(
            view1_id=self.view1_id,
            view2_id=self.view2_id,
            points1=self.points1.to(device),
            points2=self.points2.to(device),
            confidence_scores=self.confidence_scores.to(device),
        )

    def cpu(self) -> CalibrationPointsAnnotation:
        return self.to(torch.device("cpu"))


class CalibrationPointsAnnotations(Annotations[CalibrationPointsAnnotation]):
    _metadata: CalibrationPointsAnnotationsMetadata

    def __init__(
        self,
        metadata: CalibrationPointsAnnotationsMetadata,
        annotations: list[CalibrationPointsAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata

        # Annotations indexed by (view_i_id, view_j_id)
        self._annotations_dict: dict[tuple[str, str], CalibrationPointsAnnotation] = (
            defaultdict(
                lambda: defaultdict(lambda: defaultdict(CalibrationPointsAnnotation))
            )
        )

        for annotation in self._annotations:
            if (annotation.view1_id, annotation.view2_id) in self._annotations_dict:
                warnings.warn(
                    f"Annotation for view pair ({annotation.view1_id}, {annotation.view2_id}) already exists. Overwriting."
                )

            self._annotations_dict[(annotation.view1_id, annotation.view2_id)] = (
                annotation
            )

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CalibrationPointsAnnotations:
        return CalibrationPointsAnnotations(
            metadata=CalibrationPointsAnnotationsMetadata.from_dict(
                dict_data["metadata"]
            ),
            annotations=[
                CalibrationPointsAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def get_annotation(
        self, view_pair: tuple[str, str], default: Any = None
    ) -> CalibrationPointsAnnotation:
        if view_pair not in self._annotations_dict:
            return default

        return self._annotations_dict[view_pair]

    def to(self, device: torch.device) -> CalibrationPointsAnnotations:
        return CalibrationPointsAnnotations(
            metadata=self._metadata,
            annotations=[annotation.to(device) for annotation in self._annotations],
        )

    def cpu(self) -> CalibrationPointsAnnotations:
        return self.to(torch.device("cpu"))