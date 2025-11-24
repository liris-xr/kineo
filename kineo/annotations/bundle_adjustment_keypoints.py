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
import torch
from kineo.annotations.annotations import Annotations


@dataclass(frozen=True)
class BundleAdjustmentKeypointsAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BundleAdjustmentKeypointsAnnotationsMetadata:
        return BundleAdjustmentKeypointsAnnotationsMetadata(
            version=dict_data["version"]
        )


@dataclass(frozen=True)
class BundleAdjustmentKeypointsAnnotation:
    view_ids: list[str]
    kps_2d_xy: torch.Tensor  # (n_views, n_kps, 2)
    kps_2d_scores: torch.Tensor  # (n_views, n_kps,)
    kps_3d: torch.Tensor  # (n_kps, 3)

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if not isinstance(self.view_ids, list):
            raise ValueError(
                f"Expected view_ids to be of type list, got {type(self.view_ids)}"
            )
        if not all(isinstance(view_id, str) for view_id in self.view_ids):
            raise ValueError(
                f"Expected view_ids to be a list of strings, got {self.view_ids}"
            )
        if self.kps_2d_xy.dtype != torch.float32:
            raise ValueError(
                f"Expected kps_2d_xy to be of type torch.float32, got {self.kps_2d_xy.dtype}"
            )
        if self.kps_2d_xy.ndim != 3:
            raise ValueError(
                f"Expected kps_2d_xy to be of shape (n_views, n_points, 2), got {self.kps_2d_xy.shape}"
            )
        if self.kps_2d_scores.dtype != torch.float32:
            raise ValueError(
                f"Expected kps_2d_scores to be of type torch.float32, got {self.kps_2d_scores.dtype}"
            )
        if self.kps_2d_scores.ndim != 2:
            raise ValueError(
                f"Expected kps_2d_scores to be of shape (n_views, n_points), got {self.kps_2d_scores.shape}"
            )
        if self.kps_3d.dtype != torch.float32:
            raise ValueError(
                f"Expected kps_3d to be of type torch.float32, got {self.kps_3d.dtype}"
            )
        if self.kps_3d.ndim != 2:
            raise ValueError(
                f"Expected kps_3d to be of shape (n_kps, 3), got {self.kps_3d.shape}"
            )

    def to_dict(self) -> dict:
        return {
            "view_ids": self.view_ids,
            "kps_2d_xy": self.kps_2d_xy.tolist(),
            "kps_2d_scores": self.kps_2d_scores.tolist(),
            "kps_3d": self.kps_3d.tolist(),
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BundleAdjustmentKeypointsAnnotation:
        return BundleAdjustmentKeypointsAnnotation(
            view_ids=dict_data["view_ids"],
            kps_2d_xy=torch.tensor(dict_data["kps_2d_xy"]),
            kps_2d_scores=torch.tensor(dict_data["kps_2d_scores"]),
            kps_3d=torch.tensor(dict_data["kps_3d"]),
        )

    def to(self, device: torch.device) -> BundleAdjustmentKeypointsAnnotation:
        return BundleAdjustmentKeypointsAnnotation(
            view_ids=self.view_ids,
            kps_2d_xy=self.kps_2d_xy.to(device),
            kps_2d_scores=self.kps_2d_scores.to(device),
            kps_3d=self.kps_3d.to(device),
        )

    def cpu(self) -> BundleAdjustmentKeypointsAnnotation:
        return self.to(torch.device("cpu"))


class BundleAdjustmentKeypointsAnnotations(
    Annotations[BundleAdjustmentKeypointsAnnotation]
):
    _metadata: BundleAdjustmentKeypointsAnnotationsMetadata

    def __init__(
        self,
        metadata: BundleAdjustmentKeypointsAnnotationsMetadata,
        annotations: list[BundleAdjustmentKeypointsAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BundleAdjustmentKeypointsAnnotations:
        return BundleAdjustmentKeypointsAnnotations(
            metadata=BundleAdjustmentKeypointsAnnotationsMetadata.from_dict(
                dict_data["metadata"]
            ),
            annotations=[
                BundleAdjustmentKeypointsAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def to(self, device: torch.device) -> BundleAdjustmentKeypointsAnnotations:
        return BundleAdjustmentKeypointsAnnotations(
            metadata=self._metadata,
            annotations=[annotation.to(device) for annotation in self._annotations],
        )

    def cpu(self) -> BundleAdjustmentKeypointsAnnotations:
        return self.to(torch.device("cpu"))

    def first_or_default(self) -> BundleAdjustmentKeypointsAnnotation:
        return self._annotations[0] if len(self._annotations) > 0 else None
