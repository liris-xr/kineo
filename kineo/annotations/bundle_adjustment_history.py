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
class BundleAdjustmentHistoryAnnotationsMetadata:
    version: str = "1"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BundleAdjustmentHistoryAnnotationsMetadata:
        return BundleAdjustmentHistoryAnnotationsMetadata(
            version=dict_data["version"]
        )


@dataclass(frozen=True)
class BundleAdjustmentHistoryAnnotation:
    stage_name: str
    stage_order: int
    iteration: int
    view_ids: list[str]
    Ks: torch.Tensor  # (n_views, 3, 3)
    dist_coeffs: torch.Tensor  # (n_views, n_coeffs)
    Rts: torch.Tensor  # (n_views, 3, 4)
    loss: float | None

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if not isinstance(self.stage_name, str):
            raise ValueError(
                f"Expected stage_name to be of type str, got {type(self.stage_name)}"
            )
        if not isinstance(self.stage_order, int):
            raise ValueError(
                f"Expected iteration to be of type int, got {type(self.stage_order)}"
            )
        if not isinstance(self.iteration, int):
            raise ValueError(
                f"Expected iteration to be of type int, got {type(self.iteration)}"
            )
        if not isinstance(self.view_ids, list):
            raise ValueError(
                f"Expected view_ids to be of type list, got {type(self.view_ids)}"
            )
        if not all(isinstance(view_id, str) for view_id in self.view_ids):
            raise ValueError(
                f"Expected view_ids to be a list of strings, got {self.view_ids}"
            )
        if self.Ks.dtype != torch.float32:
            raise ValueError(
                f"Expected Ks to be of type torch.float32, got {self.Ks.dtype}"
            )
        if self.Ks.ndim != 3 or self.Ks.shape[1:] != (3, 3):
            raise ValueError(
                f"Expected Ks to be of shape (n_views, 3, 3), got {self.Ks.shape}"
            )
        if self.dist_coeffs.dtype != torch.float32:
            raise ValueError(
                f"Expected dist_coeffs to be of type torch.float32, got {self.dist_coeffs.dtype}"
            )
        if self.dist_coeffs.ndim != 2:
            raise ValueError(
                f"Expected dist_coeffs to be of shape (n_views, n_coeffs), got {self.dist_coeffs.shape}"
            )
        if self.Rts.dtype != torch.float32:
            raise ValueError(
                f"Expected Rts to be of type torch.float32, got {self.Rts.dtype}"
            )
        if self.Rts.ndim != 3 or self.Rts.shape[1:] != (3, 4):
            raise ValueError(
                f"Expected Rts to be of shape (n_views, 3, 4), got {self.Rts.shape}"
            )
        if self.loss is not None and not isinstance(self.loss, float):
            raise ValueError(
                f"Expected loss to be of type float or None, got {type(self.loss)}"
            )

    def to_dict(self) -> dict:
        return {
            "stage_name": self.stage_name,
            "stage_order": self.stage_order,
            "iteration": self.iteration,
            "view_ids": self.view_ids,
            "Ks": self.Ks.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "Rts": self.Rts.tolist(),
            "loss": self.loss,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BundleAdjustmentHistoryAnnotation:
        return BundleAdjustmentHistoryAnnotation(
            stage_name=dict_data["stage_name"],
            stage_order=dict_data["stage_order"],
            iteration=dict_data["iteration"],
            view_ids=dict_data["view_ids"],
            Ks=torch.tensor(dict_data["Ks"]),
            dist_coeffs=torch.tensor(dict_data["dist_coeffs"]),
            Rts=torch.tensor(dict_data["Rts"]),
            loss=dict_data["loss"],
        )

    def to(self, device: torch.device) -> BundleAdjustmentHistoryAnnotation:
        return BundleAdjustmentHistoryAnnotation(
            stage_name=self.stage_name,
            stage_order=self.stage_order,
            iteration=self.iteration,
            view_ids=self.view_ids,
            Ks=self.Ks.to(device),
            dist_coeffs=self.dist_coeffs.to(device),
            Rts=self.Rts.to(device),
            loss=self.loss,
        )

    def cpu(self) -> BundleAdjustmentHistoryAnnotation:
        return self.to(torch.device("cpu"))


class BundleAdjustmentHistoryAnnotations(
    Annotations[BundleAdjustmentHistoryAnnotation]
):
    _metadata: BundleAdjustmentHistoryAnnotationsMetadata

    def __init__(
        self,
        metadata: BundleAdjustmentHistoryAnnotationsMetadata,
        annotations: list[BundleAdjustmentHistoryAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BundleAdjustmentHistoryAnnotations:
        return BundleAdjustmentHistoryAnnotations(
            metadata=BundleAdjustmentHistoryAnnotationsMetadata.from_dict(
                dict_data["metadata"]
            ),
            annotations=[
                BundleAdjustmentHistoryAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def to(self, device: torch.device) -> BundleAdjustmentHistoryAnnotations:
        return BundleAdjustmentHistoryAnnotations(
            metadata=self._metadata,
            annotations=[annotation.to(device) for annotation in self._annotations],
        )

    def cpu(self) -> BundleAdjustmentHistoryAnnotations:
        return self.to(torch.device("cpu"))

    def first_or_default(self) -> BundleAdjustmentHistoryAnnotation:
        return self._annotations[0] if len(self._annotations) > 0 else None
