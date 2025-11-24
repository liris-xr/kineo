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


@dataclass(frozen=True)
class BackgroundImageAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BackgroundImageAnnotationsMetadata:
        return BackgroundImageAnnotationsMetadata(
            version=dict_data["version"],
        )


@dataclass(frozen=True)
class BackgroundImageAnnotation:
    view_id: str
    image: torch.Tensor  # (3, H, W)
    mask: torch.Tensor  # (H, W)
    sky_mask: torch.Tensor  # (H, W)

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if not isinstance(self.image, torch.Tensor):
            raise ValueError(
                f"Expected image to be of type torch.Tensor, got {type(self.image)}"
            )
        if self.image.dtype != torch.uint8:
            raise ValueError(
                f"Expected image to be of type torch.uint8, got {self.image.dtype}"
            )
        if not isinstance(self.view_id, str):
            raise ValueError(
                f"Expected view_id to be of type str, got {type(self.view_id)}"
            )
        if not isinstance(self.mask, torch.Tensor):
            raise ValueError(
                f"Expected mask to be of type torch.Tensor, got {type(self.mask)}"
            )
        if self.mask.dtype != torch.bool:
            raise ValueError(
                f"Expected mask to be of type torch.bool, got {self.mask.dtype}"
            )

    def to_dict(self) -> dict:
        return {
            "view_id": self.view_id,
            "image": self.image.tolist(),
            "mask": self.mask.tolist(),
            "sky_mask": self.sky_mask.tolist(),
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BackgroundImageAnnotation:
        return BackgroundImageAnnotation(
            view_id=dict_data["view_id"],
            image=torch.tensor(dict_data["image"], dtype=torch.uint8),
            mask=torch.tensor(dict_data["mask"], dtype=torch.bool),
            sky_mask=torch.tensor(dict_data["sky_mask"], dtype=torch.bool),
        )


class BackgroundImageAnnotations(Annotations[BackgroundImageAnnotation]):
    def __init__(
        self,
        annotations: list[BackgroundImageAnnotation],
        metadata: BackgroundImageAnnotationsMetadata = BackgroundImageAnnotationsMetadata(),
    ):
        super().__init__(annotations)
        self._metadata = metadata

    @property
    def metadata(self) -> BackgroundImageAnnotationsMetadata:
        return self._metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BackgroundImageAnnotations:
        return BackgroundImageAnnotations(
            metadata=BackgroundImageAnnotationsMetadata.from_dict(
                dict_data["metadata"]
            ),
            annotations=[
                BackgroundImageAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def filter_by_view_id(self, view_id: str) -> BackgroundImageAnnotations:
        return BackgroundImageAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.view_id == view_id
            ],
        )

    def first_or_default(self, default: Any = None) -> BackgroundImageAnnotation:
        return self._annotations[0] if self._annotations else default
