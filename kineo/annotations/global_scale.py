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
from kineo.annotations.annotations import Annotations


@dataclass(frozen=True)
class GlobalScaleAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> GlobalScaleAnnotationsMetadata:
        return GlobalScaleAnnotationsMetadata(
            version=dict_data["version"],
        )


@dataclass(frozen=True)
class GlobalScaleAnnotation:
    frame_idx: int
    scale: float  # global scale factor (scalar)

    def __post_init__(self):
        assert isinstance(self.scale, float)

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "scale": self.scale,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> GlobalScaleAnnotation:
        return GlobalScaleAnnotation(
            frame_idx=dict_data["frame_idx"],
            scale=dict_data["scale"],
        )


class GlobalScaleAnnotations(Annotations[GlobalScaleAnnotation]):
    def __init__(
        self,
        metadata: GlobalScaleAnnotationsMetadata,
        annotations: list[GlobalScaleAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata

    @property
    def metadata(self) -> GlobalScaleAnnotationsMetadata:
        return self._metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> GlobalScaleAnnotations:
        return GlobalScaleAnnotations(
            metadata=GlobalScaleAnnotationsMetadata.from_dict(dict_data["metadata"]),
            annotations=[
                GlobalScaleAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def first_or_default(self, default: Any = None) -> GlobalScaleAnnotation:
        return self._annotations[0] if self._annotations else default
