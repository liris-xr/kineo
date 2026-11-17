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
class FrameTiming:
    """Online loop times for one frame, total includes untimed work."""

    frame_idx: int
    read_seconds: float
    detection_seconds: float
    keypoints_seconds: float
    triangulation_seconds: float
    total_seconds: float

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "read_seconds": self.read_seconds,
            "detection_seconds": self.detection_seconds,
            "keypoints_seconds": self.keypoints_seconds,
            "triangulation_seconds": self.triangulation_seconds,
            "total_seconds": self.total_seconds,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> FrameTiming:
        return FrameTiming(
            frame_idx=dict_data["frame_idx"],
            read_seconds=dict_data["read_seconds"],
            detection_seconds=dict_data["detection_seconds"],
            keypoints_seconds=dict_data["keypoints_seconds"],
            triangulation_seconds=dict_data["triangulation_seconds"],
            total_seconds=dict_data["total_seconds"],
        )


class FrameTimingsAnnotations(Annotations[FrameTiming]):
    def __init__(self, annotations: list[FrameTiming]):
        super().__init__(annotations)

    def to_dict(self) -> dict:
        return {
            "annotations": [
                annotation.to_dict() for annotation in self._annotations
            ],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> FrameTimingsAnnotations:
        return FrameTimingsAnnotations(
            annotations=[
                FrameTiming.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def first_or_default(self, default: Any = None) -> FrameTiming:
        return self._annotations[0] if self._annotations else default
