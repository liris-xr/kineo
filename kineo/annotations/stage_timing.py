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
class StageTiming:
    stage_name: str
    stage_idx: int
    duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "stage_name": self.stage_name,
            "stage_idx": self.stage_idx,
            "duration_seconds": self.duration_seconds,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> StageTiming:
        return StageTiming(
            stage_name=dict_data["stage_name"],
            stage_idx=dict_data["stage_idx"],
            duration_seconds=dict_data["duration_seconds"],
        )


class StageTimingsAnnotations(Annotations[StageTiming]):
    def __init__(self, annotations: list[StageTiming]):
        super().__init__(annotations)

    def to_dict(self) -> dict:
        return {
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> StageTimingsAnnotations:
        return StageTimingsAnnotations(
            annotations=[
                StageTiming.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def first_or_default(self, default: Any = None) -> StageTiming:
        return self._annotations[0] if self._annotations else default
