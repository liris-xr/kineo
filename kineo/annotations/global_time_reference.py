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
class GlobalTimeReferenceAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> GlobalTimeReferenceAnnotationsMetadata:
        return GlobalTimeReferenceAnnotationsMetadata(
            version=dict_data["version"],
        )


@dataclass(frozen=True)
class GlobalTimeReferenceAnnotation:
    timestamps: torch.Tensor  # timestamps in seconds
    closest_local_frame_idx: dict[
        str, torch.Tensor
    ]  # For a given view_id, provide the closest local frame index for each global frame index

    @property
    def fps(self) -> float:
        return self.timestamps.shape[0] / (self.timestamps[-1] - self.timestamps[0])

    def to_dict(self) -> dict:
        return {
            "timestamps": self.timestamps.tolist(),
            "closest_local_frame_idx": {
                k: v.tolist() for k, v in self.closest_local_frame_idx.items()
            },
        }

    @staticmethod
    def from_dict(dict_data: dict) -> GlobalTimeReferenceAnnotation:
        return GlobalTimeReferenceAnnotation(
            timestamps=torch.tensor(dict_data["timestamps"]),
            closest_local_frame_idx=dict(
                (k, torch.tensor(v))
                for k, v in dict_data["closest_local_frame_idx"].items()
            ),
        )

    def cpu(self) -> GlobalTimeReferenceAnnotation:
        return GlobalTimeReferenceAnnotation(
            timestamps=self.timestamps.cpu(),
            closest_local_frame_idx=self.closest_local_frame_idx,
        )


class GlobalTimeReferenceAnnotations(Annotations[GlobalTimeReferenceAnnotation]):
    def __init__(
        self,
        metadata: GlobalTimeReferenceAnnotationsMetadata,
        annotations: list[GlobalTimeReferenceAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata

    @property
    def metadata(self) -> GlobalTimeReferenceAnnotationsMetadata:
        return self._metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> GlobalTimeReferenceAnnotations:
        return GlobalTimeReferenceAnnotations(
            metadata=GlobalTimeReferenceAnnotationsMetadata.from_dict(
                dict_data["metadata"]
            ),
            annotations=[
                GlobalTimeReferenceAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def first_or_default(self, default: Any = None) -> GlobalTimeReferenceAnnotation:
        return self._annotations[0] if self._annotations else default

    def cpu(self) -> GlobalTimeReferenceAnnotations:
        return GlobalTimeReferenceAnnotations(
            metadata=self._metadata,
            annotations=[annotation.cpu() for annotation in self._annotations],
        )
