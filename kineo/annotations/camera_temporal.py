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
from typing import Any, Literal
import warnings
from collections import defaultdict
from kineo.annotations.annotations import Annotations


@dataclass(frozen=True)
class CameraTemporalAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CameraTemporalAnnotationsMetadata:
        return CameraTemporalAnnotationsMetadata(**dict_data)


@dataclass(frozen=True)
class CameraTemporalAnnotation:
    view_id: str
    frame_idx: int  # frame index in the video/image sequence (local time reference)
    time_offset: float  # time offset in seconds

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if not isinstance(self.time_offset, float):
            raise ValueError(
                f"Expected time_offset to be of type float, got {type(self.time_offset)}"
            )

    def to_dict(self) -> dict:
        return {
            "view_id": self.view_id,
            "frame_idx": self.frame_idx,
            "time_offset": self.time_offset,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CameraTemporalAnnotation:
        return CameraTemporalAnnotation(**dict_data)


class CameraTemporalAnnotations(Annotations[CameraTemporalAnnotation]):
    _metadata: CameraTemporalAnnotationsMetadata

    def __init__(
        self,
        metadata: CameraTemporalAnnotationsMetadata,
        annotations: list[CameraTemporalAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata
        self._update_annotations_dict()

    def _update_annotations_dict(self):
        # Annotations indexed by view_id -> frame_idx
        self._annotations_by_view: dict[str, dict[int, CameraTemporalAnnotation]] = (
            defaultdict(lambda: defaultdict(CameraTemporalAnnotation))
        )

        for annotation in self._annotations:
            if annotation.frame_idx in self._annotations_by_view[annotation.view_id]:
                warnings.warn(
                    f"Annotation for frame {annotation.frame_idx} of view {annotation.view_id} already exists. Overwriting."
                )

            self._annotations_by_view[annotation.view_id][annotation.frame_idx] = (
                annotation
            )

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CameraTemporalAnnotations:
        return CameraTemporalAnnotations(
            metadata=CameraTemporalAnnotationsMetadata.from_dict(dict_data["metadata"]),
            annotations=[
                CameraTemporalAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def filter_by_view_id(self, view_ids: str | list[str]) -> CameraTemporalAnnotations:
        if isinstance(view_ids, str):
            view_ids = [view_ids]
        return CameraTemporalAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.view_id in view_ids
            ],
        )

    def filter_by_frame_idx(
        self, frame_idxs: int | list[int]
    ) -> CameraTemporalAnnotations:
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]
        return CameraTemporalAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.frame_idx in frame_idxs
            ],
        )

    def first_or_default(self, default: Any = None) -> CameraTemporalAnnotation:
        return self._annotations[0] if len(self._annotations) > 0 else default

    def get_annotation(
        self,
        view_id: str,
        frame_idx: int,
        default: Any = None,
        selection_mode: Literal["exact", "before", "after", "closest"] = "closest",
    ) -> CameraTemporalAnnotation:
        if view_id not in self._annotations_by_view:
            return default

        view_annotations = self._annotations_by_view[view_id]

        if selection_mode == "exact":
            if frame_idx not in view_annotations:
                return default
            return view_annotations[frame_idx]

        # Get all available frames for this view
        available_frames = sorted(view_annotations.keys())
        if not available_frames:
            return default

        if selection_mode == "before":
            # Find the closest frame before the requested one
            valid_frames = [f for f in available_frames if f <= frame_idx]
            if not valid_frames:
                return default
            return view_annotations[max(valid_frames)]

        elif selection_mode == "after":
            # Find the closest frame after the requested one
            valid_frames = [f for f in available_frames if f >= frame_idx]
            if not valid_frames:
                return default
            return view_annotations[min(valid_frames)]

        elif selection_mode == "closest":
            # Find the frame with minimum absolute difference
            closest_frame = min(available_frames, key=lambda x: abs(x - frame_idx))
            return view_annotations[closest_frame]

        raise ValueError(f"Invalid selection_mode: {selection_mode}")

    def has_static_time_offset(self, view_id: str) -> bool:
        # Returns True if the time offset is constant for all frames (no clock drift)
        if view_id not in self._annotations_by_view:
            return True

        view_annotations = self._annotations_by_view[view_id]
        if not view_annotations:
            return True

        if len(view_annotations) == 1:
            return True

        # Get the time offset of the first frame
        first_offset = next(iter(view_annotations.values())).time_offset

        # Check if any frame has a different time offset
        return not any(
            annotation.time_offset != first_offset
            for annotation in view_annotations.values()
        )
