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
from bisect import bisect_left, bisect_right
from kineo.maths import clamp

from kineo.annotations.annotations import Annotations


@dataclass(frozen=True)
class BBox2DAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BBox2DAnnotationsMetadata:
        return BBox2DAnnotationsMetadata(
            version=dict_data["version"],
        )


@dataclass(frozen=True)
class BBox2DAnnotation:
    view_id: str
    frame_idx: int  # frame index in the video/image sequence (local time reference)
    subject_id: str
    category_id: int
    xyxy: torch.Tensor  # (4,)
    score: float

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if not isinstance(self.xyxy, torch.Tensor):
            raise ValueError(
                f"Expected xyxy to be of type torch.Tensor, got {type(self.xyxy)}"
            )
        if self.xyxy.dtype != torch.float32:
            raise ValueError(
                f"Expected xyxy to be of type torch.float32, got {self.xyxy.dtype}"
            )
        if self.xyxy.shape != (4,):
            raise ValueError(
                f"Expected xyxy to be of shape (4,), got {self.xyxy.shape}"
            )
        if not isinstance(self.score, float):
            raise ValueError(
                f"Expected score to be of type float, got {type(self.score)}"
            )
        if not isinstance(self.category_id, int):
            raise ValueError(
                f"Expected category_id to be of type int, got {type(self.category_id)}"
            )
        if not isinstance(self.subject_id, str):
            raise ValueError(
                f"Expected subject_id to be of type str, got {type(self.subject_id)}"
            )
        if not isinstance(self.view_id, str):
            raise ValueError(
                f"Expected view_id to be of type str, got {type(self.view_id)}"
            )
        if not isinstance(self.frame_idx, int):
            raise ValueError(
                f"Expected frame_idx to be of type int, got {type(self.frame_idx)}"
            )

    def to_dict(self) -> dict:
        return {
            "view_id": self.view_id,
            "frame_idx": self.frame_idx,
            "subject_id": self.subject_id,
            "category_id": self.category_id,
            "xyxy": self.xyxy.tolist(),
            "score": self.score,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> BBox2DAnnotation:
        return BBox2DAnnotation(
            view_id=dict_data["view_id"],
            frame_idx=dict_data["frame_idx"],
            subject_id=dict_data["subject_id"],
            category_id=dict_data["category_id"],
            xyxy=torch.tensor(dict_data["xyxy"]),
            score=dict_data["score"],
        )

    def cpu(self) -> BBox2DAnnotation:
        return BBox2DAnnotation(
            view_id=self.view_id,
            frame_idx=self.frame_idx,
            subject_id=self.subject_id,
            category_id=self.category_id,
            xyxy=self.xyxy.cpu(),
            score=self.score,
        )


class BBox2DAnnotations(Annotations[BBox2DAnnotation]):
    def __init__(
        self,
        metadata: BBox2DAnnotationsMetadata,
        annotations: list[BBox2DAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata

    @property
    def metadata(self) -> BBox2DAnnotationsMetadata:
        return self._metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @property
    def frames(self) -> list[int]:
        return sorted(
            list(set([annotation.frame_idx for annotation in self._annotations]))
        )

    @property
    def subjects_ids(self) -> list[str]:
        return sorted(
            list(set([annotation.subject_id for annotation in self._annotations]))
        )

    @property
    def view_ids(self) -> list[str]:
        return sorted(
            list(set([annotation.view_id for annotation in self._annotations]))
        )

    @staticmethod
    def from_dict(dict_data: dict) -> BBox2DAnnotations:
        return BBox2DAnnotations(
            metadata=BBox2DAnnotationsMetadata.from_dict(dict_data["metadata"]),
            annotations=[
                BBox2DAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def filter_by_view_id(self, view_id: str) -> BBox2DAnnotations:
        return BBox2DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.view_id == view_id
            ],
        )

    def filter_by_view_ids(self, view_ids: list[str]) -> BBox2DAnnotations:
        return BBox2DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.view_id in view_ids
            ],
        )

    def filter_by_frame_idx(self, frame_idx: int) -> BBox2DAnnotations:
        return BBox2DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.frame_idx == frame_idx
            ],
        )

    def filter_by_subject_id(self, subject_ids: str | list[str]) -> BBox2DAnnotations:
        if isinstance(subject_ids, str):
            subject_ids = [subject_ids]

        return BBox2DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.subject_id in subject_ids
            ],
        )

    def filter_by_subject_ids(self, subject_ids: list[str]) -> BBox2DAnnotations:
        return BBox2DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.subject_id in subject_ids
            ],
        )

    def filter_by_category_id(self, category_id: int) -> BBox2DAnnotations:
        return BBox2DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.category_id == category_id
            ],
        )

    def first_or_default(self, default: Any = None) -> BBox2DAnnotation:
        return self._annotations[0] if self._annotations else default

    def cpu(self) -> BBox2DAnnotations:
        return BBox2DAnnotations(
            metadata=self._metadata,
            annotations=[annotation.cpu() for annotation in self._annotations],
        )

    def interpolate_by_frame_indices(
        self, target_frame_indices: list[int], max_frame_idx_diff: int = -1
    ) -> BBox2DAnnotations:
        """
        Interpolate the bboxes for the given frame indices.
        """
        interpolated_annotations = []

        for subject_id in self.subjects_ids:
            subject_annotations = self.filter_by_subject_id(subject_id)

            for view_id in subject_annotations.view_ids:
                view_annotations = subject_annotations.filter_by_view_id(view_id)

                # We need at least 2 annotations to interpolate
                if len(view_annotations) <= 1:
                    continue

                annotations: list[BBox2DAnnotation] = sorted(
                    view_annotations._annotations, key=lambda x: x.frame_idx
                )
                frame_indices = list(a.frame_idx for a in annotations)
                frame_idx_to_list_idx = {
                    annotation.frame_idx: idx
                    for idx, annotation in enumerate(annotations)
                }

                for frame_idx in target_frame_indices:
                    if frame_idx in frame_idx_to_list_idx:
                        interpolated_annotations.append(
                            annotations[frame_idx_to_list_idx[frame_idx]]
                        )
                    else:
                        prev_annotation_idx = clamp(
                            bisect_left(frame_indices, frame_idx) - 1,
                            0,
                            len(annotations) - 1,
                        )
                        next_annotation_idx = clamp(
                            bisect_right(frame_indices, frame_idx),
                            0,
                            len(annotations) - 1,
                        )
                        prev_annotation = annotations[prev_annotation_idx]
                        next_annotation = annotations[next_annotation_idx]

                        if (
                            max_frame_idx_diff != -1
                            and frame_idx - prev_annotation.frame_idx
                            > max_frame_idx_diff
                        ):
                            continue

                        if (
                            max_frame_idx_diff != -1
                            and next_annotation.frame_idx - frame_idx
                            > max_frame_idx_diff
                        ):
                            continue

                        frame_idx_diff = (
                            next_annotation.frame_idx - prev_annotation.frame_idx
                        )

                        if frame_idx_diff == 0:
                            lerp_weight = 0.0
                        else:
                            lerp_weight = (frame_idx - prev_annotation.frame_idx) / (
                                frame_idx_diff
                            )

                        interpolated_annotation = linear_interpolate_bbox(
                            prev_annotation, next_annotation, frame_idx, lerp_weight
                        )
                        interpolated_annotations.append(interpolated_annotation)

        return BBox2DAnnotations(
            metadata=self._metadata,
            annotations=interpolated_annotations,
        )


def convert_bbox_xyxy_to_xywh(bbox_xyxy: torch.Tensor) -> torch.Tensor:
    """
    Convert a bbox from xyxy to xywh.
    """
    return torch.cat(
        [bbox_xyxy[..., :2], bbox_xyxy[..., 2:] - bbox_xyxy[..., :2]], dim=-1
    )


def convert_bbox_xywh_to_xyxy(bbox_xywh: torch.Tensor) -> torch.Tensor:
    """
    Convert a bbox from xywh to xyxy.
    """
    return torch.cat(
        [bbox_xywh[..., :2], bbox_xywh[..., :2] + bbox_xywh[..., 2:]], dim=-1
    )


def linear_interpolate_bbox(
    prev_annotation: BBox2DAnnotation,
    next_annotation: BBox2DAnnotation,
    frame_idx: int,
    lerp_weight: float,
) -> BBox2DAnnotation:
    """
    Interpolate a bbox between two annotations.
    """
    assert (
        prev_annotation.view_id == next_annotation.view_id
    ), f"Expected the same view_id for the previous and next annotations, got {prev_annotation.view_id} and {next_annotation.view_id}"
    assert (
        prev_annotation.subject_id == next_annotation.subject_id
    ), f"Expected the same subject_id for the previous and next annotations, got {prev_annotation.subject_id} and {next_annotation.subject_id}"
    assert (
        prev_annotation.category_id == next_annotation.category_id
    ), f"Expected the same category_id for the previous and next annotations, got {prev_annotation.category_id} and {next_annotation.category_id}"

    prev_annotation_score = prev_annotation.score
    next_annotation_score = next_annotation.score

    prev_annotation_xywh = convert_bbox_xyxy_to_xywh(prev_annotation.xyxy)
    next_annotation_xywh = convert_bbox_xyxy_to_xywh(next_annotation.xyxy)

    interpolated_xywh = (
        1 - lerp_weight
    ) * prev_annotation_xywh + lerp_weight * next_annotation_xywh

    interpolated_xyxy = convert_bbox_xywh_to_xyxy(interpolated_xywh)

    interpolated_score = (
        1 - lerp_weight
    ) * prev_annotation_score + lerp_weight * next_annotation_score

    return BBox2DAnnotation(
        view_id=prev_annotation.view_id,
        frame_idx=frame_idx,
        subject_id=prev_annotation.subject_id,
        category_id=prev_annotation.category_id,
        xyxy=interpolated_xyxy,
        score=interpolated_score,
    )
