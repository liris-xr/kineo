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
from dataclasses import dataclass, field
from typing import Any
import torch
from bisect import bisect_left, bisect_right
from kineo.maths import clamp
from kineo.annotations.annotations import Annotations
from kineo.annotations.keypoints_format import KeypointsFormat

# Pixel coordinates come from a float32 projection; a full float64 repr is 16
# chars of noise. A milli-pixel is below the accepted drift (<=0.00146 px).
XY_DECIMALS = 3


@dataclass(frozen=True)
class Keypoints2DAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes
    formats: list[KeypointsFormat] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "formats": [format.to_dict() for format in self.formats],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> Keypoints2DAnnotationsMetadata:
        return Keypoints2DAnnotationsMetadata(
            version=dict_data["version"],
            formats=[
                KeypointsFormat.from_dict(keypoints_format)
                for keypoints_format in dict_data["formats"]
            ],
        )


@dataclass(frozen=True)
class Keypoints2DAnnotation:
    view_id: str
    frame_idx: int  # frame index in the video/image sequence (local time reference)
    subject_id: str
    xy: torch.Tensor  # (n_keypoints, 2)
    scores: torch.Tensor  # (n_keypoints,)
    format: str  # format of the keypoints (e.g., "h36m", "coco")

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if self.xy.dtype != torch.float32:
            raise ValueError(
                f"Expected xy to be of type torch.float32, got {self.xy.dtype}"
            )
        if self.scores.dtype != torch.float32:
            raise ValueError(
                f"Expected scores to be of type torch.float32, got {self.scores.dtype}"
            )
        if self.xy.ndim != 2:
            raise ValueError(
                f"Expected xy to be of shape (n_keypoints, 2), got {self.xy.shape}"
            )
        n_keypoints = self.xy.shape[0]
        if self.scores.shape != (n_keypoints,):
            raise ValueError(
                f"Expected scores to be of shape ({n_keypoints},), got {self.scores.shape}"
            )
        if not isinstance(self.format, str):
            raise ValueError(
                f"Expected format to be of type str, got {type(self.format)}"
            )

    def to_dict(self) -> dict:
        return {
            "view_id": self.view_id,
            "frame_idx": self.frame_idx,
            "subject_id": self.subject_id,
            "xy": [
                [round(x, XY_DECIMALS), round(y, XY_DECIMALS)]
                for x, y in self.xy.tolist()
            ],
            "scores": self.scores.tolist(),
            "format": self.format,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> Keypoints2DAnnotation:
        return Keypoints2DAnnotation(
            view_id=dict_data["view_id"],
            frame_idx=dict_data["frame_idx"],
            subject_id=dict_data["subject_id"],
            xy=torch.tensor(dict_data["xy"], dtype=torch.float32),
            scores=torch.tensor(dict_data["scores"], dtype=torch.float32),
            format=dict_data["format"],
        )

    def cpu(self) -> Keypoints2DAnnotation:
        return Keypoints2DAnnotation(
            view_id=self.view_id,
            frame_idx=self.frame_idx,
            subject_id=self.subject_id,
            xy=self.xy.cpu(),
            scores=self.scores.cpu(),
            format=self.format,
        )


class Keypoints2DAnnotations(Annotations[Keypoints2DAnnotation]):
    _metadata: Keypoints2DAnnotationsMetadata

    def __init__(
        self,
        metadata: Keypoints2DAnnotationsMetadata,
        annotations: list[Keypoints2DAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata
        self._check_validity()

    @property
    def metadata(self) -> Keypoints2DAnnotationsMetadata:
        return self._metadata

    @property
    def view_ids(self) -> list[str]:
        return sorted(
            list(set([annotation.view_id for annotation in self._annotations]))
        )

    @property
    def frames(self) -> set[int]:
        return sorted(set([annotation.frame_idx for annotation in self._annotations]))

    def _check_validity(self):
        for annotation in self._annotations:
            keypoints_format_metadata = next(
                (
                    format_metadata
                    for format_metadata in self._metadata.formats
                    if format_metadata.name == annotation.format
                ),
                None,
            )

            if keypoints_format_metadata is None:
                raise ValueError(
                    f"Keypoint format {annotation.format} not found in metadata"
                )

            n_keypoints = keypoints_format_metadata.n_keypoints

            if annotation.xy.shape != (n_keypoints, 2):
                raise ValueError(
                    f"Expected xy to be of shape ({n_keypoints}, 2), got {annotation.xy.shape}"
                )
            if annotation.scores.shape != (n_keypoints,):
                raise ValueError(
                    f"Expected scores to be of shape ({n_keypoints},), got {annotation.scores.shape}"
                )

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> Keypoints2DAnnotations:
        return Keypoints2DAnnotations(
            metadata=Keypoints2DAnnotationsMetadata.from_dict(dict_data["metadata"]),
            annotations=[
                Keypoints2DAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def filter_by_view_id(self, view_ids: str | list[str]) -> Keypoints2DAnnotations:
        if isinstance(view_ids, str):
            view_ids = [view_ids]
        return Keypoints2DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.view_id in view_ids
            ],
        )

    def filter_by_view_ids(self, view_ids: list[str]) -> Keypoints2DAnnotations:
        return Keypoints2DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.view_id in view_ids
            ],
        )

    def filter_by_frame_idx(
        self, frame_idxs: int | list[int]
    ) -> Keypoints2DAnnotations:
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]
        return Keypoints2DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.frame_idx in frame_idxs
            ],
        )

    def filter_by_subject_id(
        self, subject_ids: str | list[str]
    ) -> Keypoints2DAnnotations:
        if isinstance(subject_ids, str):
            subject_ids = [subject_ids]
        return Keypoints2DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.subject_id in subject_ids
            ],
        )

    def first_or_default(self, default: Any = None) -> Keypoints2DAnnotation:
        return self._annotations[0] if len(self._annotations) > 0 else default

    @property
    def subjects_ids(self) -> list[str]:
        # We sort the subjects ids to make the order deterministic
        return sorted(
            list(set([annotation.subject_id for annotation in self._annotations]))
        )

    @property
    def views_ids(self) -> list[str]:
        # We sort the views ids to make the order deterministic
        return sorted(
            list(set([annotation.view_id for annotation in self._annotations]))
        )

    def cpu(self) -> Keypoints2DAnnotations:
        return Keypoints2DAnnotations(
            metadata=self._metadata,
            annotations=[annotation.cpu() for annotation in self._annotations],
        )

    def interpolate_by_frame_indices(
        self, target_frame_indices: list[int], max_frame_idx_diff: int = -1
    ) -> Keypoints2DAnnotations:
        """
        Interpolate the keypoints for the given frame indices.
        """
        interpolated_annotations = []

        for subject_id in self.subjects_ids:
            subject_annotations = self.filter_by_subject_id(subject_id)

            for view_id in subject_annotations.view_ids:
                view_annotations = subject_annotations.filter_by_view_id(view_id)

                # We need at least 2 annotations to interpolate
                if len(view_annotations) <= 1:
                    continue

                annotations: list[Keypoints2DAnnotation] = sorted(
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

                        interpolated_annotation = linear_interpolate_keypoints(
                            prev_annotation, next_annotation, frame_idx, lerp_weight
                        )
                        interpolated_annotations.append(interpolated_annotation)

        return Keypoints2DAnnotations(
            metadata=self._metadata,
            annotations=interpolated_annotations,
        )

    def convert_to_format(
        self, target_format: KeypointsFormat
    ) -> Keypoints2DAnnotations:
        # Assuming all subjects have the same format
        current_format = self.metadata.formats[0]

        if current_format.name == target_format.name:
            return self

        if current_format.name == "coco_wholebody" and target_format.name == "coco":
            mapping = torch.arange(17)
            coco_annotations = []
            for annotation in self._annotations:
                coco_annotations.append(
                    Keypoints2DAnnotation(
                        view_id=annotation.view_id,
                        frame_idx=annotation.frame_idx,
                        subject_id=annotation.subject_id,
                        xy=annotation.xy[mapping],
                        scores=annotation.scores[mapping],
                        format="coco",
                    )
                )
            return Keypoints2DAnnotations(
                metadata=Keypoints2DAnnotationsMetadata(formats=[target_format]),
                annotations=coco_annotations,
            )
        elif current_format.name == "coco_19" and target_format.name == "coco":
            mapping = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
            coco_annotations = []
            for annotation in self._annotations:
                coco_annotations.append(
                    Keypoints2DAnnotation(
                        view_id=annotation.view_id,
                        frame_idx=annotation.frame_idx,
                        subject_id=annotation.subject_id,
                        xy=annotation.xy[mapping],
                        scores=annotation.scores[mapping],
                        format="coco",
                    )
                )
            return Keypoints2DAnnotations(
                metadata=Keypoints2DAnnotationsMetadata(formats=[target_format]),
                annotations=coco_annotations,
            )
        else:
            raise NotImplementedError(
                f"Conversion from {current_format.name} to {target_format.name} is not supported"
            )


def linear_interpolate_keypoints(
    prev_annotation: Keypoints2DAnnotation,
    next_annotation: Keypoints2DAnnotation,
    frame_idx: int,
    lerp_weight: float,
) -> Keypoints2DAnnotation:
    """
    Interpolate a keypoints between two annotations.
    """
    assert prev_annotation.view_id == next_annotation.view_id, (
        f"Expected the same view_id for the previous and next annotations, got {prev_annotation.view_id} and {next_annotation.view_id}"
    )
    assert prev_annotation.subject_id == next_annotation.subject_id, (
        f"Expected the same subject_id for the previous and next annotations, got {prev_annotation.subject_id} and {next_annotation.subject_id}"
    )

    prev_annotation_xy = prev_annotation.xy
    next_annotation_xy = next_annotation.xy
    prev_annotation_scores = prev_annotation.scores
    next_annotation_scores = next_annotation.scores

    interpolated_xy = (
        1 - lerp_weight
    ) * prev_annotation_xy + lerp_weight * next_annotation_xy

    interpolated_scores = (
        1 - lerp_weight
    ) * prev_annotation_scores + lerp_weight * next_annotation_scores

    return Keypoints2DAnnotation(
        view_id=prev_annotation.view_id,
        frame_idx=frame_idx,
        subject_id=prev_annotation.subject_id,
        xy=interpolated_xy,
        scores=interpolated_scores,
        format=prev_annotation.format,
    )
