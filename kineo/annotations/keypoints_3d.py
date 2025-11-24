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
from kineo.annotations.annotations import Annotations
from kineo.annotations.keypoints_format import KeypointsFormat

from kineo.geometry.transformations import (
    apply_similarity_transform_to_points,
    compute_similarity_transform,
)


@dataclass(frozen=True)
class Keypoints3DAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes
    formats: list[KeypointsFormat] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "formats": [format.to_dict() for format in self.formats],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> Keypoints3DAnnotationsMetadata:
        return Keypoints3DAnnotationsMetadata(
            version=dict_data["version"],
            formats=[
                KeypointsFormat.from_dict(keypoints_format)
                for keypoints_format in dict_data["formats"]
            ],
        )


@dataclass(frozen=True)
class Keypoints3DAnnotation:
    frame_idx: int  # frame index in the global time reference
    subject_id: str
    xyz: torch.Tensor  # (n_keypoints, 3)
    annotated: torch.Tensor  # (n_keypoints,)
    scores: torch.Tensor  # (n_keypoints,)
    format: str  # format of the keypoints (e.g., "h36m", "coco")

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if self.xyz.dtype != torch.float32:
            raise ValueError(
                f"Expected xyz to be of type torch.float32, got {self.xyz.dtype}"
            )
        if self.scores.dtype != torch.float32:
            raise ValueError(
                f"Expected score to be of type torch.float32, got {self.scores.dtype}"
            )
        if self.annotated.dtype != torch.bool:
            raise ValueError(
                f"Expected annotated to be of type torch.bool, got {self.annotated.dtype}"
            )
        if self.xyz.ndim != 2:
            raise ValueError(
                f"Expected xyz to be of shape (n_keypoints, 3), got {self.xyz.shape}"
            )
        n_keypoints = self.xyz.shape[0]
        if self.annotated.shape != (n_keypoints,):
            raise ValueError(
                f"Expected annotated to be of shape ({n_keypoints},), got {self.annotated.shape}"
            )
        if not isinstance(self.format, str):
            raise ValueError(
                f"Expected format to be of type str, got {type(self.format)}"
            )

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "subject_id": self.subject_id,
            "xyz": self.xyz.tolist(),
            "annotated": self.annotated.tolist(),
            "scores": self.scores.tolist(),
            "format": self.format,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> Keypoints3DAnnotation:
        return Keypoints3DAnnotation(
            frame_idx=dict_data["frame_idx"],
            subject_id=dict_data["subject_id"],
            xyz=torch.tensor(dict_data["xyz"]),
            annotated=torch.tensor(dict_data["annotated"]),
            scores=torch.tensor(dict_data["scores"]),
            format=dict_data["format"],
        )

    def apply_similarity_transform(
        self, R: torch.Tensor, t: torch.Tensor, s: torch.Tensor
    ) -> Keypoints3DAnnotation:
        device = self.xyz.device
        R = R.to(device)
        t = t.to(device)
        s = s.to(device)
        new_xyz = apply_similarity_transform_to_points(self.xyz, R, t, s)

        return Keypoints3DAnnotation(
            frame_idx=self.frame_idx,
            subject_id=self.subject_id,
            xyz=new_xyz,
            annotated=self.annotated.clone(),
            scores=self.scores.clone(),
            format=self.format,
        )

    def cpu(self) -> Keypoints3DAnnotation:
        return self.to(torch.device("cpu"))

    def to(self, device: torch.device) -> Keypoints3DAnnotation:
        return Keypoints3DAnnotation(
            frame_idx=self.frame_idx,
            subject_id=self.subject_id,
            xyz=self.xyz.to(device),
            annotated=self.annotated.to(device),
            scores=self.scores.to(device),
            format=self.format,
        )


class Keypoints3DAnnotations(Annotations[Keypoints3DAnnotation]):
    _metadata: Keypoints3DAnnotationsMetadata

    def __init__(
        self,
        metadata: Keypoints3DAnnotationsMetadata,
        annotations: list[Keypoints3DAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata
        self._check_validity()

    @property
    def metadata(self) -> Keypoints3DAnnotationsMetadata:
        return self._metadata

    @property
    def frames(self) -> set[int]:
        return sorted(set([annotation.frame_idx for annotation in self._annotations]))

    @property
    def n_frames(self) -> int:
        frames = self.frames
        return frames[-1] + 1

    def _check_validity(self):
        for annotation in self._annotations:
            format_metadata = next(
                (f for f in self._metadata.formats if f.name == annotation.format),
                None,
            )

            if format_metadata is None:
                raise ValueError(
                    f"Keypoint format {annotation.format} not found in metadata"
                )

            n_keypoints = format_metadata.n_keypoints

            if annotation.xyz.shape != (n_keypoints, 3):
                raise ValueError(
                    f"Expected xyz to be of shape ({n_keypoints}, 3), got {annotation.xyz.shape}"
                )
            if annotation.annotated.shape != (n_keypoints,):
                raise ValueError(
                    f"Expected annotated to be of shape ({n_keypoints},), got {annotation.annotated.shape}"
                )

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> Keypoints3DAnnotations:
        return Keypoints3DAnnotations(
            metadata=Keypoints3DAnnotationsMetadata.from_dict(dict_data["metadata"]),
            annotations=[
                Keypoints3DAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    @property
    def subjects_ids(self) -> list[str]:
        # We sort the subjects ids to make the order deterministic
        return sorted(
            list(set([annotation.subject_id for annotation in self._annotations]))
        )

    def get_closest_by_frame_idx(self, frame_idx: int) -> Keypoints3DAnnotations:
        frames = torch.tensor(self.frames, dtype=torch.int32)
        closest_frame_idx = frames[torch.argmin(torch.abs(frames - frame_idx))]
        return self.filter_by_frame_idx(closest_frame_idx.item())

    def filter_by_subject_id(self, subject_id: str) -> Keypoints3DAnnotations:
        return Keypoints3DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.subject_id == subject_id
            ],
        )

    def filter_by_frame_idx(self, frame_idx: int) -> Keypoints3DAnnotations:
        return Keypoints3DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.frame_idx == frame_idx
            ],
        )

    def filter_by_frames_idxs(self, frame_idxs: list[int]) -> Keypoints3DAnnotations:
        return Keypoints3DAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.frame_idx in frame_idxs
            ],
        )

    def apply_frame_idx_shift(self, shift: int) -> Keypoints3DAnnotations:
        new_annotations = []
        for annotation in self._annotations:
            new_annotations.append(
                Keypoints3DAnnotation(
                    frame_idx=annotation.frame_idx + shift,
                    subject_id=annotation.subject_id,
                    xyz=annotation.xyz.clone(),
                    annotated=annotation.annotated.clone(),
                    scores=annotation.scores.clone(),
                    format=annotation.format,
                )
            )
        return Keypoints3DAnnotations(self._metadata, new_annotations)

    def first_or_default(self, default: Any = None) -> Keypoints3DAnnotation:
        return self._annotations[0] if self._annotations else default

    def compute_similarity_transform(
        self,
        other: Keypoints3DAnnotations,
        estimate_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_subject_ids = list(self.subjects_ids)
        other_subject_ids = list(other.subjects_ids)
        self_n_frames = self.n_frames
        other_n_frames = other.n_frames

        n_frames = min(self_n_frames, other_n_frames)

        n_keypoints = other.metadata.formats[0].n_keypoints

        if set(self_subject_ids) != set(other_subject_ids):
            raise ValueError(
                f"Subject IDs do not match: {self_subject_ids} != {other_subject_ids}"
            )

        subject_ids = list(self_subject_ids)

        self_xyz = torch.zeros((n_frames, len(subject_ids), n_keypoints, 3))
        other_xyz = torch.zeros((n_frames, len(subject_ids), n_keypoints, 3))

        for f in range(n_frames):
            for i, subject_id in enumerate(subject_ids):
                self_xyz_frame = (
                    self.filter_by_subject_id(subject_id)
                    .filter_by_frame_idx(f)
                    .first_or_default()
                )
                other_xyz_frame = (
                    other.filter_by_subject_id(subject_id)
                    .filter_by_frame_idx(f)
                    .first_or_default()
                )

                if self_xyz_frame is not None:
                    self_xyz[f, i] = self_xyz_frame.xyz

                if other_xyz_frame is not None:
                    other_xyz[f, i] = other_xyz_frame.xyz

        R, t, s = compute_similarity_transform(
            X=self_xyz.reshape(-1, 3),
            Y=other_xyz.reshape(-1, 3),
            estimate_scale=estimate_scale,
        )

        return R, t, s

    def apply_similarity_transform(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> Keypoints3DAnnotations:
        new_annotations = [
            a.apply_similarity_transform(R, t, s) for a in self._annotations
        ]

        return Keypoints3DAnnotations(self._metadata, new_annotations)

    def convert_to_format(
        self, target_format: KeypointsFormat
    ) -> Keypoints3DAnnotations:
        # Assuming all subjects have the same format
        current_format = self.metadata.formats[0]

        if current_format.name == target_format.name:
            return self

        if current_format.name == "coco_wholebody" and target_format.name == "coco":
            mapping = torch.arange(17)
            coco_annotations = []
            for annotation in self._annotations:
                coco_annotations.append(
                    Keypoints3DAnnotation(
                        frame_idx=annotation.frame_idx,
                        subject_id=annotation.subject_id,
                        xyz=annotation.xyz[mapping],
                        annotated=annotation.annotated[mapping],
                        scores=annotation.scores[mapping],
                        format="coco",
                    )
                )
            return Keypoints3DAnnotations(
                metadata=Keypoints3DAnnotationsMetadata(formats=[target_format]),
                annotations=coco_annotations,
            )
        elif current_format.name == "coco_19" and target_format.name == "coco":
            mapping = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]
            coco_annotations = []
            for annotation in self._annotations:
                coco_annotations.append(
                    Keypoints3DAnnotation(
                        frame_idx=annotation.frame_idx,
                        subject_id=annotation.subject_id,
                        xyz=annotation.xyz[mapping],
                        annotated=annotation.annotated[mapping],
                        scores=annotation.scores[mapping],
                        format="coco",
                    )
                )
            return Keypoints3DAnnotations(
                metadata=Keypoints3DAnnotationsMetadata(formats=[target_format]),
                annotations=coco_annotations,
            )
        else:
            raise NotImplementedError(
                f"Conversion from {current_format.name} to {target_format.name} is not supported"
            )

    def cpu(self) -> Keypoints3DAnnotations:
        return self.to(torch.device("cpu"))

    def to(self, device: torch.device) -> Keypoints3DAnnotations:
        return Keypoints3DAnnotations(
            metadata=self._metadata,
            annotations=[annotation.to(device) for annotation in self._annotations],
        )

    def resample(self, original_fps: int, dst_timestamps: torch.Tensor) -> Keypoints3DAnnotations:
        
        dt = 1 / original_fps

        sorted_annotations = sorted(self._annotations, key=lambda x: x.frame_idx)

        original_timestamps = torch.tensor([annotation.frame_idx * dt for annotation in sorted_annotations])
        first_frame_idx = sorted_annotations[0].frame_idx
        new_frame_idxs = torch.arange(first_frame_idx, first_frame_idx + len(dst_timestamps))
        
        new_annotations = []

        for subject_id in self.subjects_ids:
            subject_annotations = [annotation for annotation in sorted_annotations if annotation.subject_id == subject_id]

            kps_xyz = torch.stack([annotation.xyz for annotation in subject_annotations])
            kps_scores = torch.stack([annotation.scores for annotation in subject_annotations])

            resampled_kps_xyz, resampled_kps_scores = _resample_keypoints(
                src_keypoints=kps_xyz,
                src_scores=kps_scores,
                src_timestamps=original_timestamps,
                dst_timestamps=dst_timestamps,
            )

            for i in range(len(dst_timestamps)):
                new_annotations.append(
                    Keypoints3DAnnotation(
                        frame_idx=new_frame_idxs[i].item(),
                        subject_id=subject_id,
                        xyz=resampled_kps_xyz[i],
                        annotated=torch.ones_like(resampled_kps_xyz[i, ..., 0], dtype=torch.bool),
                        scores=resampled_kps_scores[i],
                        format=subject_annotations[0].format,
                    )
                )

        return Keypoints3DAnnotations(
            metadata=self._metadata,
            annotations=new_annotations,
        )


def _resample_keypoints(
    src_keypoints: torch.Tensor,
    src_scores: torch.Tensor,
    src_timestamps: torch.Tensor,
    dst_timestamps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resample keypoints to a target timestamp grid.
    """
    idx = torch.searchsorted(src_timestamps, dst_timestamps)
    idx = torch.clamp(idx, 0, src_timestamps.numel() - 1)

    closest_src_left_idx = torch.clamp(idx - 1, 0, src_timestamps.numel() - 1)
    closest_src_right_idx = torch.clamp(idx, 0, src_timestamps.numel() - 1)

    src_keypoints_left = src_keypoints[closest_src_left_idx, :]
    src_keypoints_right = src_keypoints[closest_src_right_idx, :]

    src_scores_left = src_scores[closest_src_left_idx, :]
    src_scores_right = src_scores[closest_src_right_idx, :]

    dt_left = torch.abs(dst_timestamps - src_timestamps[closest_src_left_idx])
    dt_right = torch.abs(dst_timestamps - src_timestamps[closest_src_right_idx])

    # We do a simple linear interpolation between the two closest timestamps
    w_l = torch.where(
        closest_src_left_idx == closest_src_right_idx,
        1.0,
        dt_right / (dt_left + dt_right),
    )
    w_r = torch.where(
        closest_src_left_idx == closest_src_right_idx,
        0.0,
        dt_left / (dt_left + dt_right),
    )

    resampled_kps_xy = (
        w_l.view(-1, 1, 1) * src_keypoints_left
        + w_r.view(-1, 1, 1) * src_keypoints_right
    )

    resampled_kps_scores = w_l.view(-1, 1) * src_scores_left + w_r.view(-1, 1) * src_scores_right
    return resampled_kps_xy, resampled_kps_scores