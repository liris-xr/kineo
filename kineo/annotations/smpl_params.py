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
from enum import Enum
import torch
from kineo.annotations.annotations import Annotations

from kineo.geometry.transformations import apply_similarity_transform_to_points


class SMPLPoseFormat(Enum):
    SMPL = 0
    SMPLX = 1
    SMPLH = 2

    def num_body_joints(self) -> int:
        if self == SMPLPoseFormat.SMPL:
            return 23
        elif self == SMPLPoseFormat.SMPLX:
            return 21
        elif self == SMPLPoseFormat.SMPLH:
            return 21

    def num_hand_joints(self) -> int:
        if self == SMPLPoseFormat.SMPL:
            return 0
        elif self == SMPLPoseFormat.SMPLX:
            return 15
        elif self == SMPLPoseFormat.SMPLH:
            return 15

    def num_face_joints(self) -> int:
        if self == SMPLPoseFormat.SMPL:
            return 0
        elif self == SMPLPoseFormat.SMPLX:
            return 3
        elif self == SMPLPoseFormat.SMPLH:
            return 0

    def num_joints(self) -> int:
        return (
            self.num_body_joints() + self.num_hand_joints() * 2 + self.num_face_joints()
        )

    def split_joints(self, joints: torch.Tensor, dim: int = -2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        root_joint, body_joints, left_hand_joints, right_hand_joints, face_joints = torch.split(
            joints,
            [
                1, # root joint
                self.num_body_joints(),
                self.num_hand_joints(),
                self.num_hand_joints(),
                self.num_face_joints(),
            ],
            dim=dim,
        )
        return root_joint, body_joints, left_hand_joints, right_hand_joints, face_joints

    def split_poses(
        self, poses: torch.Tensor, dim: int = -2
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        poses_root, poses_body, poses_left_hand, poses_right_hand, poses_face = (
            torch.split(
                poses,
                [
                    1,
                    self.num_body_joints(),
                    self.num_hand_joints(),
                    self.num_hand_joints(),
                    self.num_face_joints(),
                ],
                dim=dim,
            )
        )
        return poses_root, poses_body, poses_left_hand, poses_right_hand, poses_face


class SMPLShapeFormat(Enum):
    SMPL_10 = 0
    SMPL_300 = 1
    SMPLX_10 = 2
    SMPLX_300 = 3
    SMPLH_10 = 4
    SMPLH_300 = 5

    def num_betas(self) -> int:
        if self == SMPLShapeFormat.SMPL_10:
            return 10
        elif self == SMPLShapeFormat.SMPL_300:
            return 300
        elif self == SMPLShapeFormat.SMPLX_10:
            return 10
        elif self == SMPLShapeFormat.SMPLX_300:
            return 300
        elif self == SMPLShapeFormat.SMPLH_10:
            return 10
        elif self == SMPLShapeFormat.SMPLH_300:
            return 300


@dataclass(frozen=True)
class SMPLPoseAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> SMPLPoseAnnotationsMetadata:
        return SMPLPoseAnnotationsMetadata(
            version=dict_data["version"],
        )


@dataclass(frozen=True)
class SMPLShapeAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> SMPLShapeAnnotationsMetadata:
        return SMPLShapeAnnotationsMetadata(
            version=dict_data["version"],
        )


@dataclass(frozen=True)
class SMPLPoseAnnotation:
    frame_idx: int
    subject_id: str
    pose: torch.Tensor  # (n_joints, 3)
    trans: torch.Tensor  # (3,)
    format: SMPLPoseFormat

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if self.pose.dtype != torch.float32:
            raise ValueError(
                f"Expected pose to be of type torch.float32, got {self.pose.dtype}"
            )
        if self.pose.ndim != 2:
            raise ValueError(
                f"Expected pose to be of shape (n_joints, 3), got {self.pose.shape}"
            )
        if self.trans.dtype != torch.float32:
            raise ValueError(
                f"Expected trans to be of type torch.float32, got {self.trans.dtype}"
            )
        if self.trans.ndim != 1:
            raise ValueError(
                f"Expected trans to be of shape (3,), got {self.trans.shape}"
            )
        if self.format not in SMPLPoseFormat:
            raise ValueError(
                f"Invalid format: {self.format}. Expected {SMPLPoseFormat.SMPL}, {SMPLPoseFormat.SMPLX}, or {SMPLPoseFormat.SMPLH}."
            )

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "subject_id": self.subject_id,
            "pose": self.pose.tolist(),
            "trans": self.trans.tolist(),
            "format": self.format.value,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> SMPLPoseAnnotation:
        return SMPLPoseAnnotation(
            frame_idx=dict_data["frame_idx"],
            subject_id=dict_data["subject_id"],
            pose=torch.tensor(dict_data["pose"]),
            trans=torch.tensor(dict_data["trans"]),
            format=SMPLPoseFormat(dict_data["format"]),
        )

    def apply_rigid_transform(
        self, R: torch.Tensor, t: torch.Tensor
    ) -> SMPLPoseAnnotation:
        new_trans = apply_similarity_transform_to_points(
            self.trans, R, t, torch.ones_like(self.trans[..., 0])
        )
        return SMPLPoseAnnotation(
            frame_idx=self.frame_idx,
            subject_id=self.subject_id,
            pose=self.pose.clone(),
            trans=new_trans,
            format=self.format,
        )

    def to(self, device: torch.device) -> SMPLPoseAnnotation:
        return SMPLPoseAnnotation(
            frame_idx=self.frame_idx,
            subject_id=self.subject_id,
            pose=self.pose.to(device),
            trans=self.trans.to(device),
            format=self.format,
        )
    
    def cpu(self) -> SMPLPoseAnnotation:
        return self.to(torch.device("cpu"))

@dataclass(frozen=True)
class SMPLShapeAnnotation:
    frame_idx: int
    subject_id: str
    betas: torch.Tensor  # (n_betas,)
    format: SMPLShapeFormat

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if self.betas.dtype != torch.float32:
            raise ValueError(
                f"Expected betas to be of type torch.float32, got {self.betas.dtype}"
            )
        if self.betas.ndim != 1:
            raise ValueError(
                f"Expected betas to be of shape (n_betas,), got {self.betas.shape}"
            )
        if self.format not in SMPLShapeFormat:
            raise ValueError(
                f"Invalid format: {self.format}. Expected {SMPLShapeFormat.SMPL_10}, {SMPLShapeFormat.SMPL_300}, {SMPLShapeFormat.SMPLX_10}, {SMPLShapeFormat.SMPLX_300}, {SMPLShapeFormat.SMPLH_10}, or {SMPLShapeFormat.SMPLH_300}."
            )

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "subject_id": self.subject_id,
            "betas": self.betas.tolist(),
            "format": self.format.value,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> SMPLShapeAnnotation:
        return SMPLShapeAnnotation(
            frame_idx=dict_data["frame_idx"],
            subject_id=dict_data["subject_id"],
            betas=torch.tensor(dict_data["betas"]),
            format=SMPLShapeFormat(dict_data["format"]),
        )

    def cpu(self) -> SMPLShapeAnnotation:
        return SMPLShapeAnnotation(
            frame_idx=self.frame_idx,
            subject_id=self.subject_id,
            betas=self.betas.cpu(),
            format=self.format,
        )

    def to(self, device: torch.device) -> SMPLShapeAnnotation:
        return SMPLShapeAnnotation(
            frame_idx=self.frame_idx,
            subject_id=self.subject_id,
            betas=self.betas.to(device),
            format=self.format,
        )


class SMPLPoseAnnotations(Annotations[SMPLPoseAnnotation]):
    _metadata: SMPLPoseAnnotationsMetadata

    def __init__(
        self,
        metadata: SMPLPoseAnnotationsMetadata,
        annotations: list[SMPLPoseAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> SMPLPoseAnnotations:
        return SMPLPoseAnnotations(
            metadata=SMPLPoseAnnotationsMetadata.from_dict(dict_data["metadata"]),
            annotations=[
                SMPLPoseAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def first_or_default(self, default: Any = None) -> SMPLPoseAnnotation:
        return self._annotations[0] if self._annotations else default

    @property
    def n_frames(self) -> int:
        return self.frames[-1] + 1

    @property
    def frames(self) -> list[int]:
        return sorted(list(set([annotation.frame_idx for annotation in self._annotations])))

    @property
    def subjects_ids(self) -> list[str]:
        return sorted(
            list(set([annotation.subject_id for annotation in self._annotations]))
        )

    def filter_by_subject_id(self, subject_id: str) -> SMPLPoseAnnotations:
        return SMPLPoseAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.subject_id == subject_id
            ],
        )

    def filter_by_frame_idx(self, frame_idx: int) -> SMPLPoseAnnotations:
        return SMPLPoseAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.frame_idx == frame_idx
            ],
        )

    def apply_rigid_transform(
        self, R: torch.Tensor, t: torch.Tensor
    ) -> SMPLPoseAnnotations:
        new_annotations = [
            annotation.apply_rigid_transform(R, t) for annotation in self._annotations
        ]
        return SMPLPoseAnnotations(
            metadata=self._metadata,
            annotations=new_annotations,
        )

    def cpu(self) -> SMPLPoseAnnotations:
        return SMPLPoseAnnotations(
            metadata=self._metadata,
            annotations=[annotation.cpu() for annotation in self._annotations],
        )

    def to(self, device: torch.device) -> SMPLPoseAnnotations:
        return SMPLPoseAnnotations(
            metadata=self._metadata,
            annotations=[annotation.to(device) for annotation in self._annotations],
        )


class SMPLShapeAnnotations(Annotations[SMPLShapeAnnotation]):
    _metadata: SMPLShapeAnnotationsMetadata

    def __init__(
        self,
        metadata: SMPLShapeAnnotationsMetadata,
        annotations: list[SMPLShapeAnnotation],
    ):
        super().__init__(annotations)
        self._metadata = metadata

    @property
    def metadata(self) -> SMPLShapeAnnotationsMetadata:
        return self._metadata

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> SMPLShapeAnnotations:
        return SMPLShapeAnnotations(
            metadata=SMPLShapeAnnotationsMetadata.from_dict(dict_data["metadata"]),
            annotations=[
                SMPLShapeAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def first_or_default(self, default: Any = None) -> SMPLShapeAnnotation:
        return self._annotations[0] if self._annotations else default

    @property
    def subjects_ids(self) -> list[str]:
        return sorted(
            list(set([annotation.subject_id for annotation in self._annotations]))
        )

    def filter_by_subject_id(self, subject_id: str) -> SMPLShapeAnnotations:
        return SMPLShapeAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.subject_id == subject_id
            ],
        )

    def filter_by_frame_idx(self, frame_idx: int) -> SMPLShapeAnnotations:
        return SMPLShapeAnnotations(
            metadata=self._metadata,
            annotations=[
                annotation
                for annotation in self._annotations
                if annotation.frame_idx == frame_idx
            ],
        )

    def cpu(self) -> SMPLShapeAnnotations:
        return SMPLShapeAnnotations(
            metadata=self._metadata,
            annotations=[annotation.cpu() for annotation in self._annotations],
        )

    def to(self, device: torch.device) -> SMPLShapeAnnotations:
        return SMPLShapeAnnotations(
            metadata=self._metadata,
            annotations=[annotation.to(device) for annotation in self._annotations],
        )
