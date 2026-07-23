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
from kineo.geometry.transformations import (
    inverse_Rt,
    compute_similarity_transform,
    apply_similarity_transform_to_Rt,
)


@dataclass(frozen=True)
class CameraExtrinsicsAnnotationsMetadata:
    version: str = "1"  # version of the annotation format to be bumped when it changes

    def to_dict(self) -> dict:
        return {
            "version": self.version,
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CameraExtrinsicsAnnotationsMetadata:
        return CameraExtrinsicsAnnotationsMetadata(**dict_data)


@dataclass(frozen=True)
class CameraExtrinsicsAnnotation:
    view_id: str
    frame_idx: int  # frame index in the video/image sequence (local time reference)
    R: torch.Tensor  # (3, 3)
    t: torch.Tensor  # (3,)

    def __post_init__(self):
        self.check_validity()

    def check_validity(self):
        if self.R.dtype != torch.float32:
            raise ValueError(
                f"Expected R to be of type torch.float32, got {self.R.dtype}"
            )
        if self.R.shape != (3, 3):
            raise ValueError(f"Expected R to be of shape (3, 3), got {self.R.shape}")
        if self.t.dtype != torch.float32:
            raise ValueError(
                f"Expected t to be of type torch.float32, got {self.t.dtype}"
            )
        if self.t.shape != (3,):
            raise ValueError(f"Expected t to be of shape (3,), got {self.t.shape}")

    @property
    def Rt(self) -> torch.Tensor:
        return torch.cat([self.R, self.t.unsqueeze(1)], dim=1)

    def to_dict(self) -> dict:
        return {
            "view_id": self.view_id,
            "frame_idx": self.frame_idx,
            "R": self.R.tolist(),
            "t": self.t.tolist(),
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CameraExtrinsicsAnnotation:
        return CameraExtrinsicsAnnotation(
            view_id=dict_data["view_id"],
            frame_idx=dict_data["frame_idx"],
            R=torch.tensor(dict_data["R"]),
            t=torch.tensor(dict_data["t"]),
        )

    def apply_similarity_transform(
        self, R: torch.Tensor, t: torch.Tensor, s: torch.Tensor
    ) -> CameraExtrinsicsAnnotation:
        device = self.Rt.device
        R = R.to(device)
        t = t.to(device)
        s = s.to(device)
        new_Rt = apply_similarity_transform_to_Rt(self.Rt, R, t, s)

        return CameraExtrinsicsAnnotation(
            view_id=self.view_id,
            frame_idx=self.frame_idx,
            R=new_Rt[:3, :3],
            t=new_Rt[:3, 3],
        )

    def to(self, device: torch.device) -> CameraExtrinsicsAnnotation:
        return CameraExtrinsicsAnnotation(
            view_id=self.view_id,
            frame_idx=self.frame_idx,
            R=self.R.to(device),
            t=self.t.to(device),
        )

    def cpu(self) -> CameraExtrinsicsAnnotation:
        return self.to(torch.device("cpu"))


class CameraExtrinsicsAnnotations(Annotations[CameraExtrinsicsAnnotation]):
    _metadata: CameraExtrinsicsAnnotationsMetadata

    def __init__(
        self,
        metadata: CameraExtrinsicsAnnotationsMetadata,
        annotations: list[CameraExtrinsicsAnnotation],
    ):
        super(CameraExtrinsicsAnnotations, self).__init__(annotations)
        self._metadata = metadata

    @property
    def metadata(self) -> CameraExtrinsicsAnnotationsMetadata:
        return self._metadata

    @property
    def views_ids(self) -> list[str]:
        # We sort the views ids to make the order deterministic
        return sorted(list(set([a.view_id for a in self._annotations])))

    def to_dict(self) -> dict:
        return {
            "metadata": self._metadata.to_dict(),
            "annotations": [annotation.to_dict() for annotation in self._annotations],
        }

    @staticmethod
    def from_dict(dict_data: dict) -> CameraExtrinsicsAnnotations:
        return CameraExtrinsicsAnnotations(
            metadata=CameraExtrinsicsAnnotationsMetadata.from_dict(
                dict_data["metadata"]
            ),
            annotations=[
                CameraExtrinsicsAnnotation.from_dict(annotation)
                for annotation in dict_data["annotations"]
            ],
        )

    def get_closest_by_frame_idx(self, frame_idx: int) -> CameraExtrinsicsAnnotations:
        # Find the closest annotation to the given frame_idx
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        frames = torch.tensor(self.frames, dtype=torch.int32, device=device)
        closest_frame_idx = frames[torch.argmin(torch.abs(frames - frame_idx))]
        return self.filter_by_frame_idx(closest_frame_idx.item())

    def filter_active_by_frame_idx(
        self, frame_idx: int
    ) -> CameraExtrinsicsAnnotations:
        # Per view: annotation whose onset frame_idx is the most recent one at or
        # before frame_idx (earliest as fallback). Yields one pose per view, i.e.
        # the camera pose active at that frame for non-static (moving) cameras.
        active = []
        for view_id in self.views_ids:
            anns = [a for a in self._annotations if a.view_id == view_id]
            eligible = [a for a in anns if a.frame_idx <= frame_idx]
            active.append(
                max(eligible, key=lambda a: a.frame_idx)
                if eligible
                else min(anns, key=lambda a: a.frame_idx)
            )
        return CameraExtrinsicsAnnotations(self._metadata, active)

    def filter_by_view_id(
        self, view_ids: str | list[str]
    ) -> CameraExtrinsicsAnnotations:
        if isinstance(view_ids, str):
            view_ids = [view_ids]
        return CameraExtrinsicsAnnotations(
            metadata=self._metadata,
            annotations=[a for a in self._annotations if a.view_id in view_ids],
        )
    
    def filter_by_view_ids(self, view_ids: list[str]) -> CameraExtrinsicsAnnotations:
        return CameraExtrinsicsAnnotations(
            metadata=self._metadata,
            annotations=[a for a in self._annotations if a.view_id in view_ids],
        )

    def filter_by_frame_idx(self, frame_idx: int) -> CameraExtrinsicsAnnotations:
        return CameraExtrinsicsAnnotations(
            metadata=self._metadata,
            annotations=[a for a in self._annotations if a.frame_idx == frame_idx],
        )

    def first_or_default(self, default: Any = None) -> CameraExtrinsicsAnnotation:
        if not self._annotations:
            return default
        return self._annotations[0]

    def is_view_static(self, view_id: str) -> bool:
        # View extrinsics are static if there is only one annotation for this view.
        return len([a for a in self._annotations if a.view_id == view_id]) == 1

    def compute_similarity_transform(
        self,
        other: CameraExtrinsicsAnnotations,
        estimate_scale: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_cam_ids = list(self.views_ids)
        other_cam_ids = list(other.views_ids)

        if set(self_cam_ids) != set(other_cam_ids):
            raise ValueError(
                f"Camera IDs do not match: {self_cam_ids} != {other_cam_ids}"
            )

        cam_ids = list(self_cam_ids)

        device = self._annotations[0].R.device
        self_world2cam = torch.empty((len(cam_ids), 3, 4), device=device)
        other_world2cam = torch.empty((len(cam_ids), 3, 4), device=device)

        for i, cam_id in enumerate(cam_ids):
            # Assuming static cameras
            self_cam = self.filter_by_view_id(cam_id).first_or_default()
            other_cam = other.filter_by_view_id(cam_id).first_or_default()
            assert self_cam is not None and other_cam is not None
            self_world2cam[i] = self_cam.Rt
            other_world2cam[i] = other_cam.Rt

        self_cam2world = inverse_Rt(self_world2cam)
        other_cam2world = inverse_Rt(other_world2cam)

        self_cam_pos = self_cam2world[:, :3, 3]
        other_cam_pos = other_cam2world[:, :3, 3]

        R, t, s = compute_similarity_transform(
            X=self_cam_pos,
            Y=other_cam_pos,
            estimate_scale=estimate_scale,
        )

        return R, t, s

    def apply_similarity_transform(
        self,
        R: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> CameraExtrinsicsAnnotations:
        new_annotations = [
            a.apply_similarity_transform(R, t, s) for a in self._annotations
        ]
        return CameraExtrinsicsAnnotations(self._metadata, new_annotations)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def frames(self) -> list[int]:
        return sorted(list(set([a.frame_idx for a in self._annotations])))

    def cpu(self) -> CameraExtrinsicsAnnotations:
        return self.to(torch.device("cpu"))

    def to(self, device: torch.device) -> CameraExtrinsicsAnnotations:
        return CameraExtrinsicsAnnotations(
            metadata=self._metadata,
            annotations=[annotation.to(device) for annotation in self._annotations],
        )
