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

import dataclasses
import os
from pathlib import Path
from typing import Any, Callable

import orjson
import torch
from tqdm import tqdm

from kineo.annotations.bboxes_2d import BBox2DAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.camera_temporal import (
    CameraTemporalAnnotation,
    CameraTemporalAnnotations,
    CameraTemporalAnnotationsMetadata,
)
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations


@dataclasses.dataclass(frozen=True)
class AnnotationKind:
    filename: str
    annotations_type: type
    build_default: Callable[[list[str]], Any] | None = None


def build_synchronized_camera_temporal(
    view_ids: list[str],
) -> CameraTemporalAnnotations:
    return CameraTemporalAnnotations(
        metadata=CameraTemporalAnnotationsMetadata(),
        annotations=[
            CameraTemporalAnnotation(
                view_id=view_id, frame_idx=0, time_offset=0.0
            )
            for view_id in view_ids
        ],
    )


ANNOTATION_KINDS = {
    "keypoints_2d": AnnotationKind(
        "keypoints_2d.json", Keypoints2DAnnotations
    ),
    "keypoints_3d": AnnotationKind(
        "keypoints_3d.json", Keypoints3DAnnotations
    ),
    "bboxes_2d": AnnotationKind("bboxes_2d.json", BBox2DAnnotations),
    "cameras_temporal": AnnotationKind(
        "cameras_temporal.json",
        CameraTemporalAnnotations,
        build_default=build_synchronized_camera_temporal,
    ),
    "cameras_intrinsics": AnnotationKind(
        "cameras_intrinsics.json", CameraIntrinsicsAnnotations
    ),
    "cameras_extrinsics": AnnotationKind(
        "cameras_extrinsics.json", CameraExtrinsicsAnnotations
    ),
}

ANNOTATION_FILENAMES = {
    key: kind.filename for key, kind in ANNOTATION_KINDS.items()
}


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def write_sequence_annotations(
    dataset_dir: str,
    annotations_reldir: str,
    annotations: dict[str, Any],
) -> dict[str, str]:
    """Serialize a sequence's annotations and return their relative paths."""
    unknown_keys = set(annotations) - set(ANNOTATION_FILENAMES)
    if unknown_keys:
        raise KeyError(
            f"Unknown annotation kinds: {sorted(unknown_keys)}. "
            f"Expected any of {sorted(ANNOTATION_FILENAMES)}."
        )

    os.makedirs(os.path.join(dataset_dir, annotations_reldir), exist_ok=True)

    relpaths: dict[str, str] = {}

    for key, filename in ANNOTATION_FILENAMES.items():
        if key not in annotations:
            continue

        relpath = os.path.join(annotations_reldir, filename)
        abspath = os.path.join(dataset_dir, relpath)

        with open(abspath, "wb") as f:
            f.write(
                orjson.dumps(
                    annotations[key].to_dict(), default=_json_fallback
                )
            )
        tqdm.write(f"Saved {key} annotations to {abspath}")

        relpaths[key] = Path(relpath).as_posix()

    return relpaths


def load_sequence_annotations(
    dataset_dir: str, sequence: dict[str, Any]
) -> dict[str, Any]:
    """Read a sequence's annotations, defaulting those it does not provide."""
    view_ids = list(sequence["views"])
    relpaths = sequence.get("annotations", {})

    annotations: dict[str, Any] = {}

    for key, kind in ANNOTATION_KINDS.items():
        relpath = relpaths.get(key)
        abspath = (
            os.path.join(dataset_dir, relpath) if relpath is not None else None
        )

        if abspath is not None and os.path.exists(abspath):
            with open(abspath, "rb") as f:
                annotations[key] = kind.annotations_type.from_dict(
                    orjson.loads(f.read())
                )
        elif kind.build_default is not None:
            annotations[key] = kind.build_default(view_ids)

    return annotations
