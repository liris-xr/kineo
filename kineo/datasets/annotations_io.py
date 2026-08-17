# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Shared on-disk layout for per-sequence dataset annotations.

Dataset preprocessors differ in how they load their raw data but agree on what
they emit: one JSON file per kind of annotation, in a per-sequence directory,
listed by relative path in the dataset's `sequences.json`. This module owns that
agreement so EgoHumans and Human3.6M cannot drift apart.
"""

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
    """How one kind of annotation is stored, parsed and, if possible, defaulted.

    Attributes:
        filename: Name of the JSON file holding this kind of annotation.
        annotations_type: Class exposing the `to_dict()` / `from_dict()` pair
            used to serialize and parse it.
        build_default: Builds the annotation a dataset that does not provide
            this kind should be read as having, given the sequence's view ids.
            `None` when no default is honest — there is no such thing as a
            default camera pose or a default set of keypoints, and inventing one
            would silently corrupt whatever consumes it.
    """

    filename: str
    annotations_type: type
    build_default: Callable[[list[str]], Any] | None = None


def build_synchronized_camera_temporal(
    view_ids: list[str],
) -> CameraTemporalAnnotations:
    """Camera temporal annotations for perfectly synchronized views.

    Emits a single zero time offset per view, which is the correct annotation for
    a genlocked or frame-aligned capture (EgoHumans exo cameras, H3.6M). Datasets
    with real clock drift build their own annotations with several entries per
    view instead.

    Args:
        view_ids: View identifiers, in the order annotations should be emitted.

    Returns:
        One annotation per view, at frame 0 with a zero offset.
    """
    return CameraTemporalAnnotations(
        metadata=CameraTemporalAnnotationsMetadata(),
        annotations=[
            CameraTemporalAnnotation(
                view_id=view_id, frame_idx=0, time_offset=0.0
            )
            for view_id in view_ids
        ],
    )


# Canonical annotation kinds. Insertion order is the order annotations are
# written and listed in sequences.json. Adding a kind here is all a new kind
# needs: every dataset writes it if it has it, and reads the default if not.
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

# Kind -> filename, the view of the registry the write path needs.
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
    """Serialize a sequence's annotations and return their relative paths.

    Writes one JSON file per entry of `annotations`, using the canonical filename
    for its kind. Kinds absent from `annotations` are not written: a dataset that
    has no such annotation lists fewer paths rather than emitting an empty file.

    Args:
        dataset_dir: Absolute path to the dataset root.
        annotations_reldir: Directory holding this sequence's annotation files,
            relative to `dataset_dir`.
        annotations: Maps a key of `ANNOTATION_FILENAMES` to an object exposing
            `to_dict()`.

    Returns:
        Maps each written kind to its POSIX path relative to `dataset_dir`, in
        `ANNOTATION_FILENAMES` order.

    Raises:
        KeyError: If `annotations` holds a key absent from
            `ANNOTATION_FILENAMES`.
    """
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
    """Read a sequence's annotations, defaulting the kinds it does not provide.

    A dataset only writes the kinds it actually has, so what is on disk varies by
    dataset and by when it was preprocessed. This reads whatever is there and
    fills in the rest from `ANNOTATION_KINDS`, so callers see one uniform set
    without knowing which dataset they are looking at. A kind whose file is
    listed but missing is defaulted too, which is what makes data preprocessed
    before a kind existed readable without regenerating it.

    Kinds that have no honest default and no file on disk are left out rather
    than invented, so using one raises `KeyError` at the point of use instead of
    silently returning a made-up camera pose.

    Args:
        dataset_dir: Absolute path to the dataset root.
        sequence: An entry of the dataset's `sequences.json`, read for its
            `"annotations"` paths and its `"views"` keys.

    Returns:
        Maps each available kind to its annotations object, in
        `ANNOTATION_KINDS` order.
    """
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
