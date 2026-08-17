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

import os
from pathlib import Path
from typing import Any

import orjson
import torch
from tqdm import tqdm

from kineo.annotations.camera_temporal import (
    CameraTemporalAnnotation,
    CameraTemporalAnnotations,
    CameraTemporalAnnotationsMetadata,
)

# Canonical annotation kinds and their filenames. Insertion order is the order
# annotations are written and listed in sequences.json.
ANNOTATION_FILENAMES = {
    "keypoints_2d": "keypoints_2d.json",
    "keypoints_3d": "keypoints_3d.json",
    "bboxes_2d": "bboxes_2d.json",
    "cameras_temporal": "cameras_temporal.json",
    "cameras_intrinsics": "cameras_intrinsics.json",
    "cameras_extrinsics": "cameras_extrinsics.json",
}


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
