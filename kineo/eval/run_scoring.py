# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import os
import pickle

import orjson

from kineo.annotations.camera_temporal import CameraTemporalAnnotations
from kineo.eval.temporal_metrics import compute_time_offset_errors


def load_sequence_list(filepath: str) -> set[str]:
    with open(filepath) as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        }


def load_sequence_names(dataset_file: str) -> list[str]:
    with open(dataset_file, "rb") as f:
        sequences = orjson.loads(f.read())
    return [sequence["sequence_name"] for sequence in sequences]


def score_time_offsets(
    benchmark_dir: str, dataset_file: str, excluded: set[str] = frozenset()
) -> dict[str, dict[str, float]]:
    dataset_dir = os.path.dirname(dataset_file)

    with open(dataset_file, "rb") as f:
        sequences = orjson.loads(f.read())

    errors = {}

    for sequence in sequences:
        name = sequence["sequence_name"]
        if name in excluded:
            continue
        predicted_path = os.path.join(
            benchmark_dir, "annotations", name, "cameras_temporal.pkl"
        )
        gt_relpath = sequence.get("annotations", {}).get("cameras_temporal")

        if gt_relpath is None or not os.path.isfile(predicted_path):
            continue

        with open(os.path.join(dataset_dir, gt_relpath), "rb") as f:
            gt = CameraTemporalAnnotations.from_dict(orjson.loads(f.read()))

        with open(predicted_path, "rb") as f:
            predicted = CameraTemporalAnnotations.from_dict(pickle.load(f))

        errors[name] = compute_time_offset_errors(gt, predicted)

    return errors
