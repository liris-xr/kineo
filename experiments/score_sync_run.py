# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Score the time offsets and confidences of a sync-only run."""

import argparse
import os
import pickle

import orjson

from kineo.annotations.camera_temporal import CameraTemporalAnnotations
from kineo.datasets.aistpp.aistpp_dataset import VIDEO_FPS
from kineo.eval.run_scoring import (
    load_sequence_list,
    load_sequence_names,
    score_time_offsets,
)
from kineo.eval.temporal_metrics import (
    print_time_offset_statistics,
    summarize_time_offset_errors,
)


def read_confidences(
    run_dir: str, sequence_names: list[str]
) -> dict[str, dict[str, float]]:
    confidences = {}
    for name in sequence_names:
        path = os.path.join(run_dir, "annotations", name, "cameras_temporal.pkl")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            annotations = CameraTemporalAnnotations.from_dict(pickle.load(f))
        confidences[name] = {
            a.view_id: a.confidence
            for a in annotations.annotations
            if a.confidence is not None
        }
    return confidences


def main(run_dir: str, dataset_file: str, fps: float, exclude_list: str | None):
    excluded = load_sequence_list(exclude_list) if exclude_list else set()
    listed = [n for n in load_sequence_names(dataset_file) if n not in excluded]

    errors = score_time_offsets(run_dir, dataset_file, excluded)
    confidences = read_confidences(run_dir, listed)
    summary = summarize_time_offset_errors(errors, fps)
    print_time_offset_statistics(summary, fps)
    print(f"  missing sequences    {len(set(listed) - set(errors))}")

    with open(os.path.join(run_dir, "time_offsets_summary.json"), "wb") as f:
        f.write(
            orjson.dumps(
                {
                    "statistics": summary,
                    "errors_by_sequence": errors,
                    "confidences_by_sequence": confidences,
                },
                option=orjson.OPT_INDENT_2,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=str, help="Directory holding annotations/")
    parser.add_argument(
        "--dataset-file",
        type=str,
        required=True,
        help="The dataset's sequences.json, read for the ground-truth offsets",
    )
    parser.add_argument("--fps", type=float, default=VIDEO_FPS)
    parser.add_argument(
        "--exclude-list",
        type=str,
        default="kineo/datasets/aistpp/aistpp_ignore_list.txt",
        help="Sequences to leave out, one per line",
    )
    args = parser.parse_args()
    main(args.run_dir, args.dataset_file, args.fps, args.exclude_list)
