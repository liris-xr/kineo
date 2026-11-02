# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Compute pair sampling hull coverage per EgoHumans sequence."""

import argparse
import json
import os
import time

import numpy as np
from scipy.spatial import QhullError

from experiments.figures.generate_pair_sampling_figure import (
    hull_share,
    points_per_image,
)
from experiments.figures.generate_sampling_rows import category_of

ARMS = ("Random", "FPS", "Top-P")


def sequence_coverage(run_dirs: dict[str, str], sequence_name: str) -> dict[str, float]:
    candidates, picks, _ = points_per_image(run_dirs, sequence_name)
    shares = {arm: [] for arm in run_dirs}
    for key, pool in candidates.items():
        try:
            values = {arm: hull_share(picks[arm][key], pool) for arm in run_dirs}
        except QhullError:
            # Collinear candidates have no hull
            continue
        for arm, value in values.items():
            shares[arm].append(value)
    return {arm: float(np.mean(v)) for arm, v in shares.items()}


def summary(results: dict[str, dict[str, float]]) -> dict:
    groups = {"All": sorted(results)}
    for name in sorted(results):
        groups.setdefault(category_of(name), []).append(name)
    return {
        group: {
            "n_sequences": len(names),
            **{
                arm: {
                    "median": float(np.median([results[s][arm] for s in names])),
                    "p10": float(np.percentile([results[s][arm] for s in names], 10)),
                }
                for arm in ARMS
            },
        }
        for group, names in groups.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("random_run_dir")
    parser.add_argument("fps_run_dir")
    parser.add_argument("topp_run_dir")
    parser.add_argument("output_json")
    parser.add_argument("--sequences", nargs="+")
    args = parser.parse_args()

    run_dirs = dict(zip(ARMS, (args.random_run_dir, args.fps_run_dir, args.topp_run_dir)))
    sequences = args.sequences or sorted(set.intersection(*(
        set(os.listdir(os.path.join(d, "annotations"))) for d in run_dirs.values())))
    results = {}
    if os.path.exists(args.output_json):
        with open(args.output_json) as f:
            results = json.load(f)["sequences"]
    for i, name in enumerate(sequences):
        if name in results:
            continue
        start = time.perf_counter()
        results[name] = sequence_coverage(run_dirs, name)
        print(f"[{i + 1}/{len(sequences)}] {name} {time.perf_counter() - start:.0f}s "
              + " ".join(f"{arm} {100 * v:.1f}%" for arm, v in results[name].items()), flush=True)
        with open(args.output_json, "w") as f:
            json.dump({"run_dirs": run_dirs, "sequences": results}, f, indent=1)

    table = summary(results)
    with open(args.output_json, "w") as f:
        json.dump({"run_dirs": run_dirs, "sequences": results, "summary": table}, f, indent=1)
    for group, stats in table.items():
        print(f"{group:16s} ({stats['n_sequences']:3d}) " + "  ".join(
            f"{arm} {100 * stats[arm]['median']:.1f}% [p10 {100 * stats[arm]['p10']:.1f}%]"
            for arm in ARMS))
