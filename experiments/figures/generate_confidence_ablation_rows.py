# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Generate the detection confidence ablation table."""

import argparse
import json
import os

import numpy as np

from experiments.figures.generate_confidence_filter_rows import (
    METRICS,
    metric_cells,
    shared_sequences,
)

_TEMPLATE = os.path.join(os.path.dirname(__file__), "confidence_ablation_rows_template.tex")
COUNTED = ("s-TE", "AE", "pa-mpjpe")
N_COLUMNS = 4 + 2 * len(METRICS)
ABLATION_CAPTIONS = {
    "weights": "With confidence, the detection scores weight bundle adjustment, "
               "triangulation and scale estimation; without it, every detected keypoint "
               "has the same weight. In both, correspondences and bundle adjustment "
               "keypoints are sampled among those scoring above the thresholds",
    "all": "With confidence, the detection scores filter the correspondences and "
           "bundle adjustment keypoints and weight the later stages; without it, every "
           "score is set to one",
}


def load_metrics(run_dir: str) -> dict[str, str]:
    metrics_dir = os.path.join(run_dir, "metrics")
    return {f[:-5]: os.path.join(metrics_dir, f)
            for f in os.listdir(metrics_dir) if f.endswith(".json")}


def read_metric(path: str, section: str, name: str) -> float:
    with open(path) as f:
        return json.load(f)[section][name]["mean"]


def pair_cells(off_run: str, on_run: str,
               restrict_to: list[str] = ()) -> tuple[list, list, dict, int]:
    scored = [load_metrics(off_run), load_metrics(on_run)]
    common = sorted(shared_sequences([off_run, on_run, *restrict_to]))
    if not common:
        raise FileNotFoundError(f"No sequence scored by both {off_run} and {on_run}")
    pairs, improved = [], {}
    for metric, section in METRICS:
        off, on = (np.array([read_metric(run[s], section, metric) for s in common])
                   for run in scored)
        pairs.append(metric_cells(*((float(np.median(v)), float(np.percentile(v, 90)))
                                    for v in (off, on)), metric))
        if metric in COUNTED:
            improved[metric] = int((on < off).sum())
    off_cells = [cell for pair in pairs for cell in pair[0]]
    on_cells = [cell for pair in pairs for cell in pair[1]]
    return off_cells, on_cells, improved, len(common)


def main(args: argparse.Namespace) -> None:
    rows, comments = [], []
    for i, (dataset, detector, inp, off_run, on_run) in enumerate(args.pair):
        off_cells, on_cells, improved, n = pair_cells(off_run, on_run, args.restrict_to)
        comments.append(f"% {dataset}, {detector}, {inp} ({n} sequences), lower with confidence: "
                        + ", ".join(f"{m} {k}/{n}" for m, k in improved.items()))
        previous = args.pair[i - 1] if i else None
        same_dataset = previous is not None and previous[0] == dataset
        same_detector = same_dataset and previous[1] == detector
        n_dataset = sum(1 for p in args.pair if p[0] == dataset) * 2
        n_detector = sum(1 for p in args.pair if p[:2] == (dataset, detector)) * 2
        if previous is not None:
            first = 2 if same_detector else 1 if same_dataset else None
            rows.append(rf"        \cmidrule{{{first + 1}-{N_COLUMNS}}}" if first else r"        \midrule")
        for i_arm, (mark, cells) in enumerate((("--", off_cells), (r"\checkmark", on_cells))):
            lead = i_arm == 0
            dataset_cell = rf"\multirow{{{n_dataset}}}{{*}}{{{dataset}}}" if lead and not same_dataset else ""
            detector_cell = rf"\multirow{{{n_detector}}}{{*}}{{{detector}}}" if lead and not same_detector else ""
            input_cell = rf"\multirow{{2}}{{*}}{{{inp}}}" if lead else ""
            rows.append("        " + " & ".join(
                [dataset_cell, detector_cell, input_cell, mark, *cells]) + r" \\")
    print("\n".join(comments))
    with open(_TEMPLATE) as f:
        table = f.read().replace("{rows}", "\n".join(rows))
    print(table.replace("{ablation}", ABLATION_CAPTIONS[args.caption_ablation]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", nargs=5, action="append", required=True,
                        metavar=("DATASET", "DETECTOR", "INPUT", "OFF_RUN", "ON_RUN"))
    parser.add_argument("--caption-ablation", choices=ABLATION_CAPTIONS, default="weights")
    parser.add_argument("--restrict-to", nargs="+", default=[], metavar="RUN_DIR")
    main(parser.parse_args())
