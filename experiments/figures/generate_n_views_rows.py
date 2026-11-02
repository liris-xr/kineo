# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Generate the EgoHumans number-of-views table rows."""

import argparse

import numpy as np
import orjson

from experiments.figures.generate_n_views_figure import (
    ALL_VIEWS,
    HSFM_PA_MPJPE_MM,
    n_views_of,
)
from experiments.figures.generate_sampling_rows import scored_sequences


def load_metrics(run_dir: str) -> dict[str, tuple[float, float]]:
    out = {}
    for name, path in scored_sequences(run_dir).items():
        with open(path, "rb") as f:
            metrics = orjson.loads(f.read())
        out[name] = (metrics["human_stats"]["pa-mpjpe"]["mean"] * 1000,
                     metrics["cam_stats"]["AE"]["mean"])
    return out


def table_rows(est_runs: list[str], gt_runs: list[str]) -> tuple[list[str], int]:
    results = [{n_views_of(d): load_metrics(d) for d in runs} for runs in (est_runs, gt_runs)]
    views = sorted(results[0], key=lambda n: (n == ALL_VIEWS, n if n != ALL_VIEWS else 0))
    common = sorted(set.intersection(*(set(r) for runs in results for r in runs.values())))
    rows = []
    for n in views:
        cells = [str(n)]
        for runs in results:
            values = np.array([runs[n][s] for s in common])
            pa, ae = values[:, 0], values[:, 1]
            cells += [f"{pa.mean():.1f} $\\pm$ {pa.std():.1f}", f"{np.median(pa):.1f}",
                      f"{np.median(ae):.2f}"]
        cells.append(f"{HSFM_PA_MPJPE_MM[n]:.0f}" if n in HSFM_PA_MPJPE_MM else "--")
        if n == ALL_VIEWS:
            rows.append(r"    \midrule")
        rows.append("    " + " & ".join(cells) + r" \\")
    return rows, len(common)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--est-runs", nargs="+", required=True)
    parser.add_argument("--gt-runs", nargs="+", required=True)
    args = parser.parse_args()
    rows, _ = table_rows(args.est_runs, args.gt_runs)
    print("\n".join(rows))
