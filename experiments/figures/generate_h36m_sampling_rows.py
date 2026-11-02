# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Generate the Human3.6M sampling strategy table."""

import argparse
import os

import orjson

from experiments.figures.generate_confidence_filter_rows import shared_sequences
from experiments.figures.generate_sampling_rows import (
    format_cells,
    load_sequence,
    scored_sequences,
    summarize,
    table_rows,
)

_TEMPLATE = os.path.join(os.path.dirname(__file__), "h36m_sampling_rows_template.tex")
ARMS = ("Random", "FPS", "Top-P")


def input_rows(label: str, runs: list[str], runtime_json: str,
               restrict_to: list[str] = ()) -> tuple[list[str], int]:
    scored = {arm: scored_sequences(run) for arm, run in zip(ARMS, runs)}
    common = set.intersection(*(set(files) for files in scored.values()))
    if restrict_to:
        common &= shared_sequences(list(restrict_to))
    common = sorted(common)
    if not common:
        raise FileNotFoundError(f"No {label} sequence was scored by every run")
    # Written by experiments/benchmark_sampling_runtime.py
    with open(runtime_json, "rb") as f:
        timed = orjson.loads(f.read())["per_sequence"]
    rows = {arm: {s: load_sequence(files[s]) for s in common} for arm, files in scored.items()}
    for arm, arm_rows in rows.items():
        for name, row in arm_rows.items():
            timing = timed[arm].get(name)
            row["sampling_seconds"] = timing["pairs"] + timing["ba"] if timing else None
    cells = format_cells({arm: summarize(r, common) for arm, r in rows.items()})
    lines = table_rows(cells)
    lines = [("        " + rf"\multirow{{{len(lines)}}}{{*}}{{{label}}}" if i == 0 else "        ")
             + " & " + line.strip() for i, line in enumerate(lines)]
    return lines, len(common)


def main(args: argparse.Namespace) -> None:
    rows, counts = [], []
    for label, *paths in args.input:
        lines, n = input_rows(label, paths[:3], paths[3], args.restrict_to)
        if rows:
            rows.append(r"        \midrule")
        rows += lines
        counts.append(f"{n} {label.lower()}")
    with open(_TEMPLATE) as f:
        table = f.read().replace("{rows}", "\n".join(rows))
    print(table.replace("{n_sequences}", " and ".join(counts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs=5, action="append", required=True,
                        metavar=("LABEL", "RANDOM_RUN", "FPS_RUN", "TOPP_RUN", "RUNTIME_JSON"))
    parser.add_argument("--restrict-to", nargs="+", default=[], metavar="RUN_DIR")
    main(parser.parse_args())
