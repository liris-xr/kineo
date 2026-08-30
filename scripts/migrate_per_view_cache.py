# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Split a whole-sequence annotation cache into the per-view layout."""

from __future__ import annotations

import argparse
import collections
import os
import pickle


def split_by_view(payload: dict) -> dict[str, dict]:
    by_view = collections.defaultdict(list)

    for annotation in payload["annotations"]:
        by_view[annotation["view_id"]].append(annotation)

    return {
        view_id: {"metadata": payload["metadata"], "annotations": annotations}
        for view_id, annotations in by_view.items()
    }


def migrate(sources: list[str], target: str, apply: bool) -> None:
    written = 0
    already_present = 0
    conflicts = []

    for source in sources:
        for sequence_name in sorted(os.listdir(source)):
            sequence_dir = os.path.join(source, sequence_name)
            if not os.path.isdir(sequence_dir):
                continue

            for filename in sorted(os.listdir(sequence_dir)):
                if not filename.endswith(".pkl"):
                    continue

                annotation_key = filename[: -len(".pkl")]

                with open(os.path.join(sequence_dir, filename), "rb") as f:
                    payload = pickle.load(f)

                view_dir = os.path.join(target, sequence_name, annotation_key)

                for view_id, view_payload in split_by_view(payload).items():
                    filepath = os.path.join(view_dir, f"{view_id}.pkl")
                    blob = pickle.dumps(view_payload)

                    if os.path.exists(filepath):
                        with open(filepath, "rb") as f:
                            if f.read() != blob:
                                conflicts.append(filepath)
                        already_present += 1
                        continue

                    if apply:
                        os.makedirs(view_dir, exist_ok=True)
                        with open(filepath, "wb") as f:
                            f.write(blob)
                    written += 1

    print(f"written: {written}, already present: {already_present}")

    if conflicts:
        print(f"MISMATCH on {len(conflicts)} existing files, left untouched:")
        for filepath in conflicts[:20]:
            print("   ", filepath)

    if not apply:
        print("DRY RUN, pass --apply to write")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    migrate(args.sources, args.target, args.apply)


if __name__ == "__main__":
    main()
