import argparse
import os

import orjson

from kineo.datasets.h36m.h36m_dataset import H36MSequenceDataset
from kineo.visualization.sequence_preview import (
    DEFAULT_DOWNSCALE_FACTOR,
    find_sequence,
    preview_sequence,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preview a Human3.6M sequence and its ground truth in rerun"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the directory the dataset was preprocessed into",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="protocol1",
        help="Evaluation protocol whose sequence listing is read",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="val",
        help="Split the sequence is taken from. Protocol 1 evaluates on S9 "
        "and S11, and holds S1, S5, S6, S7 and S8 in the train split",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Name of the sequence to preview, the first one by default",
    )
    parser.add_argument(
        "--downscale-factor",
        type=int,
        default=DEFAULT_DOWNSCALE_FACTOR,
        help="How much smaller than the dataset's the footage is shown, the "
        "annotations resized with it. Pass 1 to keep the dataset's own size",
    )
    parser.add_argument(
        "--up-axis",
        type=str,
        choices=["x", "y", "z"],
        default="z",
        help="Axis pointing up in the dataset's world",
    )
    args = parser.parse_args()

    sequences_file = os.path.join(
        args.dataset_dir, f"h36m_{args.protocol}_sequences.json"
    )

    # Names come from the listing rather than the dataset, which opens a
    # reader per view of every sequence as it is built.
    with open(sequences_file, "rb") as f:
        names_by_split: dict[str, list[str]] = {}
        for entry in orjson.loads(f.read()):
            names_by_split.setdefault(entry["split"], []).append(
                entry["sequence_name"]
            )

    names = names_by_split.get(args.split, [])

    try:
        index = find_sequence(names, args.sequence)
    except LookupError as error:
        # A subject sits in one split only, so a name missing here is usually
        # a name that lives in the other one.
        other_split = "train" if args.split == "val" else "val"
        other_names = names_by_split.get(other_split, [])

        try:
            found = other_names[find_sequence(other_names, args.sequence)]
        except LookupError:
            parser.error(str(error))

        parser.error(
            f"{error}. '{found}' is in the {other_split} split, reachable "
            f"with --split {other_split}"
        )

    dataset = H36MSequenceDataset(sequences_file, split=args.split)

    preview_sequence(
        dataset[index],
        downscale_factor=args.downscale_factor,
        up_axis=args.up_axis,
    )
