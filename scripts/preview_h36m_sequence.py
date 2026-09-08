import argparse
import os

import orjson

from kineo.datasets.h36m.h36m_dataset import H36MSequenceDataset
from kineo.visualization.sequence_preview import (
    DEFAULT_DOWNSCALE_FACTOR,
    add_dataset_dir_argument,
    find_sequence,
    preview_sequence,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preview a Human3.6M sequence and its ground truth in rerun"
    )
    add_dataset_dir_argument(parser, "H3.6M")
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
        help="Split the sequence is taken from",
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
        help="Downscale factor of the footage and annotations, 1 for full size",
    )
    parser.add_argument(
        "--up-axis",
        type=str,
        choices=["x", "y", "z"],
        default="z",
        help="Axis pointing up in the dataset's world",
    )
    args = parser.parse_args()

    if args.dataset_dir is None:
        parser.error(
            "pass --dataset-dir, or set KINEO_DATASETS_DIR to the directory "
            "holding H3.6M"
        )

    sequences_file = os.path.join(
        args.dataset_dir, f"h36m_{args.protocol}_sequences.json"
    )

    # Read names from the JSON, the dataset opens every video
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
