import argparse
import os

from kineo.datasets.egohumans.egohumans_dataset import EgoHumansSequenceDataset
from kineo.visualization.sequence_preview import (
    DEFAULT_DOWNSCALE_FACTOR,
    find_sequence,
    preview_sequence,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preview an EgoHumans sequence and its ground truth in rerun"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the directory the dataset was preprocessed into",
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

    dataset = EgoHumansSequenceDataset(
        os.path.join(args.dataset_dir, "egohumans_sequences.json")
    )

    names = [sequence["sequence_name"] for sequence in dataset.sequences_data]

    try:
        index = find_sequence(names, args.sequence)
    except LookupError as error:
        parser.error(str(error))

    preview_sequence(
        dataset[index],
        downscale_factor=args.downscale_factor,
        up_axis=args.up_axis,
    )
