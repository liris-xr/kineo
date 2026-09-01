import argparse
import os

from kineo.datasets.aistpp.aistpp_dataset import AISTPPSequenceDataset
from kineo.datasets.aistpp.aistpp_download import AISTPP_SPLITS, VIDEO_VARIANTS
from kineo.visualization.sequence_preview import (
    DEFAULT_DOWNSCALE_FACTOR,
    preview_sequence,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preview an AIST++ sequence and its ground truth in rerun"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the directory the dataset was preprocessed into",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=AISTPP_SPLITS,
        default="pose_test",
        help="Split the sequence is taken from",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=VIDEO_VARIANTS,
        default="raw",
        help="Video variant to read, the untrimmed recordings by default",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Name of the sequence to preview, the first one by default",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Write the recording to this .rrd file instead of spawning the "
        "viewer",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Number of frames to log, the whole sequence by default",
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
        default="y",
        help="Axis pointing up in the dataset's world",
    )
    args = parser.parse_args()

    dataset = AISTPPSequenceDataset(
        os.path.join(
            args.dataset_dir,
            f"aistpp_{args.split}_{args.variant}_sequences.json",
        )
    )

    names = [sequence["sequence_name"] for sequence in dataset.sequences_data]
    if args.sequence and args.sequence not in names:
        parser.error(
            f"unknown sequence '{args.sequence}'. The listing holds "
            f"{len(names)}, such as {', '.join(names[:3])}"
        )

    index = names.index(args.sequence) if args.sequence else 0

    preview_sequence(
        dataset[index],
        output_path=args.save,
        max_frames=args.max_frames,
        downscale_factor=args.downscale_factor,
        up_axis=args.up_axis,
    )
