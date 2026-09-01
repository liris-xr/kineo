import argparse

from kineo.datasets.h36m.h36m_dataset import H36MSequenceDataset
from kineo.visualization.sequence_preview import (
    MAX_PREVIEW_SIDE,
    preview_sequence,
)

# Human3.6M was captured at 50 Hz and its listings do not carry the rate.
VIDEO_FPS = 50.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequences_file",
        type=str,
        help="Path to an h36m_<protocol>_sequences.json file written by the "
        "preprocessing",
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
        "--max-side",
        type=int,
        default=MAX_PREVIEW_SIDE,
        help="Longest side the footage is resized to, the annotations resized "
        "with it. Pass a value above the source size to keep it native",
    )
    args = parser.parse_args()

    dataset = H36MSequenceDataset(args.sequences_file, split=args.split)
    names = [sequence["sequence_name"] for sequence in dataset.sequences]
    index = names.index(args.sequence) if args.sequence else 0

    preview_sequence(
        dataset[index],
        fps=VIDEO_FPS,
        output_path=args.save,
        max_frames=args.max_frames,
        max_side=args.max_side,
    )
