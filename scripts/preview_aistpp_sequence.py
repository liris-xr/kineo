import argparse

from kineo.datasets.aistpp.aistpp_dataset import (
    VIDEO_FPS,
    AISTPPSequenceDataset,
)
from kineo.visualization.sequence_preview import (
    MAX_PREVIEW_SIDE,
    preview_sequence,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequences_file",
        type=str,
        help="Path to an aistpp_<split>_<variant>_sequences.json file written "
        "by the preprocessing",
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

    dataset = AISTPPSequenceDataset(args.sequences_file)
    names = [sequence["sequence_name"] for sequence in dataset.sequences_data]
    index = names.index(args.sequence) if args.sequence else 0

    preview_sequence(
        dataset[index],
        fps=VIDEO_FPS,
        output_path=args.save,
        max_frames=args.max_frames,
        max_side=args.max_side,
    )
