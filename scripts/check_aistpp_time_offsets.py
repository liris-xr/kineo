import argparse

from kineo.datasets.aistpp.aistpp_audio_check import (
    DEFAULT_TOLERANCE_S,
    check_time_offsets_against_audio,
)
from kineo.datasets.aistpp.aistpp_download import AISTPP_SPLITS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-check the preprocessed AIST++ time offsets against "
        "the raw recordings' audio, without re-matching any video frame"
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory the dataset was preprocessed into",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=AISTPP_SPLITS,
        default="pose_test",
        help="AIST++ split whose offsets are checked",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_S,
        help="Largest disagreement between the frame match and the audio, in "
        "seconds, taken as agreement",
    )
    parser.add_argument(
        "--sequences-filter",
        nargs="+",
        default=None,
        help="Sequences to check, every measured one by default",
    )
    args = parser.parse_args()

    check_time_offsets_against_audio(
        args.directory,
        split=args.split,
        tolerance_s=args.tolerance,
        sequence_names=args.sequences_filter,
    )
