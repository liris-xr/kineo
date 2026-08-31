import argparse

from kineo.datasets.chime6.chime6_download import CHIME6_SPLITS
from kineo.datasets.chime6.chime6_preprocess import (
    WINDOW_LENGTHS_S,
    WINDOWS_PER_CELL,
    preprocess_chime6,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory where the raw dataset was downloaded",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=CHIME6_SPLITS,
        default="dev",
        help="CHiME-6 split to preprocess",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        nargs="+",
        default=[],
        help="Sessions to preprocess, defaults to every session of the split",
    )
    parser.add_argument(
        "--lengths",
        type=float,
        nargs="+",
        default=WINDOW_LENGTHS_S,
        help="Window lengths to select, in seconds",
    )
    parser.add_argument(
        "--windows-per-cell",
        type=int,
        default=WINDOWS_PER_CELL,
        help="Windows to select per (length, content class) pair",
    )
    args = parser.parse_args()

    preprocess_chime6(
        args.directory,
        split=args.split,
        sessions=args.sessions,
        lengths=args.lengths,
        windows_per_cell=args.windows_per_cell,
    )
