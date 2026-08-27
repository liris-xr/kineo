import argparse

from kineo.datasets.aistpp.aistpp_download import AISTPP_SPLITS
from kineo.datasets.aistpp.aistpp_preprocess import preprocess_aistpp

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
        choices=AISTPP_SPLITS,
        default="pose_test",
        help="AIST++ split to preprocess",
    )
    parser.add_argument(
        "--keep-ignored",
        action="store_true",
        help="Keep the sequences AIST++ flags as poorly reconstructed in its "
        "ignore_list.txt",
    )
    parser.add_argument(
        "--min-match-margin",
        type=float,
        default=1.5,
        help="Least decisive raw-to-refined frame match accepted. Sequences "
        "matching less decisively than this on any view are dropped.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Whether to skip unpacking the annotation archives",
    )
    args = parser.parse_args()

    preprocess_aistpp(
        args.directory,
        split=args.split,
        drop_ignored=not args.keep_ignored,
        min_match_margin=args.min_match_margin,
        skip_extract=args.skip_extract,
    )
