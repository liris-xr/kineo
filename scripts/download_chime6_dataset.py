import argparse

from kineo.datasets.chime6.chime6_download import CHIME6_SPLITS, download_chime6

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory where the dataset is downloaded",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        choices=CHIME6_SPLITS,
        default=["dev"],
        help="CHiME-6 splits to download",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download archives already present instead of resuming or "
        "skipping them",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Whether to skip unpacking the archives",
    )
    args = parser.parse_args()

    download_chime6(
        args.directory,
        splits=args.splits,
        force_download=args.force_download,
        skip_extract=args.skip_extract,
    )
