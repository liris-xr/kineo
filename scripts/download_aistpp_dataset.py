"""Script to download the AIST++ dataset.

Downloading the videos implies accepting the Terms of Use of the AIST Dance
Video Database, available at https://aistdancedb.ongaaccel.jp/terms_of_use/.
"""

import argparse
import os
import sys

from kineo.datasets.aistpp.aistpp_download import (
    AIST_APPLICATION_FORM_URL,
    AIST_TERMS_OF_USE_URL,
    AISTPP_SPLITS,
    VIDEO_BITRATES,
    VIDEO_VARIANTS,
    download_aistpp,
)


def _accept_terms_of_use() -> bool:
    if not sys.stdin.isatty():
        return False

    print(TERMS_OF_USE_NOTICE)
    answer = input(
        "Have you filed the application form, and do you agree with the "
        "Terms of Use? [y/N] "
    ).strip()
    return answer.lower() in ("y", "yes")


TERMS_OF_USE_NOTICE = f"""The videos come from the AIST Dance Video Database, whose Terms of Use require,
before any use of the database:

  1. Filing their application form: {AIST_APPLICATION_FORM_URL}
  2. Agreeing to the Terms of Use:  {AIST_TERMS_OF_USE_URL}

In short: academic research only, commercial use needs AIST's prior written
consent, redistributing any content of the database is prohibited, and work
using it must name the "AIST Dance Video Database" and cite Tsuchida et al.,
ISMIR 2019. Read the page above for the authoritative text.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory where the raw dataset will be downloaded",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        choices=VIDEO_VARIANTS,
        default=list(VIDEO_VARIANTS),
        help="Video variants to download. The AIST++ annotations are aligned "
        "with the 'refined' videos.",
    )
    parser.add_argument(
        "--bitrate",
        type=str,
        choices=VIDEO_BITRATES,
        default="10M",
        help="Video bitrate. 10M is the highest AIST publishes and the "
        "native quality; both tiers are 1920x1080. The default pose_test "
        "split weighs 128GB of raw video at 10M, 20GB at 2M.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=AISTPP_SPLITS,
        default="pose_test",
        help="Only download the videos of this AIST++ split. 'pose_*' splits "
        "are subject-disjoint and meant for pose estimation, 'crossmodal_*' "
        "ones are music-disjoint and meant for dance generation.",
    )
    parser.add_argument(
        "--keep-ignored",
        action="store_true",
        help="Keep the sequences AIST++ flags as poorly reconstructed in its "
        "ignore_list.txt. Their videos are fine, only their pose annotations "
        "are unreliable.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Whether to skip the download of the AIST++ annotations",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Whether to skip the download of the AIST video metadata CSVs",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Whether to skip the download of the videos",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of videos downloaded concurrently",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Whether to download files that are already complete",
    )
    parser.add_argument(
        "--accept-terms",
        action="store_true",
        help="Confirm that the AIST Dance Video Database application form has "
        "been filed and its Terms of Use are accepted, without being prompted",
    )
    args = parser.parse_args()

    variants = [] if args.no_videos else args.variants

    if variants and not args.accept_terms and not _accept_terms_of_use():
        print(TERMS_OF_USE_NOTICE)
        print(
            "Program exit. File the application form and accept the Terms of "
            "Use, then re-run with '--accept-terms'."
        )
        sys.exit(1)

    if not os.path.exists(args.directory):
        os.makedirs(args.directory, exist_ok=True)

    download_aistpp(
        args.directory,
        variants=variants,
        bitrate=args.bitrate,
        split=args.split,
        drop_ignored=not args.keep_ignored,
        annotations=not args.no_annotations,
        metadata=not args.no_metadata,
        num_workers=args.num_workers,
        force_download=args.force_download,
    )
