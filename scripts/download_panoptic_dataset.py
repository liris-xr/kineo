"""Script to download the CMU Panoptic Studio sequences of the TEMPO protocol.

Only the five HD views the protocol reads are fetched, along with the
calibration and the 3D body keypoints. The full dome release, 480 VGA and 31 HD
views per sequence plus face, hand and mesh annotations, is never touched.
"""

import argparse
import os

from kineo.datasets.panoptic.panoptic_download import (
    N_HD_CAMERAS,
    PANOPTIC_SPLITS,
    TEMPO_HD_CAMERAS,
    download_panoptic,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory where the raw dataset will be downloaded",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=PANOPTIC_SPLITS,
        default="test",
        help="TEMPO split to download. Over the five protocol views, 'test' "
        "weighs about 16GB and 'train' about 40GB.",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=list(TEMPO_HD_CAMERAS),
        help=f"HD camera nodes to download, in 0..{N_HD_CAMERAS - 1}. The "
        "TEMPO protocol reads 3, 6, 12, 13 and 23.",
    )
    parser.add_argument(
        "--keep-ignored",
        action="store_true",
        help="Keep the sequences of 'panoptic_ignore_list.txt'. TEMPO drops "
        "160906_band3 as corrupted; VoxelPose and VTP train on it.",
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Whether to skip the download of the calibration and the 3D body "
        "keypoint archives",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Whether to skip the download of the HD videos",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of files downloaded concurrently",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Whether to download files that are already complete",
    )
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory, exist_ok=True)

    download_panoptic(
        args.directory,
        split=args.split,
        cameras=args.cameras,
        drop_ignored=not args.keep_ignored,
        annotations=not args.no_annotations,
        videos=not args.no_videos,
        num_workers=args.num_workers,
        force_download=args.force_download,
    )
