import argparse

from kineo.datasets.panoptic.panoptic_download import (
    PANOPTIC_SPLITS,
    TEMPO_HD_CAMERAS,
)
from kineo.datasets.panoptic.panoptic_preprocess import (
    TEMPO_FRAME_INTERVAL,
    preprocess_panoptic,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the directory where the dataset was downloaded",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=PANOPTIC_SPLITS,
        default="all",
        help="TEMPO split to preprocess. The default covers both and skips "
        "the sequences whose videos are not on disk.",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=list(TEMPO_HD_CAMERAS),
        help="HD camera nodes to read. The TEMPO protocol reads 3, 6, 12, 13 "
        "and 23.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=TEMPO_FRAME_INTERVAL,
        help="Read one annotated frame out of this many. TEMPO reads every "
        "third, which is 9.99Hz out of the dome's 29.97Hz.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Whether to skip unpacking the 3D body keypoint archives",
    )
    args = parser.parse_args()

    preprocess_panoptic(
        args.dataset_dir,
        split=args.split,
        cameras=args.cameras,
        frame_interval=args.frame_interval,
        skip_extract=args.skip_extract,
    )
