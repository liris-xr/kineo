import argparse
from kineo.datasets.panoptic.panoptic_preprocess import preprocess_panoptic

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the directory where the raw dataset was downloaded",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extracting the tarfiles and only preprocess the annotations",
    )

    args = parser.parse_args()

    preprocess_panoptic(
        args.dataset_dir,
        skip_extract=args.skip_extract,
    )