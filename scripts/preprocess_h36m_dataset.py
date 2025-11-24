import argparse
from kineo.datasets.h36m.h36m_preprocess import preprocess_h36m

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the directory where the dataset was downloaded",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip extracting the tarfiles and only preprocess the annotations",
    )
    parser.add_argument(
        "--skip-annotations-generation",
        action="store_true",
        help="Skip annotations generation",
    )
    args = parser.parse_args()

    preprocess_h36m(
        args.dataset_dir,
        skip_extract=args.skip_extract,
        skip_annotations_generation=args.skip_annotations_generation,
    )
