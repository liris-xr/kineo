import argparse
from kineo.datasets.egoexo4d.egoexo4d_preprocess import preprocess_egoexo4d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the directory where the raw dataset was downloaded",
    )
    parser.add_argument(
        "splits",
        type=str,
        nargs="+",
        choices=["train", "val", "test"],
    )
    args = parser.parse_args()

    for split in args.splits:
        preprocess_egoexo4d(args.input_dir, split)
