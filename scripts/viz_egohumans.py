import argparse
from kineo.datasets.egohumans.egohumans_viz import visualize_egohumans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the directory where the dataset was downloaded",
    )
    args = parser.parse_args()

    visualize_egohumans(args.dataset_dir)
