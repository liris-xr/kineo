import argparse
from kineo.datasets.panoptic.panoptic_download import download_panoptic
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory where the raw dataset will be downloaded",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Whether to disable checksum verification",
    )
    parser.add_argument(
        "--use-snu-endpoint",
        action="store_true",
        help="Use the SNU endpoint instead of the CMU endpoint",
    )
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory, exist_ok=True)

    download_panoptic(
        output_base_dir=args.directory,
        verify=not args.no_checksum,
        use_snu_endpoint=args.use_snu_endpoint,
    )
