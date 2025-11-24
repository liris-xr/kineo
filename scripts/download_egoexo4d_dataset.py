import argparse
from kineo.datasets.egoexo4d.egoexo4d_download import download_egoexo4d
import os
import sys
from getpass import getpass
from warnings import warn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory where the raw dataset will be downloaded",
    )
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory, exist_ok=True)

    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")

    if not aws_access_key_id or not aws_secret_access_key:
        print("Please enter your credentials for the EgoExo4D dataset.")

        if not sys.stdin.isatty():
            warn("Terminal is not interactive. Password will be echoed.")

        while True:
            aws_access_key_id = (
                input("AWS Access Key ID: ")
                if aws_access_key_id is None
                else aws_access_key_id
            ).strip()

            aws_access_key_id = aws_access_key_id if aws_access_key_id != "" else None

            if aws_access_key_id is not None:
                break
            print("AWS Access Key ID cannot be empty. Please try again.")

        while True:
            if not sys.stdin.isatty():
                aws_secret_access_key = (
                    input("AWS Secret Access Key: ")
                    if aws_secret_access_key is None
                    else aws_secret_access_key
                ).strip()
            else:
                aws_secret_access_key = (
                    getpass("AWS Secret Access Key: ")
                    if aws_secret_access_key is None
                    else aws_secret_access_key
                ).strip()

            aws_secret_access_key = (
                aws_secret_access_key if aws_secret_access_key != "" else None
            )

            if aws_secret_access_key is not None:
                break
            print("AWS Secret Access Key cannot be empty. Please try again.")

    if not aws_region:
        aws_region = (
            input("AWS Region (default: None): ") if aws_region is None else aws_region
        )

    if aws_region is not None:
        aws_region = aws_region.strip()
        if aws_region == "":
            aws_region = None

    download_egoexo4d(
        args.directory,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
    )
