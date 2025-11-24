import argparse
from kineo.datasets.h36m.h36m_download import download_h36m
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
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Whether to disable checksum verification",
    )
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory, exist_ok=True)

    username = os.getenv("H36M_USERNAME")
    password = os.getenv("H36M_PASSWORD")

    if not username or not password:
        print("Please enter your login credentials for the Human3.6M dataset.")

        if not sys.stdin.isatty():
            warn("Terminal is not interactive. Password will be echoed.")

        while True:
            username = input("Username: ").strip()
            username = username if username != "" else None

            if username is not None:
                break
            print("Username cannot be empty. Please try again.")

        while True:
            if not sys.stdin.isatty():
                password = input("Password: ").strip()
            else:
                password = getpass("Password: ").strip()

            password = password if password != "" else None

            if password is not None:
                break
            print("Password cannot be empty. Please try again.")

    download_h36m(
        args.directory,
        username=username,
        password=password,
        verify=not args.no_checksum,
    )
