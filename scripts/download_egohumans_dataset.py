"""Script to download the EgoHumans dataset from Google Drive.

When running this script from a server without a GUI, you must provide a valid token.json file
since the Google OAuth login page needs to open in a browser.
"""

import argparse
from kineo.datasets.egohumans.egohumans_download import download_egohumans
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory where the raw dataset will be downloaded",
    )
    parser.add_argument(
        "--token-filepath",
        type=str,
        help="Path to the token.json file.",
        default="./token.json",
    )
    parser.add_argument(
        "--credentials-filepath",
        type=str,
        help="Path to the credentials.json file. Used to generate token.json if not provided.",
        default="./credentials.json",
    )
    args = parser.parse_args()

    if not os.path.exists(args.directory):
        os.makedirs(args.directory, exist_ok=True)

    if os.path.exists(args.token_filepath):
        print(f"Using token file at {args.token_filepath}")
    elif not os.path.exists(args.credentials_filepath):
        raise FileNotFoundError(
            f"Credentials file not found at {args.credentials_filepath} and no token is provided."
            "To download the credentials.json: "
            "   1. Go to https://console.cloud.google.com/auth/clients "
            "   2. Click Create Client"
            "   3. Click Application type > Desktop app"
            "   4. In the Name field, type a name for the credential. This name is only shown in the Google Cloud console."
            "   5. Save the downloaded JSON file as credentials.json, and move the file to your working directory."
        )

    download_egohumans(
        output_base_dir=args.directory,
    )
