# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

try:
    from ego4d.internal.download.cli import create_arg_parse, main as download_main
except ImportError:
    raise ImportError(
        "ego4d is not installed, please install it using `pip install ego4d`"
    )

try:
    from awscli.clidriver import create_clidriver
except ImportError:
    raise ImportError(
        "awscli is not installed, please install it using `pip install awscli`"
    )

from argparse import ArgumentParser


def download_egoexo4d(
    output_dir: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str | None = None,
):
    driver = create_clidriver()
    driver.main(
        [
            "configure",
            "set",
            "aws_access_key_id",
            aws_access_key_id,
        ]
    )
    driver.main(
        [
            "configure",
            "set",
            "aws_secret_access_key",
            aws_secret_access_key,
        ]
    )

    if aws_region is not None:
        driver.main(
            [
                "configure",
                "set",
                "aws_region",
                aws_region,
            ]
        )

    parser: ArgumentParser = create_arg_parse(
        script_name="egoexo",
        release_name="v2",
        base_dir="s3://ego4d-consortium-sharing/egoexo-public/",
    )
    args = parser.parse_args(
        args=[
            "-o",
            output_dir,
            "--parts",
            "annotations",
            "metadata",
            "takes",
            "take_trajectory",
            "--splits",
            "val",
            "--views",
            "exo",
            "--benchmarks",
            "egopose",
        ]
    )
    download_main(args)


if __name__ == "__main__":
    # Prompt user for aws credentials
    aws_access_key_id = input("Enter your AWS access key ID: ")
    aws_secret_access_key = input("Enter your AWS secret access key: ")
    aws_region = input("Enter your AWS region (default: None): ")

    if aws_region.strip() == "":
        aws_region = None

    download_egoexo4d(
        output_dir="/mnt/hdd_storage/Charles_JAVERLIAT/Datasets/EgoExo4D",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
    )
