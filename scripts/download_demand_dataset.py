"""Script to download DEMAND environmental noise recordings.

DEMAND is distributed under CC BY-SA 3.0 (https://zenodo.org/records/1227121);
work using it must cite Thiemann, Ito and Vincent, "The Diverse Environments
Multi-channel Acoustic Noise Database (DEMAND)", ICA 2013.
"""

import argparse

from kineo.datasets.demand.demand_download import (
    DEMAND_ENVIRONMENTS,
    DEMAND_SAMPLE_RATES,
    download_demand,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory where the dataset is downloaded",
    )
    parser.add_argument(
        "--environments",
        type=str,
        nargs="+",
        choices=DEMAND_ENVIRONMENTS,
        default=["PCAFETER", "NPARK"],
        help="Environments to download",
    )
    parser.add_argument(
        "--sample-rate-khz",
        type=int,
        choices=DEMAND_SAMPLE_RATES,
        default=48,
        help="48 kHz matches the AIST++ camera audio",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download archives already present",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Whether to skip unpacking the archives",
    )
    args = parser.parse_args()
    download_demand(
        args.directory,
        environments=args.environments,
        sample_rate_khz=args.sample_rate_khz,
        force_download=args.force_download,
        skip_extract=args.skip_extract,
    )
