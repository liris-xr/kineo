# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import os
import zipfile
from typing import Sequence

import requests

from kineo.io.download import download_file

DEMAND_ZENODO_URL = "https://zenodo.org/records/1227121/files"

DEMAND_ENVIRONMENTS = (
    "DKITCHEN", "DLIVING", "DWASHING",
    "NFIELD", "NPARK", "NRIVER",
    "OHALLWAY", "OMEETING", "OOFFICE",
    "PCAFETER", "PRESTO", "PSTATION",
    "SCAFE", "SPSQUARE", "STRAFFIC",
    "TBUS", "TCAR", "TMETRO",
)
DEMAND_SAMPLE_RATES = (48, 16)  # kHz


def download_demand(
    output_dir: str,
    environments: Sequence[str] = ("PCAFETER", "NPARK"),
    sample_rate_khz: int = 48,
    force_download: bool = False,
    skip_extract: bool = False,
):
    unknown = set(environments) - set(DEMAND_ENVIRONMENTS)
    if unknown:
        raise ValueError(
            f"Unknown environments: {sorted(unknown)}. "
            f"Expected any of {DEMAND_ENVIRONMENTS}."
        )
    if sample_rate_khz not in DEMAND_SAMPLE_RATES:
        raise ValueError(
            f"Unknown sample rate {sample_rate_khz} kHz, expected one of "
            f"{DEMAND_SAMPLE_RATES}."
        )

    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()

    for environment in environments:
        archive = f"{environment}_{sample_rate_khz}k.zip"
        archive_path = os.path.join(output_dir, archive)
        download_file(
            session,
            f"{DEMAND_ZENODO_URL}/{archive}?download=1",
            archive_path,
            verbose=True,
            force_download=force_download,
        )
        if skip_extract:
            continue
        with zipfile.ZipFile(archive_path) as zipped:
            zipped.extractall(output_dir)
