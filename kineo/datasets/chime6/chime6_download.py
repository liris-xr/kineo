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
import tarfile
from typing import Sequence

import requests
from tqdm import tqdm

from kineo.io.download import download_file

# OpenSLR resource 150 publishes CHiME-6 under CC BY-SA 4.0, with no
# registration. The three mirrors carry identical files.
CHIME6_MIRRORS = (
    "https://openslr.trmal.net/resources/150/",
    "https://openslr.elda.org/resources/150/",
    "https://openslr.magicdatatech.com/resources/150/",
)

CHIME6_SPLITS = ("dev", "train", "eval")

# The transcripts drive window selection and the floorplans are the source of
# the device positions, so both are fetched whatever the split.
CHIME6_SUPPORT_ARCHIVES = (
    "CHiME6_transcriptions.tar.gz",
    "CHiME6_floorplans.tar.gz",
)


def _archive_name(split: str) -> str:
    return f"CHiME6_{split}.tar.gz"


def _download_archive(
    session: requests.Session,
    archive: str,
    output_dir: str,
    force_download: bool,
):
    """Fetch one archive, falling back to the next mirror on failure."""
    for mirror in CHIME6_MIRRORS:
        try:
            download_file(
                session,
                mirror + archive,
                os.path.join(output_dir, archive),
                force_download=force_download,
            )
            return
        except Exception as e:
            tqdm.write(f"{mirror} failed for {archive}: {e}")
    raise RuntimeError(f"Could not download {archive} from any CHiME-6 mirror")


def _extract_archive(archive: str, output_dir: str):
    """Unpack an archive in place, keeping its own directory structure."""
    archive_path = os.path.join(output_dir, archive)
    with tarfile.open(archive_path) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=f"Extracting {archive}", unit="file"):
            tar.extract(member, output_dir)


def download_chime6(
    output_dir: str,
    splits: Sequence[str] = ("dev",),
    force_download: bool = False,
    skip_extract: bool = False,
):
    """Downloads the CHiME-6 audio, transcriptions and floorplans.

    Args:
        output_dir: Directory where the dataset is downloaded and extracted.
        splits: Splits to download, among "dev", "train" and "eval". The audio
            weighs 11GB for "dev", 97GB for "train" and 12GB for "eval".
        force_download: Re-download archives already present instead of
            resuming or skipping them.
        skip_extract: Whether to skip unpacking the archives.

    Raises:
        ValueError: If a split is not a CHiME-6 split.
        RuntimeError: If an archive could not be fetched from any mirror.
    """
    unknown_splits = set(splits) - set(CHIME6_SPLITS)
    if unknown_splits:
        raise ValueError(
            f"Unknown splits: {sorted(unknown_splits)}. "
            f"Expected any of {sorted(CHIME6_SPLITS)}."
        )

    archives = [_archive_name(split) for split in splits]
    archives.extend(CHIME6_SUPPORT_ARCHIVES)

    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()

    for archive in archives:
        _download_archive(session, archive, output_dir, force_download)

    if skip_extract:
        return

    for archive in archives:
        _extract_archive(archive, output_dir)
