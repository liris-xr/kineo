# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import tarfile
from zipfile import ZipFile
import hashlib
import os
from typing import BinaryIO, Union
from tqdm import tqdm


def compute_md5_checksum(file: Union[BinaryIO, str]) -> str:
    should_close = False

    if isinstance(file, str):
        file = open(file, "rb")
        should_close = True

    file_size = os.fstat(file.fileno()).st_size

    hash_md5 = hashlib.md5()

    pbar = tqdm(
        desc="Computing MD5 checksum",
        total=file_size,
        unit="B",
        unit_scale=True,
        leave=False,
    )

    for chunk in iter(lambda: file.read(4096), b""):
        pbar.update(len(chunk))
        hash_md5.update(chunk)

    pbar.close()

    if should_close:
        file.close()

    return hash_md5.hexdigest()


def load_checksums(checksums_filepath: str) -> dict[str, str]:
    checksums = {}
    with open(checksums_filepath, "r") as f:
        for line in f.read().splitlines(keepends=False):
            v, k = line.split(" ")
            checksums[k] = v
    return checksums


def extract_file_from_zip(zip_file: str, file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with ZipFile(zip_file, "r") as f:
        f.extract(file, output_dir)


def _track_progress(
    tar: tarfile.TarFile,
    pbar: tqdm,
    strip_components: int = 0,
    force_extract: bool = False,
):
    members = tar.getmembers()
    pbar.total = len(members)
    for member in members:
        path_parts = member.name.split("/")
        n_parts = len(path_parts)
        # Don't strip the last part of the path (file name)
        strip_components = min(strip_components, n_parts - 1)
        path_parts = path_parts[strip_components:]
        new_path = "/".join(path_parts)

        if os.path.exists(new_path) and not force_extract:
            continue

        member.name = new_path
        yield member
        pbar.update(1)
    pbar.close()


def extract_tar_with_progress(
    tar: str,
    output_dir: str,
    desc: str = "Extracting",
    strip_components: int = 0,
    force_extract: bool = False,
    leave: bool = True,
):
    if not os.path.exists(tar):
        raise FileNotFoundError(f"File {tar} not found")

    if tar.endswith(".tar.gz"):
        mode = "r:gz"
    elif tar.endswith(".tar.bz2"):
        mode = "r:bz2"
    elif tar.endswith(".tar.xz"):
        mode = "r:xz"
    elif tar.endswith(".tar"):
        mode = "r"
    else:
        raise ValueError(f"Unsupported tar file extension: {tar}")

    os.makedirs(output_dir, exist_ok=True)
    with tarfile.open(tar, mode) as tar_file:
        pbar = tqdm(desc=desc, leave=leave)
        pbar.update(0)
        tar_file.extractall(
            output_dir,
            members=_track_progress(tar_file, pbar, strip_components, force_extract),
        )
