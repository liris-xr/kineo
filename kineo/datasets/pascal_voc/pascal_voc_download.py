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

import requests

from kineo.io.download import download_file

PASCAL_VOC_2012_URL = (
    "https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/"
    "VOCtrainval_11-May-2012.tar"
)
PASCAL_VOC_2012_MD5 = "6cd6e144f989b92b3379bac3b3de84fd"

_ARCHIVE_ROOT = "VOCdevkit/VOC2012"
_SEGMENTED_LIST = f"{_ARCHIVE_ROOT}/ImageSets/Segmentation/trainval.txt"


def download_pascal_voc(
    output_dir: str,
    force_download: bool = False,
    keep_archive: bool = False,
):
    """Download Pascal VOC 2012 trainval and extract its segmented images."""
    os.makedirs(output_dir, exist_ok=True)
    archive_path = os.path.join(output_dir, os.path.basename(PASCAL_VOC_2012_URL))

    download_file(
        requests.Session(),
        PASCAL_VOC_2012_URL,
        archive_path,
        md5_checksum=PASCAL_VOC_2012_MD5,
        verbose=True,
        force_download=force_download,
    )

    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        segmented = archive.extractfile(_SEGMENTED_LIST).read().decode().split()
        wanted_images = {
            f"{_ARCHIVE_ROOT}/JPEGImages/{image_id}.jpg" for image_id in segmented
        }
        archive.extractall(
            output_dir,
            members=[
                member
                for member in members
                if member.name in wanted_images
                or member.name.startswith(
                    (
                        f"{_ARCHIVE_ROOT}/Annotations/",
                        f"{_ARCHIVE_ROOT}/SegmentationObject/",
                        f"{_ARCHIVE_ROOT}/ImageSets/Segmentation/",
                    )
                )
            ],
            filter="data",
        )

    if not keep_archive:
        os.remove(archive_path)
