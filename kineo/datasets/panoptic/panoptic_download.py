# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from tqdm import tqdm
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

from urllib.parse import urljoin

from kineo.io.download import download_file
from kineo.io.file import load_checksums

import warnings

CMU_BASE_URL = "http://domedb.perception.cs.cmu.edu/"
SNU_BASE_URL = "http://vcl.snu.ac.kr/panoptic/"

SEQUENCES = [
    # Multi-person training
    "160422_ultimatum1",
    "160224_haggling1",
    "160226_haggling1",
    "161202_haggling1",
    "160906_ian1",
    "160906_ian2",
    "160906_ian3",
    "160906_band1",
    "160906_band2",
    # Multi-person testing
    "160906_pizza1",
    "160422_haggling1",
    "160906_ian5",
    "160906_band4",
    # Single-person testing
    "171204_pose1",
    "171204_pose2",
    "171204_pose3",
    "171204_pose4",
    "171204_pose5",
    "171204_pose6",
    "171026_pose1",
    "171026_pose2",
    "171026_pose3",
]


def _download_hd_video(
    session: requests.Session,
    base_url: str,
    dataset_name: str,
    camera_id: int,
    output_base_dir: str,
    md5_checksum: str | None = None,
):
    filename = f"hd_00_{camera_id:02d}.mp4"

    try:
        download_file(
            session=session,
            file_url=urljoin(
                base_url,
                f"webdata/dataset/{dataset_name}/videos/hd_shared_crf20/{filename}",
            ),
            output_file=os.path.join(
                output_base_dir, dataset_name, "hdVideos", filename
            ),
            md5_checksum=md5_checksum,
        )
    except Exception:
        warnings.warn(f"Missing {filename} for {dataset_name}, skipping...")


def _download_calibration(
    session: requests.Session,
    base_url: str,
    dataset_name: str,
    output_base_dir: str,
    md5_checksum: str | None = None,
):
    filename = f"calibration_{dataset_name}.json"

    try:
        download_file(
            session=session,
            file_url=urljoin(
                base_url,
                f"webdata/dataset/{dataset_name}/{filename}",
            ),
            output_file=os.path.join(output_base_dir, dataset_name, filename),
            md5_checksum=md5_checksum,
        )
    except Exception:
        warnings.warn(f"Missing {filename} for {dataset_name}, skipping...")


def download_panoptic(
    output_base_dir: str,
    verify: bool = True,
    use_snu_endpoint: bool = False,
):
    base_url = SNU_BASE_URL if use_snu_endpoint else CMU_BASE_URL

    session = requests.Session()
    adapter = HTTPAdapter(
        max_retries=Retry(total=3, status_forcelist=[500, 502, 503, 504])
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    hd_cameras = [3, 6, 12, 13, 23]

    dataset_pbar = tqdm(total=len(SEQUENCES), desc="Downloading CMU Panoptic dataset")

    for dataset in SEQUENCES:
        dataset_pbar.set_description(f"Downloading CMU Panoptic dataset ({dataset})")

        try:
            download_file(
                session=session,
                file_url=urljoin(
                    base_url,
                    f"webdata/dataset/{dataset}/hdPose3d_stage1_coco19.tar",
                ),
                output_file=os.path.join(
                    output_base_dir, dataset, "hdPose3d_stage1_coco19.tar"
                ),
            )
        except Exception:
            warnings.warn(
                f"Missing hdPose3d_stage1_coco19.tar for {dataset}, skipping..."
            )

        _download_calibration(
            session=session,
            dataset_name=dataset,
            base_url=base_url,
            output_base_dir=output_base_dir,
            # md5_checksum=checksums[f"{dataset}_calibration_{dataset}.json"] if verify else None,
        )

        for camera in hd_cameras:
            _download_hd_video(
                session=session,
                dataset_name=dataset,
                base_url=base_url,
                camera_id=camera,
                output_base_dir=output_base_dir,
                # md5_checksum=checksums[f"{dataset}_hd_{camera:02d}.mp4"] if verify else None,
            )

        dataset_pbar.update(1)

    dataset_pbar.close()
