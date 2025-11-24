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
import requests
from urllib.parse import urlencode
from urllib3.util.retry import Retry

from urllib.parse import urljoin

from kineo.io.download import download_file
from kineo.io.file import load_checksums

BASE_URL = "http://vision.imar.ro/human3.6m/"

SUBJECTS_NAME_TO_FILE_ID = [
    ("S1", 1),
    ("S5", 6),
    ("S6", 7),
    ("S7", 2),
    ("S8", 3),
    ("S9", 4),
    ("S11", 5),
]


def _login(session: requests.Session, username: str, password: str):
    login_response = session.post(
        urljoin(BASE_URL, "checklogin.php"),
        data={"username": username, "password": password},
    )

    if (
        login_response.status_code != 200
        or login_response.url == "https://vision.imar.ro/human3.6m/checklogin.php"
    ):
        raise Exception("Failed to login.")


def download_h36m(
    output_dir: str,
    username: str,
    password: str,
    verify: bool = True,
):
    checksums = (
        load_checksums(
            os.path.join(os.path.dirname(__file__), "h36m_dataset_checksums.txt")
        )
        if verify
        else {}
    )

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=Retry(total=3, status_forcelist=[500, 502, 503, 504])
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        _login(session, username, password)
    except Exception as e:
        print(f"Failed to login: {e}")
        return

    code_file = os.path.join(output_dir, "code-v1.2.zip")

    download_file(
        session,
        urljoin(BASE_URL, "code-v1.2.zip"),
        code_file,
        md5_checksum=checksums["code-v1.2.zip"] if checksums else None,
    )

    pbar = tqdm(
        total=len(SUBJECTS_NAME_TO_FILE_ID),
        desc="Downloading H3.6M dataset",
        unit="subject",
    )

    for subject_name, subject_file_id in SUBJECTS_NAME_TO_FILE_ID:
        gt_poses_3d_file = os.path.join(
            output_dir, f"Poses_D3_Positions_{subject_name}.tgz"
        )
        videos_file = os.path.join(output_dir, f"Videos_{subject_name}.tgz")
        bboxes_file = os.path.join(output_dir, f"Segments_mat_gt_bb_{subject_name}.tgz")
        meshes_file = os.path.join(output_dir, f"Meshes_{subject_name}.tgz")

        # Download GT 3D poses
        download_file(
            session,
            urljoin(
                BASE_URL,
                "filebrowser.php?{}".format(
                    urlencode(
                        {
                            "download": 1,
                            "filepath": "Poses/D3_Positions",
                            "filename": f"SubjectSpecific_{subject_file_id}.tgz",
                        }
                    )
                ),
            ),
            gt_poses_3d_file,
            md5_checksum=(
                checksums[f"Poses_D3_Positions_{subject_name}.tgz"]
                if checksums
                else None
            ),
        )

        # Download GT bboxes
        download_file(
            session,
            urljoin(
                BASE_URL,
                "filebrowser.php?{}".format(
                    urlencode(
                        {
                            "download": 1,
                            "filepath": "Segments/mat_gt_bb",
                            "filename": f"SubjectSpecific_{subject_file_id}.tgz",
                        }
                    )
                ),
            ),
            bboxes_file,
            md5_checksum=(
                checksums[f"Segments_mat_gt_bb_{subject_name}.tgz"]
                if checksums
                else None
            ),
        )

        # Download GT scans
        download_file(
            session,
            urljoin(
                BASE_URL,
                "filebrowser.php?{}".format(
                    urlencode(
                        {
                            "download": 1,
                            "filepath": "Meshes",
                            "filename": f"{subject_name}.tgz",
                            "downloadname": subject_name,
                        }
                    )
                ),
            ),
            meshes_file,
            md5_checksum=(
                checksums[f"Meshes_{subject_name}.tgz"] if checksums else None
            ),
        )

        # Download videos
        download_file(
            session,
            urljoin(
                BASE_URL,
                "filebrowser.php?{}".format(
                    urlencode(
                        {
                            "download": 1,
                            "filepath": "Videos",
                            "filename": f"SubjectSpecific_{subject_file_id}.tgz",
                        }
                    )
                ),
            ),
            videos_file,
            md5_checksum=checksums[f"Videos_{subject_name}.tgz"] if checksums else None,
        )

        pbar.update(1)

    pbar.close()
