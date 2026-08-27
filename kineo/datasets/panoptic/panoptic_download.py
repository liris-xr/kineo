# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

"""Download of the CMU Panoptic Studio sequences the TEMPO protocol uses.

The Panoptic Studio publishes 480 VGA and 31 HD views per sequence, plus face,
hand and mesh annotations, which run to terabytes. The TEMPO protocol reads
five HD views of thirteen sequences, so only those videos, the calibration and
the 3D body keypoints are fetched: about 16GB for the test split.

The on-disk layout matches `panoptic-toolbox/scripts/getData.sh`, so a
directory populated by that script is a valid input to the preprocessing, and
one populated here can be extended with the toolbox.
"""

import os
import threading
import time
from concurrent import futures
from typing import Sequence

import requests
from tqdm import tqdm
from urllib3.util.retry import Retry

from kineo.io.download import download_file

PANOPTIC_BASE_URL = "http://domedb.perception.cs.cmu.edu/webdata/dataset"

PANOPTIC_CALIBRATION_URL = (
    PANOPTIC_BASE_URL + "/{sequence}/calibration_{sequence}.json"
)
PANOPTIC_KEYPOINTS_3D_URL = (
    PANOPTIC_BASE_URL + "/{sequence}/hdPose3d_stage1_coco19.tar"
)
# HD videos live under a re-encode directory rather than next to the
# annotations, which is what `getData.sh` reads and what the more obvious
# '<sequence>/hdVideos/' path 404s on.
PANOPTIC_HD_VIDEO_URL = (
    PANOPTIC_BASE_URL
    + "/{sequence}/videos/hd_shared_crf20/hd_00_{node:02d}.mp4"
)

# The Panoptic Studio holds its 31 HD cameras on a single panel.
HD_PANEL = 0
N_HD_CAMERAS = 31

# The five HD views VoxelPose introduced and TEMPO, Faster VoxelPose, MvP and
# PlaneSweepPose all reuse.
TEMPO_HD_CAMERAS = (3, 6, 12, 13, 23)

TEMPO_TRAIN_SEQUENCES = (
    "160422_ultimatum1",
    "160224_haggling1",
    "160226_haggling1",
    "161202_haggling1",
    "160906_ian1",
    "160906_ian2",
    "160906_ian3",
    "160906_band1",
    "160906_band2",
)

TEMPO_TEST_SEQUENCES = (
    "160906_pizza1",
    "160422_haggling1",
    "160906_ian5",
    "160906_band4",
)

PANOPTIC_SPLITS = ("test", "train", "all")

# Sequences Kineo excludes; the file's own header says why.
IGNORE_LIST_PATH = os.path.join(
    os.path.dirname(__file__), "panoptic_ignore_list.txt"
)

# A stalled transfer now raises rather than hanging, and a split is too long a
# download for one flaky file to end. Each attempt resumes where the last
# stopped.
DOWNLOAD_ATTEMPTS = 4


def _make_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=Retry(total=3, status_forcelist=[500, 502, 503, 504])
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def load_ignored_sequences() -> set[str]:
    """Reads the sequence names Kineo excludes from the Panoptic benchmarks.

    Returns:
        Sequence names, as the Panoptic Studio spells them.
    """
    with open(IGNORE_LIST_PATH) as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        }


def split_sequences(split: str, drop_ignored: bool = True) -> list[str]:
    """Lists the sequences of a TEMPO split.

    Args:
        split: Name of the split, one of `PANOPTIC_SPLITS`.
        drop_ignored: Whether to drop the sequences of `IGNORE_LIST_PATH`.

    Returns:
        Sequence names, in the order the TEMPO configuration lists them.

    Raises:
        ValueError: If an unknown split is requested.
    """
    if split not in PANOPTIC_SPLITS:
        raise ValueError(
            f"Unknown split '{split}', expected any of {list(PANOPTIC_SPLITS)}."
        )

    if split == "train":
        sequences = list(TEMPO_TRAIN_SEQUENCES)
    elif split == "test":
        sequences = list(TEMPO_TEST_SEQUENCES)
    else:
        sequences = list(TEMPO_TRAIN_SEQUENCES) + list(TEMPO_TEST_SEQUENCES)

    if not drop_ignored:
        return sequences

    ignored = load_ignored_sequences()
    return [sequence for sequence in sequences if sequence not in ignored]


def _download_jobs(
    jobs: list[tuple[str, str]],
    desc: str,
    num_workers: int,
    force_download: bool,
):
    """Downloads (url, output path) pairs concurrently, retrying each."""
    # requests sessions are not thread-safe, so give each worker its own.
    worker_state = threading.local()

    def _download(job: tuple[str, str]):
        url, output_path = job

        if not hasattr(worker_state, "session"):
            worker_state.session = _make_session()

        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                download_file(
                    worker_state.session,
                    url,
                    output_path,
                    force_download=force_download and attempt == 0,
                )
                return
            except Exception as error:
                if attempt == DOWNLOAD_ATTEMPTS - 1:
                    raise
                tqdm.write(
                    f"Retrying {os.path.basename(output_path)} "
                    f"({attempt + 1}/{DOWNLOAD_ATTEMPTS - 1}): {error}"
                )
                time.sleep(2**attempt)

    pbar = tqdm(total=len(jobs), desc=desc, unit="file")

    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = [executor.submit(_download, job) for job in jobs]
        for task in futures.as_completed(tasks):
            task.result()
            pbar.update(1)

    pbar.close()


def download_panoptic(
    output_dir: str,
    split: str = "test",
    cameras: Sequence[int] = TEMPO_HD_CAMERAS,
    drop_ignored: bool = True,
    annotations: bool = True,
    videos: bool = True,
    num_workers: int = 1,
    force_download: bool = False,
):
    """Downloads the CMU Panoptic sequences of a TEMPO split.

    Each sequence is laid out as '<output_dir>/<sequence>/', holding
    'calibration_<sequence>.json', 'hdPose3d_stage1_coco19.tar' and
    'hdVideos/hd_00_<node>.mp4' for each requested camera. Only the requested
    HD views are fetched: the other HD views, the 480 VGA views and the face,
    hand and mesh annotations are never touched.

    Args:
        output_dir: Directory where the dataset is downloaded.
        split: TEMPO split to download, one of `PANOPTIC_SPLITS`. Over the five
            protocol views the test split weighs about 16GB and the train split
            about 40GB.
        cameras: HD camera nodes to download, in 0..30. The TEMPO protocol
            reads `TEMPO_HD_CAMERAS`.
        drop_ignored: Whether to skip the sequences of `IGNORE_LIST_PATH`.
        annotations: Whether to download the calibration and the 3D body
            keypoint archive of each sequence.
        videos: Whether to download the HD videos.
        num_workers: Number of files downloaded concurrently.
        force_download: Whether to download files that are already complete.

    Raises:
        ValueError: If an unknown split or camera node is requested.
    """
    sequences = split_sequences(split, drop_ignored)

    unknown_cameras = set(cameras) - set(range(N_HD_CAMERAS))
    if unknown_cameras:
        raise ValueError(
            f"Unknown HD camera nodes {sorted(unknown_cameras)}, "
            f"expected values in 0..{N_HD_CAMERAS - 1}."
        )

    print(f"Split '{split}' holds {len(sequences)} sequences.")

    jobs: list[tuple[str, str]] = []

    for sequence in sequences:
        sequence_dir = os.path.join(output_dir, sequence)

        if annotations:
            jobs.append(
                (
                    PANOPTIC_CALIBRATION_URL.format(sequence=sequence),
                    os.path.join(sequence_dir, f"calibration_{sequence}.json"),
                )
            )
            jobs.append(
                (
                    PANOPTIC_KEYPOINTS_3D_URL.format(sequence=sequence),
                    os.path.join(sequence_dir, "hdPose3d_stage1_coco19.tar"),
                )
            )

        if videos:
            for node in cameras:
                jobs.append(
                    (
                        PANOPTIC_HD_VIDEO_URL.format(
                            sequence=sequence, node=node
                        ),
                        os.path.join(
                            sequence_dir,
                            "hdVideos",
                            f"hd_{HD_PANEL:02d}_{node:02d}.mp4",
                        ),
                    )
                )

    if not jobs:
        return

    _download_jobs(
        jobs,
        desc=f"Downloading CMU Panoptic ({split})",
        num_workers=num_workers,
        force_download=force_download,
    )
