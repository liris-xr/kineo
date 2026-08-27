# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from concurrent import futures
from typing import Sequence
from tqdm import tqdm
from urllib3.util.retry import Retry
import os
import re
import requests
import threading
import zipfile

from kineo.io.download import download_file

# The AIST Dance Video Database publishes, for each video variant and bitrate,
# a headerless CSV holding one video URL per line.
AIST_VIDEO_LIST_URL = (
    "https://aistdancedb.ongaaccel.jp/data/video_{variant}/{bitrate}/"
    "{variant}_{bitrate}_all_video_url.csv"
)
AIST_VIDEO_METADATA_URL = (
    "https://aistdancedb.ongaaccel.jp/data/video_{variant}/{bitrate}/"
    "{variant}_{bitrate}_{genre}_all.csv"
)
AIST_TERMS_OF_USE_URL = "https://aistdancedb.ongaaccel.jp/terms_of_use/"
# The Terms of Use require an application form to be filed before any use
# of the database, which no amount of terminal prompting can stand in for.
AIST_APPLICATION_FORM_URL = "https://forms.gle/9nVAxPFUhXNPrKQ5A"

AISTPP_ANNOTATIONS_BASE_URL = (
    "https://github.com/google/aistplusplus_dataset/releases/download/v1.0/"
)
AISTPP_ANNOTATION_FILES = (
    "cameras.zip",
    "motions.zip",
    "keypoints2d.zip",
    "keypoints3d.zip",
    "splits.zip",
    "ignore_list.txt",
)

# "raw" videos hold the full recording, "refined" ones are trimmed to the
# dancing part and their noisy audio is replaced by the original music track.
# The AIST++ annotations are aligned with the "refined" videos.
VIDEO_VARIANTS = ("raw", "refined")
VIDEO_BITRATES = ("10M", "2M")

# Dance situations, used to index the per-genre metadata CSVs. Only "sBM" and
# "sFM" are covered by the AIST++ annotations.
VIDEO_GENRES = ("sBM", "sBT", "sCY", "sFM", "sGR", "sMM", "sSH")

# AIST++ ships two unrelated split families: "pose_*" partitions the annotated
# sequences for 3D pose estimation, "crossmodal_*" splits them for music-to-
# dance generation and covers only 1020 of the 1408 sequences.
AISTPP_SPLITS = (
    "all",
    "pose_train",
    "pose_val",
    "pose_test",
    "crossmodal_train",
    "crossmodal_val",
    "crossmodal_test",
)

_CAMERA_FIELD_RE = re.compile(r"^c\d+$")


def _make_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        max_retries=Retry(total=3, status_forcelist=[500, 502, 503, 504])
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _sequence_name(video_name: str) -> str:
    """Maps a video base name to the sequence name the annotations use.

    Args:
        video_name: Video base name, e.g. "gBR_sBM_c01_d04_mBR0_ch01".

    Returns:
        The same name with its camera field replaced by "cAll", e.g.
        "gBR_sBM_cAll_d04_mBR0_ch01". Names holding no camera field, which
        belong to situations AIST++ does not annotate, are returned unchanged.
    """
    fields = video_name.split("_")
    return "_".join(
        "cAll" if _CAMERA_FIELD_RE.match(field) else field for field in fields
    )


def _load_split_sequences(
    session: requests.Session,
    annotations_dir: str,
    split: str,
    drop_ignored: bool,
    force_download: bool,
) -> set[str]:
    """Reads the sequence names of a split, dropping the ignored ones.

    Downloads the two small annotation files the split is read from if they are
    not already there, so a split can be selected without pulling the multi-
    gigabyte keypoint archives.

    Args:
        session: Session used to fetch the split files if they are missing.
        annotations_dir: Directory holding the AIST++ annotation files.
        split: Name of the split, one of `AISTPP_SPLITS`.
        drop_ignored: Whether to drop the sequences AIST++ flags as poorly
            reconstructed in its `ignore_list.txt`.
        force_download: Whether to re-download the split files.

    Returns:
        Sequence names of the split, e.g. "gBR_sBM_cAll_d04_mBR0_ch01".
    """
    os.makedirs(annotations_dir, exist_ok=True)

    for filename in ("splits.zip", "ignore_list.txt"):
        download_file(
            session,
            AISTPP_ANNOTATIONS_BASE_URL + filename,
            os.path.join(annotations_dir, filename),
            force_download=force_download,
        )

    with zipfile.ZipFile(os.path.join(annotations_dir, "splits.zip")) as archive:
        sequences = set(archive.read(f"splits/{split}.txt").decode().split())

    if not drop_ignored:
        return sequences

    with open(os.path.join(annotations_dir, "ignore_list.txt")) as f:
        ignored = set(f.read().split())

    return sequences - ignored


def _fetch_video_urls(
    session: requests.Session,
    variant: str,
    bitrate: str,
) -> list[str]:
    list_url = AIST_VIDEO_LIST_URL.format(variant=variant, bitrate=bitrate)
    response = session.get(list_url)

    if response.status_code != 200:
        raise Exception(
            f"Failed to fetch the video list {list_url}: "
            f"{response.status_code} {response.reason}"
        )

    return [line.strip() for line in response.text.splitlines() if line.strip()]


def _download_videos(
    output_dir: str,
    variant: str,
    bitrate: str,
    sequences: set[str] | None,
    num_workers: int,
    force_download: bool,
):
    video_urls = _fetch_video_urls(_make_session(), variant, bitrate)

    if sequences is not None:
        video_urls = [
            url
            for url in video_urls
            if _sequence_name(os.path.basename(url)[: -len(".mp4")]) in sequences
        ]

    os.makedirs(output_dir, exist_ok=True)

    # requests sessions are not thread-safe, so give each worker its own.
    worker_state = threading.local()

    def _download_video(video_url: str):
        if not hasattr(worker_state, "session"):
            worker_state.session = _make_session()
        download_file(
            worker_state.session,
            video_url,
            os.path.join(output_dir, os.path.basename(video_url)),
            force_download=force_download,
        )

    pbar = tqdm(
        total=len(video_urls),
        desc=f"Downloading AIST {variant} videos ({bitrate})",
        unit="video",
    )

    with futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = [executor.submit(_download_video, url) for url in video_urls]
        for task in futures.as_completed(tasks):
            task.result()
            pbar.update(1)

    pbar.close()


def _download_metadata(output_dir: str, bitrate: str, force_download: bool):
    session = _make_session()
    os.makedirs(output_dir, exist_ok=True)

    metadata_files = [
        (variant, genre) for variant in VIDEO_VARIANTS for genre in VIDEO_GENRES
    ]

    pbar = tqdm(
        total=len(metadata_files),
        desc="Downloading AIST video metadata",
        unit="file",
    )

    for variant, genre in metadata_files:
        metadata_url = AIST_VIDEO_METADATA_URL.format(
            variant=variant, bitrate=bitrate, genre=genre
        )
        download_file(
            session,
            metadata_url,
            os.path.join(output_dir, os.path.basename(metadata_url)),
            force_download=force_download,
        )
        pbar.update(1)

    pbar.close()


def _download_annotations(output_dir: str, force_download: bool):
    session = _make_session()
    os.makedirs(output_dir, exist_ok=True)

    pbar = tqdm(
        total=len(AISTPP_ANNOTATION_FILES),
        desc="Downloading AIST++ annotations",
        unit="file",
    )

    for annotation_file in AISTPP_ANNOTATION_FILES:
        download_file(
            session,
            AISTPP_ANNOTATIONS_BASE_URL + annotation_file,
            os.path.join(output_dir, annotation_file),
            force_download=force_download,
        )
        pbar.update(1)

    pbar.close()


def download_aistpp(
    output_dir: str,
    variants: Sequence[str] = VIDEO_VARIANTS,
    bitrate: str = "10M",
    split: str = "pose_test",
    drop_ignored: bool = True,
    annotations: bool = True,
    metadata: bool = True,
    num_workers: int = 1,
    force_download: bool = False,
):
    """Downloads the AIST++ annotations and the AIST Dance Database videos.

    Videos are laid out as '<output_dir>/videos/<variant>/*.mp4', the AIST
    per-genre video metadata as '<output_dir>/metadata/*.csv' and the
    annotation archives as '<output_dir>/annotations/*'. Downloading the videos
    implies accepting the AIST Dance Video Database Terms of Use, available at
    https://aistdancedb.ongaaccel.jp/terms_of_use/.

    Args:
        output_dir: Directory where the dataset is downloaded.
        variants: Video variants to download, among "raw" and "refined". The
            AIST++ annotations are aligned with the "refined" videos.
        bitrate: Video bitrate, either "10M" or "2M". Raw videos weigh 421GB
            at 10M and 66GB at 2M over the whole annotated set.
        split: Only download the videos of this AIST++ split, minus the
            sequences of its `ignore_list.txt`. "all" keeps every annotated
            sequence; the videos of the situations AIST++ does not annotate are
            never downloaded.
        drop_ignored: Whether to skip the sequences AIST++ flags as poorly
            reconstructed. Their videos are fine, so a study of the videos
            alone rather than of the pose annotations can keep them.
        annotations: Whether to download the AIST++ annotation archives.
        metadata: Whether to download the per-genre video metadata CSVs. They
            hold the duration of every raw and refined video, from which the
            trimmed-away pre-roll and the tempo of each musical piece
            (BPM = 960 / refined duration) are derived.
        num_workers: Number of videos downloaded concurrently.
        force_download: Whether to download files that are already complete.

    Raises:
        ValueError: If an unknown video variant, bitrate or split is requested.
    """
    unknown_variants = set(variants) - set(VIDEO_VARIANTS)
    if unknown_variants:
        raise ValueError(
            f"Unknown video variants {sorted(unknown_variants)}, "
            f"expected any of {list(VIDEO_VARIANTS)}."
        )

    if bitrate not in VIDEO_BITRATES:
        raise ValueError(
            f"Unknown video bitrate '{bitrate}', "
            f"expected any of {list(VIDEO_BITRATES)}."
        )

    if split not in AISTPP_SPLITS:
        raise ValueError(
            f"Unknown split '{split}', expected any of {list(AISTPP_SPLITS)}."
        )

    if annotations:
        _download_annotations(os.path.join(output_dir, "annotations"), force_download)

    if metadata:
        _download_metadata(
            os.path.join(output_dir, "metadata"), bitrate, force_download
        )

    if not variants:
        return

    sequences = _load_split_sequences(
        _make_session(),
        os.path.join(output_dir, "annotations"),
        split,
        drop_ignored,
        force_download,
    )
    print(f"Split '{split}' holds {len(sequences)} sequences.")

    for variant in variants:
        _download_videos(
            os.path.join(output_dir, "videos", variant),
            variant,
            bitrate,
            sequences,
            num_workers,
            force_download,
        )
