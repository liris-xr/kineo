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
import time
import zipfile

from kineo.io.download import HTTP_TIMEOUT_S, download_file

AIST_VIDEO_LIST_URL = (
    "https://aistdancedb.ongaaccel.jp/data/video_{variant}/{bitrate}/"
    "{variant}_{bitrate}_all_video_url.csv"
)
AIST_VIDEO_METADATA_URL = (
    "https://aistdancedb.ongaaccel.jp/data/video_{variant}/{bitrate}/"
    "{variant}_{bitrate}_{genre}_all.csv"
)
AIST_TERMS_OF_USE_URL = "https://aistdancedb.ongaaccel.jp/terms_of_use/"
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

# Refined videos carry music, not camera audio
VIDEO_VARIANTS = ("raw", "refined")
VIDEO_BITRATES = ("10M", "2M")

# AIST++ only annotates sBM and sFM
VIDEO_GENRES = ("sBM", "sBT", "sCY", "sFM", "sGR", "sMM", "sSH")

DOWNLOAD_ATTEMPTS = 4

# AIST++ ignore list plus Kineo additions
IGNORE_LIST_PATH = os.path.join(os.path.dirname(__file__), "aistpp_ignore_list.txt")

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


def load_ignored_sequences() -> set[str]:
    with open(IGNORE_LIST_PATH) as f:
        return {line.strip() for line in f if line.strip() and not line.startswith("#")}


def _sequence_name(video_name: str) -> str:
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
    os.makedirs(annotations_dir, exist_ok=True)

    download_file(
        session,
        AISTPP_ANNOTATIONS_BASE_URL + "splits.zip",
        os.path.join(annotations_dir, "splits.zip"),
        force_download=force_download,
    )

    with zipfile.ZipFile(os.path.join(annotations_dir, "splits.zip")) as archive:
        sequences = set(archive.read(f"splits/{split}.txt").decode().split())

    if not drop_ignored:
        return sequences

    return sequences - load_ignored_sequences()


def _fetch_video_urls(
    session: requests.Session,
    variant: str,
    bitrate: str,
) -> list[str]:
    list_url = AIST_VIDEO_LIST_URL.format(variant=variant, bitrate=bitrate)
    response = session.get(list_url, timeout=HTTP_TIMEOUT_S)

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
    sequences: set[str],
    num_workers: int,
    force_download: bool,
):
    video_urls = [
        url
        for url in _fetch_video_urls(_make_session(), variant, bitrate)
        if _sequence_name(os.path.basename(url)[: -len(".mp4")]) in sequences
    ]

    os.makedirs(output_dir, exist_ok=True)

    # requests sessions are not thread-safe
    worker_state = threading.local()

    def _download_video(video_url: str):
        if not hasattr(worker_state, "session"):
            worker_state.session = _make_session()

        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                download_file(
                    worker_state.session,
                    video_url,
                    os.path.join(output_dir, os.path.basename(video_url)),
                    force_download=force_download and attempt == 0,
                )
                return
            except Exception as error:
                if attempt == DOWNLOAD_ATTEMPTS - 1:
                    raise
                tqdm.write(
                    f"Retrying {os.path.basename(video_url)} "
                    f"({attempt + 1}/{DOWNLOAD_ATTEMPTS - 1}): {error}"
                )
                time.sleep(2**attempt)

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
    """Download the AIST++ annotations and the AIST Dance Database videos."""
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
