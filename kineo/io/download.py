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
import requests
import os
from typing import Optional
import warnings
import re
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


from kineo.io.file import compute_md5_checksum

GDRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def sanitize_windows_path_part(part: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', part)

def sanitize_windows_path(path: Path) -> Path:
    path = path.absolute()
    parts = path.parts
    sanitized_parts = [parts[0]] + [sanitize_windows_path_part(p) for p in parts[1:]]
    return Path(*sanitized_parts)


def download_gdrive_file_or_folder(
    file_id: str,
    ouput_filepath: Path | str,
    recursive: bool = True,
    credentials_filepath: str = "./credentials.json",
    token_filepath: str = "./token.json",
    force_download: bool = False,
) -> bytes:
    """Shows basic usage of the Docs API.
    Prints the title of a sample document.
    """

    if isinstance(ouput_filepath, str):
        ouput_filepath = Path(ouput_filepath)

    if os.name == "nt":
        ouput_filepath = sanitize_windows_path(ouput_filepath)

    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(token_filepath):
        creds = Credentials.from_authorized_user_file(token_filepath, GDRIVE_SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_filepath, GDRIVE_SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(token_filepath, "w") as token:
            token.write(creds.to_json())

    service = build("drive", "v3", credentials=creds)

    file = service.files().get(fileId=file_id).execute()
    if file["mimeType"] == "application/vnd.google-apps.folder":
        for child in (
            service.files().list(q=f"'{file_id}' in parents").execute()["files"]
        ):
            if (
                child["mimeType"] == "application/vnd.google-apps.folder"
                and not recursive
            ):
                continue

            download_gdrive_file_or_folder(
                child["id"], ouput_filepath / child["name"], recursive
            )
        return

    remote_file_size = (
        service.files().get(fileId=file_id, fields="size").execute().get("size", None)
    )
    remote_file_size = int(remote_file_size) if remote_file_size is not None else None

    os.makedirs(ouput_filepath.parent, exist_ok=True)

    local_file_size = (
        os.path.getsize(ouput_filepath) if os.path.exists(ouput_filepath) else None
    )

    if local_file_size == remote_file_size and not force_download:
        print(
            f"Skipping '{ouput_filepath.name}' because it is already downloaded. "
            f"Use '--force-download' to download it again."
        )
        return

    resume_download = (
        local_file_size is not None
        and remote_file_size is not None
        and local_file_size < remote_file_size
        and not force_download
    )

    if resume_download:
        print(
            f"Resuming download of '{ouput_filepath.name}' (partially downloaded at {local_file_size / remote_file_size * 100:.2f}%)"
        )

    request = service.files().get_media(fileId=file_id)

    pbar = tqdm(
        initial=local_file_size if resume_download else 0,
        total=remote_file_size,
        desc=f"Downloading '{ouput_filepath.name}'",
        unit="B",
        unit_scale=True,
        leave=False,
    )

    with open(ouput_filepath, "ab" if resume_download else "wb") as f:
        if resume_download:
            f.seek(local_file_size)
        downloader = MediaIoBaseDownload(f, request)
        done = False

        while done is False:
            status, done = downloader.next_chunk()
            if pbar.total is None:
                pbar.total = status.total_size
            pbar.n = f.tell()
            pbar.refresh()

    pbar.close()


def get_file_size(session: requests.Session, file_url: str) -> int:
    head = session.head(file_url, allow_redirects=True)

    if head.status_code != 200:
        raise Exception(
            f"Failed to get headers for {file_url}: {head.status_code} {head.reason}"
        )

    file_size = head.headers.get("Content-Length", None)
    return int(file_size) if file_size is not None else None


def download_file(
    session: requests.Session,
    file_url: str,
    output_file: str,
    md5_checksum: Optional[str] = None,
    use_remote_file_size: bool = True,
    verbose: bool = False,
    force_download: bool = False,
):
    remote_file_size = None
    if use_remote_file_size:
        try:
            remote_file_size = get_file_size(session, file_url)
        except Exception as e:
            warnings.warn(f"Unable to get the file size: {e}")

    local_file_exists = os.path.exists(output_file)
    local_file_size = os.path.getsize(output_file) if local_file_exists else None

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    if (
        remote_file_size is not None
        and local_file_size is not None
        and local_file_size == remote_file_size
        and not force_download
    ):
        # Ensure checksum is correct
        if md5_checksum is not None:
            with open(output_file, "rb") as f:
                file_checksum = compute_md5_checksum(f)
            if file_checksum != md5_checksum:
                # File already downloaded but checksum mismatch. Redownloading.
                local_file_size = None
                if verbose:
                    print(
                        f"Checksum mismatch for {output_file}. Expected {md5_checksum}, got {file_checksum}. Redownloading."
                    )
            else:
                # File already downloaded. Skipping.
                if verbose:
                    print(
                        f"Skipping {output_file} because it already exists and checksum is correct."
                    )
                return
        else:
            # File already downloaded. Skipping.
            if verbose:
                print(f"Skipping {output_file} because it is already downloaded.")
            return

    if local_file_size is None:
        local_file_size = 0

    partial_download = remote_file_size is not None and not force_download

    if partial_download:
        download_response = session.get(
            file_url,
            headers={"Range": f"bytes={local_file_size}-"},
            stream=True,
            allow_redirects=True,
        )
    else:
        download_response = session.get(file_url, stream=True, allow_redirects=True)
    if download_response.status_code != 200 and download_response.status_code != 206:
        raise Exception(
            f"Failed to download {file_url}: {download_response.status_code} {download_response.reason}"
        )

    if remote_file_size is not None:
        if download_response.status_code == 206:
            # Partial content response
            total_file_size = remote_file_size + local_file_size
        else:
            total_file_size = remote_file_size
    else:
        total_file_size = -1

    mode = "ab" if partial_download else "wb"
    with open(output_file, mode) as f:
        pbar = tqdm(
            total=total_file_size,
            initial=local_file_size,
            desc=f"Downloading {file_url}",
            unit="B",
            unit_scale=True,
            leave=False,
        )
        for chunk in download_response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
        pbar.close()

    if remote_file_size is not None:
        local_file_size = (
            os.path.getsize(output_file) if os.path.exists(output_file) else None
        )

        if local_file_size != remote_file_size:
            raise Exception(
                f"File size mismatch for {output_file}. Expected {remote_file_size}B, got {local_file_size}B"
            )

    # Ensure checksum is correct
    if md5_checksum is not None:
        with open(output_file, "rb") as f:
            file_checksum = compute_md5_checksum(f)
        if file_checksum != md5_checksum:
            raise Exception(
                f"Checksum mismatch for {output_file}. Expected {md5_checksum}, got {file_checksum}"
            )
