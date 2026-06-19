"""Download the SMPLX neutral body model.

Requires a free account at https://smplx.is.tue.mpg.de/register.php
"""
import getpass
import os
import shutil
import sys
import urllib.parse
import zipfile
from tqdm import tqdm

DOWNLOAD_URL = "https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip&resume=1"
PKL_IN_ZIP = "models/smplx/SMPLX_NEUTRAL.pkl"
NPZ_IN_ZIP = "models/smplx/SMPLX_NEUTRAL.npz"

DEST_DIR = os.path.join("body_models", "smplx")
DEST_PKL = os.path.join(DEST_DIR, "SMPLX_NEUTRAL.pkl")
DEST_NPZ = os.path.join(DEST_DIR, "SMPLX_NEUTRAL.npz")
ZIP_PATH = os.path.join(DEST_DIR, "models_smplx_v1_1.zip")


def main():
    try:
        import requests
    except ImportError:
        print("ERROR: pip install requests")
        sys.exit(1)

    if os.path.exists(DEST_PKL) and os.path.exists(DEST_NPZ):
        print(f"{DEST_PKL} and {DEST_NPZ} already exist, skipping.")
        return

    print("SMPL-X — register at https://smplx.is.tue.mpg.de/register.php")
    username = urllib.parse.quote(input("  Email: ").strip(), safe="")
    password = urllib.parse.quote(getpass.getpass("  Password: "), safe="")

    os.makedirs(DEST_DIR, exist_ok=True)

    print("  Downloading...")
    try:
        with requests.post(
                DOWNLOAD_URL,
                data=f"username={username}&password={password}",
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "User-Agent": "Mozilla/5.0",
                },
                stream=True,
                timeout=120,
                verify=False,
        ) as resp:
            resp.raise_for_status()

            if "text/html" in resp.headers.get("content-type", ""):
                raise RuntimeError(f"Login failed — server returned HTML:\n{resp.text[:2000]}")

            total = int(resp.headers.get("content-length", 0))

            with open(ZIP_PATH, "wb") as f:
                with tqdm(
                        total=total if total > 0 else None,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc="Downloading",
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))

        print()
    except Exception as e:
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        print(f"  ERROR: {e}")
        sys.exit(1)

    zip_size = os.path.getsize(ZIP_PATH)
    if not zipfile.is_zipfile(ZIP_PATH):
        with open(ZIP_PATH, "rb") as f:
            head = f.read(512)
        os.remove(ZIP_PATH)
        print(f"  ERROR: Downloaded file is not a valid ZIP ({zip_size} bytes).")
        if b"<html" in head.lower() or b"<!doctype" in head.lower():
            print("  Looks like an HTML page — login likely failed. Check credentials.")
        else:
            print(f"  First bytes: {head[:64]!r}")
        sys.exit(1)

    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            with zf.open(PKL_IN_ZIP) as src, open(DEST_PKL, "wb") as dst:
                shutil.copyfileobj(src, dst)
                print(f"  Saved to {DEST_PKL}")
            with zf.open(NPZ_IN_ZIP) as src, open(DEST_NPZ, "wb") as dst:
                shutil.copyfileobj(src, dst)
                print(f"  Saved to {DEST_NPZ}")
    except KeyError as e:
        os.remove(ZIP_PATH)
        print(f"  ERROR: Expected file not found in ZIP: {e}")
        sys.exit(1)
    os.remove(ZIP_PATH)


if __name__ == "__main__":
    main()
