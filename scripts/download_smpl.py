"""Download the SMPL neutral body model.

Requires a free account at https://smpl.is.tue.mpg.de/register.php
"""
import getpass
import os
import shutil
import sys
import urllib.parse
import zipfile
from tqdm import tqdm

DOWNLOAD_URL = "https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip&resume=1"
PKL_IN_ZIP = "SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"

DEST_DIR = os.path.join("body_models", "smpl")
DEST_PKL = os.path.join(DEST_DIR, "SMPL_NEUTRAL.pkl")
ZIP_PATH = os.path.join(DEST_DIR, "SMPL_python_v.1.1.0.zip")


def main():
    try:
        import requests
    except ImportError:
        print("ERROR: pip install requests")
        sys.exit(1)

    if os.path.exists(DEST_PKL):
        print(f"{DEST_PKL} already exists, skipping.")
        return

    print("SMPL — register at https://smpl.is.tue.mpg.de/register.php")
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

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        with zf.open(PKL_IN_ZIP) as src, open(DEST_PKL, "wb") as dst:
            shutil.copyfileobj(src, dst)
    os.remove(ZIP_PATH)
    print(f"  Saved to {DEST_PKL}")


if __name__ == "__main__":
    main()
