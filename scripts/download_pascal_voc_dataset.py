"""Script to download the Pascal VOC 2012 objects used as synthetic occluders.

Pascal VOC is distributed for research use; work using it must cite
Everingham et al., "The PASCAL Visual Object Classes (VOC) Challenge",
IJCV 2010.
"""

import argparse

from kineo.datasets.pascal_voc.pascal_voc_download import download_pascal_voc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=str,
        help="Path to the directory receiving VOCdevkit/VOC2012",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the archive even if present",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the 2 GB tar once extracted",
    )
    args = parser.parse_args()
    download_pascal_voc(
        args.directory,
        force_download=args.force_download,
        keep_archive=args.keep_archive,
    )
