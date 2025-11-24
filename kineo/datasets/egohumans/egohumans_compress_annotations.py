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
import zipfile
import argparse


def zip_annotations(root_dir, output_zip, compression=zipfile.ZIP_STORED):
    """
    Recursively find all 'annotations' folders in root_dir and zip them
    while preserving the folder hierarchy.
    """
    with zipfile.ZipFile(output_zip, "w", compression) as zipf:
        # Add the main JSON file containing the reference to the sequences
        file_path = os.path.join(root_dir, "egohumans_sequences.json")
        name = os.path.relpath(file_path, root_dir)
        zipf.write(file_path, name)

        for foldername, subfolders, filenames in os.walk(root_dir):
            if os.path.basename(foldername) == "annotations":
                for filename in filenames:
                    file_path = os.path.join(foldername, filename)
                    # Preserve the folder hierarchy relative to root_dir
                    name = os.path.relpath(file_path, root_dir)
                    zipf.write(file_path, name)
                print(f"Added folder: {foldername}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zip all 'annotations' folders recursively"
    )
    parser.add_argument(
        "root_dir", help="Root directory to search for 'annotations' folders"
    )
    parser.add_argument(
        "-o",
        "--output-path",
        default="egohumans_annotations.zip",
        help="Path to the output zip file (default: egohumans_annotations.zip)",
    )
    args = parser.parse_args()

    zip_annotations(args.root_dir, args.output_path)
    print(f"All 'annotations' folders zipped into {args.output_path}")
