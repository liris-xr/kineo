# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from pathlib import Path

from kineo.io.download import download_gdrive_file_or_folder

EGOHUMANS_GDRIVE_ID = "1JD963urzuzV_R_6FOVOtlx8UupwUuknR"


def download_egohumans(
    output_base_dir: Path | str,
):
    download_gdrive_file_or_folder(EGOHUMANS_GDRIVE_ID, output_base_dir, recursive=True)
