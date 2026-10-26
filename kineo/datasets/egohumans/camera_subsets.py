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

from omegaconf import OmegaConf

CAMERA_SUBSETS_PATH = os.path.join(os.path.dirname(__file__), "camera_subsets.yaml")


def load_camera_subsets(n_views: int) -> dict[str, list[str]]:
    subsets = OmegaConf.to_container(OmegaConf.load(CAMERA_SUBSETS_PATH))
    if n_views not in subsets:
        raise ValueError(
            f"No {n_views}-view camera subsets, only {sorted(subsets)}"
        )
    return subsets[n_views]
