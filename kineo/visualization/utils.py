# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import random
import colorsys


def get_subject_color_rgba(subject_id: str) -> tuple[float, float, float, float]:
    rng = random.Random(subject_id)
    subject_hue = rng.random()
    subject_color = (*colorsys.hsv_to_rgb(subject_hue, 1.0, 1.0), 1.0)
    return subject_color
