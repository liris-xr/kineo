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

import orjson

CAMERA_SELECTIONS_FILE = os.path.join(
    os.path.dirname(__file__), "hsfm_eval_camera_selections.json"
)


def load_hsfm_view_selection(
    sequence_name: str,
    n_views: int,
    available_view_ids: list[str],
) -> list[str]:
    """Return the views to evaluate a sequence on.

    HSfM (Muller & Choi et al., 'Reconstructing People, Places, and Cameras',
    CVPR 2025) evaluates EgoHumans on a fixed per-activity camera subset for
    each view count (supplementary S.4.2). Sequences whose snapshot lacks one
    of those cameras carry a per-sequence override, which wins over the
    activity selection; see the notes in the selections file.

    Args:
        sequence_name: Sequence name, e.g. "fencing_001".
        n_views: View-count setting, one of the counts the paper reports.
        available_view_ids: View ids the sequence provides, used to resolve the
            "all" selection.

    Returns:
        The selected view ids, in the order HSfM lists them.

    Raises:
        ValueError: If the view count or the sequence activity has no
            selection, or if the sequence lacks one of the selected views.
    """
    with open(CAMERA_SELECTIONS_FILE, "rb") as f:
        selections_file = orjson.loads(f.read())

    selections = selections_file["camera_selections"]
    overrides = selections_file.get("sequence_overrides", {})

    if str(n_views) not in selections:
        raise ValueError(
            f"No HSfM camera selection for {n_views} views "
            f"(available: {sorted(int(k) for k in selections)})"
        )
    activity_selections = selections[str(n_views)]

    activity = sequence_name.rsplit("_", 1)[0]
    if activity not in activity_selections:
        raise ValueError(
            f"No HSfM {n_views}-view camera selection for activity "
            f"'{activity}' (sequence '{sequence_name}')"
        )
    selected_view_ids = overrides.get(str(n_views), {}).get(
        sequence_name, activity_selections[activity]
    )

    if selected_view_ids == "all":
        return list(available_view_ids)

    missing_view_ids = [
        view_id
        for view_id in selected_view_ids
        if view_id not in available_view_ids
    ]
    if missing_view_ids:
        raise ValueError(
            f"Sequence '{sequence_name}' lacks the {n_views}-view "
            f"cameras {missing_view_ids}"
        )
    return selected_view_ids
