# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from kineo.annotations import Annotations

if TYPE_CHECKING:
    from kineo.datasets.keypoints_sequence_dataset import ViewInput


@dataclass(frozen=True)
class PerViewCacheSpec:
    annotations_cls: type[Annotations]
    metadata: Any


def load_or_infer_per_view(
    views: list[ViewInput],
    specs: dict[str, PerViewCacheSpec],
    infer_missing: Callable[[list[ViewInput]], dict[str, Annotations]],
    sequence_name: str,
    cache_output_path_template: str,
    use_cache: bool,
) -> dict[str, Annotations]:
    annotation_keys = ", ".join(sorted(specs))

    if not use_cache:
        print(
            f"[cache] {sequence_name} {annotation_keys}: disabled, inferring "
            f"{len(views)} views"
        )
        return _check_inferred(infer_missing(views), specs)

    if "{view_id}" not in cache_output_path_template:
        raise ValueError(
            "The cache holds one file per view, so cache_output_path_template "
            f"must contain a {{view_id}} field, got {cache_output_path_template!r}"
        )

    cached: dict[str, dict[str, list]] = {key: {} for key in specs}
    missing_views: list[ViewInput] = []

    for view in views:
        view_id = view["view_id"]
        filepaths = {
            key: _cache_filepath(
                cache_output_path_template, sequence_name, key, view_id
            )
            for key in specs
        }

        if not all(os.path.exists(filepath) for filepath in filepaths.values()):
            missing_views.append(view)
            continue

        for key, filepath in filepaths.items():
            with open(filepath, "rb") as f:
                loaded = specs[key].annotations_cls.from_dict(pickle.load(f))

            if loaded.metadata != specs[key].metadata:
                raise ValueError(
                    f"Cache file {filepath} holds {loaded.metadata}, but this "
                    f"stage produces {specs[key].metadata}. The cache directory "
                    "is shared with an incompatible configuration."
                )

            cached[key][view_id] = list(loaded.annotations)

    hit = f"{len(views) - len(missing_views)}/{len(views)} views hit"

    if missing_views:
        missing_ids = ", ".join(view["view_id"] for view in missing_views)
        print(
            f"[cache] {sequence_name} {annotation_keys}: {hit}, inferring {missing_ids}"
        )

        inferred = _check_inferred(infer_missing(missing_views), specs)
        _save_per_view(
            inferred, missing_views, sequence_name, cache_output_path_template
        )

        for key, inferred_annotations in inferred.items():
            for view in missing_views:
                view_id = view["view_id"]
                cached[key][view_id] = list(
                    inferred_annotations.filter_by_view_id(view_id).annotations
                )
    else:
        print(f"[cache] {sequence_name} {annotation_keys}: {hit}")

    return {
        key: spec.annotations_cls(
            metadata=spec.metadata,
            annotations=[
                annotation
                for view in views
                for annotation in cached[key].get(view["view_id"], [])
            ],
        )
        for key, spec in specs.items()
    }


def _cache_filepath(
    template: str, sequence_name: str, annotation_key: str, view_id: str
) -> str:
    return template.format(
        sequence_name=sequence_name, annotation_key=annotation_key, view_id=view_id
    )


def _check_inferred(
    inferred: dict[str, Annotations], specs: dict[str, PerViewCacheSpec]
) -> dict[str, Annotations]:
    if set(inferred) != set(specs):
        raise ValueError(
            f"Inference returned annotations {sorted(inferred)}, expected "
            f"{sorted(specs)}"
        )
    return inferred


def _save_per_view(
    inferred: dict[str, Annotations],
    views: list[ViewInput],
    sequence_name: str,
    template: str,
) -> None:
    for key, inferred_annotations in inferred.items():
        inferred_annotations = inferred_annotations.cpu()

        for view in views:
            view_id = view["view_id"]
            filepath = _cache_filepath(template, sequence_name, key, view_id)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, "wb") as f:
                pickle.dump(
                    inferred_annotations.filter_by_view_id(view_id).to_dict(), f
                )
