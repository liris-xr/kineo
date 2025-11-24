# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

from kineo.pipeline.pipeline import PipelineStage
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline.pipeline import ViewInput
from kineo.pipeline.pipeline import Annotations
from typing import Iterable

from dataclasses import dataclass


@dataclass(frozen=True)
class TransferGroundTruthAnnotationsRuntimeConfig:
    annotations_keys: list[str] | str | None = None
    raise_error_if_not_found: bool = False
    overwrite: bool = True


class TransferGroundTruthAnnotationsStage(
    PipelineStage[TransferGroundTruthAnnotationsRuntimeConfig]
):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: TransferGroundTruthAnnotationsRuntimeConfig,
        dynamic_runtime_cfg: (
            dict[str, TransferGroundTruthAnnotationsRuntimeConfig] | None
        ) = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: TransferGroundTruthAnnotationsRuntimeConfig,
    ):
        if runtime_cfg.annotations_keys is None:
            return

        if not isinstance(runtime_cfg.annotations_keys, (Iterable, str)):
            raise ValueError(
                f"annotations_keys must be a list of strings or a string, got {type(runtime_cfg.annotations_keys)}"
            )

        if isinstance(runtime_cfg.annotations_keys, str):
            runtime_cfg.annotations_keys = [runtime_cfg.annotations_keys]

        for annotation_key in runtime_cfg.annotations_keys:
            if annotation_key not in gt_annotations:
                if runtime_cfg.raise_error_if_not_found:
                    raise ValueError(
                        f"annotation {annotation_key} not found in gt_annotations"
                    )
                continue

            if annotation_key in annotations and not runtime_cfg.overwrite:
                continue

            print(f"Transferring ground truth annotation {annotation_key}")
            annotations[annotation_key] = gt_annotations[annotation_key]
