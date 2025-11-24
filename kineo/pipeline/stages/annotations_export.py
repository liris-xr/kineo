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
import orjson
import torch
import os
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path
import pickle


def default(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass(frozen=True)
class AnnotationsExportRuntimeConfig:
    annotations_keys: str | list[str] = "*"
    excluded_annotations_keys: list[str] = field(default_factory=list)
    not_found_error: bool = True
    output_path_template: str = (
        "./outputs/annotations/{sequence_name}/{annotation_key}.pkl"
    )

    def __post_init__(self):
        file_ext = Path(self.output_path_template).suffix
        if file_ext not in [".json", ".pkl"]:
            raise ValueError("Output path template must end with .json or .pkl")


class AnnotationsExportStage(PipelineStage[AnnotationsExportRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: AnnotationsExportRuntimeConfig,
        dynamic_runtime_cfg: dict[str, AnnotationsExportRuntimeConfig] | None = None,
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
        runtime_cfg: AnnotationsExportRuntimeConfig,
    ):
        if runtime_cfg.annotations_keys == "*":
            annotations_keys = list(annotations.keys())
        elif isinstance(runtime_cfg.annotations_keys, str):
            annotations_keys = [runtime_cfg.annotations_keys]
        else:
            annotations_keys = runtime_cfg.annotations_keys

        annotations_keys = [
            k
            for k in annotations_keys
            if k not in runtime_cfg.excluded_annotations_keys
        ]

        pbar = tqdm(annotations_keys, desc="Exporting annotations", leave=False)

        for k in pbar:
            pbar.set_postfix(annotation_key=k)

            ann: Annotations = annotations.get(k)

            if ann is None and runtime_cfg.not_found_error:
                raise ValueError(f"Annotations with key {k} not found")

            annotations_dict = ann.to_dict()

            formatted_output_path = runtime_cfg.output_path_template.format(
                sequence_name=sequence_name, annotation_key=k
            )
            os.makedirs(os.path.dirname(formatted_output_path), exist_ok=True)

            file_ext = Path(formatted_output_path).suffix

            if file_ext == ".json":
                with open(formatted_output_path, "wb") as f:
                    f.write(orjson.dumps(annotations_dict, default=default))
            elif file_ext == ".pkl":
                with open(formatted_output_path, "wb") as f:
                    pickle.dump(annotations_dict, f)

            print(f"Annotations '{k}' saved to {formatted_output_path}")

        pbar.close()
