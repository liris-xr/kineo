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
import os
from dataclasses import dataclass, field
from tqdm import tqdm
from pathlib import Path
import pickle
import warnings
from kineo.annotations.global_time_reference import GlobalTimeReferenceAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.reconstructed_scene import WorldReconstructedSceneAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.bboxes_2d import BBox2DAnnotations


@dataclass(frozen=True)
class AnnotationsLoadRuntimeConfig:
    annotations_keys: str | list[str] = field(
        default_factory=lambda: [
            "global_time_reference",
            "camera_extrinsics",
            "camera_intrinsics",
            "keypoints_3d",
            "world_reconstructed_scene",
            "keypoints_2d",
            "bboxes_2d",
        ]
    )
    not_found_error: bool = True
    input_path_template: str = (
        "./outputs/annotations/{sequence_name}/{annotation_key}.pkl"
    )

    def __post_init__(self):
        file_ext = Path(self.input_path_template).suffix
        if file_ext != ".pkl":
            raise ValueError("Input path template must end with .pkl")


class AnnotationsLoadStage(PipelineStage[AnnotationsLoadRuntimeConfig]):
    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: AnnotationsLoadRuntimeConfig,
        dynamic_runtime_cfg: dict[str, AnnotationsLoadRuntimeConfig] | None = None,
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
        runtime_cfg: AnnotationsLoadRuntimeConfig,
    ):
        if isinstance(runtime_cfg.annotations_keys, str):
            annotations_keys = [runtime_cfg.annotations_keys]
        elif isinstance(runtime_cfg.annotations_keys, Iterable):
            annotations_keys = list(runtime_cfg.annotations_keys)
        else:
            raise ValueError("Annotations keys must be a string or a list")

        pbar = tqdm(annotations_keys, desc="Loading annotations", leave=False)

        for k in pbar:
            pbar.set_postfix(annotation_key=k)

            formatted_output_path = runtime_cfg.input_path_template.format(
                sequence_name=sequence_name, annotation_key=k
            )

            if os.path.exists(formatted_output_path):
                with open(formatted_output_path, "rb") as f:
                    annotations_dict = pickle.load(f)

                    if k == "global_time_reference":
                        annotations[k] = GlobalTimeReferenceAnnotations.from_dict(
                            annotations_dict
                        )
                    elif k == "camera_extrinsics":
                        annotations[k] = CameraExtrinsicsAnnotations.from_dict(
                            annotations_dict
                        )
                    elif k == "camera_intrinsics":
                        annotations[k] = CameraIntrinsicsAnnotations.from_dict(
                            annotations_dict
                        )
                    elif k == "keypoints_3d":
                        annotations[k] = Keypoints3DAnnotations.from_dict(
                            annotations_dict
                        )
                    elif k == "world_reconstructed_scene":
                        annotations[k] = WorldReconstructedSceneAnnotations.from_dict(
                            annotations_dict
                        )
                    elif k == "keypoints_2d":
                        annotations[k] = Keypoints2DAnnotations.from_dict(
                            annotations_dict
                        )
                    elif k == "bboxes_2d":
                        annotations[k] = BBox2DAnnotations.from_dict(annotations_dict)
                    else:
                        raise ValueError(f"Annotations '{k}' not supported by loader")

                print(f"Annotations '{k}' loaded from {formatted_output_path}")
            else:
                if runtime_cfg.not_found_error:
                    raise ValueError(f"Annotations '{k}' not found")
                else:
                    warnings.warn(f"Annotations '{k}' not found")

            pbar.update(1)

        pbar.close()