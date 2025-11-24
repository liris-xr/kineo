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
import torch
import os
from tqdm import tqdm
import pickle

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    Annotations,
    Keypoints2DAnnotations,
    Keypoints2DAnnotationsMetadata,
    Keypoints2DAnnotation,
    BBox2DAnnotations,
    BBox2DAnnotation,
    KeypointsFormat,
)
from kineo.pipeline.pipeline import Pipeline
from rtmlib.visualization.skeleton import coco17, coco133
from dataclasses import dataclass

import numpy as np

from rtmlib import RTMPose
import numpy as np


@dataclass
class RtmlibKeypointsDetectionRuntimeConfig:
    use_half_precision: bool = True
    use_cache: bool = True
    cache_output_path_template: str = "cache/{sequence_name}/{annotation_key}.pkl"


class RtmlibKeypointsDetectionStage(
    PipelineStage[RtmlibKeypointsDetectionRuntimeConfig]
):
    """
    Stage for detecting keypoints from images using RTMLib.

    Produces :class:`Keypoints2DAnnotations` with the detected keypoints for each view with key "keypoints_2d".
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: RtmlibKeypointsDetectionRuntimeConfig,
        keypoints_model_onnx: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip",
        keypoints_model_input_shape_hw: tuple[int, int] = (288, 384),
        keypoints_model_backend: str = "onnxruntime",
        dynamic_runtime_cfg: (
            dict[str, RtmlibKeypointsDetectionRuntimeConfig] | None
        ) = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )

        self.keypoints_model = RTMPose(
            onnx_model=keypoints_model_onnx,
            model_input_size=keypoints_model_input_shape_hw,
            backend=keypoints_model_backend,
            device=device,
        )

        dummy_output, _ = self.keypoints_model(
            np.random.randn(*keypoints_model_input_shape_hw, 3)
        )

        if dummy_output.ndim == 2:
            dummy_output = dummy_output[None, ...]

        n_keypoints = dummy_output.shape[1]
        if n_keypoints == 17:
            keypoints_format_name = "coco"
            keypoint_info = coco17["keypoint_info"]
            skeleton_info = coco17["skeleton_info"]
        elif n_keypoints == 133:
            keypoints_format_name = "coco-wholebody"
            keypoint_info = coco133["keypoint_info"]
            skeleton_info = coco133["skeleton_info"]
        else:
            raise ValueError(f"Expected 17 or 133 keypoints, got {n_keypoints}")

        keypoints_names = [keypoint_info[idx]["name"] for idx in range(n_keypoints)]
        keypoints_connectivity = []

        for link_idx in range(len(skeleton_info)):
            bone1, bone2 = skeleton_info[link_idx]["link"]
            bone1_idx = keypoints_names.index(bone1)
            bone2_idx = keypoints_names.index(bone2)
            keypoints_connectivity.append((bone1_idx, bone2_idx))

        self.keypoints_format = KeypointsFormat(
            name=keypoints_format_name,
            n_keypoints=n_keypoints,
            keypoints_names=keypoints_names,
            keypoints_connectivity=keypoints_connectivity,
        )

        self.keypoints_metadata = Keypoints2DAnnotationsMetadata(
            formats=[self.keypoints_format]
        )

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: RtmlibKeypointsDetectionRuntimeConfig,
    ):
        device = pipeline.device

        bboxes_annotations: BBox2DAnnotations | None = annotations.get(
            "bboxes_2d", None
        )

        # Sanity check that the bboxes annotations are correctly provided.
        if bboxes_annotations is None:
            raise ValueError("Expected bboxes annotations but none were provided")

        if runtime_cfg.use_cache:
            kps2d_cache_filepath = runtime_cfg.cache_output_path_template.format(
                sequence_name=sequence_name, annotation_key="keypoints_2d"
            )

            bboxes_cache_filepath = runtime_cfg.cache_output_path_template.format(
                sequence_name=sequence_name, annotation_key="bboxes_2d"
            )

            if os.path.exists(kps2d_cache_filepath):
                with open(kps2d_cache_filepath, "rb") as f:
                    keypoints_annotations = Keypoints2DAnnotations.from_dict(
                        pickle.load(f)
                    )
                print(
                    f"Loaded keypoints annotations from cache: {kps2d_cache_filepath}"
                )
            if os.path.exists(bboxes_cache_filepath):
                with open(bboxes_cache_filepath, "rb") as f:
                    bboxes_annotations = BBox2DAnnotations.from_dict(pickle.load(f))
                print(f"Loaded bboxes annotations from cache: {bboxes_cache_filepath}")

        if keypoints_annotations is not None:
            annotations["keypoints_2d"] = keypoints_annotations
            return

        keypoints_annotations = self._infer_keypoints(
            views=views,
            bboxes_annotations=bboxes_annotations,
            use_half_precision=runtime_cfg.use_half_precision,
        )

        if runtime_cfg.use_cache:
            os.makedirs(os.path.dirname(kps2d_cache_filepath), exist_ok=True)
            os.makedirs(os.path.dirname(bboxes_cache_filepath), exist_ok=True)

            if not os.path.exists(kps2d_cache_filepath):
                with open(kps2d_cache_filepath, "wb") as f:
                    print(
                        f"Saving keypoints annotations to cache: {kps2d_cache_filepath}"
                    )
                    pickle.dump(keypoints_annotations.to_dict(), f)

        annotations["keypoints_2d"] = keypoints_annotations

    def _infer_keypoints(
        self,
        views: list[ViewInput],
        bboxes_annotations: BBox2DAnnotations,
        use_half_precision: bool = True,
    ) -> tuple[Keypoints2DAnnotations]:
        """
        Infer keypoints and bounding boxes for all views.
        """

        all_keypoints_annotations: list[Keypoints2DAnnotation] = []
        all_bboxes_annotations: list[BBox2DAnnotation] = []

        n_total_frames = sum(view["frame_loader"].n_frames for view in views)
        pbar = tqdm(total=n_total_frames, desc="Inferring keypoints", leave=False)

        for view in views:

            frame_loader = view["frame_loader"]
            view_id = view["view_id"]
            view_n_frames = frame_loader.n_frames

            view_bboxes_annotations = bboxes_annotations.filter_by_view_id(
                view["view_id"]
            )

            for frame_idx in range(view_n_frames):
                frame_rgb = frame_loader.load_frame_at(frame_idx)
                frame_bgr = frame_rgb.permute(1, 2, 0).flip(-1).cpu().numpy()

                frame_bboxes_annotations: list[BBox2DAnnotation] = []
                frame_bboxes_annotations = view_bboxes_annotations.filter_by_frame_idx(
                    frame_idx
                ).annotations
                if len(frame_bboxes_annotations) == 0:
                    continue

                all_bboxes_annotations.extend(frame_bboxes_annotations)

                bboxes_xyxy = np.stack(
                    [
                        bbox.xyxy.cpu().numpy().reshape(4)
                        for bbox in frame_bboxes_annotations
                    ]
                )
                bboxes_xyxy = bboxes_xyxy.reshape(-1, 4)
                bboxes_subject_ids = [
                    bbox.subject_id for bbox in frame_bboxes_annotations
                ]

                with (
                    torch.no_grad(),
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=use_half_precision,
                    ),
                ):
                    keypoints, keypoints_scores = self.keypoints_model(
                        frame_bgr, bboxes_xyxy
                    )

                keypoints = torch.from_numpy(keypoints)
                keypoints_scores = torch.from_numpy(keypoints_scores)

                for i in range(len(keypoints)):
                    subject_id = bboxes_subject_ids[i]
                    subject_keypoints_xy = keypoints[i].float()
                    subject_keypoints_scores = keypoints_scores[i].float()

                    keypoints_annotation = Keypoints2DAnnotation(
                        view_id=view_id,
                        frame_idx=frame_idx,
                        subject_id=subject_id,
                        xy=subject_keypoints_xy,
                        scores=subject_keypoints_scores,
                        annotated=torch.ones(
                            subject_keypoints_xy.shape[0], dtype=torch.bool
                        ),
                        format=self.keypoints_format.name,
                    )
                    all_keypoints_annotations.append(keypoints_annotation)

                pbar.update(1)

        pbar.close()

        keypoints_annotations = Keypoints2DAnnotations(
            metadata=Keypoints2DAnnotationsMetadata(formats=[self.keypoints_format]),
            annotations=all_keypoints_annotations,
        )

        return keypoints_annotations
