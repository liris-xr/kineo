# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import torch
from tqdm import tqdm

import cv2

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    Annotations,
    BBox2DAnnotations,
    BBox2DAnnotation,
    Keypoints2DAnnotations,
    Keypoints2DAnnotation,
    KeypointsFormat,
    Keypoints2DAnnotationsMetadata,
)
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline import per_view_cache

from dataclasses import dataclass

from kineo.maths import clamp
import numpy as np

from kineo.pipeline.stages.rtmlib.tools.pose_estimation.rtmpose import RTMPose
from kineo.pipeline.stages.rtmlib.skeleton.coco17 import coco17
from kineo.pipeline.stages.rtmlib.skeleton.coco133 import coco133
from kineo.pipeline.stages.rtmlib.skeleton.openpose18 import openpose18
from kineo.pipeline.stages.rtmlib.skeleton.openpose134 import openpose134
from kineo.pipeline.stages.rtmlib.skeleton.halpe26 import halpe26
from kineo.pipeline.stages.rtmlib.skeleton.hand21 import hand21

from kineo.visualization.viz_2d import draw_bboxes, draw_keypoints

from collections import namedtuple

BboxDetectionResult = namedtuple("BboxDetectionResult", ["bboxes", "scores"])
KeypointsDetectionResult = namedtuple("KeypointsDetectionResult", ["keypoints", "scores"])

@dataclass
class RtmlibKeypointsDetectionRuntimeConfig:
    use_cache: bool = True
    cache_output_path_template: str = (
        "cache/{sequence_name}/{annotation_key}/{view_id}.pkl"
    )
    frame_step: int = 1
    show: bool = False


class RtmlibKeypointsDetectionStage(
    PipelineStage[RtmlibKeypointsDetectionRuntimeConfig]
):
    """
    Stage for detecting bounding boxes from images by using MMDet.

    Produces :class:`BBox2DAnnotations` with the detected bounding boxes for each view with key "bboxes_2d".
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: RtmlibKeypointsDetectionRuntimeConfig,
        dynamic_runtime_cfg: dict[str, RtmlibKeypointsDetectionRuntimeConfig]
        | None = None,
        keypoints_model: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        keypoints_model_input_shape_hw: tuple[int, int] = (288, 384),
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )

        self.keypoints_model = RTMPose(
            onnx_model=keypoints_model,
            model_input_size=keypoints_model_input_shape_hw,
            device="cuda",
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
        elif n_keypoints == 18:
            keypoints_format_name = "openpose-wholebody"
            keypoint_info = openpose18["keypoint_info"]
            skeleton_info = openpose18["skeleton_info"]
        elif n_keypoints == 134:
            keypoints_format_name = "openpose-wholebody"
            keypoint_info = openpose134["keypoint_info"]
            skeleton_info = openpose134["skeleton_info"]
        elif n_keypoints == 21:
            keypoints_format_name = "hand21"
            keypoint_info = hand21["keypoint_info"]
            skeleton_info = hand21["skeleton_info"]
        elif n_keypoints == 26:
            keypoints_format_name = "halpe26"
            keypoint_info = halpe26["keypoint_info"]
            skeleton_info = halpe26["skeleton_info"]
        else:
            raise ValueError(f"Expected 17, 133, 18, 134, 21 or 26 keypoints, got {n_keypoints}")

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
        bboxes_annotations: BBox2DAnnotations = annotations.get("bboxes_2d")

        if bboxes_annotations is None:
            raise ValueError("Expected bboxes annotations but none were provided")

        def infer_missing(missing_views: list[ViewInput]) -> dict[str, Annotations]:
            return {
                "keypoints_2d": self._infer_keypoints(
                    views=missing_views,
                    bboxes_annotations=bboxes_annotations,
                    keypoints_model=self.keypoints_model,
                    frame_step=runtime_cfg.frame_step,
                    show=runtime_cfg.show,
                )
            }

        cached = per_view_cache.load_or_infer_per_view(
            views=views,
            specs={
                "keypoints_2d": per_view_cache.PerViewCacheSpec(
                    annotations_cls=Keypoints2DAnnotations,
                    metadata=self.keypoints_metadata,
                )
            },
            infer_missing=infer_missing,
            sequence_name=sequence_name,
            cache_output_path_template=runtime_cfg.cache_output_path_template,
            use_cache=runtime_cfg.use_cache,
        )

        annotations["keypoints_2d"] = cached["keypoints_2d"].cpu()

    def _infer_keypoints(
        self,
        views: list[ViewInput],
        bboxes_annotations: BBox2DAnnotations,
        keypoints_model: RTMPose,
        frame_step: int = 1,
        show: bool = False,
    ) -> Keypoints2DAnnotations:
        """
        Infer keypoints for all views.
        """
        all_keypoints_annotations: list[Keypoints2DAnnotation] = []

        n_total_frames = sum(view["frame_loader"].n_frames for view in views)
        n_total_inference_frames = sum(
            len(_get_frames_batch(view["frame_loader"].n_frames, frame_step))
            for view in views
        )
        pbar = tqdm(
            total=n_total_frames, desc="Inferring keypoints", leave=False, unit="frames"
        )

        batch_size = 1
        n_inference_frames_processed = 0

        for view in views:

            frame_loader = view["frame_loader"]
            view_id = view["view_id"]
            view_n_frames = frame_loader.n_frames

            view_bboxes_annotations: list[BBox2DAnnotation] = bboxes_annotations.filter_by_view_id(view_id)
            view_keypoints_annotations: list[Keypoints2DAnnotation] = []

            inference_frames = _get_frames_batch(view_n_frames, frame_step)

            for batch_start in range(0, len(inference_frames), batch_size):
                batch_end = min(batch_start + batch_size, len(inference_frames))
                batch_frames = inference_frames[batch_start:batch_end]
                actual_batch_size = len(batch_frames)

                # Load batch of frames
                frames_rgb = frame_loader.load_frames_at(
                    frame_indices=torch.tensor(batch_frames)
                )
                # (B, C, H, W) -> (B, H, W, C)
                frames_bgr = frames_rgb.permute(0, 2, 3, 1).flip(-1).cpu().numpy()

                batch_bboxes_annotations = view_bboxes_annotations.filter_by_frame_indices(batch_frames)

                batch_keypoints_annotations = detect_keypoints(
                    view_id=view_id,
                    frames_bgr=frames_bgr,
                    batch_frames=batch_frames,
                    bboxes_annotations=batch_bboxes_annotations,
                    keypoints_model=keypoints_model,
                    keypoints_metadata=self.keypoints_metadata
                )

                if show:
                    for i in range(actual_batch_size):
                        frame_idx = batch_frames[i]
                        frame_bgr = frames_bgr[i].copy()

                        frame_bboxes_annotations = batch_bboxes_annotations.filter_by_frame_idx(frame_idx).first_or_default()
                        frame_keypoints_annotations = batch_keypoints_annotations.filter_by_frame_idx(frame_idx).first_or_default()

                        if frame_bboxes_annotations is None or frame_keypoints_annotations is None:
                            continue

                        frame_bboxes = frame_bboxes_annotations.xyxy.reshape(-1, 4).cpu().numpy()
                        frame_keypoints = frame_keypoints_annotations.xy.cpu().numpy()
                        frame_bgr = draw_bboxes(frame_bgr, frame_bboxes)
                        frame_bgr = draw_keypoints(frame_bgr, frame_keypoints, connectivity=self.keypoints_format.keypoints_connectivity)
                        cv2.imshow("Frame", frame_bgr)
                        cv2.waitKey(1)

                view_keypoints_annotations.extend(batch_keypoints_annotations)

                n_inference_frames_processed += actual_batch_size
                progress = int(
                    clamp(
                        n_total_frames
                        * (n_inference_frames_processed / n_total_inference_frames),
                        0,
                        n_total_frames,
                    )
                )
                pbar.update(progress - pbar.n)

            view_keypoints_annotations = Keypoints2DAnnotations(
                metadata=self.keypoints_metadata,
                annotations=view_keypoints_annotations,
            )

            if frame_step > 1:
                # Interpolate bboxes to all frames
                all_frames = list(range(view_n_frames))
                view_keypoints_annotations = (
                    view_keypoints_annotations.interpolate_by_frame_indices(
                        target_frame_indices=all_frames, max_frame_idx_diff=frame_step
                    )
                )
            all_keypoints_annotations.extend(view_keypoints_annotations._annotations)

        pbar.close()

        keypoints_annotations = Keypoints2DAnnotations(
            metadata=self.keypoints_metadata,
            annotations=all_keypoints_annotations,
        ).cpu()

        if show:
            cv2.destroyAllWindows()

        return keypoints_annotations


def detect_keypoints(
    view_id: str,
    frames_bgr: torch.Tensor,
    batch_frames: list[int],
    bboxes_annotations: BBox2DAnnotations,
    keypoints_model: RTMPose,
    keypoints_metadata: Keypoints2DAnnotationsMetadata
) -> Keypoints2DAnnotations:
    assert frames_bgr.ndim in [3, 4] and frames_bgr.shape[-1] == 3, (
        f"Expected frames_bgr to have shape (B, H, W, C) or (H, W, C), got {frames_bgr.shape}"
    )
    assert frames_bgr.dtype == np.uint8, "Expected frames_bgr to be uint8"

    annotations: list[Keypoints2DAnnotation] = []

    for batch_idx, frame_idx in enumerate(batch_frames):
        frame_bboxes = bboxes_annotations.filter_by_view_id(
            view_id
        ).filter_by_frame_idx(frame_idx)

        if len(frame_bboxes) == 0:
            continue

        bboxes_array = np.stack(
            [bbox.xyxy.cpu().numpy() for bbox in frame_bboxes],
            axis=0,
        )
        subjects_ids = [bbox.subject_id for bbox in frame_bboxes]

        keypoints, scores = keypoints_model(frames_bgr[batch_idx], bboxes_array)

        annotations.extend(_create_keypoints_annotations(
            keypoints_result=KeypointsDetectionResult(keypoints=keypoints, scores=scores),
            view_id=view_id,
            frame_idx=frame_idx,
            keypoints_format_name=keypoints_metadata.formats[0].name,
            subjects_ids=subjects_ids,
        ))

    return Keypoints2DAnnotations(
        metadata=keypoints_metadata,
        annotations=annotations,
    )


def _create_keypoints_annotations(
    keypoints_result: KeypointsDetectionResult,
    view_id: str,
    frame_idx: int,
    keypoints_format_name: str,
    subjects_ids: list[str]
) -> list[Keypoints2DAnnotation]:
    """Create keypoints annotations from inference results."""
    annotations = []
    n_instances = len(keypoints_result.keypoints)

    for instance_idx in range(n_instances):
        keypoints_xy = keypoints_result.keypoints[instance_idx]
        keypoints_scores = keypoints_result.scores[instance_idx]

        kps_xy = torch.from_numpy(keypoints_xy).to(torch.float32)
        scores = torch.from_numpy(keypoints_scores).to(torch.float32)
        subject_id = subjects_ids[instance_idx]

        keypoint_annotation = Keypoints2DAnnotation(
            view_id=view_id,
            frame_idx=frame_idx,
            subject_id=subject_id,
            xy=kps_xy.cpu(),
            scores=scores.cpu(),
            format=keypoints_format_name,
        )
        annotations.append(keypoint_annotation)

    return annotations

def _get_frames_batch(n_frames: int, frame_step: int) -> list[int]:
    """Get list of frame indices to run inference on."""
    if frame_step == 1:
        return list(range(n_frames))

    frames_batch = list(range(0, n_frames, frame_step))

    # Always include the last frame if it's not already included
    if (n_frames - 1) not in frames_batch and n_frames > 0:
        frames_batch.append(n_frames - 1)

    return sorted(frames_batch)
