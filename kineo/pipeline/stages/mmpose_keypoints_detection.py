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
import os
import os.path as osp
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
from kineo.maths import clamp
from mmpose.apis import Pose2DInferencer
from mmpose.structures import PoseDataSample
from mmpose.structures.utils import merge_data_samples

from dataclasses import dataclass

import numpy as np

from mmengine.runner import CheckpointLoader


# Register a custom checkpoint loader to avoid safety errors when loading the model weights.
@CheckpointLoader.register_scheme(prefixes="", force=True)
def load_from_local(filename, map_location):
    """Load checkpoint by local file path.

    Args:
        filename (str): local checkpoint file path
        map_location (str, optional): Same as :func:`torch.load`.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = osp.expanduser(filename)
    if not osp.isfile(filename):
        raise FileNotFoundError(f"{filename} can not be found.")
    checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    return checkpoint


@dataclass
class MMPoseKeypointsDetectionRuntimeConfig:
    batch_size: int = 16
    use_half_precision: bool = True
    use_cache: bool = True
    cache_output_path_template: str = "cache/{sequence_name}/{annotation_key}.pkl"
    force_zero_scores_outside_bbox: bool = True
    use_flip_test: bool = True
    frame_step: int = (
        1  # Infer every N frames (1 = every frame, 2 = every other frame, etc.)
    )
    disable_confidence: bool = False


class MMPoseKeypointsDetectionStage(
    PipelineStage[MMPoseKeypointsDetectionRuntimeConfig]
):
    """
    Stage for detecting keypoints and bounding boxes from images by jointly using MMPose and MMDet (prevents the need for decoding the video twice).

    Produces :class:`Keypoints2DAnnotations` and :class:`BBox2DAnnotations` with the detected keypoints and bounding boxes for each view with keys "keypoints_2d" and "bboxes_2d" respectively.
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: MMPoseKeypointsDetectionRuntimeConfig,
        dynamic_runtime_cfg: (
            dict[str, MMPoseKeypointsDetectionRuntimeConfig] | None
        ) = None,
        keypoints_model: str = "rtmpose-l_8xb256-420e_h36m-384x288",
        keypoints_model_weights: str | None = None,
        keypoints_model_scope: str = "mmpose",
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )
        self.keypoints_inferencer = Pose2DInferencer(
            keypoints_model,
            weights=keypoints_model_weights,
            det_model="whole-image",
            scope=keypoints_model_scope,
            show_progress=False,
        )

        self.keypoints_inferencer.model = self.keypoints_inferencer.model.cpu()
        self.keypoints_inferencer.model = self.keypoints_inferencer.model.eval()

        try:
            # Make sure the model decodes visibility (otherwise the keypoints scores might be out of the range [0, 1])
            self.keypoints_inferencer.model.head.decoder.decode_visibility = True
        except AttributeError:
            pass

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: MMPoseKeypointsDetectionRuntimeConfig,
    ):
        device = pipeline.device

        bboxes_annotations: BBox2DAnnotations = annotations.get("bboxes_2d")

        if bboxes_annotations is None:
            raise ValueError("Expected bboxes annotations but none were provided")

        if runtime_cfg.use_cache:
            kps2d_cache_filepath = runtime_cfg.cache_output_path_template.format(
                sequence_name=sequence_name, annotation_key="keypoints_2d"
            )

            if os.path.exists(kps2d_cache_filepath):
                with open(kps2d_cache_filepath, "rb") as f:
                    keypoints_annotations = Keypoints2DAnnotations.from_dict(
                        pickle.load(f)
                    )
                keypoints_annotations = keypoints_annotations.filter_by_view_ids(
                    [view["view_id"] for view in views]
                )
                print(
                    f"Loaded keypoints annotations from cache: {kps2d_cache_filepath}"
                )
                annotations["keypoints_2d"] = keypoints_annotations.cpu()
                return

        self.keypoints_inferencer.model = self.keypoints_inferencer.model.to(device)

        try:
            self.keypoints_inferencer.test_cfg.flip_test = runtime_cfg.use_flip_test
        except AttributeError:
            pass

        keypoints_annotations = self._infer_keypoints(
            views=views,
            bboxes_annotations=bboxes_annotations,
            keypoints_inferencer=self.keypoints_inferencer,
            batch_size=runtime_cfg.batch_size,
            use_half_precision=runtime_cfg.use_half_precision,
            force_zero_scores_outside_bbox=runtime_cfg.force_zero_scores_outside_bbox,
            frame_step=runtime_cfg.frame_step,
            disable_confidence=runtime_cfg.disable_confidence,
        )

        self.keypoints_inferencer.model = self.keypoints_inferencer.model.cpu()

        if runtime_cfg.use_cache and not os.path.exists(kps2d_cache_filepath):
            os.makedirs(os.path.dirname(kps2d_cache_filepath), exist_ok=True)

            if not os.path.exists(kps2d_cache_filepath):
                with open(kps2d_cache_filepath, "wb") as f:
                    print(
                        f"Saving keypoints annotations to cache: {kps2d_cache_filepath}"
                    )
                    pickle.dump(keypoints_annotations.to_dict(), f)

        annotations["keypoints_2d"] = keypoints_annotations.cpu()

    def _infer_keypoints(
        self,
        views: list[ViewInput],
        bboxes_annotations: BBox2DAnnotations,
        keypoints_inferencer: Pose2DInferencer,
        batch_size: int = 16,
        use_half_precision: bool = True,
        force_zero_scores_outside_bbox: bool = True,
        frame_step: int = 1,
        disable_confidence: bool = False,
    ) -> Keypoints2DAnnotations:
        """
        Infer keypoints for all views.
        """

        keypoints_format_name = keypoints_inferencer.model.dataset_meta["dataset_name"]
        keypoints_format = KeypointsFormat.from_mmpose_dataset(keypoints_format_name)
        keypoints_metadata = Keypoints2DAnnotationsMetadata(formats=[keypoints_format])

        all_keypoints_annotations: list[Keypoints2DAnnotation] = []

        n_total_frames = sum(view["frame_loader"].n_frames for view in views)
        n_total_inference_frames = sum(
            len(_get_frames_batch(view["frame_loader"].n_frames, frame_step))
            for view in views
        )
        pbar = tqdm(
            total=n_total_frames, desc="Inferring keypoints", leave=False, unit="frames"
        )

        n_inference_frames_processed = 0

        for view in views:
            view_annotations: list[Keypoints2DAnnotation] = []

            frame_loader = view["frame_loader"]
            view_id = view["view_id"]
            view_n_frames = frame_loader.n_frames

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
                frames_bgr = frames_rgb.permute(0, 2, 3, 1).flip(-1)

                batch_bboxes: list[list[BBox2DAnnotation]] = [
                    [] for _ in range(actual_batch_size)
                ]

                for batch_idx, frame_idx in enumerate(batch_frames):
                    frame_bboxes = bboxes_annotations.filter_by_view_id(
                        view_id
                    ).filter_by_frame_idx(frame_idx)

                    frame_bboxes = list(frame_bboxes.annotations)
                    batch_bboxes[batch_idx] = frame_bboxes

                bboxes_xyxy: list[np.ndarray] = [
                    np.zeros((0, 4)) for _ in range(actual_batch_size)
                ]
                bboxes_scores: list[np.ndarray] = [
                    np.zeros((0, 1)) for _ in range(actual_batch_size)
                ]

                for batch_idx in range(actual_batch_size):
                    frame_bboxes = batch_bboxes[batch_idx]

                    if len(frame_bboxes) == 0:
                        continue

                    bboxes_xyxy[batch_idx] = np.stack(
                        [bbox.xyxy.cpu().numpy() for bbox in frame_bboxes]
                    )
                    bboxes_scores[batch_idx] = np.stack(
                        [[bbox.score] for bbox in frame_bboxes]
                    )

                pose_data_samples = batch_infer_keypoints(
                    frames_bgr=frames_bgr,
                    keypoints_inferencer=keypoints_inferencer,
                    bboxes_xyxy=bboxes_xyxy,
                    bboxes_scores=bboxes_scores,
                    use_half_precision=use_half_precision,
                    filter_bbox_with_zero_score=force_zero_scores_outside_bbox
                )

                for batch_idx in range(actual_batch_size):
                    frame_idx = batch_frames[batch_idx]
                    frame_data_sample = pose_data_samples[batch_idx]

                    keypoints_annotation = _create_keypoints_annotations(
                        frame_data_sample=frame_data_sample,
                        frame_bboxes=batch_bboxes[batch_idx],
                        view_id=view_id,
                        frame_idx=frame_idx,
                        keypoints_format_name=keypoints_format_name,
                        force_zero_scores_outside_bbox=force_zero_scores_outside_bbox,
                        disable_confidence=disable_confidence,
                    )

                    view_annotations.extend(keypoints_annotation)

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
                metadata=keypoints_metadata, annotations=view_annotations
            )

            if frame_step > 1:
                # Interpolate keypoints to all frames
                all_frames = list(range(view_n_frames))
                view_keypoints_annotations = (
                    view_keypoints_annotations.interpolate_by_frame_indices(
                        target_frame_indices=all_frames, max_frame_idx_diff=frame_step
                    )
                )
            all_keypoints_annotations.extend(view_keypoints_annotations._annotations)

        pbar.close()

        keypoints_annotations = Keypoints2DAnnotations(
            metadata=keypoints_metadata, annotations=all_keypoints_annotations
        ).cpu()

        return keypoints_annotations


def _get_frames_batch(n_frames: int, frame_step: int) -> list[int]:
    """Get list of frame indices to run inference on."""
    if frame_step == 1:
        return list(range(n_frames))

    frames_batch = list(range(0, n_frames, frame_step))

    # Always include the last frame if it's not already included
    if (n_frames - 1) not in frames_batch and n_frames > 0:
        frames_batch.append(n_frames - 1)

    return sorted(frames_batch)


def _create_keypoints_annotations(
    frame_data_sample: PoseDataSample,
    frame_bboxes: list[BBox2DAnnotation],
    view_id: str,
    frame_idx: int,
    keypoints_format_name: str,
    force_zero_scores_outside_bbox: bool,
    disable_confidence: bool = False,
) -> list[Keypoints2DAnnotation]:
    """Create keypoints annotations from inference results."""
    annotations = []
    n_instances = len(frame_data_sample.pred_instances.keypoints)

    for instance_idx in range(n_instances):
        keypoints_xy = frame_data_sample.pred_instances.keypoints[instance_idx]
        keypoints_scores = frame_data_sample.pred_instances.keypoints_visible[
            instance_idx
        ]
        bbox_annotation = frame_bboxes[instance_idx]

        kps_xy = torch.from_numpy(keypoints_xy).to(torch.float32)
        scores = torch.from_numpy(keypoints_scores).to(torch.float32)

        if force_zero_scores_outside_bbox:
            # For keypoints outside of the bbox, set the score to 0
            bbox_xyxy = bbox_annotation.xyxy
            kps_xy = kps_xy.to(bbox_xyxy.device)
            scores = scores.to(bbox_xyxy.device)

            scores = torch.where(
                (kps_xy[:, 0] < bbox_xyxy[0])
                | (kps_xy[:, 0] > bbox_xyxy[2])
                | (kps_xy[:, 1] < bbox_xyxy[1])
                | (kps_xy[:, 1] > bbox_xyxy[3]),
                torch.zeros_like(scores),
                scores,
            ).cpu()

        if disable_confidence:
            # Set all scores to 1
            scores = torch.ones_like(scores)

        keypoint_annotation = Keypoints2DAnnotation(
            view_id=view_id,
            frame_idx=frame_idx,
            subject_id=bbox_annotation.subject_id,
            xy=kps_xy.cpu(),
            scores=scores.cpu(),
            format=keypoints_format_name,
        )
        annotations.append(keypoint_annotation)

    return annotations


def batch_infer_keypoints(
    frames_bgr: torch.Tensor,
    keypoints_inferencer: Pose2DInferencer,
    use_half_precision: bool = True,
    bboxes_xyxy: list[np.ndarray] | None = None,
    bboxes_scores: list[np.ndarray] | None = None,
    filter_bbox_with_zero_score: bool = True,
) -> list[PoseDataSample]:
    assert (
        frames_bgr.ndim in [3, 4] and frames_bgr.shape[-1] == 3
    ), f"Expected frames_bgr to have shape (B, H, W, C) or (H, W, C), got {frames_bgr.shape}"
    assert frames_bgr.dtype == torch.uint8, "Expected frames_bgr to be uint8"

    if frames_bgr.ndim == 3:
        frames_bgr = frames_bgr.unsqueeze(0)

    batch_size = frames_bgr.shape[0]

    assert len(bboxes_xyxy) == batch_size
    assert len(bboxes_scores) == batch_size

    for bbox_xyxy in bboxes_xyxy:
        assert (
            bbox_xyxy.shape[1] == 4
        ), f"Expected bbox to have shape (*, 4), got {bbox_xyxy.shape}"

    for bbox_score in bboxes_scores:
        assert (
            bbox_score.shape[1] == 1
        ), f"Expected bbox score to have shape (*, 1), got {bbox_score.shape}"

    # Merge to (*, 5)
    bboxes = [
        np.concatenate((bbox_xyxy, bbox_score), axis=-1).astype(
            np.float16 if use_half_precision else np.float32
        )
        for bbox_xyxy, bbox_score in zip(bboxes_xyxy, bboxes_scores)
    ]

    with (
        torch.no_grad(),
        torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_half_precision,
        ),
    ):
        pose_results = next(
            keypoints_inferencer(
                frames_bgr,
                batch_size=batch_size,
                return_datasamples=True,
                merge_results=False,
                bboxes=bboxes,
                show=False,
                draw_bbox=False,
            )
        )["predictions"]

    # Filter out bad detections (bboxes with 0 score).
    # Usually these come from full-image detection when no bbox were detected.
    if filter_bbox_with_zero_score:
        for ds in pose_results:
            if "bbox_scores" in ds.pred_instances:
                ds.pred_instances = ds.pred_instances[ds.pred_instances.bbox_scores > 0]

    # Merge data samples based on img_path
    unique_img_paths = set([result.metainfo["img_path"] for result in pose_results])
    # Order numerically
    unique_img_paths = sorted(unique_img_paths, key=lambda x: int(x.split(".")[0]))

    merged_results = []

    for img_path in unique_img_paths:
        merged_results.append(
            merge_data_samples(
                [
                    result
                    for result in pose_results
                    if result.metainfo["img_path"] == img_path
                ]
            )
        )

    return merged_results
