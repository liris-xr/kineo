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
import os.path as osp
from tqdm import tqdm
import warnings

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    Annotations,
    BBox2DAnnotations,
    BBox2DAnnotationsMetadata,
    BBox2DAnnotation,
    Keypoints2DAnnotations,
    Keypoints2DAnnotation,
    KeypointsFormat,
    Keypoints2DAnnotationsMetadata,
)
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline import per_view_cache

from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmpose.apis import Pose2DInferencer
from mmpose.structures import PoseDataSample
from mmpose.structures.utils import merge_data_samples

from dataclasses import dataclass

from kineo.maths import clamp
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
class MMLabBboxKeypointsDetectionRuntimeConfig:
    bbox_thr: float = 0.3
    nms_iou_thr: float = 0.65
    nms_pre_top_k: int = 10
    best_bbox_only: bool = False
    det_category_id: int | None = 0
    batch_size: int = 16
    use_half_precision: bool = True
    use_cache: bool = True
    cache_output_path_template: str = (
        "cache/{sequence_name}/{annotation_key}/{view_id}.pkl"
    )
    default_subject_id: str = "subject_0"
    frame_step: int = 1
    force_zero_scores_outside_bbox: bool = True
    use_flip_test: bool = True


class MMLabBboxKeypointsDetectionStage(
    PipelineStage[MMLabBboxKeypointsDetectionRuntimeConfig]
):
    """
    Stage for detecting bounding boxes from images by using MMDet.

    Produces :class:`BBox2DAnnotations` with the detected bounding boxes for each view with key "bboxes_2d".
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: MMLabBboxKeypointsDetectionRuntimeConfig,
        dynamic_runtime_cfg: dict[str, MMLabBboxKeypointsDetectionRuntimeConfig]
        | None = None,
        det_model: str = "rtmdet_m_640-8xb32_coco-person",
        det_model_weights: str | None = None,
        det_model_scope: str = "mmdet",
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
        self.det_inferencer = DetInferencer(
            model=det_model,
            weights=det_model_weights,
            scope=det_model_scope,
            show_progress=False,
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

        self.det_inferencer.model = self.det_inferencer.model.cpu()
        self.det_inferencer.model = self.det_inferencer.model.eval()

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: MMLabBboxKeypointsDetectionRuntimeConfig,
    ):
        device = pipeline.device

        keypoints_format = KeypointsFormat.from_mmpose_dataset(
            self.keypoints_inferencer.model.dataset_meta["dataset_name"]
        )

        def infer_missing(missing_views: list[ViewInput]) -> dict[str, Annotations]:
            self.det_inferencer.model = self.det_inferencer.model.to(device)
            self.det_inferencer.model.test_cfg["nms_pre"] = runtime_cfg.nms_pre_top_k
            self.keypoints_inferencer.model = self.keypoints_inferencer.model.to(device)

            try:
                bboxes, keypoints = self._infer_bboxes_and_keypoints(
                    views=missing_views,
                    det_inferencer=self.det_inferencer,
                    keypoints_inferencer=self.keypoints_inferencer,
                    batch_size=runtime_cfg.batch_size,
                    use_half_precision=runtime_cfg.use_half_precision,
                    det_category_id=runtime_cfg.det_category_id,
                    best_bbox_only=runtime_cfg.best_bbox_only,
                    bbox_thr=runtime_cfg.bbox_thr,
                    nms_iou_thr=runtime_cfg.nms_iou_thr,
                    default_subject_id=runtime_cfg.default_subject_id,
                    frame_step=runtime_cfg.frame_step,
                    force_zero_scores_outside_bbox=runtime_cfg.force_zero_scores_outside_bbox,
                    use_flip_test=runtime_cfg.use_flip_test,
                )
            finally:
                self.det_inferencer.model = self.det_inferencer.model.cpu()
                self.keypoints_inferencer.model = self.keypoints_inferencer.model.cpu()

            return {"bboxes_2d": bboxes, "keypoints_2d": keypoints}

        cached = per_view_cache.load_or_infer_per_view(
            views=views,
            specs={
                "bboxes_2d": per_view_cache.PerViewCacheSpec(
                    annotations_cls=BBox2DAnnotations,
                    metadata=BBox2DAnnotationsMetadata(),
                ),
                "keypoints_2d": per_view_cache.PerViewCacheSpec(
                    annotations_cls=Keypoints2DAnnotations,
                    metadata=Keypoints2DAnnotationsMetadata(formats=[keypoints_format]),
                ),
            },
            infer_missing=infer_missing,
            sequence_name=sequence_name,
            cache_output_path_template=runtime_cfg.cache_output_path_template,
            use_cache=runtime_cfg.use_cache,
        )

        annotations["bboxes_2d"] = cached["bboxes_2d"].cpu()
        annotations["keypoints_2d"] = cached["keypoints_2d"].cpu()

    def _infer_bboxes_and_keypoints(
        self,
        views: list[ViewInput],
        det_inferencer: DetInferencer,
        keypoints_inferencer: Pose2DInferencer,
        batch_size: int = 16,
        use_half_precision: bool = True,
        det_category_id: int | None = None,
        bbox_thr: float = 0.3,
        nms_iou_thr: float = 0.65,
        best_bbox_only: bool = True,
        default_subject_id: str = "subject_0",
        frame_step: int = 1,
        force_zero_scores_outside_bbox: bool = True,
        use_flip_test: bool = True,
        filter_bbox_with_zero_score: bool = True,
    ) -> tuple[BBox2DAnnotations]:
        """
        Infer bounding boxes for all views.
        """

        keypoints_format_name = keypoints_inferencer.model.dataset_meta["dataset_name"]
        keypoints_format = KeypointsFormat.from_mmpose_dataset(keypoints_format_name)
        keypoints_metadata = Keypoints2DAnnotationsMetadata(formats=[keypoints_format])

        self.keypoints_inferencer.model.test_cfg.flip_test = use_flip_test

        all_bboxes_annotations: list[BBox2DAnnotation] = []
        all_keypoints_annotations: list[Keypoints2DAnnotation] = []

        n_total_frames = sum(view["frame_loader"].n_frames for view in views)
        n_total_inference_frames = sum(
            len(_get_frames_batch(view["frame_loader"].n_frames, frame_step))
            for view in views
        )
        pbar = tqdm(
            total=n_total_frames, desc="Inferring bboxes and keypoints", leave=False, unit="frames"
        )

        n_inference_frames_processed = 0

        for view in views:
            view_bboxes_annotations: list[BBox2DAnnotation] = []
            view_keypoints_annotations: list[Keypoints2DAnnotation] = []

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

                batch_bboxes_annotations = detect_bboxes(
                    view_id=view_id,
                    frames_bgr=frames_bgr,
                    batch_frames=batch_frames,
                    det_inferencer=det_inferencer,
                    use_half_precision=use_half_precision,
                    bbox_thr=bbox_thr,
                    nms_iou_thr=nms_iou_thr,
                    det_category_id=det_category_id,
                    best_bbox_only=best_bbox_only,
                    default_subject_id=default_subject_id,
                )

                batch_keypoints_annotations = detect_keypoints(
                    view_id=view_id,
                    frames_bgr=frames_bgr,
                    batch_frames=batch_frames,
                    bboxes_annotations=batch_bboxes_annotations,
                    keypoints_inferencer=keypoints_inferencer,
                    keypoints_metadata=keypoints_metadata,
                    use_half_precision=use_half_precision,
                    filter_bbox_with_zero_score=filter_bbox_with_zero_score,
                    force_zero_scores_outside_bbox=force_zero_scores_outside_bbox,
                    default_subject_id=default_subject_id,
                )

                view_bboxes_annotations.extend(batch_bboxes_annotations)
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

            view_bboxes_annotations = BBox2DAnnotations(
                metadata=BBox2DAnnotationsMetadata(),
                annotations=view_bboxes_annotations,
            )

            view_keypoints_annotations = Keypoints2DAnnotations(
                metadata=keypoints_metadata,
                annotations=view_keypoints_annotations,
            )

            if frame_step > 1:
                # Interpolate bboxes to all frames
                all_frames = list(range(view_n_frames))
                view_bboxes_annotations = (
                    view_bboxes_annotations.interpolate_by_frame_indices(
                        target_frame_indices=all_frames, max_frame_idx_diff=frame_step
                    )
                )
                view_keypoints_annotations = (
                    view_keypoints_annotations.interpolate_by_frame_indices(
                        target_frame_indices=all_frames, max_frame_idx_diff=frame_step
                    )
                )
            all_bboxes_annotations.extend(view_bboxes_annotations._annotations)
            all_keypoints_annotations.extend(view_keypoints_annotations._annotations)

        pbar.close()

        bboxes_annotations = BBox2DAnnotations(
            metadata=BBox2DAnnotationsMetadata(), annotations=all_bboxes_annotations
        ).cpu()

        keypoints_annotations = Keypoints2DAnnotations(
            metadata=keypoints_metadata,
            annotations=all_keypoints_annotations,
        ).cpu()

        return bboxes_annotations, keypoints_annotations


def detect_keypoints(
    view_id: str,
    frames_bgr: torch.Tensor,
    batch_frames: list[int],
    bboxes_annotations: BBox2DAnnotations,
    keypoints_inferencer: Pose2DInferencer,
    keypoints_metadata: Keypoints2DAnnotationsMetadata,
    use_half_precision: bool = True,
    filter_bbox_with_zero_score: bool = True,
    force_zero_scores_outside_bbox: bool = True,
    default_subject_id: str = "subject_0",
) -> Keypoints2DAnnotations:
    assert frames_bgr.ndim in [3, 4] and frames_bgr.shape[-1] == 3, (
        f"Expected frames_bgr to have shape (B, H, W, C) or (H, W, C), got {frames_bgr.shape}"
    )
    assert frames_bgr.dtype == torch.uint8, "Expected frames_bgr to be uint8"

    actual_batch_size = len(batch_frames)

    bboxes: list[np.ndarray] = [np.zeros((0, 5)) for _ in range(actual_batch_size)]

    for batch_idx in range(actual_batch_size):
        frame_idx = batch_frames[batch_idx]

        frame_bboxes = bboxes_annotations.filter_by_view_id(
            view_id
        ).filter_by_frame_idx(frame_idx)

        if len(frame_bboxes) == 0:
            continue

        bboxes[batch_idx] = np.stack(
            [
                np.concatenate(
                    (bbox.xyxy.cpu().numpy(), np.array([bbox.score])),
                    axis=-1,
                )
                for bbox in frame_bboxes
            ],
            axis=0,
        )

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
                batch_size=actual_batch_size,
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

    if force_zero_scores_outside_bbox:
        for ds in pose_results:
            if "bboxes" in ds.pred_instances:
                # For keypoints outside of the bbox, set the score to 0
                for instance_idx in range(len(ds.pred_instances.keypoints)):
                    bbox_xyxy = ds.pred_instances.bboxes[instance_idx]
                    kps_xy = ds.pred_instances.keypoints[instance_idx]
                    kps_scores = ds.pred_instances.keypoints_visible[instance_idx]

                    outside_mask = (
                        (kps_xy[:, 0] < bbox_xyxy[0])
                        | (kps_xy[:, 0] > bbox_xyxy[2])
                        | (kps_xy[:, 1] < bbox_xyxy[1])
                        | (kps_xy[:, 1] > bbox_xyxy[3])
                    )

                    kps_scores[outside_mask] = 0

    # Merge data samples based on img_path
    unique_img_paths = set([result.metainfo["img_path"] for result in pose_results])
    # Order numerically
    unique_img_paths = sorted(unique_img_paths, key=lambda x: int(x.split(".")[0]))

    pose_data_samples = []

    for img_path in unique_img_paths:
        pose_data_samples.append(
            merge_data_samples(
                [
                    result
                    for result in pose_results
                    if result.metainfo["img_path"] == img_path
                ]
            )
        )

    annotations: list[Keypoints2DAnnotation] = []

    for batch_idx in range(actual_batch_size):
        frame_idx = batch_frames[batch_idx]
        frame_data_sample = pose_data_samples[batch_idx]

        keypoints_annotations = _create_keypoints_annotations(
            frame_data_sample=frame_data_sample,
            view_id=view_id,
            frame_idx=frame_idx,
            keypoints_format_name=keypoints_metadata.formats[0].name,
            default_subject_id=default_subject_id,
        )
        annotations.extend(keypoints_annotations)

    return Keypoints2DAnnotations(
        metadata=keypoints_metadata,
        annotations=annotations,
    )


def detect_bboxes(
    view_id: str,
    frames_bgr: torch.Tensor,
    batch_frames: list[int],
    det_inferencer: DetInferencer,
    use_half_precision: bool = True,
    bbox_thr: float = 0.3,
    nms_iou_thr: float = 0.65,
    det_category_id: int | None = None,
    best_bbox_only: bool = True,
    default_subject_id: str = "subject_0",
) -> BBox2DAnnotations:
    assert frames_bgr.ndim in [3, 4] and frames_bgr.shape[-1] == 3, (
        f"Expected frames_bgr to have shape (B, H, W, C) or (H, W, C), got {frames_bgr.shape}"
    )
    assert frames_bgr.dtype == torch.uint8, "Expected frames_bgr to be uint8"

    actual_batch_size = len(batch_frames)

    if frames_bgr.ndim == 3:
        frames_bgr = frames_bgr.unsqueeze(0)
    batch_size = frames_bgr.shape[0]

    with (
        torch.no_grad(),
        torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_half_precision,
        ),
    ):
        det_data_samples = det_inferencer(
            frames_bgr,
            batch_size=batch_size,
            return_datasamples=True,
            cat_ids=[det_category_id] if det_category_id is not None else None,
            bbox_thr=bbox_thr,
            nms_thr=nms_iou_thr,
            show=False,
        )["predictions"]

    # Keep only one bbox per frame based on the score
    if best_bbox_only:
        det_data_samples = [
            (keep_best_bbox(det_data_sample) if det_data_sample is not None else None)
            for det_data_sample in det_data_samples
        ]

    annotations: list[BBox2DAnnotation] = []

    # Convert the DetDataSample to BBox2DAnnotation
    for batch_idx in range(actual_batch_size):
        det_data_sample = det_data_samples[batch_idx]
        frame_idx = batch_frames[batch_idx]

        bboxes_annotations = _create_bboxes_annotations(
            det_data_sample=det_data_sample,
            view_id=view_id,
            frame_idx=frame_idx,
            default_subject_id=default_subject_id,
        )
        annotations.extend(bboxes_annotations)

    return BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(),
        annotations=annotations,
    )


def keep_best_bbox(det_result: DetDataSample) -> DetDataSample:
    result = det_result.clone()
    bboxes = result.pred_instances.bboxes
    scores = result.pred_instances.scores
    labels = result.pred_instances.labels

    if len(bboxes) == 0:
        return result

    best_idx = torch.argmax(scores)

    bboxes = bboxes[best_idx].unsqueeze(0)
    scores = scores[best_idx].unsqueeze(0)
    labels = labels[best_idx].unsqueeze(0)

    pred_instances = InstanceData(metainfo=result.metainfo)
    pred_instances.bboxes = bboxes
    pred_instances.scores = scores
    pred_instances.labels = labels
    result.pred_instances = pred_instances
    return result


def _create_bboxes_annotations(
    det_data_sample: DetDataSample,
    view_id: str,
    frame_idx: int,
    default_subject_id: str,
) -> list[BBox2DAnnotation]:
    annotations = []
    n_instances = len(det_data_sample.pred_instances.bboxes)

    if n_instances > 1:
        warnings.warn(
            "MMDetBboxDetectionStage only supports one bbox per frame for now (no tracker implemented). The best bbox will be used."
        )
        det_data_sample = keep_best_bbox(det_data_sample)
        n_instances = 1

    for instance_idx in range(n_instances):
        bbox_xyxy = det_data_sample.pred_instances.bboxes[instance_idx]
        score = det_data_sample.pred_instances.scores[instance_idx]
        label = det_data_sample.pred_instances.labels[instance_idx]
        bbox_annotation = BBox2DAnnotation(
            view_id=view_id,
            frame_idx=frame_idx,
            subject_id=default_subject_id,
            category_id=label.item(),
            xyxy=bbox_xyxy.reshape(4).to(torch.float32).cpu(),
            score=score.item(),
        )
        annotations.append(bbox_annotation)
    return annotations


def _create_keypoints_annotations(
    frame_data_sample: PoseDataSample,
    view_id: str,
    frame_idx: int,
    keypoints_format_name: str,
    default_subject_id: str,
) -> list[Keypoints2DAnnotation]:
    """Create keypoints annotations from inference results."""
    annotations = []
    n_instances = len(frame_data_sample.pred_instances.keypoints)

    for instance_idx in range(n_instances):
        keypoints_xy = frame_data_sample.pred_instances.keypoints[instance_idx]
        keypoints_scores = frame_data_sample.pred_instances.keypoints_visible[
            instance_idx
        ]

        kps_xy = torch.from_numpy(keypoints_xy).to(torch.float32)
        scores = torch.from_numpy(keypoints_scores).to(torch.float32)

        keypoint_annotation = Keypoints2DAnnotation(
            view_id=view_id,
            frame_idx=frame_idx,
            subject_id=default_subject_id,
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
