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
import warnings

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    Annotations,
    BBox2DAnnotations,
    BBox2DAnnotationsMetadata,
    BBox2DAnnotation,
)
from kineo.pipeline.pipeline import Pipeline

from mmdet.apis import DetInferencer
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

from dataclasses import dataclass

from kineo.maths import clamp

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
class MMDetBboxDetectionRuntimeConfig:
    min_bbox_score: float = 0.3
    nms_iou_thr: float = 0.65
    nms_pre_top_k: int = 10
    best_bbox_only: bool = False
    det_category_id: int | None = 0
    batch_size: int = 16
    use_half_precision: bool = True
    use_cache: bool = True
    cache_output_path_template: str = "cache/{sequence_name}/{annotation_key}.pkl"
    default_subject_id: str = "subject_0"
    frame_step: int = 1


class MMDetBboxDetectionStage(PipelineStage[MMDetBboxDetectionRuntimeConfig]):
    """
    Stage for detecting bounding boxes from images by using MMDet.

    Produces :class:`BBox2DAnnotations` with the detected bounding boxes for each view with key "bboxes_2d".
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: MMDetBboxDetectionRuntimeConfig,
        dynamic_runtime_cfg: dict[str, MMDetBboxDetectionRuntimeConfig] | None = None,
        det_model: str = "rtmdet_m_640-8xb32_coco-person",
        det_model_weights: str | None = None,
        det_model_scope: str = "mmdet",
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

        self.det_inferencer.model = self.det_inferencer.model.cpu()
        self.det_inferencer.model = self.det_inferencer.model.eval()

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: MMDetBboxDetectionRuntimeConfig,
    ):
        device = pipeline.device

        bboxes_annotations: BBox2DAnnotations | None = None

        if runtime_cfg.use_cache:
            bboxes_cache_filepath = runtime_cfg.cache_output_path_template.format(
                sequence_name=sequence_name, annotation_key="bboxes_2d"
            )

            if os.path.exists(bboxes_cache_filepath):
                with open(bboxes_cache_filepath, "rb") as f:
                    bboxes_annotations = BBox2DAnnotations.from_dict(pickle.load(f))
                bboxes_annotations = bboxes_annotations.filter_by_view_ids(
                    [view["view_id"] for view in views]
                )
                print(f"Loaded bboxes annotations from cache: {bboxes_cache_filepath}")
                annotations["bboxes_2d"] = bboxes_annotations.cpu()
                return

        self.det_inferencer.model = self.det_inferencer.model.to(device)
        self.det_inferencer.model.test_cfg["nms_pre"] = runtime_cfg.nms_pre_top_k

        bboxes_annotations = self._infer_bboxes(
            views=views,
            det_inferencer=self.det_inferencer,
            batch_size=runtime_cfg.batch_size,
            use_half_precision=runtime_cfg.use_half_precision,
            det_category_id=runtime_cfg.det_category_id,
            best_bbox_only=runtime_cfg.best_bbox_only,
            min_bbox_score=runtime_cfg.min_bbox_score,
            nms_iou_thr=runtime_cfg.nms_iou_thr,
            default_subject_id=runtime_cfg.default_subject_id,
            frame_step=runtime_cfg.frame_step,
        )

        if runtime_cfg.use_cache and not os.path.exists(bboxes_cache_filepath):
            os.makedirs(os.path.dirname(bboxes_cache_filepath), exist_ok=True)

            if not os.path.exists(bboxes_cache_filepath):
                with open(bboxes_cache_filepath, "wb") as f:
                    print(
                        f"Saving bboxes annotations to cache: {bboxes_cache_filepath}"
                    )
                    pickle.dump(bboxes_annotations.to_dict(), f)

        self.det_inferencer.model = self.det_inferencer.model.cpu()
        annotations["bboxes_2d"] = bboxes_annotations.cpu()

    def _infer_bboxes(
        self,
        views: list[ViewInput],
        det_inferencer: DetInferencer,
        batch_size: int = 16,
        use_half_precision: bool = True,
        det_category_id: int | None = None,
        min_bbox_score: float = 0.3,
        nms_iou_thr: float = 0.65,
        best_bbox_only: bool = True,
        default_subject_id: str = "subject_0",
        frame_step: int = 1,
    ) -> tuple[BBox2DAnnotations]:
        """
        Infer bounding boxes for all views.
        """
        all_bboxes_annotations: list[BBox2DAnnotation] = []

        n_total_frames = sum(view["frame_loader"].n_frames for view in views)
        n_total_inference_frames = sum(
            len(_get_frames_batch(view["frame_loader"].n_frames, frame_step))
            for view in views
        )
        pbar = tqdm(
            total=n_total_frames, desc="Inferring bboxes", leave=False, unit="frames"
        )

        n_inference_frames_processed = 0

        for view in views:

            view_annotations: list[BBox2DAnnotation] = []

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

                det_data_samples = batch_infer_bboxes(
                    frames_bgr=frames_bgr,
                    det_inferencer=det_inferencer,
                    min_bbox_score=min_bbox_score,
                    nms_iou_thr=nms_iou_thr,
                    det_category_id=det_category_id,
                    use_half_precision=use_half_precision,
                )

                # Keep only one bbox per frame based on the score
                if best_bbox_only:
                    det_data_samples = [
                        (
                            keep_best_bbox(det_data_sample)
                            if det_data_sample is not None
                            else None
                        )
                        for det_data_sample in det_data_samples
                    ]

                # Convert the DetDataSample to BBox2DAnnotation
                for batch_idx in range(actual_batch_size):
                    det_data_sample = det_data_samples[batch_idx]
                    frame_idx = batch_frames[batch_idx]

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

                        # TODO: get the subject_id from the track_id

                        bbox_annotation = BBox2DAnnotation(
                            view_id=view_id,
                            frame_idx=frame_idx,
                            subject_id=default_subject_id,  # TODO: replace with track_id
                            category_id=label.item(),
                            xyxy=bbox_xyxy.reshape(4).to(torch.float32).cpu(),
                            score=score.item(),
                        )
                        view_annotations.append(bbox_annotation)

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
                metadata=BBox2DAnnotationsMetadata(), annotations=view_annotations
            )

            if frame_step > 1:
                # Interpolate bboxes to all frames
                all_frames = list(range(view_n_frames))
                view_bboxes_annotations = (
                    view_bboxes_annotations.interpolate_by_frame_indices(
                        target_frame_indices=all_frames, max_frame_idx_diff=frame_step
                    )
                )
            all_bboxes_annotations.extend(view_bboxes_annotations._annotations)

        pbar.close()

        bboxes_annotations = BBox2DAnnotations(
            metadata=BBox2DAnnotationsMetadata(), annotations=all_bboxes_annotations
        ).cpu()

        return bboxes_annotations


def batch_infer_bboxes(
    frames_bgr: torch.Tensor,
    det_inferencer: DetInferencer,
    use_half_precision: bool = True,
    min_bbox_score: float = 0.3,
    nms_iou_thr: float = 0.65,
    det_category_id: int | None = None,
    show: bool = False,
) -> list[DetDataSample]:
    assert (
        frames_bgr.ndim in [3, 4] and frames_bgr.shape[-1] == 3
    ), f"Expected frames_bgr to have shape (B, H, W, C) or (H, W, C), got {frames_bgr.shape}"
    assert frames_bgr.dtype == torch.uint8, "Expected frames_bgr to be uint8"

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
        det_results = det_inferencer(
            frames_bgr,
            batch_size=batch_size,
            return_datasamples=True,
            cat_ids=[det_category_id] if det_category_id is not None else None,
            min_bbox_score=min_bbox_score,
            nms_thr=nms_iou_thr,
            show=show,
        )["predictions"]

    return det_results


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


def _get_frames_batch(n_frames: int, frame_step: int) -> list[int]:
    """Get list of frame indices to run inference on."""
    if frame_step == 1:
        return list(range(n_frames))

    frames_batch = list(range(0, n_frames, frame_step))

    # Always include the last frame if it's not already included
    if (n_frames - 1) not in frames_batch and n_frames > 0:
        frames_batch.append(n_frames - 1)

    return sorted(frames_batch)
