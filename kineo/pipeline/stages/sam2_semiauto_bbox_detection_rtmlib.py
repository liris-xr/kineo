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
from tqdm import tqdm
import pickle
from math import sqrt

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    Annotations,
    BBox2DAnnotations,
    BBox2DAnnotationsMetadata,
    BBox2DAnnotation,
)
from kineo.pipeline.pipeline import Pipeline
from kineo.visualization.sam_prompt_widget import SAMVideoWidget
from sam2.sam2_generic_video_predictor import (
    SAM2GenericVideoPredictorState,
    SAM2GenericVideoPredictor,
)
from sam2.modeling.sam2_forgetful_memory import SAM2ForgetfulObjectMemoryBank

from sam2.build_sam import build_sam2_generic_video_predictor

from kineo.pipeline.stages.rtmlib.tools.object_detection.yolox import YOLOX

from dataclasses import dataclass
import cv2
from kineo.visualization.viz_2d import draw_bboxes
import matplotlib.pyplot as plt
import matplotlib
from kineo.maths import clamp
from collections import namedtuple

matplotlib.use("TkAgg")

BboxDetectionResult = namedtuple("BboxDetectionResult", ["bboxes", "scores"])

@dataclass
class SAM2SemiAutoBboxDetectionRtmlibRuntimeConfig:
    min_bbox_score: float = 0.3
    nms_iou_thr: float = 0.65
    nms_pre_top_k: int = 10
    det_category_id: int | None = 0
    batch_size: int = 16
    use_half_precision: bool = True
    use_cache: bool = True
    cache_output_path_template: str = "cache/{sequence_name}/{annotation_key}.pkl"
    show: bool = False
    frame_step: int = 1


class SAM2SemiAutoBboxDetectionRtmlibStage(
    PipelineStage[SAM2SemiAutoBboxDetectionRtmlibRuntimeConfig]
):
    """
    Stage for detecting bounding boxes from images by using MMDet.

    Produces :class:`BBox2DAnnotations` with the detected bounding boxes for each view with key "bboxes_2d".
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: SAM2SemiAutoBboxDetectionRtmlibRuntimeConfig,
        dynamic_runtime_cfg: (
            dict[str, SAM2SemiAutoBboxDetectionRtmlibRuntimeConfig] | None
        ) = None,
        det_model: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        det_model_input_shape_hw: tuple[int, int] = (416, 416),
        sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        sam2_model_weights: str = "./checkpoints/sam2.1_hiera_small.pt",
        sam2_vos_optimized: bool = False,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )
        self.det_model = YOLOX(
            onnx_model=det_model,
            model_input_size=det_model_input_shape_hw,
            device="cuda",
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam2_video_predictor = build_sam2_generic_video_predictor(
            sam2_model_cfg, sam2_model_weights, device=device, vos_optimized=sam2_vos_optimized
        )

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: SAM2SemiAutoBboxDetectionRtmlibRuntimeConfig,
    ):
        device = pipeline.device

        if runtime_cfg.use_cache:
            bboxes_cache_filepath = runtime_cfg.cache_output_path_template.format(
                sequence_name=sequence_name, annotation_key="bboxes_2d"
            )

            if os.path.exists(bboxes_cache_filepath):
                with open(bboxes_cache_filepath, "rb") as f:
                    bboxes_annotations = BBox2DAnnotations.from_dict(pickle.load(f))
                print(f"Loaded bboxes annotations from cache: {bboxes_cache_filepath}")
                annotations["bboxes_2d"] = bboxes_annotations.cpu()
                return

        self.sam2_video_predictor = self.sam2_video_predictor.to(device)

        states = _init_video_states(views, self.sam2_video_predictor)
        bboxes_annotations = _infer_bboxes(
            views, states, frame_step=runtime_cfg.frame_step, det_model=self.det_model, sam2_video_predictor=self.sam2_video_predictor, show=runtime_cfg.show, min_bbox_score=runtime_cfg.min_bbox_score, nms_iou_thr=runtime_cfg.nms_iou_thr
        )

        if runtime_cfg.use_cache:
            os.makedirs(os.path.dirname(bboxes_cache_filepath), exist_ok=True)

            if not os.path.exists(bboxes_cache_filepath):
                with open(bboxes_cache_filepath, "wb") as f:
                    print(
                        f"Saving bboxes annotations to cache: {bboxes_cache_filepath}"
                    )
                    pickle.dump(bboxes_annotations.to_dict(), f)

        self.sam2_video_predictor = self.sam2_video_predictor.cpu()
        annotations["bboxes_2d"] = bboxes_annotations.cpu()


def _init_video_states(
    views: list[ViewInput], sam2_video_predictor: SAM2GenericVideoPredictor
) -> list[SAM2GenericVideoPredictorState]:
    video_resolutions_hw = [view["frame_loader"].resolution_hw for view in views]
    states = [
        SAM2GenericVideoPredictorState(video_hw=video_resolution_hw, memory_bank=SAM2ForgetfulObjectMemoryBank(memory_window_size=10))
        for video_resolution_hw in video_resolutions_hw
    ]
    for view, predictor_state in zip(views, states):

        def continue_callback(widget: SAMVideoWidget):
            widget.close()

        prompt_widget = SAMVideoWidget(
            frame_sequence_loader=view["frame_loader"],
            sam2_video_predictor=sam2_video_predictor,
            sam2_video_predictor_state=predictor_state,
            continue_callback=continue_callback,
        )
        prompt_widget.set_title(f"View {view['view_id']}")
        plt.show()

    return states


def _batch_infer_bboxes(
    frames_rgb: torch.Tensor,
    det_model: YOLOX,
    min_bbox_score: float = 0.3,
    nms_iou_thr: float = 0.65
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    assert (
        frames_rgb.ndim in [3, 4] and frames_rgb.shape[-3] == 3
    ), f"Expected frames_bgr to have shape (C, H, W) or (B, C, H, W) got {frames_rgb.shape}"

    if frames_rgb.dtype != torch.uint8:
        frames_rgb = frames_rgb / 255.0

    has_batch_dim = frames_rgb.ndim == 4

    if not has_batch_dim:
        frames_rgb = frames_rgb.unsqueeze(0)

    frames_bgr = frames_rgb.flip(-3).permute(0, 2, 3, 1)
    batch_size = frames_rgb.shape[0]

    det_data_samples: list[BboxDetectionResult] = []

    det_model.nms_thr = nms_iou_thr
    det_model.score_thr = min_bbox_score

    for frame_bgr in frames_bgr:
        bboxes, scores = det_model(frame_bgr.cpu().numpy())
        bboxes = torch.from_numpy(bboxes).to(frames_rgb.device)
        scores = torch.from_numpy(scores).to(frames_rgb.device)
        det_data_samples.append(BboxDetectionResult(bboxes=bboxes, scores=scores))

    batch_bboxes = []
    batch_scores = []

    for i in range(batch_size):
        bboxes = det_data_samples[i].bboxes
        scores = det_data_samples[i].scores
        batch_bboxes.append(bboxes)
        batch_scores.append(scores)

    return batch_bboxes, batch_scores


def _infer_bboxes(
    views: list[ViewInput],
    states: list[SAM2GenericVideoPredictorState],
    det_model: YOLOX,
    sam2_video_predictor: SAM2GenericVideoPredictor,
    frame_step: int = 1,
    show: bool = False,
    min_bbox_score: float = 0.3,
    nms_iou_thr: float = 0.65,
    default_subject_id="subject_0"
) -> BBox2DAnnotations:

    all_bboxes_annotations: list[BBox2DAnnotation] = []

    n_total_frames = sum(view["frame_loader"].n_frames for view in views)
    n_total_inference_frames = sum(
        len(_get_frames_batch(view["frame_loader"].n_frames, frame_step))
        for view in views
    )
    pbar = tqdm(total=n_total_frames, desc="Propagating SAM2 masks to bboxes")

    batch_size = 1 # SAM doesn't support batch inference
    n_inference_frames_processed = 0

    for view, state in zip(views, states):
        view_bboxes_annotations: list[BBox2DAnnotation] = []

        frame_loader = view["frame_loader"]
        view_n_frames = frame_loader.n_frames
        inference_frames = _get_frames_batch(view_n_frames, frame_step)

        inference_frames = _get_frames_batch(view_n_frames, frame_step)

        for batch_start in range(0, len(inference_frames), batch_size):
            batch_end = min(batch_start + batch_size, len(inference_frames))
            batch_frames = inference_frames[batch_start:batch_end]
            actual_batch_size = len(batch_frames)

            # Load batch of frames
            frames_rgb = frame_loader.load_frames_at(
                frame_indices=torch.tensor(batch_frames)
            ).to(sam2_video_predictor.device)

            for frame_idx, frame_rgb in zip(batch_frames, frames_rgb):

                results = sam2_video_predictor.forward(
                    state=state,
                    frame_idx=frame_idx,
                    frame=frame_rgb,
                    prompts=[],
                    multimask_output=True,
                    reverse_tracking=False,
                    create_memory=True,
                )
                obj_results = results.get(0, None)

                if obj_results is None:
                    continue

                mask = (obj_results.best_mask_logits > 0).squeeze()

                batch_bboxes, batch_scores = _batch_infer_bboxes(
                    frame_rgb, det_model, min_bbox_score=min_bbox_score, nms_iou_thr=nms_iou_thr
                )

                bboxes = batch_bboxes[0]
                scores = batch_scores[0]

                best_bbox = None
                best_bbox_tracking_score = 0
                best_bbox_score = 0

                bbox_score_thr = 0.5
                tracking_score_thr = 0.5

                for bbox, score in zip(bboxes, scores):
                    iou = _mask_bbox_iou(mask, bbox)
                    intersection = _mask_bbox_intersection(mask, bbox)

                    tracking_score = sqrt(iou * intersection)

                    if (
                        tracking_score > tracking_score_thr
                        and tracking_score > best_bbox_tracking_score
                    ):
                        best_bbox = bbox
                        best_bbox_tracking_score = tracking_score
                        best_bbox_score = score

                if best_bbox is not None and best_bbox_score > bbox_score_thr:
                    bbox_annotation = BBox2DAnnotation(
                        view_id=view["view_id"],
                        frame_idx=frame_idx,
                        subject_id=default_subject_id,
                        xyxy=best_bbox.reshape(4).cpu(),
                        score=best_bbox_score.item(),
                        category_id=0,
                    )

                    if show:
                        frame_bgr = frame_rgb.flip(0).permute(1, 2, 0).contiguous().cpu().numpy()
                        frame_bgr = draw_bboxes(frame_bgr, best_bbox.reshape(4).cpu().numpy())
                        cv2.imshow("Frame", frame_bgr)
                        cv2.waitKey(1)

                    view_bboxes_annotations.append(bbox_annotation)

                    n_inference_frames_processed += 1
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
        metadata=BBox2DAnnotationsMetadata(),
        annotations=all_bboxes_annotations,
    )

    if show:
        cv2.destroyAllWindows()

    return bboxes_annotations


def _mask_bbox_intersection(mask: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    assert mask.ndim == 2, f"Expected mask to have shape (H, W), got {mask.shape}"
    x1, y1, x2, y2 = bbox
    x1 = x1.floor().clamp(0, mask.shape[1]).int()
    y1 = y1.floor().clamp(0, mask.shape[0]).int()
    x2 = x2.ceil().clamp(0, mask.shape[1]).int()
    y2 = y2.ceil().clamp(0, mask.shape[0]).int()
    bbox_mask = torch.zeros_like(mask)
    bbox_mask[y1:y2, x1:x2] = 1
    return (mask * bbox_mask).sum() / mask.sum()


def _mask_bbox_iou(mask: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    assert mask.ndim == 2, f"Expected mask to have shape (H, W), got {mask.shape}"
    x1, y1, x2, y2 = bbox
    x1 = x1.floor().clamp(0, mask.shape[1]).int()
    y1 = y1.floor().clamp(0, mask.shape[0]).int()
    x2 = x2.ceil().clamp(0, mask.shape[1]).int()
    y2 = y2.ceil().clamp(0, mask.shape[0]).int()

    bbox_mask = torch.zeros_like(mask)
    bbox_mask[y1:y2, x1:x2] = 1

    intersection = mask * bbox_mask
    union = (mask + bbox_mask).clamp(0, 1)

    if union.sum() == 0:
        return 0

    return intersection.sum() / union.sum()

def _get_frames_batch(n_frames: int, frame_step: int) -> list[int]:
    """Get list of frame indices to run inference on."""
    if frame_step == 1:
        return list(range(n_frames))

    frames_batch = list(range(0, n_frames, frame_step))

    # Always include the last frame if it's not already included
    if (n_frames - 1) not in frames_batch and n_frames > 0:
        frames_batch.append(n_frames - 1)

    return sorted(frames_batch)
