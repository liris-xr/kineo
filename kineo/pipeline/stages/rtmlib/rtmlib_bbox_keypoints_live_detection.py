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

import cv2

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

from dataclasses import dataclass

from kineo.geometry.triangulation import triangulate_points
from kineo.geometry.transformations import undistort_points
from kineo.geometry.metrics import pairwise_reprojection_consensus_score
from kineo.maths import clamp
import numpy as np

from kineo.pipeline.stages.rtmlib.tools.object_detection.yolox import YOLOX
from kineo.pipeline.stages.rtmlib.tools.pose_estimation.rtmpose import RTMPose
from kineo.pipeline.stages.rtmlib.skeleton.coco17 import coco17
from kineo.pipeline.stages.rtmlib.skeleton.coco133 import coco133
from kineo.pipeline.stages.rtmlib.skeleton.openpose18 import openpose18
from kineo.pipeline.stages.rtmlib.skeleton.openpose134 import openpose134
from kineo.pipeline.stages.rtmlib.skeleton.halpe26 import halpe26
from kineo.pipeline.stages.rtmlib.skeleton.hand21 import hand21

from kineo.visualization.viz_2d import draw_bboxes, draw_keypoints

from collections import namedtuple

import rerun as rr

BboxDetectionResult = namedtuple("BboxDetectionResult", ["bboxes", "scores"])
KeypointsDetectionResult = namedtuple("KeypointsDetectionResult", ["keypoints", "scores"])

@dataclass
class RtmlibBboxKeypointsLiveDetectionRuntimeConfig:
    bbox_thr: float = 0.3
    nms_iou_thr: float = 0.65
    best_bbox_only: bool = False
    det_category_id: int | None = 0
    default_subject_id: str = "subject_0"
    frame_step: int = 1
    show: bool = False


class RtmlibBboxKeypointsLiveDetectionStage(
    PipelineStage[RtmlibBboxKeypointsLiveDetectionRuntimeConfig]
):
    """
    Stage for detecting bounding boxes from images by using MMDet.

    Produces :class:`BBox2DAnnotations` with the detected bounding boxes for each view with key "bboxes_2d".
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: RtmlibBboxKeypointsLiveDetectionRuntimeConfig,
        dynamic_runtime_cfg: dict[str, RtmlibBboxKeypointsLiveDetectionRuntimeConfig]
        | None = None,
        bbox_model: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        bbox_model_input_shape_hw: tuple[int, int] = (416, 416),
        keypoints_model: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        keypoints_model_input_shape_hw: tuple[int, int] = (288, 384),
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )

        self.det_model = YOLOX(
            onnx_model=bbox_model,
            model_input_size=bbox_model_input_shape_hw,
            device="cuda",
        )
        self.keypoints_model = RTMPose(
            onnx_model=keypoints_model,
            model_input_size=keypoints_model_input_shape_hw,
            device="cuda",
        )

        rr.init("live_rtmlib_bbox_keypoints_detection")

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
        runtime_cfg: RtmlibBboxKeypointsLiveDetectionRuntimeConfig,
    ):
        device = pipeline.device

        kps2d_score_threshold = 0.3
        kps3d_score_threshold = 0.4
        skeleton_color = np.array([0, 255, 0, 255])
        world_points_score_threshold = 0.2

        rr.spawn()

        camera_intrinsics: CameraIntrinsicsAnnotations = annotations["cameras_intrinsics"]
        camera_extrinsics: CameraExtrinsicsAnnotations = annotations["cameras_extrinsics"]

        n_views = len(views)
        Ks = torch.empty((n_views, 3, 3), dtype=torch.float32, device=device)
        Rts = torch.empty((n_views, 3, 4), dtype=torch.float32, device=device)
        Ps = torch.empty((n_views, 3, 4), dtype=torch.float32, device=device)
        distortion_coefficients = []
        distortion_models = []

        for view_idx, view in enumerate(views):
            cam_intrinsics = camera_intrinsics.filter_by_view_id(view["view_id"]).first_or_default()
            cam_extrinsics = camera_extrinsics.filter_by_view_id(view["view_id"]).first_or_default()
            Rts[view_idx] = cam_extrinsics.Rt
            Ks[view_idx] = cam_intrinsics.K
            Ps[view_idx] = Ks[view_idx] @ Rts[view_idx]
            distortion_coefficients.append(cam_intrinsics.distortion_coefficients)
            distortion_models.append(cam_intrinsics.distortion_model.value)

        world_reconstructed_scene_annotations: WorldReconstructedSceneAnnotations = annotations.get(
            "world_reconstructed_scene", None)

        if world_reconstructed_scene_annotations is not None:
            world_reconstructed_scene = world_reconstructed_scene_annotations.first_or_default()
            world_points_xyz = world_reconstructed_scene.points_xyz
            world_points_colors = world_reconstructed_scene.points_colors
            world_points_confidences = world_reconstructed_scene.points_confidences

            valid_world_points_mask = world_points_confidences > world_points_score_threshold
            valid_world_points_xyz = world_points_xyz[valid_world_points_mask]
            valid_world_points_colors = world_points_colors[valid_world_points_mask]

            rr.set_time("frame", sequence=0)
            rr.log("3d/world", rr.Points3D(positions=valid_world_points_xyz.cpu().numpy(),
                                           colors=valid_world_points_colors.cpu().numpy()))

        self._infer_bboxes_and_keypoints(
            views=views,
            Ks=Ks,
            Rts=Rts,
            Ps=Ps,
            distortion_coefficients=distortion_coefficients,
            distortion_models=distortion_models,
            det_model=self.det_model,
            keypoints_model=self.keypoints_model,
            det_category_id=runtime_cfg.det_category_id,
            best_bbox_only=runtime_cfg.best_bbox_only,
            kps2d_score_threshold=kps2d_score_threshold,
            kps3d_score_threshold=kps3d_score_threshold,
            skeleton_color=skeleton_color,
            bbox_thr=runtime_cfg.bbox_thr,
            nms_iou_thr=runtime_cfg.nms_iou_thr,
            default_subject_id=runtime_cfg.default_subject_id,
            frame_step=runtime_cfg.frame_step,
            show=runtime_cfg.show,
            device=device,
        )

    def _infer_bboxes_and_keypoints(
        self,
        views: list[ViewInput],
        Ks: torch.Tensor,
        Rts: torch.Tensor,
        Ps: torch.Tensor,
        distortion_coefficients: list[torch.Tensor],
        distortion_models: list[str],
        det_model: YOLOX,
        keypoints_model: RTMPose,
        det_category_id: int | None = None,
        bbox_thr: float = 0.3,
        nms_iou_thr: float = 0.65,
        kps2d_score_threshold: float = 0.3,
        kps3d_score_threshold: float = 0.2,
        skeleton_color: np.ndarray = np.array([0, 255, 0, 255]),
        best_bbox_only: bool = True,
        default_subject_id: str = "subject_0",
        frame_step: int = 1,
        show: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Infer bounding boxes for all views.
        """

        keypoints_format = self.keypoints_format
        keypoints_format_name = keypoints_format.name
        keypoints_metadata = self.keypoints_metadata
        n_keypoints = keypoints_format.n_keypoints
        keypoints_names = keypoints_format.keypoints_names
        keypoints_connectivity = keypoints_format.keypoints_connectivity

        n_views = len(views)
        video_loaders: list[LiveVideoLoader] = [view["frame_loader"] for view in views]
        view_ids = [view["view_id"] for view in views]
        view_id_to_idx = {view["view_id"]: view_idx for view_idx, view in enumerate(views)}

        frame_idx = 0

        while all(v.is_opened() for v in video_loaders):
            rr.set_time("frame", sequence=frame_idx)

            frames = []

            for view, video_loader in zip(views, video_loaders):
                frame = video_loader.read_frame()
                # (C, H, W) -> (H, W, C)
                frame_bgr = frame.permute(1, 2, 0).flip(-1)
                frames.append(frame_bgr)

            frames = torch.stack(frames, dim=0)
            frames_indices = [frame_idx] * n_views

            batch_bboxes_annotations = detect_bboxes(
                view_ids=view_ids,
                frames_bgr=frames,
                batch_frames=frames_indices,
                det_model=det_model,
                bbox_thr=bbox_thr,
                nms_iou_thr=nms_iou_thr,
                det_category_id=det_category_id,
                best_bbox_only=best_bbox_only,
                default_subject_id=default_subject_id,
            )

            batch_keypoints_annotations = detect_keypoints(
                view_ids=view_ids,
                frames_bgr=frames,
                batch_frames=frames_indices,
                bboxes_annotations=batch_bboxes_annotations,
                keypoints_model=keypoints_model,
                keypoints_metadata=self.keypoints_metadata,
                default_subject_id=default_subject_id,
            )

            # Triangulate keypoints and log to rerun
            for subject_id in batch_keypoints_annotations.subjects_ids:
                subject_keypoints_annotations = batch_keypoints_annotations.filter_by_subject_id(
                    subject_id
                )

                view_kps_xy = torch.empty((n_views, n_keypoints, 2), dtype=torch.float32, device=device)
                view_kps_xy_undistorted = torch.empty((n_views, n_keypoints, 2), dtype=torch.float32, device=device)
                view_kps_scores = torch.empty((n_views, n_keypoints), dtype=torch.float32, device=device)

                for a in subject_keypoints_annotations.annotations:
                    view_idx = view_id_to_idx[a.view_id]

                    xy = a.xy
                    xy_undistorted = undistort_points(
                        points=xy,
                        K=Ks[view_idx],
                        D=distortion_coefficients[view_idx],
                        distortion_model=distortion_models[view_idx]
                    )

                    view_kps_xy[view_idx] = xy
                    view_kps_xy_undistorted[view_idx] = xy_undistorted
                    view_kps_scores[view_idx] = a.scores

                kps_xyz = triangulate_points(
                    Ps=Ps,
                    points=view_kps_xy_undistorted,
                    points_weights=view_kps_scores,
                )
                kps_scores = pairwise_reprojection_consensus_score(
                    kps_3d=kps_xyz,
                    kps_2d=view_kps_xy_undistorted,
                    kps_2d_scores=view_kps_scores,
                    Rts=Rts,
                    Ks=Ks,
                    Ds=torch.stack(distortion_coefficients, dim=0),
                    distortion_model=distortion_models[0]
                )

                frames_bgr = []

                for view_idx, view in enumerate(views):
                    valid_kps_2d_mask = view_kps_scores[view_idx] > kps2d_score_threshold
                    valid_kps_2d = view_kps_xy[view_idx][valid_kps_2d_mask]

                    bones_2d = np.empty((0, 2, 2), dtype=np.float32)
                    for bone_idx, (start_kp_idx, end_kp_idx) in enumerate(keypoints_connectivity):
                        if not valid_kps_2d_mask[start_kp_idx] or not valid_kps_2d_mask[end_kp_idx]:
                            continue
                        kp1_pos = view_kps_xy[view_idx][start_kp_idx].cpu().numpy()
                        kp2_pos = view_kps_xy[view_idx][end_kp_idx].cpu().numpy()
                        bone_2d = np.stack([kp1_pos, kp2_pos], axis=0)
                        bones_2d = np.concatenate((bones_2d, bone_2d[None, ...]), axis=0)

                    # Draw the valid_kps_2d and bones2d on the frames
                    frame_bgr = frames[view_idx].cpu().numpy()
                    frames_bgr.append(frame_bgr)

                    rr.log(f"2d/{view['view_id']}/skeleton/kps",
                           rr.Points2D(positions=valid_kps_2d.cpu().numpy(), colors=skeleton_color))
                    rr.log(f"2d/{view['view_id']}/skeleton/bones", rr.LineStrips2D(
                        strips=bones_2d,
                        colors=skeleton_color,
                    ))

                valid_kps_3d_mask = kps_scores > kps3d_score_threshold
                valid_kps_3d = kps_xyz[valid_kps_3d_mask]
                kps_3d = valid_kps_3d.cpu().numpy()

                bones_3d = np.empty((0, 2, 3), dtype=np.float32)
                for bone_idx, (start_kp_idx, end_kp_idx) in enumerate(keypoints_connectivity):
                    if not valid_kps_3d_mask[start_kp_idx] or not valid_kps_3d_mask[end_kp_idx]:
                        continue

                    kp1_pos = kps_xyz[start_kp_idx].cpu().numpy()
                    kp2_pos = kps_xyz[end_kp_idx].cpu().numpy()
                    bone_3d = np.stack([kp1_pos, kp2_pos], axis=0)
                    bones_3d = np.concatenate((bones_3d, bone_3d[None, ...]), axis=0)

                rr.log(f"3d/skeleton/kps", rr.Points3D(positions=kps_3d, colors=skeleton_color, radii=0.01))
                rr.log(f"3d/skeleton/bones", rr.LineStrips3D(strips=bones_3d, colors=skeleton_color, radii=0.005))

            frame_idx += 1


def detect_keypoints(
    view_ids: list[str],
    frames_bgr: torch.Tensor,
    batch_frames: list[int],
    bboxes_annotations: BBox2DAnnotations,
    keypoints_model: RTMPose,
    keypoints_metadata: Keypoints2DAnnotationsMetadata,
    filter_bbox_with_zero_score: bool = True,
    force_zero_scores_outside_bbox: bool = True,
    default_subject_id: str = "subject_0",
) -> Keypoints2DAnnotations:
    frames_bgr = frames_bgr.cpu().numpy()

    assert frames_bgr.ndim in [3, 4] and frames_bgr.shape[-1] == 3, (
        f"Expected frames_bgr to have shape (B, H, W, C) or (H, W, C), got {frames_bgr.shape}"
    )
    assert frames_bgr.dtype == np.uint8, "Expected frames_bgr to be uint8"

    actual_batch_size = len(batch_frames)

    bboxes: list[np.ndarray] = [np.zeros((0, 5)) for _ in range(actual_batch_size)]

    for batch_idx in range(actual_batch_size):
        frame_idx = batch_frames[batch_idx]

        frame_bboxes = bboxes_annotations.filter_by_view_id(
            view_ids[batch_idx]
        ).filter_by_frame_idx(frame_idx)

        if len(frame_bboxes) == 0:
            continue

        bboxes[batch_idx] = np.stack(
            [
                bbox.xyxy.cpu().numpy()
                for bbox in frame_bboxes
            ],
            axis=0,
        )

    keypoints_results: list[KeypointsDetectionResult] = []
    for i in range(actual_batch_size):
        frame_bgr = frames_bgr[i]
        frame_bboxes = bboxes[i]
        keypoints, scores = keypoints_model(frame_bgr, frame_bboxes)
        keypoints_results.append(KeypointsDetectionResult(keypoints=keypoints, scores=scores))

    annotations: list[Keypoints2DAnnotation] = []

    for batch_idx in range(actual_batch_size):
        frame_idx = batch_frames[batch_idx]
        keypoints_result = keypoints_results[batch_idx]

        keypoints_annotations = _create_keypoints_annotations(
            keypoints_result=keypoints_result,
            view_id=view_ids[batch_idx],
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
    view_ids: list[str],
    frames_bgr: torch.Tensor,
    batch_frames: list[int],
    det_model: YOLOX,
    bbox_thr: float = 0.3,
    nms_iou_thr: float = 0.65,
    det_category_id: int | None = None,
    best_bbox_only: bool = True,
    default_subject_id: str = "subject_0",
) -> BBox2DAnnotations:
    frames_bgr = frames_bgr.cpu().numpy()

    assert frames_bgr.ndim in [3, 4] and frames_bgr.shape[-1] == 3, (
        f"Expected frames_bgr to have shape (B, H, W, C) or (H, W, C), got {frames_bgr.shape}"
    )
    assert frames_bgr.dtype == np.uint8, "Expected frames_bgr to be uint8"

    actual_batch_size = len(batch_frames)

    if frames_bgr.ndim == 3:
        frames_bgr = frames_bgr[None, ...]
    batch_size = frames_bgr.shape[0]

    det_data_samples: list[BboxDetectionResult] = []

    for frame_bgr in frames_bgr:
        bboxes, scores = det_model(frame_bgr)
        det_data_samples.append(BboxDetectionResult(bboxes=bboxes, scores=scores))

    # Keep only one bbox per frame based on the score
    if best_bbox_only:
        det_data_samples = [
            keep_best_bbox(det_data_sample)
            for det_data_sample in det_data_samples
        ]

    annotations: list[BBox2DAnnotation] = []

    # Convert the DetDataSample to BBox2DAnnotation
    for batch_idx in range(actual_batch_size):
        det_data_sample = det_data_samples[batch_idx]
        frame_idx = batch_frames[batch_idx]

        bboxes_annotations = _create_bboxes_annotations(
            det_data_sample=det_data_sample,
            view_id=view_ids[batch_idx],
            frame_idx=frame_idx,
            default_subject_id=default_subject_id,
        )
        annotations.extend(bboxes_annotations)

    return BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(),
        annotations=annotations,
    )

def keep_best_bbox(det_data_sample: BboxDetectionResult) -> BboxDetectionResult:
    if len(det_data_sample.bboxes) == 0:
        return det_data_sample

    best_idx = np.argmax(det_data_sample.scores)
    return BboxDetectionResult(bboxes=det_data_sample.bboxes[best_idx][None, ...], scores=det_data_sample.scores[best_idx][None, ...])


def _create_bboxes_annotations(
    det_data_sample: tuple[np.ndarray, np.ndarray],
    view_id: str,
    frame_idx: int,
    default_subject_id: str,
) -> list[BBox2DAnnotation]:
    annotations = []
    n_instances = len(det_data_sample[0])

    if n_instances > 1:
        warnings.warn(
            "MMDetBboxDetectionStage only supports one bbox per frame for now (no tracker implemented). The best bbox will be used."
        )
        det_data_sample = keep_best_bbox(det_data_sample)
        n_instances = 1

    for instance_idx in range(n_instances):
        bbox_xyxy = det_data_sample.bboxes[instance_idx]
        score = det_data_sample.scores[instance_idx]
        bbox_annotation = BBox2DAnnotation(
            view_id=view_id,
            frame_idx=frame_idx,
            subject_id=default_subject_id,
            category_id=0,
            xyxy=torch.from_numpy(bbox_xyxy.reshape(4)).to(torch.float32),
            score=score.item(),
        )
        annotations.append(bbox_annotation)
    return annotations


def _create_keypoints_annotations(
    keypoints_result: KeypointsDetectionResult,
    view_id: str,
    frame_idx: int,
    keypoints_format_name: str,
    default_subject_id: str,
) -> list[Keypoints2DAnnotation]:
    """Create keypoints annotations from inference results."""
    annotations = []
    n_instances = len(keypoints_result.keypoints)

    for instance_idx in range(n_instances):
        keypoints_xy = keypoints_result.keypoints[instance_idx]
        keypoints_scores = keypoints_result.scores[instance_idx]

        kps_xy = torch.from_numpy(keypoints_xy).to(torch.float32)
        scores = torch.from_numpy(keypoints_scores).to(torch.float32)

        keypoint_annotation = Keypoints2DAnnotation(
            view_id=view_id,
            frame_idx=frame_idx,
            subject_id=default_subject_id,
            xy=kps_xy.cpu(),
            scores=scores.cpu(),
            annotated=torch.ones(keypoints_xy.shape[0], dtype=torch.bool),
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
