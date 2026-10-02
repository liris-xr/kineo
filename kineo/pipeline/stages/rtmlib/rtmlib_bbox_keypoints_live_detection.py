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
import warnings
from time import perf_counter

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    Annotations,
    BBox2DAnnotations,
    BBox2DAnnotationsMetadata,
    BBox2DAnnotation,
    Keypoints2DAnnotations,
    Keypoints2DAnnotation,
    Keypoints3DAnnotations,
    Keypoints3DAnnotation,
    Keypoints3DAnnotationsMetadata,
    KeypointsFormat,
    Keypoints2DAnnotationsMetadata,
)
from kineo.annotations.frame_timing import FrameTiming, FrameTimingsAnnotations
from kineo.pipeline.stages.rtmlib.keypoints_formats import (
    NAMED_KEYPOINTS_FORMATS,
)
from kineo.pipeline.pipeline import Pipeline

from dataclasses import dataclass

from kineo.geometry.metrics import pairwise_visibility_score
from kineo.geometry.triangulation import triangulate_points_vectorized
from kineo.geometry.transformations import undistort_points
import numpy as np

from kineo.pipeline.stages.rtmlib.tools.object_detection.yolox_person import (
    YOLOXPerson,
)
from kineo.pipeline.stages.rtmlib.tools.pose_estimation.rtmpose import RTMPose
from kineo.pipeline.stages.rtmlib.skeleton.coco17 import coco17
from kineo.pipeline.stages.rtmlib.skeleton.coco133 import coco133
from kineo.pipeline.stages.rtmlib.skeleton.openpose18 import openpose18
from kineo.pipeline.stages.rtmlib.skeleton.openpose134 import openpose134
from kineo.pipeline.stages.rtmlib.skeleton.halpe26 import halpe26
from kineo.pipeline.stages.rtmlib.skeleton.hand21 import hand21

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
        bbox_model: str = "./checkpoints/yolox_tiny_humanart_person.onnx",
        bbox_model_input_shape_hw: tuple[int, int] = (416, 416),
        keypoints_model: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        keypoints_model_input_shape_hw: tuple[int, int] = (288, 384),
        use_tensorrt: bool = False,
        fp16: bool = False,
        engine_cache_dir: str | None = None,
        keypoints_format_name: str | None = None,
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )

        self.det_model = YOLOXPerson(
            onnx_model=bbox_model,
            model_input_size=bbox_model_input_shape_hw,
            score_thr=runtime_cfg.bbox_thr,
            nms_thr=runtime_cfg.nms_iou_thr,
            device="cuda",
            use_tensorrt=use_tensorrt,
            fp16=fp16,
            engine_cache_dir=engine_cache_dir,
            strict_providers=True,
        )
        self.keypoints_model = RTMPose(
            onnx_model=keypoints_model,
            model_input_size=keypoints_model_input_shape_hw,
            device="cuda",
            use_tensorrt=use_tensorrt,
            fp16=fp16,
            engine_cache_dir=engine_cache_dir,
            strict_providers=True,
        )

        rr.init("live_rtmlib_bbox_keypoints_detection")

        n_keypoints = self._warmup(bbox_model_input_shape_hw)

        self.keypoints_format = resolve_keypoints_format(
            keypoints_format_name, n_keypoints
        )
        self.keypoints_metadata = Keypoints2DAnnotationsMetadata(
            formats=[self.keypoints_format]
        )

    def _warmup(self, bbox_model_input_shape_hw: tuple[int, int]) -> int:
        frame = torch.zeros(
            (3, *bbox_model_input_shape_hw), dtype=torch.uint8, device="cuda"
        )
        self.det_model.forward_gpu([frame])
        box = torch.tensor(
            [[0.0, 0.0, float(bbox_model_input_shape_hw[1]),
              float(bbox_model_input_shape_hw[0])]],
            dtype=torch.float32, device="cuda",
        )
        keypoints, _ = self.keypoints_model.forward_gpu_batch([frame], [box])[0]

        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]

        return keypoints.shape[1]

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

        if runtime_cfg.show:
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

        if world_reconstructed_scene_annotations is not None and runtime_cfg.show:
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

        keypoints_3d, frame_timings = self._infer_bboxes_and_keypoints(
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

        annotations["keypoints_3d"] = keypoints_3d
        annotations["frame_timings"] = frame_timings

    def _infer_bboxes_and_keypoints(
        self,
        views: list[ViewInput],
        Ks: torch.Tensor,
        Rts: torch.Tensor,
        Ps: torch.Tensor,
        distortion_coefficients: list[torch.Tensor],
        distortion_models: list[str],
        det_model: YOLOXPerson,
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
    ) -> tuple[Keypoints3DAnnotations, FrameTimingsAnnotations]:
        """
        Stream every view, triangulating each frame as it arrives.
        """

        keypoints_format = self.keypoints_format
        keypoints_format_name = keypoints_format.name
        n_keypoints = keypoints_format.n_keypoints
        keypoints_connectivity = keypoints_format.keypoints_connectivity

        n_views = len(views)
        video_loaders: list[LiveVideoLoader] = [view["frame_loader"] for view in views]
        view_ids = [view["view_id"] for view in views]
        view_id_to_idx = {view["view_id"]: view_idx for view_idx, view in enumerate(views)}

        frame_idx = 0
        keypoints_3d_annotations: list[Keypoints3DAnnotation] = []
        frame_timings: list[FrameTiming] = []

        while all(v.is_opened() for v in video_loaders):
            if show:
                rr.set_time("frame", sequence=frame_idx)

            frame_start_time = perf_counter()

            frames = []

            for view, video_loader in zip(views, video_loaders):
                frame = video_loader.read_frame()
                frames.append(frame.flip(0))

            frames_indices = [frame_idx] * n_views

            torch.cuda.synchronize()
            read_time = perf_counter()

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

            detection_time = perf_counter()

            batch_keypoints_annotations = detect_keypoints(
                view_ids=view_ids,
                frames_bgr=frames,
                batch_frames=frames_indices,
                bboxes_annotations=batch_bboxes_annotations,
                keypoints_model=keypoints_model,
                keypoints_metadata=self.keypoints_metadata,
                default_subject_id=default_subject_id,
            )

            keypoints_time = perf_counter()
            triangulation_seconds = 0.0

            for subject_id in batch_keypoints_annotations.subjects_ids:
                triangulation_start_time = perf_counter()

                subject_keypoints_annotations = (
                    batch_keypoints_annotations.filter_by_subject_id(subject_id)
                )

                view_kps_xy, view_kps_xy_undistorted, view_kps_scores = (
                    gather_view_keypoints(
                        keypoints_annotations=subject_keypoints_annotations,
                        view_id_to_idx=view_id_to_idx,
                        Ks=Ks,
                        distortion_coefficients=distortion_coefficients,
                        distortion_models=distortion_models,
                        n_views=n_views,
                        n_keypoints=n_keypoints,
                        device=device,
                    )
                )

                # CUDA svd is slow past ~32 rows
                kps_xyz = triangulate_points_vectorized(
                    Ps=Ps,
                    points=view_kps_xy_undistorted,
                    points_weights=view_kps_scores,
                    use_eigendecomposition=True,
                )
                kps_scores = pairwise_visibility_score(view_kps_scores)

                torch.cuda.synchronize()
                triangulation_seconds += perf_counter() - triangulation_start_time

                # xyz is NaN if no keypoint has two views
                if not bool((kps_scores > 0).any()):
                    continue

                keypoints_3d_annotations.append(
                    Keypoints3DAnnotation(
                        frame_idx=frame_idx,
                        subject_id=subject_id,
                        xyz=kps_xyz,
                        scores=kps_scores,
                        format=keypoints_format_name,
                    )
                )

                if show:
                    log_skeletons(
                        view_ids=view_ids,
                        view_kps_xy=view_kps_xy,
                        view_kps_scores=view_kps_scores,
                        kps_xyz=kps_xyz,
                        kps_scores=kps_scores,
                        keypoints_connectivity=keypoints_connectivity,
                        kps2d_score_threshold=kps2d_score_threshold,
                        kps3d_score_threshold=kps3d_score_threshold,
                        skeleton_color=skeleton_color,
                    )

            frame_timings.append(
                FrameTiming(
                    frame_idx=frame_idx,
                    read_seconds=read_time - frame_start_time,
                    detection_seconds=detection_time - read_time,
                    keypoints_seconds=keypoints_time - detection_time,
                    triangulation_seconds=triangulation_seconds,
                    total_seconds=perf_counter() - frame_start_time,
                )
            )

            frame_idx += 1

        return (
            Keypoints3DAnnotations(
                metadata=Keypoints3DAnnotationsMetadata(
                    formats=[keypoints_format]
                ),
                annotations=keypoints_3d_annotations,
            ),
            FrameTimingsAnnotations(annotations=frame_timings),
        )


KEYPOINTS_FORMATS_BY_COUNT = {
    17: ("coco", coco17),
    18: ("openpose-wholebody", openpose18),
    21: ("hand21", hand21),
    26: ("halpe26", halpe26),
    133: ("coco-wholebody", coco133),
    134: ("openpose-wholebody", openpose134),
}


def resolve_keypoints_format(
    keypoints_format_name: str | None, n_keypoints: int
) -> KeypointsFormat:
    if keypoints_format_name is not None:
        if keypoints_format_name not in NAMED_KEYPOINTS_FORMATS:
            raise ValueError(
                f"Unknown keypoints format {keypoints_format_name}, "
                f"expected one of {sorted(NAMED_KEYPOINTS_FORMATS)}"
            )
        named_format = NAMED_KEYPOINTS_FORMATS[keypoints_format_name]
        if named_format.n_keypoints != n_keypoints:
            raise ValueError(
                f"{keypoints_format_name} holds "
                f"{named_format.n_keypoints} keypoints but the model "
                f"emits {n_keypoints}"
            )
        return named_format

    if n_keypoints not in KEYPOINTS_FORMATS_BY_COUNT:
        raise ValueError(
            f"Expected one of "
            f"{sorted(KEYPOINTS_FORMATS_BY_COUNT)} keypoints, got "
            f"{n_keypoints}"
        )

    name, metainfo = KEYPOINTS_FORMATS_BY_COUNT[n_keypoints]
    keypoints_names = [
        metainfo["keypoint_info"][idx]["name"] for idx in range(n_keypoints)
    ]

    return KeypointsFormat(
        name=name,
        n_keypoints=n_keypoints,
        keypoints_names=keypoints_names,
        keypoints_connectivity=[
            (keypoints_names.index(bone1), keypoints_names.index(bone2))
            for bone1, bone2 in (
                link["link"] for link in metainfo["skeleton_info"].values()
            )
        ],
    )


def gather_view_keypoints(
    keypoints_annotations: Keypoints2DAnnotations,
    view_id_to_idx: dict[str, int],
    Ks: torch.Tensor,
    distortion_coefficients: list[torch.Tensor],
    distortion_models: list[str],
    n_views: int,
    n_keypoints: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Views without the subject get zero weight
    view_kps_xy = torch.zeros(
        (n_views, n_keypoints, 2), dtype=torch.float32, device=device
    )
    view_kps_xy_undistorted = torch.zeros(
        (n_views, n_keypoints, 2), dtype=torch.float32, device=device
    )
    view_kps_scores = torch.zeros(
        (n_views, n_keypoints), dtype=torch.float32, device=device
    )

    for annotation in keypoints_annotations.annotations:
        view_idx = view_id_to_idx[annotation.view_id]

        view_kps_xy[view_idx] = annotation.xy
        view_kps_xy_undistorted[view_idx] = undistort_points(
            points=annotation.xy,
            K=Ks[view_idx],
            D=distortion_coefficients[view_idx],
            distortion_model=distortion_models[view_idx],
        )
        view_kps_scores[view_idx] = annotation.scores

    return view_kps_xy, view_kps_xy_undistorted, view_kps_scores


def skeleton_bones(
    keypoints: torch.Tensor,
    valid_mask: torch.Tensor,
    keypoints_connectivity: list[tuple[int, int]],
) -> np.ndarray:
    bones = [
        keypoints[[start_kp_idx, end_kp_idx]]
        for start_kp_idx, end_kp_idx in keypoints_connectivity
        if valid_mask[start_kp_idx] and valid_mask[end_kp_idx]
    ]

    if not bones:
        return np.empty((0, 2, keypoints.shape[-1]), dtype=np.float32)

    return torch.stack(bones).cpu().numpy()


def log_skeletons(
    view_ids: list[str],
    view_kps_xy: torch.Tensor,
    view_kps_scores: torch.Tensor,
    kps_xyz: torch.Tensor,
    kps_scores: torch.Tensor,
    keypoints_connectivity: list[tuple[int, int]],
    kps2d_score_threshold: float,
    kps3d_score_threshold: float,
    skeleton_color: np.ndarray,
):
    for view_idx, view_id in enumerate(view_ids):
        valid_mask = view_kps_scores[view_idx] > kps2d_score_threshold
        rr.log(f"2d/{view_id}/skeleton/kps", rr.Points2D(
            positions=view_kps_xy[view_idx][valid_mask].cpu().numpy(),
            colors=skeleton_color,
        ))
        rr.log(f"2d/{view_id}/skeleton/bones", rr.LineStrips2D(
            strips=skeleton_bones(
                view_kps_xy[view_idx], valid_mask, keypoints_connectivity
            ),
            colors=skeleton_color,
        ))

    valid_mask = kps_scores > kps3d_score_threshold
    rr.log("3d/skeleton/kps", rr.Points3D(
        positions=kps_xyz[valid_mask].cpu().numpy(),
        colors=skeleton_color,
        radii=0.01,
    ))
    rr.log("3d/skeleton/bones", rr.LineStrips3D(
        strips=skeleton_bones(kps_xyz, valid_mask, keypoints_connectivity),
        colors=skeleton_color,
        radii=0.005,
    ))


def detect_keypoints(
    view_ids: list[str],
    frames_bgr: list[torch.Tensor],
    batch_frames: list[int],
    bboxes_annotations: BBox2DAnnotations,
    keypoints_model: RTMPose,
    keypoints_metadata: Keypoints2DAnnotationsMetadata,
    filter_bbox_with_zero_score: bool = True,
    force_zero_scores_outside_bbox: bool = True,
    default_subject_id: str = "subject_0",
) -> Keypoints2DAnnotations:
    # Views may differ in resolution
    assert all(f.ndim == 3 and f.shape[0] == 3 for f in frames_bgr), (
        f"Expected each frame to have shape (C, H, W), got "
        f"{[tuple(f.shape) for f in frames_bgr]}"
    )

    actual_batch_size = len(batch_frames)
    device = frames_bgr[0].device

    bboxes: list[torch.Tensor] = [
        torch.zeros((0, 4), dtype=torch.float32, device=device)
        for _ in range(actual_batch_size)
    ]

    for batch_idx in range(actual_batch_size):
        frame_idx = batch_frames[batch_idx]

        frame_bboxes = bboxes_annotations.filter_by_view_id(
            view_ids[batch_idx]
        ).filter_by_frame_idx(frame_idx)

        if len(frame_bboxes) == 0:
            continue

        bboxes[batch_idx] = torch.stack(
            [bbox.xyxy for bbox in frame_bboxes], dim=0
        ).to(device=device, dtype=torch.float32)

    keypoints_results = [
        KeypointsDetectionResult(keypoints=keypoints, scores=scores)
        for keypoints, scores in keypoints_model.forward_gpu_batch(
            frames_bgr, bboxes
        )
    ]

    annotations: list[Keypoints2DAnnotation] = []

    for batch_idx in range(actual_batch_size):
        annotations.extend(
            _create_keypoints_annotations(
                keypoints_result=keypoints_results[batch_idx],
                view_id=view_ids[batch_idx],
                frame_idx=batch_frames[batch_idx],
                keypoints_format_name=keypoints_metadata.formats[0].name,
                default_subject_id=default_subject_id,
            )
        )

    return Keypoints2DAnnotations(
        metadata=keypoints_metadata,
        annotations=annotations,
    )


def detect_bboxes(
    view_ids: list[str],
    frames_bgr: list[torch.Tensor],
    batch_frames: list[int],
    det_model: YOLOXPerson,
    bbox_thr: float = 0.3,
    nms_iou_thr: float = 0.65,
    det_category_id: int | None = None,
    best_bbox_only: bool = True,
    default_subject_id: str = "subject_0",
) -> BBox2DAnnotations:
    # Views may differ in resolution
    assert all(f.ndim == 3 and f.shape[0] == 3 for f in frames_bgr), (
        f"Expected each frame to have shape (C, H, W), got "
        f"{[tuple(f.shape) for f in frames_bgr]}"
    )

    actual_batch_size = len(batch_frames)

    det_data_samples = [
        BboxDetectionResult(bboxes=bboxes, scores=scores)
        for bboxes, scores in det_model.forward_gpu(frames_bgr)
    ]

    # Keep only one bbox per frame based on the score
    if best_bbox_only:
        det_data_samples = [
            keep_best_bbox(det_data_sample)
            for det_data_sample in det_data_samples
        ]

    annotations: list[BBox2DAnnotation] = []

    for batch_idx in range(actual_batch_size):
        annotations.extend(
            _create_bboxes_annotations(
                det_data_sample=det_data_samples[batch_idx],
                view_id=view_ids[batch_idx],
                frame_idx=batch_frames[batch_idx],
                default_subject_id=default_subject_id,
            )
        )

    return BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(),
        annotations=annotations,
    )

def keep_best_bbox(det_data_sample: BboxDetectionResult) -> BboxDetectionResult:
    if len(det_data_sample.bboxes) == 0:
        return det_data_sample

    best_idx = torch.argmax(det_data_sample.scores)
    return BboxDetectionResult(bboxes=det_data_sample.bboxes[best_idx][None, ...], scores=det_data_sample.scores[best_idx][None, ...])


def _create_bboxes_annotations(
    det_data_sample: BboxDetectionResult,
    view_id: str,
    frame_idx: int,
    default_subject_id: str,
) -> list[BBox2DAnnotation]:
    annotations = []
    n_instances = len(det_data_sample.bboxes)

    if n_instances > 1:
        warnings.warn(
            "Only one bbox per view is supported for now (no tracker "
            "implemented). The best bbox will be used."
        )
        det_data_sample = keep_best_bbox(det_data_sample)
        n_instances = 1

    for instance_idx in range(n_instances):
        annotations.append(
            BBox2DAnnotation(
                view_id=view_id,
                frame_idx=frame_idx,
                subject_id=default_subject_id,
                category_id=0,
                xyxy=det_data_sample.bboxes[instance_idx]
                .reshape(4)
                .to(torch.float32)
                .cpu(),
                score=det_data_sample.scores[instance_idx].item(),
            )
        )
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

    for instance_idx in range(len(keypoints_result.keypoints)):
        annotations.append(
            Keypoints2DAnnotation(
                view_id=view_id,
                frame_idx=frame_idx,
                subject_id=default_subject_id,
                xy=keypoints_result.keypoints[instance_idx]
                .to(torch.float32)
                .cpu(),
                scores=keypoints_result.scores[instance_idx]
                .to(torch.float32)
                .cpu(),
                format=keypoints_format_name,
            )
        )

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
