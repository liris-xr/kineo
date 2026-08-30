from __future__ import annotations
import torch
from tqdm import tqdm

from kineo.pipeline.pipeline import PipelineStage
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations import (
    Annotations,
    Keypoints2DAnnotations,
    Keypoints2DAnnotationsMetadata,
    Keypoints2DAnnotation,
    BBox2DAnnotations,
    CameraIntrinsicsAnnotations,
)
from kineo.pipeline.pipeline import Pipeline
from kineo.pipeline import per_view_cache
from kineo.pipeline.stages.nlf.model_wrapper import NLFModelWrapper
from kineo.pipeline.stages.nlf.skeleton_keypoints_format import (
    H36M_17_KEYPOINTS_FORMAT,
    COCO_19_KEYPOINTS_FORMAT,
    COCO_17_KEYPOINTS_FORMAT,
    SMPL_24_KEYPOINTS_FORMAT,
)

from dataclasses import dataclass

KEYPOINTS_FORMAT = {
    "smpl_24": SMPL_24_KEYPOINTS_FORMAT,
    "h36m_17": H36M_17_KEYPOINTS_FORMAT,
    "coco_17": COCO_17_KEYPOINTS_FORMAT,
    "coco_19": COCO_19_KEYPOINTS_FORMAT,
}


@dataclass
class NLFSkeletonKeypointsDetectionRuntimeConfig:
    batch_size: int = 16
    use_half_precision: bool = True
    use_cache: bool = True
    cache_output_path_template: str = (
        "cache/{sequence_name}/{annotation_key}/{view_id}.pkl"
    )
    frame_step: int = (
        1  # Infer every N frames (1 = every frame, 2 = every other frame, etc.)
    )
    skeleton_name: str = "smpl_24"


# TODO: in case of single view, export keypoints 3d as well
class NLFSkeletonKeypointsDetectionStage(
    PipelineStage[NLFSkeletonKeypointsDetectionRuntimeConfig]
):
    """
    Stage for detecting keypoints from images by jointly using NLF.

    Produces :class:`Keypoints2DAnnotations` with the detected keypoints for each view with key "keypoints_2d".
    """

    def __init__(
        self,
        name: str,
        order: int,
        runtime_cfg: NLFSkeletonKeypointsDetectionRuntimeConfig,
        dynamic_runtime_cfg: (
            dict[str, NLFSkeletonKeypointsDetectionRuntimeConfig] | None
        ) = None,
        torchscript_model_path: str = "./checkpoints/nlf_l_multi_0.3.2.torchscript",
    ):
        super().__init__(
            name=name,
            order=order,
            runtime_cfg=runtime_cfg,
            dynamic_runtime_cfg=dynamic_runtime_cfg,
        )
        self.model = NLFModelWrapper(torchscript_model_path)
        self.model = self.model.cpu()
        self.model = self.model.eval()

        self.skeleton_names = self.model.skeleton_names

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: NLFSkeletonKeypointsDetectionRuntimeConfig,
    ):
        device = pipeline.device

        if runtime_cfg.skeleton_name not in self.skeleton_names:
            raise ValueError(
                f"Invalid skeleton name: {runtime_cfg.skeleton_name}. Expected one of {self.skeleton_names}."
            )

        if runtime_cfg.skeleton_name not in KEYPOINTS_FORMAT:
            raise NotImplementedError(
                f"Skeleton format {runtime_cfg.skeleton_name} not implemented yet."
            )

        bboxes_annotations: BBox2DAnnotations = annotations.get("bboxes_2d")

        intrinsics_annotations: CameraIntrinsicsAnnotations = annotations.get(
            "camera_intrinsics"
        )

        if bboxes_annotations is None:
            raise ValueError("Expected bboxes annotations but none were provided")

        def infer_missing(missing_views: list[ViewInput]) -> dict[str, Annotations]:
            self.model = self.model.to(device)
            try:
                return {
                    "keypoints_2d": self._infer_keypoints(
                        views=missing_views,
                        device=device,
                        bboxes_annotations=bboxes_annotations,
                        intrinsics_annotations=intrinsics_annotations,
                        model=self.model,
                        batch_size=runtime_cfg.batch_size,
                        use_half_precision=runtime_cfg.use_half_precision,
                        frame_step=runtime_cfg.frame_step,
                        skeleton_name=runtime_cfg.skeleton_name,
                    )
                }
            finally:
                self.model = self.model.cpu()

        cached = per_view_cache.load_or_infer_per_view(
            views=views,
            specs={
                "keypoints_2d": per_view_cache.PerViewCacheSpec(
                    annotations_cls=Keypoints2DAnnotations,
                    metadata=Keypoints2DAnnotationsMetadata(
                        formats=[KEYPOINTS_FORMAT[runtime_cfg.skeleton_name]]
                    ),
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
        model: NLFModelWrapper,
        views: list[ViewInput],
        device: torch.device,
        bboxes_annotations: BBox2DAnnotations,
        intrinsics_annotations: CameraIntrinsicsAnnotations | None = None,
        batch_size: int = 16,
        use_half_precision: bool = True,
        frame_step: int = 1,
        skeleton_name: str = "smpl_24",
    ) -> Keypoints2DAnnotations:
        """
        Infer keypoints for all views.
        """

        if skeleton_name not in self.skeleton_names:
            raise ValueError(
                f"Invalid model name: {skeleton_name}. Expected one of {self.skeleton_names}."
            )

        keypoints_format = KEYPOINTS_FORMAT[skeleton_name]

        all_keypoints_annotations: list[Keypoints2DAnnotation] = []

        n_total_inference_frames = sum(
            len(_get_inference_frames(view["frame_loader"].n_frames, frame_step))
            for view in views
        )
        pbar = tqdm(
            total=n_total_inference_frames, desc="Inferring keypoints", leave=False
        )

        for view in views:
            frame_loader = view["frame_loader"]
            view_id = view["view_id"]
            view_n_frames = frame_loader.n_frames

            K = (
                intrinsics_annotations.filter_by_view_id(view_id)
                .first_or_default()
                .K.to(device)
                if intrinsics_annotations is not None
                else None
            )

            D = (
                intrinsics_annotations.filter_by_view_id(view_id)
                .first_or_default()
                .distortion_coefficients.to(device)
                if intrinsics_annotations is not None
                else None
            )

            world_up_vector = torch.tensor(
                [0, -1, 0], dtype=torch.float32, device=device
            )

            inference_frames = _get_inference_frames(view_n_frames, frame_step)

            for batch_start in range(0, len(inference_frames), batch_size):
                batch_end = min(batch_start + batch_size, len(inference_frames))
                batch_frames = inference_frames[batch_start:batch_end]
                actual_batch_size = len(batch_frames)

                # Load batch of frames
                frames_rgb = frame_loader.load_frames_at(
                    frame_indices=torch.tensor(batch_frames)
                )

                batch_bboxes_xywhs: list[torch.Tensor] = [
                    torch.zeros((0, 5), dtype=torch.float32, device=device)
                    for _ in range(actual_batch_size)
                ]

                batch_subjects_ids: list[list[str]] = [
                    [] for _ in range(actual_batch_size)
                ]

                for batch_idx, frame_idx in enumerate(batch_frames):
                    frame_bboxes = bboxes_annotations.filter_by_view_id(
                        view_id
                    ).filter_by_frame_idx(frame_idx)

                    frame_annotations = frame_bboxes.annotations

                    if len(frame_annotations) == 0:
                        continue

                    frame_subjects_ids = [bbox.subject_id for bbox in frame_annotations]
                    frame_bboxes_xyxy = torch.stack(
                        [bbox.xyxy.to(device) for bbox in frame_annotations]
                    )
                    frame_bboxes_xywh = torch.empty_like(frame_bboxes_xyxy)
                    frame_bboxes_xywh[..., :2] = frame_bboxes_xyxy[..., :2]
                    frame_bboxes_xywh[..., 2:] = (
                        frame_bboxes_xyxy[..., 2:] - frame_bboxes_xyxy[..., :2]
                    )
                    frame_bboxes_scores = torch.tensor(
                        [[bbox.score] for bbox in frame_annotations],
                        dtype=torch.float32,
                        device=device,
                    )
                    frame_bboxes_xywhs = torch.cat(
                        [frame_bboxes_xywh, frame_bboxes_scores], dim=-1
                    )

                    batch_bboxes_xywhs[batch_idx] = frame_bboxes_xywhs
                    batch_subjects_ids[batch_idx] = frame_subjects_ids

                batch_results = _batch_infer_keypoints(
                    frames_rgb=frames_rgb,
                    model=model,
                    bboxes_xywhs=batch_bboxes_xywhs,
                    K=K,
                    D=D,
                    world_up_vector=world_up_vector,
                    skeleton_name=skeleton_name,
                    use_half_precision=use_half_precision,
                )

                for batch_idx in range(actual_batch_size):
                    frame_idx = batch_frames[batch_idx]
                    batch_joints2d = batch_results["joints2d"][batch_idx]
                    batch_joints2d_confidences = batch_results["joints_confidences"][
                        batch_idx
                    ]

                    n_subjects = batch_joints2d.shape[0]

                    for subject_idx in range(n_subjects):
                        subject_id = batch_subjects_ids[batch_idx][subject_idx]
                        subject_joints2d_xy = batch_joints2d[subject_idx]
                        subject_joints2d_confidence = batch_joints2d_confidences[
                            subject_idx
                        ]

                        keypoints_annotation = Keypoints2DAnnotation(
                            view_id=view_id,
                            frame_idx=frame_idx,
                            subject_id=subject_id,
                            xy=subject_joints2d_xy.cpu(),
                            scores=subject_joints2d_confidence.cpu(),
                            annotated=torch.ones(
                                subject_joints2d_xy.shape[0], dtype=torch.bool
                            ),
                            format=keypoints_format.name,
                        )

                        all_keypoints_annotations.append(keypoints_annotation)

                pbar.update(actual_batch_size)

        pbar.close()

        # TODO: interpolate keypoints annotations if necessary

        keypoints_annotations = Keypoints2DAnnotations(
            metadata=Keypoints2DAnnotationsMetadata(formats=[keypoints_format]),
            annotations=all_keypoints_annotations,
        ).cpu()

        return keypoints_annotations


def _get_inference_frames(n_frames: int, frame_step: int) -> list[int]:
    """Get list of frame indices to run inference on."""
    if frame_step == 1:
        return list(range(n_frames))

    inference_frames = list(range(0, n_frames, frame_step))

    # Always include the last frame if it's not already included
    if (n_frames - 1) not in inference_frames and n_frames > 0:
        inference_frames.append(n_frames - 1)

    return sorted(inference_frames)


def _batch_infer_keypoints(
    frames_rgb: torch.Tensor,
    model: NLFModelWrapper,
    bboxes_xywhs: list[torch.Tensor],
    K: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    world_up_vector: torch.Tensor | None = None,
    internal_batch_size: int = 64,
    antialias_factor: int = 1,
    num_aug: int = 1,
    rot_aug_max_degrees: float = 25.0,
    skeleton_name: str = "smpl_24",
    use_half_precision: bool = True,
):
    assert frames_rgb.ndim in [3, 4] and frames_rgb.shape[-3] == 3, (
        f"Expected frames_bgr to have shape (B, C, H, W) or (C, H, W), got {frames_rgb.shape}"
    )

    if frames_rgb.dtype == torch.uint8:
        frames_rgb = frames_rgb / 255.0

    if K is not None:
        assert K.shape == (3, 3), f"Expected K to have shape (3, 3), got {K.shape}"
        K = K.unsqueeze(0)

    if D is not None:
        if D.shape != (5,):
            # If the distortion coefficients are opencv fisheye (most probably coming from GT), ignore
            D = None
        else:
            assert D.shape == (5,), f"Expected D to have shape (5,), got {D.shape}"
            D = D.unsqueeze(0)

    if frames_rgb.ndim == 3:
        frames_rgb = frames_rgb.unsqueeze(0)

    batch_size = frames_rgb.shape[0]

    assert len(bboxes_xywhs) == batch_size

    for bbox_xyxys in bboxes_xywhs:
        assert bbox_xyxys.shape[1] == 5, (
            f"Expected bboxes_xyxys to have shape (*, 5), got {bbox_xyxys.shape}"
        )

    pred = model.infer_skeleton_keypoints(
        images=frames_rgb,
        boxes=bboxes_xywhs,
        intrinsic_matrix=K,
        distortion_coeffs=D,
        extrinsic_matrix=None,
        world_up_vector=world_up_vector,
        internal_batch_size=internal_batch_size,
        antialias_factor=antialias_factor,
        num_aug=num_aug,
        rot_aug_max_degrees=rot_aug_max_degrees,
        skeleton_name=skeleton_name,
        use_half_precision=use_half_precision,
    )

    batch_joints2d = pred["joints2d"]
    batch_joints3d = pred["joints3d"]
    batch_joints_confidences = pred["joints_confidences"]

    return {
        "joints2d": batch_joints2d,
        "joints3d": batch_joints3d,
        "joints_confidences": batch_joints_confidences,
    }
