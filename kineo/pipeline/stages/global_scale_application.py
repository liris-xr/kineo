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
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotation
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotation
from kineo.annotations.global_scale import GlobalScaleAnnotation


class GlobalScaleApplicationStage(PipelineStage):
    def __init__(self, name: str, order: int):
        super().__init__(
            name=name, order=order, runtime_cfg=None, dynamic_runtime_cfg=None
        )

    def forward(
        self,
        sequence_name: str,
        pipeline: Pipeline,
        views: list[ViewInput],
        annotations: dict[str, Annotations],
        gt_annotations: dict[str, Annotations],
        runtime_cfg: None = None,
    ):
        camera_extrinsics: CameraExtrinsicsAnnotations = annotations[
            "camera_extrinsics"
        ]
        keypoints_3d: Keypoints3DAnnotations = annotations["keypoints_3d"]
        global_scale: GlobalScaleAnnotation = annotations[
            "global_scale"
        ].first_or_default()

        if global_scale is None:
            raise ValueError("Global scale is not available")

        global_scale_factor = global_scale.scale

        new_keypoints_3d: list[Keypoints3DAnnotation] = []
        new_camera_extrinsics: list[CameraExtrinsicsAnnotation] = []

        for cam_extrinsics_annot in camera_extrinsics.annotations:
            new_camera_extrinsics.append(
                CameraExtrinsicsAnnotation(
                    view_id=cam_extrinsics_annot.view_id,
                    frame_idx=cam_extrinsics_annot.frame_idx,
                    R=cam_extrinsics_annot.R,
                    t=cam_extrinsics_annot.t * global_scale_factor,
                )
            )

        for keypoints_3d_annot in keypoints_3d.annotations:
            new_keypoints_3d.append(
                Keypoints3DAnnotation(
                    frame_idx=keypoints_3d_annot.frame_idx,
                    subject_id=keypoints_3d_annot.subject_id,
                    annotated=keypoints_3d_annot.annotated,
                    scores=keypoints_3d_annot.scores,
                    xyz=keypoints_3d_annot.xyz * global_scale_factor,
                    format=keypoints_3d_annot.format,
                )
            )

        annotations["keypoints_3d"] = Keypoints3DAnnotations(
            metadata=keypoints_3d.metadata,
            annotations=new_keypoints_3d,
        ).cpu()

        annotations["camera_extrinsics"] = CameraExtrinsicsAnnotations(
            metadata=camera_extrinsics.metadata,
            annotations=new_camera_extrinsics,
        ).cpu()
