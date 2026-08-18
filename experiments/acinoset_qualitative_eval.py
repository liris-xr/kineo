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
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.pipeline.pipeline import Pipeline
from kineo.annotations.keypoints_format import KeypointsFormat
from kineo.annotations.keypoints_2d import Keypoints2DAnnotation
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotationsMetadata
from kineo.annotations.bboxes_2d import BBox2DAnnotations
from kineo.annotations.bboxes_2d import BBox2DAnnotationsMetadata
from kineo.annotations.bboxes_2d import BBox2DAnnotation
from kineo.io.frame_sequence_loader import VideoLoader
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

CHEETAH_KPS_FORMAT = KeypointsFormat(
    name="cheetah",
    n_keypoints=20,
    keypoints_names=[
        "r_eye",           # 0
        "l_eye",           # 1
        "r_shoulder",      # 2
        "r_front_knee",    # 3
        "r_front_ankle",   # 4
        "spine",           # 5
        "r_hip",           # 6
        "r_back_knee",     # 7
        "r_back_ankle",    # 8
        "tail1",           # 9
        "tail2",           # 10
        "l_shoulder",      # 11
        "l_front_knee",    # 12
        "l_front_ankle",   # 13
        "l_hip",           # 14
        "l_back_knee",     # 15
        "l_back_ankle",    # 16
        "tail_base",       # 17
        "nose",            # 18
        "neck_base",       # 19
    ],
    keypoints_connectivity=[
        # Eyes to nose and neck_base
        (0, 18),  # r_eye to nose
        (1, 18),  # l_eye to nose
        (18, 19),  # nose to neck_base
        # Neck_base to shoulders
        (19, 2),  # neck_base to r_shoulder
        (19, 11),  # neck_base to l_shoulder
        (19, 5),  # neck_base to spine
        # Shoulders to front legs
        (2, 3),  # r_shoulder to r_front_knee
        (3, 4),  # r_front_knee to r_front_ankle
        (11, 12),  # l_shoulder to l_front_knee
        (12, 13),  # l_front_knee to l_front_ankle
        # Shoulders to spine and tail_base
        (2, 5),  # r_shoulder to spine
        (11, 5),  # l_shoulder to spine
        (5, 17),  # spine to tail_base
        # Spine to hips
        (5, 6),  # spine to r_hip
        (5, 14),  # spine to l_hip
        # Hips to back legs
        (6, 7),  # r_hip to r_back_knee
        (7, 8),  # r_back_knee to r_back_ankle
        (14, 15),  # l_hip to l_back_knee
        (15, 16),  # l_back_knee to l_back_ankle
        # Tail
        (17, 9),  # tail_base to tail1
        (9, 10),  # tail1 to tail2
    ],
)


def load_fte_annotations(
    annotations_file: str,
    camera_id: str,
    subject_id: str,
    bbox_kp_score_thr: float = 0.5,
) -> tuple[list[Keypoints2DAnnotation], list[BBox2DAnnotation]]:
    df = pd.read_hdf(annotations_file)

    full_index = range(0, df.index.max() + 1)
    df = df.reindex(full_index, fill_value=np.nan)

    bodyparts = CHEETAH_KPS_FORMAT.keypoints_names

    n_frames = len(df)

    xs = df.loc[:, (bodyparts, "x")].to_numpy()
    ys = df.loc[:, (bodyparts, "y")].to_numpy()

    kps_2d_annotations: list[Keypoints2DAnnotation] = []
    bboxes_2d_annotations: list[BBox2DAnnotation] = []

    for frame_idx in tqdm(
        range(n_frames),
        desc=f"Loading camera {camera_id} keypoints annotations",
        leave=False,
    ):
        kps_xy = torch.tensor(
            np.stack([xs[frame_idx], ys[frame_idx]], axis=1), dtype=torch.float32
        )
        valid_mask = torch.isfinite(kps_xy).all(dim=-1)
        kps_xy[~valid_mask] = 0
        score = torch.where(valid_mask, 1.0, 0.0)

        mask = score > bbox_kp_score_thr
        if mask.any():
            valid_xy = kps_xy[mask]
            min_x, min_y = valid_xy.min(dim=0).values
            max_x, max_y = valid_xy.max(dim=0).values
            bbox_score = 1.0
            bbox_xyxy = torch.tensor([min_x - 1, min_y - 1, max_x + 1, max_y + 1], dtype=torch.float32)
        else:
            bbox_xyxy = torch.zeros(4, dtype=torch.float32)
            bbox_score = 0.0

        bboxes_2d_annotations.append(
            BBox2DAnnotation(
                view_id=camera_id,
                frame_idx=frame_idx,
                subject_id=subject_id,
                category_id=-1,
                xyxy=bbox_xyxy,
                score=bbox_score,
            )
        )

        kps_2d_annotations.append(
            Keypoints2DAnnotation(
                view_id=camera_id,
                frame_idx=frame_idx,
                subject_id=subject_id,
                xy=kps_xy,
                scores=score,
                format=CHEETAH_KPS_FORMAT.name,
            )
        )

    return kps_2d_annotations, bboxes_2d_annotations

def load_dlc_annotations(
    annotations_file: str,
    camera_id: str,
    subject_id: str,
    bbox_kp_score_thr: float = 0.5,
) -> tuple[list[Keypoints2DAnnotation], list[BBox2DAnnotation]]:
    df = pd.read_hdf(annotations_file)

    scorer = df.columns.get_level_values(0).unique()[0]
    bodyparts = CHEETAH_KPS_FORMAT.keypoints_names

    n_frames = len(df)

    xs = df.loc[:, (scorer, bodyparts, "x")].to_numpy()
    ys = df.loc[:, (scorer, bodyparts, "y")].to_numpy()
    scores = df.loc[:, (scorer, bodyparts, "likelihood")].to_numpy()

    kps_2d_annotations: list[Keypoints2DAnnotation] = []
    bboxes_2d_annotations: list[BBox2DAnnotation] = []

    for frame_idx in tqdm(
        range(n_frames),
        desc=f"Loading camera {camera_id} keypoints annotations",
        leave=False,
    ):
        kps_xy = torch.tensor(
            np.stack([xs[frame_idx], ys[frame_idx]], axis=1), dtype=torch.float32
        )
        score = torch.tensor(scores[frame_idx], dtype=torch.float32)

        mask = score > bbox_kp_score_thr
        if mask.any():
            valid_xy = kps_xy[mask]
            valid_scores = score[mask]
            min_x, min_y = valid_xy.min(dim=0).values
            max_x, max_y = valid_xy.max(dim=0).values
            bbox_score = valid_scores.mean().item()
            bbox_xyxy = torch.tensor([min_x, min_y, max_x, max_y], dtype=torch.float32)
        else:
            bbox_xyxy = torch.zeros(4, dtype=torch.float32)
            bbox_score = 0.0

        bboxes_2d_annotations.append(
            BBox2DAnnotation(
                view_id=camera_id,
                frame_idx=frame_idx,
                subject_id=subject_id,
                category_id=-1,
                xyxy=bbox_xyxy,
                score=bbox_score,
            )
        )

        kps_2d_annotations.append(
            Keypoints2DAnnotation(
                view_id=camera_id,
                frame_idx=frame_idx,
                subject_id=subject_id,
                xy=kps_xy,
                scores=score,
                format=CHEETAH_KPS_FORMAT.name,
            )
        )

    return kps_2d_annotations, bboxes_2d_annotations


if __name__ == "__main__":
    config_file = "configs/experiments/acinoset_qualitative_eval.yaml"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = Pipeline.build_pipeline_from_config(config_file, device)

    data_dir = "data/flick1"
    cameras = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]

    views = []
    for camera_id in cameras:
        view_input = ViewInput(
            view_id=camera_id,
            frame_loader=VideoLoader(
                video_path=os.path.join(
                    data_dir,
                    f"{camera_id}.mp4",
                ),
                device=device,
            ),
            audio_loader=None,
        )
        views.append(view_input)

    all_kps_2d_annotations: list[Keypoints2DAnnotation] = []
    all_bboxes_2d_annotations: list[BBox2DAnnotation] = []

    for camera_id in cameras:
        kps_2d_annotations, bboxes_2d_annotations = load_fte_annotations(
            os.path.join(
                data_dir,
                "fte_pw",
                f"{camera_id}_fte.h5",
            ),
            camera_id,
            "cheetah_0",
        )
        # kps_2d_annotations, bboxes_2d_annotations = load_dlc_annotations(
        #     os.path.join(
        #         data_dir,
        #         "dlc_pw",
        #         f"{camera_id}DLC_resnet152_CheetahOct14shuffle3_650000.h5",
        #     ),
        #     camera_id,
        #     "cheetah_0",
        # )
        all_kps_2d_annotations.extend(kps_2d_annotations)
        all_bboxes_2d_annotations.extend(bboxes_2d_annotations)

    keypoints_2d = Keypoints2DAnnotations(
        metadata=Keypoints2DAnnotationsMetadata(formats=[CHEETAH_KPS_FORMAT]),
        annotations=all_kps_2d_annotations,
    )

    bboxes_2d = BBox2DAnnotations(
        metadata=BBox2DAnnotationsMetadata(),
        annotations=all_bboxes_2d_annotations,
    )

    pipeline.run(
        sequence_name="flick1",
        views=views,
        annotations={
            "keypoints_2d": keypoints_2d,
            "bboxes_2d": bboxes_2d,
        },
        gt_annotations={},
    )
