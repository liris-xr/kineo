# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import os
import argparse
from math import atan2, degrees

import numpy as np
import orjson
import torch
from moge.model.v2 import MoGeModel
from tqdm import tqdm

from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.keypoints_2d import Keypoints2DAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.geometry.camera import transform_points_from_world_to_camera
from kineo.io.frame_sequence_loader import VideoLoader

TORSO_KEYPOINTS = [
    "root",
    "right_hip",
    "left_hip",
    "spine",
    "thorax",
    "neck_base",
    "head",
    "left_shoulder",
    "right_shoulder",
]


def load_annotations(dataset_dir: str, sequence: dict, key: str, cls):
    with open(os.path.join(dataset_dir, sequence["annotations"][key]), "rb") as f:
        return cls.from_dict(orjson.loads(f.read()))


def collect_depth_ratios(
    dataset_dir: str,
    sequence: dict,
    model: MoGeModel,
    n_frames: int,
    use_half_precision: bool,
    device: torch.device,
) -> np.ndarray:
    keypoints_3d = load_annotations(
        dataset_dir, sequence, "keypoints_3d", Keypoints3DAnnotations
    )
    keypoints_2d = load_annotations(
        dataset_dir, sequence, "keypoints_2d", Keypoints2DAnnotations
    )
    extrinsics = load_annotations(
        dataset_dir, sequence, "cameras_extrinsics", CameraExtrinsicsAnnotations
    )
    intrinsics = load_annotations(
        dataset_dir, sequence, "cameras_intrinsics", CameraIntrinsicsAnnotations
    )

    kps_3d_by_frame = {ann.frame_idx: ann.xyz for ann in keypoints_3d.annotations}
    frames = sorted(kps_3d_by_frame)
    picked_frames = [
        frames[i]
        for i in np.linspace(0, len(frames) - 1, n_frames).round().astype(int)
    ]

    rows = []
    for view_idx, (view_id, view) in enumerate(sequence["views"].items()):
        Rt = extrinsics.filter_by_view_id(view_id).first_or_default().Rt
        K = intrinsics.filter_by_view_id(view_id).first_or_default().K
        kps_2d_by_frame = {
            ann.frame_idx: ann.xy
            for ann in keypoints_2d.filter_by_view_id(view_id).annotations
        }

        selected_frames = view["selected_frames"]
        frame_loader = VideoLoader(
            video_path=os.path.join(dataset_dir, view["video_path"]),
            selected_frames=range(
                int(selected_frames["start"]),
                int(selected_frames["stop"]),
                int(selected_frames["step"]),
            ),
            device=device,
        )
        img_h, img_w = frame_loader.resolution_hw
        fov_x = degrees(2 * atan2(img_w, 2 * K[0, 0].item()))

        for frame_idx in picked_frames:
            frame_rgb = frame_loader.load_frame_at(frame_idx).float() / 255.0
            with torch.no_grad():
                depth = model.infer(
                    frame_rgb, fov_x=fov_x, use_fp16=use_half_precision
                )["depth"].cpu()

            gt_depth = transform_points_from_world_to_camera(
                kps_3d_by_frame[frame_idx].cpu(), Rt.cpu()
            )[:, 2]
            kps_xy = kps_2d_by_frame[frame_idx].cpu().round().long()

            for kp_idx, (x, y) in enumerate(kps_xy.tolist()):
                if 0 <= x < img_w and 0 <= y < img_h:
                    rows.append(
                        (
                            view_idx,
                            frame_idx,
                            kp_idx,
                            gt_depth[kp_idx].item(),
                            depth[y, x].item(),
                        )
                    )

        frame_loader.close()

    return np.array(rows, dtype=np.float64)


def summarize(rows: np.ndarray, keypoints_names: list[str]) -> dict:
    view_idx, frame_idx, kp_idx, gt_depth, moge_depth = rows.T
    finite = np.isfinite(moge_depth)
    view_idx, frame_idx, kp_idx = view_idx[finite], frame_idx[finite], kp_idx[finite]
    gt_depth, ratio = gt_depth[finite], moge_depth[finite] / gt_depth[finite]

    torso_idx = [keypoints_names.index(name) for name in TORSO_KEYPOINTS]
    is_torso = np.isin(kp_idx, torso_idx)

    def stats(values: np.ndarray) -> dict:
        return {
            "n": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "outliers_30pct": float((np.abs(values - 1) > 0.3).mean()),
        }

    return {
        "median_gt_depth_m": float(np.median(gt_depth)),
        "all": stats(ratio),
        "torso": stats(ratio[is_torso]),
        "limbs": stats(ratio[~is_torso]),
        "per_keypoint": {
            name: stats(ratio[kp_idx == i]) for i, name in enumerate(keypoints_names)
        },
        "per_view_median": [
            float(np.median(ratio[view_idx == v])) for v in np.unique(view_idx)
        ],
        "per_image_median": [
            float(np.median(ratio[(view_idx == v) & (frame_idx == f)]))
            for v in np.unique(view_idx)
            for f in np.unique(frame_idx)
        ],
    }


def main(
    dataset_dir: str,
    checkpoint: str,
    output_dir: str,
    n_frames: int,
    use_half_precision: bool,
    sequences_filter: list[str] = [],
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoGeModel.from_pretrained(checkpoint).to(device).eval()

    with open(os.path.join(dataset_dir, "h36m_protocol1_sequences.json"), "rb") as f:
        sequences = orjson.loads(f.read())

    if sequences_filter:
        sequences = [s for s in sequences if s["sequence_name"] in sequences_filter]
    sequences = [s for s in sequences if s["split"] == "val"]

    os.makedirs(output_dir, exist_ok=True)

    for sequence in tqdm(sequences, desc="Processing sequences"):
        sequence_name = sequence["sequence_name"]
        keypoints_names = load_annotations(
            dataset_dir, sequence, "keypoints_3d", Keypoints3DAnnotations
        ).metadata.formats[0].keypoints_names

        rows = collect_depth_ratios(
            dataset_dir, sequence, model, n_frames, use_half_precision, device
        )
        summary = summarize(rows, keypoints_names)

        np.save(os.path.join(output_dir, f"{sequence_name}_rows.npy"), rows)
        with open(os.path.join(output_dir, f"{sequence_name}.json"), "wb") as f:
            f.write(orjson.dumps(summary, option=orjson.OPT_INDENT_2))

        tqdm.write(
            f"{sequence_name}: median ratio {summary['all']['median']:.3f} "
            f"(torso {summary['torso']['median']:.3f}, limbs {summary['limbs']['median']:.3f}), "
            f"outliers {summary['all']['outliers_30pct']:.1%}, "
            f"per-view medians {[round(m, 3) for m in summary['per_view_median']]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument(
        "--checkpoint", type=str, default="./checkpoints/moge-2-vitl-normal.pt"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs/h36m_moge_depth_diagnostic"
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=12,
        help="Number of frames per view, evenly spaced over the sequence",
    )
    parser.add_argument("--use-half-precision", action="store_true")
    parser.add_argument(
        "--sequences-filter",
        nargs="+",
        default=[],
        help="List of sequences to process",
    )
    args = parser.parse_args()
    main(
        args.dataset_dir,
        args.checkpoint,
        args.output_dir,
        args.n_frames,
        args.use_half_precision,
        args.sequences_filter,
    )
