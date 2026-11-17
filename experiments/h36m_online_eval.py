# -----------------------------------------------------------------------------
# Kineo
# Copyright (c) Ecole Centrale de Lyon, CNRS, University Claude Bernard Lyon 1,
# and INSA Lyon. All rights reserved.
#
# Use of this software is strictly for research and evaluation purposes only.
# Commercial use or distribution without prior written consent is prohibited.
# Contact: guillaume.lavoue@enise.ec-lyon.fr
# -----------------------------------------------------------------------------

import argparse
import contextlib
import os

# Must be set before importing torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import pickle
import traceback

import orjson
import torch
from omegaconf import OmegaConf

from kineo.datasets import annotations_io
from kineo.datasets.h36m.h36m_dataset import H36MSequenceDataset
from kineo.datasets.keypoints_sequence_dataset import ViewInput
from kineo.annotations.keypoints_3d import (
    Keypoints3DAnnotation,
    Keypoints3DAnnotations,
)
from kineo.eval import sequence_metrics
from kineo.io.frame_range_loader import FrameRangeLoader
from kineo.io.frame_sequence_loader import FrameSequenceLoader
from kineo.io.nvdec_video_loader import NvdecVideoLoader
from kineo.pipeline.pipeline import Pipeline

def set_fast_backends(enabled: bool):
    torch.use_deterministic_algorithms(not enabled)
    torch.backends.cudnn.deterministic = not enabled
    torch.backends.cudnn.benchmark = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


# Autotuned kernels make BA drift
@contextlib.contextmanager
def deterministic_calibration():
    set_fast_backends(False)
    try:
        yield
    finally:
        set_fast_backends(True)


set_fast_backends(True)


def split_frame_counts(
    loader: FrameSequenceLoader,
    calibration_window_s: float | None,
    reconstruction_start_s: float | None,
) -> tuple[int, int]:
    if calibration_window_s is None and reconstruction_start_s is None:
        raise ValueError(
            "Calibrating on the whole sequence needs reconstruction_start_s, "
            "or there is no offset to start reconstructing from"
        )

    n_calibration_frames = (
        loader.n_frames
        if calibration_window_s is None
        else FrameRangeLoader.frames_for_duration(loader, calibration_window_s)
    )
    n_skipped_frames = FrameRangeLoader.frames_for_duration(
        loader,
        calibration_window_s
        if reconstruction_start_s is None
        else reconstruction_start_s,
    )

    if n_skipped_frames >= loader.n_frames:
        raise ValueError(
            f"A sequence of {loader.n_frames} frames is not longer than the "
            f"{n_skipped_frames} frames skipped before reconstruction; "
            f"nothing would be left to reconstruct"
        )

    return n_calibration_frames, n_skipped_frames


def split_views(
    views_inputs: list[ViewInput],
    calibration_window_s: float | None,
    reconstruction_start_s: float | None = None,
) -> tuple[list[ViewInput], list[ViewInput]]:
    calibration_views: list[ViewInput] = []
    stream_views: list[ViewInput] = []

    for view in views_inputs:
        loader = view["frame_loader"]
        n_calibration_frames, n_skipped_frames = split_frame_counts(
            loader, calibration_window_s, reconstruction_start_s
        )

        calibration_views.append(
            ViewInput(
                view_id=view["view_id"],
                frame_loader=FrameRangeLoader(
                    loader, start_frame=0, n_frames=n_calibration_frames
                ),
                audio_loader=None,
            )
        )

        source_indices = (
            loader.selected_frames.tolist()
            if loader.selected_frames is not None
            else list(range(loader.n_frames))
        )
        stream_views.append(
            ViewInput(
                view_id=view["view_id"],
                frame_loader=NvdecVideoLoader(
                    loader.video_path,
                    loader.device,
                    frame_indices=source_indices[n_skipped_frames:],
                ),
                audio_loader=None,
            )
        )

    return calibration_views, stream_views


def load_exported_calibration(
    output_root_dir: str, sequence_name: str
) -> dict | None:
    annotations = {}
    for key in ("cameras_intrinsics", "cameras_extrinsics"):
        path = os.path.join(
            output_root_dir, "annotations", sequence_name, f"{key}.pkl"
        )
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            kind = annotations_io.ANNOTATION_KINDS[key]
            annotations[key] = kind.annotations_type.from_dict(
                pickle.load(f)
            )
    return annotations


def shift_frame_indices(
    keypoints_3d: Keypoints3DAnnotations, offset: int
) -> Keypoints3DAnnotations:
    return Keypoints3DAnnotations(
        metadata=keypoints_3d.metadata,
        annotations=[
            Keypoints3DAnnotation(
                frame_idx=annotation.frame_idx + offset,
                subject_id=annotation.subject_id,
                xyz=annotation.xyz,
                scores=annotation.scores,
                format=annotation.format,
            )
            for annotation in keypoints_3d.annotations
        ],
    )


def score_stream(
    gt_annotations: dict,
    calibration_annotations: dict,
    stream_annotations: dict,
    n_skipped_frames: int,
) -> dict:
    predictions = shift_frame_indices(
        stream_annotations["keypoints_3d"], n_skipped_frames
    )
    streamed_frames = set(predictions.frames)

    gt_keypoints_3d = gt_annotations["keypoints_3d"]
    gt_streamed = Keypoints3DAnnotations(
        metadata=gt_keypoints_3d.metadata,
        annotations=[
            annotation
            for annotation in gt_keypoints_3d.annotations
            if annotation.frame_idx in streamed_frames
        ],
    )

    gt_cam_extrinsics = gt_annotations["cameras_extrinsics"]
    pred_cam_extrinsics = calibration_annotations["cameras_extrinsics"]

    cam_metrics = sequence_metrics.compute_camera_metrics_over_sequence(
        gt_cam_intrinsics_annotations=gt_annotations["cameras_intrinsics"],
        gt_cam_extrinsics_annotations=gt_cam_extrinsics,
        pred_cam_intrinsics_annotations=calibration_annotations[
            "cameras_intrinsics"
        ],
        pred_cam_extrinsics_annotations=pred_cam_extrinsics,
    )
    human_metrics = sequence_metrics.compute_human_metrics_over_sequence(
        gt_keypoints_3d_annotations=gt_streamed,
        gt_cam_extrinsics_annotations=gt_cam_extrinsics,
        pred_keypoints_3d_annotations=predictions,
        pred_cam_extrinsics_annotations=pred_cam_extrinsics,
    )

    return {
        "cam_stats": cam_metrics,
        "human_stats": human_metrics,
        "n_streamed_frames": len(streamed_frames),
        "n_skipped_frames": n_skipped_frames,
    }


def main(
    dataset_dir: str,
    calibration_config_file: str,
    stream_config_file: str,
    calibration_window_s: float | None,
    reconstruction_start_s: float | None,
    split: str,
    sequences_names: list[str],
    reuse_calibration: bool,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    calibration_output_root_dir = str(
        OmegaConf.to_container(
            OmegaConf.load(calibration_config_file), resolve=True
        )["output_root_dir"]
    )
    stream_output_root_dir = str(
        OmegaConf.to_container(
            OmegaConf.load(stream_config_file), resolve=True
        )["output_root_dir"]
    )

    dataset = H36MSequenceDataset(
        dataset_filepath=os.path.join(
            dataset_dir, "h36m_protocol1_sequences.json"
        ),
        split=split,
        device=device,
    )

    calibration_pipeline = None
    stream_pipeline = Pipeline.build_pipeline_from_config(
        stream_config_file, device
    )

    failed_sequences: list[str] = []

    for sequence in dataset:
        sequence_name = sequence["sequence_name"]

        if sequences_names and sequence_name not in sequences_names:
            continue

        try:
            calibration_views, stream_views = split_views(
                sequence["views_inputs"],
                calibration_window_s,
                reconstruction_start_s,
            )
            view_ids = [view["view_id"] for view in sequence["views_inputs"]]

            n_calibration_frames = len(calibration_views[0]["frame_loader"])
            _, n_skipped_frames = split_frame_counts(
                sequence["views_inputs"][0]["frame_loader"],
                calibration_window_s,
                reconstruction_start_s,
            )
            n_frames = sequence["views_inputs"][0]["frame_loader"].n_frames
            print(
                f"{sequence_name}: calibrating on "
                f"{n_calibration_frames} frames, "
                f"reconstructing {n_frames - n_skipped_frames}"
            )

            synchronized_temporal = (
                annotations_io.build_synchronized_camera_temporal(view_ids)
            )

            calibration_annotations = None
            if reuse_calibration:
                calibration_annotations = load_exported_calibration(
                    calibration_output_root_dir, sequence_name
                )

            if calibration_annotations is None:
                with deterministic_calibration():
                    if calibration_pipeline is None:
                        calibration_pipeline = (
                            Pipeline.build_pipeline_from_config(
                                calibration_config_file, device
                            )
                        )
                    calibration_annotations = calibration_pipeline.run(
                        sequence_name=sequence_name,
                        views=calibration_views,
                        annotations={
                            "cameras_temporal": synchronized_temporal,
                        },
                        gt_annotations={},
                    )

            stream_annotations = stream_pipeline.run(
                sequence_name=sequence_name,
                views=stream_views,
                annotations={
                    "cameras_intrinsics": calibration_annotations[
                        "cameras_intrinsics"
                    ],
                    "cameras_extrinsics": calibration_annotations[
                        "cameras_extrinsics"
                    ],
                },
                gt_annotations={},
            )

            if sequence["annotations"] is None:
                print(f"{sequence_name}: no ground truth, skipping")
                continue

            metrics = score_stream(
                gt_annotations=sequence["annotations"],
                calibration_annotations=calibration_annotations,
                stream_annotations=stream_annotations,
                n_skipped_frames=n_skipped_frames,
            )

            metrics_path = os.path.join(
                stream_output_root_dir, "metrics", f"{sequence_name}.json"
            )
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, "wb") as f:
                f.write(
                    orjson.dumps(
                        metrics, default=float, option=orjson.OPT_INDENT_2
                    )
                )
            print(f"{sequence_name}: metrics written to {metrics_path}")
        except Exception:
            print(f"{sequence_name} failed: " + traceback.format_exc())
            failed_sequences.append(sequence_name)

    if failed_sequences:
        print(f"Failed sequences: {failed_sequences}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/experiments/online_vs_offline/h36m_online.yaml",
        help="The arm to run: what the calibration may read, where "
        "reconstruction starts, and the two pipelines it runs",
    )
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--sequences-names", type=str, nargs="*", default=[])
    parser.add_argument("--reuse-calibration", action="store_true")
    args = parser.parse_args()

    arm = OmegaConf.to_container(OmegaConf.load(args.config_file))

    main(
        dataset_dir=args.dataset_dir,
        calibration_config_file=arm["calibration_config_file"],
        stream_config_file=arm["stream_config_file"],
        calibration_window_s=arm["calibration_window_s"],
        reconstruction_start_s=arm["reconstruction_start_s"],
        split=args.split,
        sequences_names=args.sequences_names,
        reuse_calibration=args.reuse_calibration,
    )
