import argparse
import os
import pickle
import orjson
from typing import Any
from tqdm import tqdm
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.stage_timing import StageTimingsAnnotations
from kineo.eval.dataset_metrics import compute_predictions_metrics


def load_gt_annotations(
    dataset_dir: str,
    sequences: list[dict],
) -> tuple[
    dict[str, Keypoints3DAnnotations],
    dict[str, CameraIntrinsicsAnnotations],
    dict[str, CameraExtrinsicsAnnotations],
]:
    gt_keypoints_3d_per_seq = {}
    gt_camera_intrinsics_per_seq = {}
    gt_camera_extrinsics_per_seq = {}

    for sequence in tqdm(
        sequences, desc="Loading ground truth annotations", leave=False
    ):
        seq_name = sequence["sequence_name"]

        camera_intrinsics_file = os.path.join(
            dataset_dir, sequence["annotations"]["cameras_intrinsics"]
        )
        keypoints_3d_file = os.path.join(
            dataset_dir, sequence["annotations"]["keypoints_3d"]
        )
        camera_extrinsics_file = os.path.join(
            dataset_dir, sequence["annotations"]["cameras_extrinsics"]
        )

        with open(keypoints_3d_file, "rb") as f:
            gt_keypoints_3d_per_seq[seq_name] = Keypoints3DAnnotations.from_dict(
                orjson.loads(f.read())
            )

        with open(camera_intrinsics_file, "rb") as f:
            gt_camera_intrinsics_per_seq[seq_name] = (
                CameraIntrinsicsAnnotations.from_dict(orjson.loads(f.read()))
            )

        with open(camera_extrinsics_file, "rb") as f:
            gt_camera_extrinsics_per_seq[seq_name] = (
                CameraExtrinsicsAnnotations.from_dict(orjson.loads(f.read()))
            )

    return (
        gt_keypoints_3d_per_seq,
        gt_camera_intrinsics_per_seq,
        gt_camera_extrinsics_per_seq,
    )


def load_predicted_annotations(
    predictions_dir: str,
    sequences: list[dict],
) -> tuple[
    dict[str, Keypoints3DAnnotations],
    dict[str, CameraIntrinsicsAnnotations],
    dict[str, CameraExtrinsicsAnnotations],
    dict[str, StageTimingsAnnotations],
]:
    pred_keypoints_3d_per_seq = {}
    pred_camera_intrinsics_per_seq = {}
    pred_camera_extrinsics_per_seq = {}
    pred_stage_timings_per_seq = {}

    for sequence in tqdm(sequences, desc="Loading predicted annotations", leave=False):
        seq_name = sequence["sequence_name"]

        try:
            pred_keypoints_3d_file = os.path.join(
                predictions_dir, seq_name, "keypoints_3d.pkl"
            )
            pred_camera_intrinsics_file = os.path.join(
                predictions_dir, seq_name, "cameras_intrinsics.pkl"
            )
            pred_camera_extrinsics_file = os.path.join(
                predictions_dir, seq_name, "cameras_extrinsics.pkl"
            )
            pred_stage_timings_file = os.path.join(
                predictions_dir, seq_name, "stage_timings.pkl"
            )
            with open(pred_keypoints_3d_file, "rb") as f:
                pred_keypoints_3d_per_seq[seq_name] = Keypoints3DAnnotations.from_dict(
                    pickle.load(f)
                )
            with open(pred_camera_intrinsics_file, "rb") as f:
                pred_camera_intrinsics_per_seq[seq_name] = (
                    CameraIntrinsicsAnnotations.from_dict(pickle.load(f))
                )
            with open(pred_camera_extrinsics_file, "rb") as f:
                pred_camera_extrinsics_per_seq[seq_name] = (
                    CameraExtrinsicsAnnotations.from_dict(pickle.load(f))
                )
            if os.path.exists(pred_stage_timings_file):
                with open(pred_stage_timings_file, "rb") as f:
                    pred_stage_timings_per_seq[seq_name] = (
                        StageTimingsAnnotations.from_dict(pickle.load(f))
                    )
        except Exception as e:
            print(f"Error loading predicted annotations for sequence {seq_name}: {e}")
            continue
    return (
        pred_keypoints_3d_per_seq,
        pred_camera_intrinsics_per_seq,
        pred_camera_extrinsics_per_seq,
        pred_stage_timings_per_seq,
    )


def available_pred_sequences(
    h36m_dataset_dir: str,
    h36m_annotations_dir: str,
) -> set[str]:
    """Sequence names (val split) that have a predicted keypoints_3d.pkl in the given dir."""
    sequences_file = os.path.join(h36m_dataset_dir, "h36m_protocol1_sequences.json")
    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())
    sequences = [s for s in sequences if s["split"] == "val"]
    return {
        s["sequence_name"]
        for s in sequences
        if os.path.isfile(
            os.path.join(h36m_annotations_dir, s["sequence_name"], "keypoints_3d.pkl")
        )
    }


def compute_h36m_stats(
    h36m_dataset_dir: str,
    h36m_annotations_dir: str,
    allowed_sequences: set[str] | None = None,
) -> dict[str, Any]:
    sequences_file = os.path.join(h36m_dataset_dir, "h36m_protocol1_sequences.json")
    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())
    sequences = [s for s in sequences if s["split"] == "val"]
    (
        gt_keypoints_3d_per_seq,
        gt_camera_intrinsics_per_seq,
        gt_camera_extrinsics_per_seq,
    ) = load_gt_annotations(h36m_dataset_dir, sequences)
    (
        pred_keypoints_3d_per_seq,
        pred_camera_intrinsics_per_seq,
        pred_camera_extrinsics_per_seq,
        pred_stage_timings_per_seq,
    ) = load_predicted_annotations(h36m_annotations_dir, sequences)
    if allowed_sequences is not None:
        # Restrict predictions to a common set for a fair (intersection) comparison.
        allowed = set(allowed_sequences)
        pred_keypoints_3d_per_seq = {
            k: v for k, v in pred_keypoints_3d_per_seq.items() if k in allowed
        }
        pred_camera_intrinsics_per_seq = {
            k: v for k, v in pred_camera_intrinsics_per_seq.items() if k in allowed
        }
        pred_camera_extrinsics_per_seq = {
            k: v for k, v in pred_camera_extrinsics_per_seq.items() if k in allowed
        }
        pred_stage_timings_per_seq = {
            k: v for k, v in pred_stage_timings_per_seq.items() if k in allowed
        }
    return compute_predictions_metrics(
        gt_keypoints_3d_annotations=gt_keypoints_3d_per_seq,
        gt_cam_intrinsics_annotations=gt_camera_intrinsics_per_seq,
        gt_cam_extrinsics_annotations=gt_camera_extrinsics_per_seq,
        pred_keypoints_3d_annotations=pred_keypoints_3d_per_seq,
        pred_cam_intrinsics_annotations=pred_camera_intrinsics_per_seq,
        pred_cam_extrinsics_annotations=pred_camera_extrinsics_per_seq,
        stage_timings_annotations=pred_stage_timings_per_seq,
    )


def print_h36m_stats(
    camera_metrics_stats: dict[str, dict[str, float]],
    human_metrics_stats: dict[str, dict[str, float]],
    stage_timings_stats: dict[str, dict[str, float]],
    n_total_frames: int,
    missing_sequences_predictions: list[str],
):
    print("\n=== Statistics Report ===\n")
    print("Camera Metrics:")
    for metric_name, metric_stats in camera_metrics_stats.items():
        print(f"- {metric_name}:")
        for key, value in metric_stats.items():
            print(f"\t- {key:<10}: {value:.4f}")

    print("\nHuman Metrics:")
    for metric_name, metric_stats in human_metrics_stats.items():
        print(f"- {metric_name}:")
        for key, value in metric_stats.items():
            print(f"\t- {key:<10}: {value:.4f}")

    print("\nStage Timings:")
    for stage_name, stage_stats in stage_timings_stats.items():
        print(f"- {stage_name}:")
        for key, value in stage_stats.items():
            print(f"\t- {key:<10}: {value:.4f}")

    print(f"\nTotal Number of Frames: {n_total_frames}")

    if missing_sequences_predictions:
        print("\nFailed Sequences:")
        for seq in missing_sequences_predictions:
            print(f"  - {seq}")

    print("\n=========================\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Print Human3.6M evaluation stats for a given set of predictions"
    )
    parser.add_argument(
        "h36m_dataset_dir", type=str, help="Path to Human3.6M dataset directory"
    )
    parser.add_argument(
        "h36m_annotations_dir",
        type=str,
        help="Path to the generated predictions annotations directory (should contain one folder for each sequence)",
    )
    args = parser.parse_args()

    stats = compute_h36m_stats(args.h36m_dataset_dir, args.h36m_annotations_dir)
    camera_metrics_stats = stats["camera_metrics"]
    human_metrics_stats = stats["human_metrics"]
    stage_timings_stats = stats["stage_timings"]
    n_total_frames = stats["n_total_frames"]
    missing_sequences_predictions = stats["missing_sequences_predictions"]
    print_h36m_stats(
        camera_metrics_stats,
        human_metrics_stats,
        stage_timings_stats,
        n_total_frames,
        missing_sequences_predictions,
    )
