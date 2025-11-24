import pickle
import orjson
import os
from tqdm import tqdm
from kineo.annotations.camera_extrinsics import CameraExtrinsicsAnnotations
from kineo.annotations.camera_intrinsics import CameraIntrinsicsAnnotations
from kineo.annotations.keypoints_3d import Keypoints3DAnnotations
from kineo.eval.sequence_metrics import compute_sequence_metrics, compute_all_sequences_metrics
from collections import defaultdict

def print_statistics(
    cam_metrics_stats: dict[str, dict[str, float]],
    human_metrics_stats: dict[str, dict[str, float]],
    failed_sequences: list[str],
):
    print("\n=== Statistics Report ===\n")
    print("📷 Camera Metrics:")
    for metric_name, metric_stats in cam_metrics_stats.items():
        print(f"- {metric_name}:")
        for key, value in metric_stats.items():
            # Print with 4 significant digits
            print(f"\t- {key:<10}: {value:.4g}")

    print("\n🧑 Human Metrics:")
    for metric_name, metric_stats in human_metrics_stats.items():
        print(f"- {metric_name}:")
        for key, value in metric_stats.items():
            # Print with 4 significant digits
            print(f"\t- {key:<10}: {value:.4g}")

    if failed_sequences:
        print("\n❌ Failed Sequences:")
        for seq in failed_sequences:
            print(f"  - {seq}")

    print("\n=========================\n")

def load_kineo_predictions_in_folder(
    folder_path: str,
) -> tuple[
    CameraIntrinsicsAnnotations,
    CameraExtrinsicsAnnotations,
    Keypoints3DAnnotations,
]:
    camera_intrinsics_file = os.path.join(folder_path, "camera_intrinsics.pkl")
    camera_extrinsics_file = os.path.join(folder_path, "camera_extrinsics.pkl")
    keypoints_3d_file = os.path.join(folder_path, "keypoints_3d.pkl")

    if not os.path.exists(camera_intrinsics_file):
        raise FileNotFoundError(
            f"Camera intrinsics file not found: {camera_intrinsics_file}"
        )

    if not os.path.exists(camera_extrinsics_file):
        raise FileNotFoundError(
            f"Camera extrinsics file not found: {camera_extrinsics_file}"
        )

    if not os.path.exists(keypoints_3d_file):
        raise FileNotFoundError(f"Keypoints 3D file not found: {keypoints_3d_file}")

    with open(camera_intrinsics_file, "rb") as f:
        kineo_camera_intrinsics = CameraIntrinsicsAnnotations.from_dict(pickle.load(f))

    with open(camera_extrinsics_file, "rb") as f:
        kineo_camera_extrinsics = CameraExtrinsicsAnnotations.from_dict(pickle.load(f))

    with open(keypoints_3d_file, "rb") as f:
        kineo_keypoints_3d = Keypoints3DAnnotations.from_dict(pickle.load(f))

    return kineo_camera_intrinsics, kineo_camera_extrinsics, kineo_keypoints_3d


def load_gt_keypoints_3d(
    dataset_dir: str,
    sequence: dict,
) -> Keypoints3DAnnotations:
    keypoints_3d_file = sequence["annotations"]["keypoints_3d"]
    keypoints_3d_file = os.path.join(dataset_dir, keypoints_3d_file)

    with open(keypoints_3d_file, "rb") as f:
        gt_keypoints_3d = Keypoints3DAnnotations.from_dict(orjson.loads(f.read()))

    return gt_keypoints_3d


def load_gt_camera_intrinsics(
    dataset_dir: str,
    sequence: dict,
) -> CameraIntrinsicsAnnotations:
    camera_intrinsics_file = sequence["annotations"]["cameras_intrinsics"]
    camera_intrinsics_file = os.path.join(dataset_dir, camera_intrinsics_file)

    with open(camera_intrinsics_file, "rb") as f:
        gt_camera_intrinsics = CameraIntrinsicsAnnotations.from_dict(
            orjson.loads(f.read())
        )

    return gt_camera_intrinsics


def load_gt_camera_extrinsics(
    dataset_dir: str,
    sequence: dict,
) -> CameraExtrinsicsAnnotations:
    camera_extrinsics_file = sequence["annotations"]["cameras_extrinsics"]
    camera_extrinsics_file = os.path.join(dataset_dir, camera_extrinsics_file)

    with open(camera_extrinsics_file, "rb") as f:
        gt_camera_extrinsics = CameraExtrinsicsAnnotations.from_dict(
            orjson.loads(f.read())
        )

    return gt_camera_extrinsics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "egohumans_dataset_dir", type=str, help="Path to dataset directory"
    )
    parser.add_argument(
        "egohumans_pred_annotations_dir",
        type=str,
        help="Path to predicted annotations directory",
    )
    args = parser.parse_args()
    dataset_dir = args.egohumans_dataset_dir
    pred_annotations_dir = args.egohumans_pred_annotations_dir

    sequences_file = os.path.join(dataset_dir, "egohumans_sequences.json")

    with open(sequences_file, "rb") as f:
        sequences = orjson.loads(f.read())

    all_seq_cam_metrics = defaultdict(list)
    all_seq_human_metrics = defaultdict(list)
    failed_sequences = []

    pbar = tqdm(sequences, desc="Processing sequences")

    for sequence in pbar:
        sequence_name = sequence["sequence_name"]
        pbar.set_postfix(sequence_name=sequence_name)

        try:
            gt_keypoints_3d = load_gt_keypoints_3d(dataset_dir, sequence)
            gt_camera_intrinsics = load_gt_camera_intrinsics(dataset_dir, sequence)
            gt_camera_extrinsics = load_gt_camera_extrinsics(dataset_dir, sequence)

            views_ids = gt_camera_intrinsics.views_ids
            views_resolution_hw = [
                gt_camera_intrinsics.filter_by_view_id(view_id)
                .first_or_default()
                .resolution_hw
                for view_id in views_ids
            ]
            subject_id = gt_keypoints_3d.subjects_ids[0]

            kineo_camera_intrinsics, kineo_camera_extrinsics, kineo_keypoints_3d = (
                load_kineo_predictions_in_folder(
                    folder_path=os.path.join(
                        args.egohumans_pred_annotations_dir, sequence["sequence_name"]
                    ),
                )
            )

            cam_metrics, human_metrics = compute_sequence_metrics(
                gt_keypoints_3d_annotations=gt_keypoints_3d,
                gt_cam_intrinsics_annotations=gt_camera_intrinsics,
                gt_cam_extrinsics_annotations=gt_camera_extrinsics,
                pred_keypoints_3d_annotations=kineo_keypoints_3d,
                pred_cam_intrinsics_annotations=kineo_camera_intrinsics,
                pred_cam_extrinsics_annotations=kineo_camera_extrinsics,
            )
            all_seq_cam_metrics[sequence_name] = cam_metrics
            all_seq_human_metrics[sequence_name] = human_metrics

        except FileNotFoundError as e:
            print(f"File not found for sequence {sequence_name}: {e}")
            failed_sequences.append(sequence_name)
            continue

    pbar.close()
    cam_metrics_stats, human_metrics_stats = compute_all_sequences_metrics(
        all_seq_cam_metrics, all_seq_human_metrics
    )
    print_statistics(cam_metrics_stats, human_metrics_stats, failed_sequences)