"""
This script computes the parameters of the affine transformation that converts
the height from a H3.6M skeleton to the real height of the subject.
"""

import torch
from tqdm import tqdm

from kineo.datasets.h36m.h36m_sequence_dataset import H36MSequenceDataset
from kineo.geometry.metrics import compute_height_from_h36m_keypoints

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = H36MSequenceDataset(
        sequences_annotations_fp="E:/Datasets/H3.6M/processed/sequences_annotations.npy",
        protocol=1,
        train=True,
        device=device,
    )

    val_dataset = H36MSequenceDataset(
        sequences_annotations_fp="E:/Datasets/H3.6M/processed/sequences_annotations.npy",
        protocol=1,
        train=False,
        device=device,
    )

    heights: dict[str, list[torch.Tensor]] = {}
    gt_height: dict[str, float] = {}

    for sequence in tqdm(train_dataset, desc="Train Sequences"):
        kps_3d = sequence.gt_keypoints_3d["pos3d"]
        height = compute_height_from_h36m_keypoints(
            kps_3d, torch.ones(kps_3d.shape[:2], device=kps_3d.device)
        )

        if sequence.subject not in gt_height:
            gt_height[sequence.subject] = sequence.gt_subject_height

        if sequence.subject not in heights:
            heights[sequence.subject] = []

        heights[sequence.subject].append(height)

    for subject in heights:
        heights[subject] = torch.stack(heights[subject]).median(dim=0).values

    X = torch.tensor([heights[subject].item() for subject in heights]).reshape(-1, 1)
    y = torch.tensor([gt_height[subject] for subject in heights])
    X_with_bias = torch.cat([X, torch.ones_like(X)], dim=1)
    solution = torch.linalg.lstsq(X_with_bias, y.unsqueeze(1)).solution
    a, b = solution.flatten()

    avg_height_error = 0

    for sequence in tqdm(val_dataset, desc="Val Sequences"):
        kps_3d = sequence.gt_keypoints_3d["pos3d"]
        computed_height = compute_height_from_h36m_keypoints(
            kps_3d, torch.ones(kps_3d.shape[:2], device=kps_3d.device)
        )

        if sequence.subject not in gt_height:
            gt_height[sequence.subject] = sequence.gt_subject_height

        # Apply the linear transformation: height = a * computed_height + b
        calibrated_height = a * computed_height + b

        avg_height_error += (gt_height[sequence.subject] - calibrated_height).abs().item()

    avg_height_error /= len(val_dataset)
    print(f"Average height error: {avg_height_error * 1000:.2f}mm")
    print(f"Linear relation: height = {a:.4f} * computed_height + {b:.4f}")
