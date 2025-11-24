import argparse
import torch
import numpy as np
from pathlib import Path

# Fix to properly load smpl model without errors
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.long = np.int_
np.complex = np.complex_
np.object = np.object_
np.str = np.str_
np.unicode = np.unicode_

COCO17_TO_COCO19_MAP = [1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14]

def main(
    coco17_regressor_path: str,
    output_path: str,
):
    extension = Path(coco17_regressor_path).suffix

    if extension == ".pt":
        coco17_joints_regressor = torch.load(coco17_regressor_path)
    elif extension == ".npy":
        coco17_joints_regressor = torch.from_numpy(np.load(coco17_regressor_path))
    elif extension == ".npz":
        coco17_joints_regressor = torch.from_numpy(np.load(coco17_regressor_path))
    else:
        raise ValueError(f"Unsupported joint regressor file extension: {extension}")

    n_vertices = coco17_joints_regressor.shape[1]

    left_shoulder_coeffs = coco17_joints_regressor[5]
    right_shoulder_coeffs = coco17_joints_regressor[6]

    left_hip_coeffs = coco17_joints_regressor[11]
    right_hip_coeffs = coco17_joints_regressor[12]

    neck_coeffs = (left_shoulder_coeffs + right_shoulder_coeffs) / 2
    pelvis_coeffs = (left_hip_coeffs + right_hip_coeffs) / 2

    coco19_joints_regressor = torch.zeros((19, n_vertices))

    for i, coco19_joint_idx in enumerate(COCO17_TO_COCO19_MAP):
        coco19_joints_regressor[coco19_joint_idx] = coco17_joints_regressor[i]

    coco19_joints_regressor[0] = neck_coeffs
    coco19_joints_regressor[2] = pelvis_coeffs

    torch.save(coco19_joints_regressor, output_path)
    print(f"Saved the COCO-19 joints regressor to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco17-regressor-path", type=str, default="./body_models/smpl/J_regressor_coco.npy")
    parser.add_argument(
        "--output-path", type=str, default="./body_models/smpl/J_regressor_coco19.pt"
    )
    args = parser.parse_args()

    main(args.coco17_regressor_path, args.output_path)
