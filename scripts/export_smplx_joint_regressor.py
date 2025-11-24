from smplx import SMPLLayer
import argparse
import torch
import numpy as np

# Fix to properly load smpl model without errors
np.bool = np.bool_
np.int = np.int_
np.float = np.float_
np.long = np.int_
np.complex = np.complex_
np.object = np.object_
np.str = np.str_
np.unicode = np.unicode_

def main(
    model_path: str,
    gender: str,
    n_joints: int,
    output_path: str,
):
    smpl_layer = SMPLLayer(
        model_path=model_path, gender=gender
    )
    J_regressor = smpl_layer.J_regressor
    J_regressor = J_regressor[:n_joints]
    print(J_regressor.shape)
    torch.save(J_regressor, output_path)
    print(f"Saved the J_regressor to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./body_models/smplx/SMPLX_NEUTRAL.pkl")
    parser.add_argument("--gender", type=str, default="neutral")
    parser.add_argument("--n-joints", type=int, default=24)
    parser.add_argument(
        "--output-path", type=str, default="./body_models/smplx/J_regressor.pt"
    )
    args = parser.parse_args()

    main(args.model_path, args.gender, args.n_joints, args.output_path)
