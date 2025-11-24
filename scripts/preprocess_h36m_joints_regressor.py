import torch
import argparse


def load_h36m_joints_regressor(
    joints_regressor_path: str,
    keep_original_legs_order: bool = False,
) -> torch.Tensor:
    # From https://github.com/ubc-vision/joint-regressor-refinement/tree/master
    J_regressor: torch.Tensor = torch.load(joints_regressor_path, weights_only=True)
    J_regressor = J_regressor.float().cpu().detach()
    J_regressor = torch.nn.ReLU()(J_regressor)
    J_regressor = J_regressor / torch.sum(J_regressor, dim=1).unsqueeze(1).expand(
        J_regressor.shape
    )
    if not keep_original_legs_order:
        # Swap the legs to match GT annotations order (following MMPose's order)
        J_regressor[[1, 2, 3, 4, 5, 6]] = J_regressor[[4, 5, 6, 1, 2, 3]]
    return J_regressor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the joints regressor file",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the joints regressor file",
    )
    parser.add_argument(
        "--keep-original-legs-order",
        action="store_true",
        help="Keep the original legs order",
    )
    args = parser.parse_args()

    J_regressor = load_h36m_joints_regressor(
        joints_regressor_path=args.input_path,
        keep_original_legs_order=args.keep_original_legs_order,
    )

    # Save the J_regressor to a file
    torch.save(J_regressor.cpu(), args.output_path)
    print(f"Saved the J_regressor to {args.output_path}")
