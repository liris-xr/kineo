"""
This script converts the weights from the PoseResNet model used in the following papers:
- "VoxelPose: Towards Multi-Camera 3D Human Pose Estimation in Wild Environment" by Tu et al.
- "Learnable Triangulation of Human Pose" by Iskakov et al. to a format that can
be used by MMPose.
"""

import argparse
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("state_dict_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    # Load the state dict
    state_dict = torch.load(args.state_dict_path)

    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module.deconv_layers"):
            new_state_dict[key.replace("module.", "head.")] = value
        elif key.startswith("module.final_layer"):
            new_state_dict[key.replace("module.", "head.")] = value
        elif (
            key.startswith("module.conv1")
            or key.startswith("module.bn1")
            or key.startswith("module.relu")
            or key.startswith("module.maxpool")
            or key.startswith("module.layer")
        ):
            new_state_dict[key.replace("module.", "backbone.")] = value
        else:
            raise ValueError(f"Unknown key: {key}")

    # Default model outputs 33 keypoints but discard the last 16 when loading.
    # We only keep the first 17 to match the H36M expected output for MMPose.
    final_layer_weight = new_state_dict["head.final_layer.weight"][:17]
    final_layer_bias = new_state_dict["head.final_layer.bias"][:17]

    # Permute the output order to match the H3.6M layout as described in MMPose documentation.
    permute = [6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10]

    final_layer_weight = final_layer_weight[permute]
    final_layer_bias = final_layer_bias[permute]

    new_state_dict["head.final_layer.weight"] = final_layer_weight
    new_state_dict["head.final_layer.bias"] = final_layer_bias

    torch.save(new_state_dict, args.output_path)
    print(f"Successfully converted state dict to {args.output_path}")
