# Generate the rows for the camera parameters ablation study table for Kineo on EgoHumans and Human3.6M

import os
import argparse

from h36m_stats_utils import compute_h36m_stats
from egohumans_stats_utils import compute_egohumans_stats

def generate_h36m_ablation_study_camera_params_rows(h36m_dataset_dir: str, kineo_eval_data_dir: str, n_decimals: int = 2, with_std: bool = True):

    with open(
        os.path.join(os.path.dirname(__file__), "ablation_study_camera_params_h36m_rows_template.tex"), "r"
    ) as f:
        ablation_study_h36m_rows_template = f.read()

    kineo_h36m_camera_params_all_gt_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_cam_params_gtKps_gtRt_gtK_gtD", "annotations"
    )
    kineo_h36m_camera_params_extrinsics_annotations_path = os.path.join(
        kineo_eval_data_dir, "h36m_cam_params_gtKps_estRt_gtK_gtD", "annotations"
    )
    kineo_h36m_camera_params_extrinsics_intrinsics_without_distortions_annotations_path = os.path.join(
        kineo_eval_data_dir,
        "h36m_cam_params_gtKps_estRt_estK_omitD",
        "annotations",
    )
    kineo_h36m_camera_params_extrinsics_intrinsics_with_distortions_annotations_path = (
        os.path.join(
            kineo_eval_data_dir,
            "h36m_cam_params_gtKps_estRt_estK_estD",
            "annotations",
        )
    )
    kineo_h36m_camera_params_extrinsics_intrinsics_with_radial_distortions_annotations_path = (
        os.path.join(
            kineo_eval_data_dir,
            "h36m_cam_params_gtKps_estRt_estK_estDk1k2",
            "annotations",
        )
    )

    kineo_h36m_results = {
        "GtRt_GtK_GtD": compute_h36m_stats(
            h36m_dataset_dir, kineo_h36m_camera_params_all_gt_annotations_path
        ),
        "EstRt_GtK_GtD": compute_h36m_stats(
            h36m_dataset_dir, kineo_h36m_camera_params_extrinsics_annotations_path
        ),
        "EstRt_EstK_OmitD": compute_h36m_stats(
            h36m_dataset_dir,
            kineo_h36m_camera_params_extrinsics_intrinsics_without_distortions_annotations_path,
        ),
        "EstRt_EstK_EstD": compute_h36m_stats(
            h36m_dataset_dir,
            kineo_h36m_camera_params_extrinsics_intrinsics_with_distortions_annotations_path,
        ),
        "EstRt_EstK_EstDk1k2": compute_h36m_stats(
            h36m_dataset_dir,
            kineo_h36m_camera_params_extrinsics_intrinsics_with_radial_distortions_annotations_path,
        ),
    }

    ablation_study_h36m_rows = str(ablation_study_h36m_rows_template)
    for config_name in kineo_h36m_results:
        for human_metric_name in kineo_h36m_results[config_name]["human_metrics"]:
            metric_stats = kineo_h36m_results[config_name]['human_metrics'][human_metric_name]
            if with_std:
                ablation_study_h36m_rows = ablation_study_h36m_rows.replace(
                    f"{{kineo_{human_metric_name}_h36m_{config_name}}}",
                    f"{metric_stats['mean']:.{n_decimals}f}$\pm${metric_stats['std']:.{n_decimals}f}",
                )
            else:
                ablation_study_h36m_rows = ablation_study_h36m_rows.replace(
                    f"{{kineo_{human_metric_name}_h36m_{config_name}}}",
                    f"{metric_stats['mean']:.{n_decimals}f}",
                )
        for camera_metric_name in kineo_h36m_results[config_name]["camera_metrics"]:
            metric_stats = kineo_h36m_results[config_name]['camera_metrics'][camera_metric_name]
            if with_std:
                ablation_study_h36m_rows = ablation_study_h36m_rows.replace(
                    f"{{kineo_{camera_metric_name}_h36m_{config_name}}}",
                    f"{kineo_h36m_results[config_name]['camera_metrics'][camera_metric_name]['mean']:.{n_decimals}f}$\pm${kineo_h36m_results[config_name]['camera_metrics'][camera_metric_name]['std']:.{n_decimals}f}",
                )
            else:
                ablation_study_h36m_rows = ablation_study_h36m_rows.replace(
                    f"{{kineo_{camera_metric_name}_h36m_{config_name}}}",
                    f"{metric_stats['mean']:.{n_decimals}f}",
                )

    return ablation_study_h36m_rows

def generate_egohumans_ablation_study_camera_params_rows(egohumans_dataset_dir: str, kineo_eval_data_dir: str, n_decimals: int = 2, with_std: bool = True):
    with open(
        os.path.join(os.path.dirname(__file__), "ablation_study_camera_params_egohumans_rows_template.tex"), "r"
    ) as f:
        ablation_study_egohumans_rows_template = f.read()

    kineo_egohumans_all_gt_annotations_path = os.path.join(
        kineo_eval_data_dir, "egohumans_cam_params_gtKps_gtRt_gtK_gtD", "annotations"
    )
    kineo_egohumans_extrinsics_annotations_path = os.path.join(
        kineo_eval_data_dir, "egohumans_cam_params_gtKps_estRt_gtK_gtD", "annotations"
    )
    kineo_egohumans_extrinsics_intrinsics_without_distortions_annotations_path = (
        os.path.join(
            kineo_eval_data_dir,
            "egohumans_cam_params_gtKps_estRt_estK_omitD",
            "annotations",
        )
    )
    kineo_egohumans_extrinsics_intrinsics_with_distortions_annotations_path = (
        os.path.join(
            kineo_eval_data_dir,
            "egohumans_cam_params_gtKps_estRt_estK_estD",
            "annotations",
        )
    )
    kineo_egohumans_extrinsics_intrinsics_with_radial_distortions_annotations_path = (
        os.path.join(
            kineo_eval_data_dir,
            "egohumans_cam_params_gtKps_estRt_estK_estDk1k2",
            "annotations",
        )
    )

    kineo_egohumans_results = {
        "GtRt_GtK_GtD": compute_egohumans_stats(
            egohumans_dataset_dir, kineo_egohumans_all_gt_annotations_path
        ),
        "EstRt_GtK_GtD": compute_egohumans_stats(
            egohumans_dataset_dir, kineo_egohumans_extrinsics_annotations_path
        ),
        "EstRt_EstK_OmitD": compute_egohumans_stats(
            egohumans_dataset_dir,
            kineo_egohumans_extrinsics_intrinsics_without_distortions_annotations_path,
        ),
        "EstRt_EstK_EstD": compute_egohumans_stats(
            egohumans_dataset_dir,
            kineo_egohumans_extrinsics_intrinsics_with_distortions_annotations_path,
        ),
        "EstRt_EstK_EstDk1k2": compute_egohumans_stats(
            egohumans_dataset_dir,
            kineo_egohumans_extrinsics_intrinsics_with_radial_distortions_annotations_path,
        ),
    }

    ablation_study_egohumans_rows = str(ablation_study_egohumans_rows_template)
    for config_name in kineo_egohumans_results:
        for human_metric_name in kineo_egohumans_results[config_name]["human_metrics"]:
            metric_stats = kineo_egohumans_results[config_name]['human_metrics'][human_metric_name]
            if with_std:
                ablation_study_egohumans_rows = ablation_study_egohumans_rows.replace(
                    f"{{kineo_{human_metric_name}_egohumans_{config_name}}}",
                    f"{metric_stats['mean']:.{n_decimals}f}$\pm${metric_stats['std']:.{n_decimals}f}",
                )
            else:
                ablation_study_egohumans_rows = ablation_study_egohumans_rows.replace(
                f"{{kineo_{human_metric_name}_egohumans_{config_name}}}",
                f"{metric_stats['mean']:.{n_decimals}f}",
            )
        for camera_metric_name in kineo_egohumans_results[config_name][
            "camera_metrics"
        ]:
            metric_stats = kineo_egohumans_results[config_name]['camera_metrics'][camera_metric_name]
            if with_std:
                ablation_study_egohumans_rows = ablation_study_egohumans_rows.replace(
                    f"{{kineo_{camera_metric_name}_egohumans_{config_name}}}",
                    f"{metric_stats['mean']:.{n_decimals}f}$\pm${metric_stats['std']:.{n_decimals}f}",
                )
            else:
                ablation_study_egohumans_rows = ablation_study_egohumans_rows.replace(
                f"{{kineo_{camera_metric_name}_egohumans_{config_name}}}",
                f"{metric_stats['mean']:.{n_decimals}f}",
            )

    return ablation_study_egohumans_rows



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate human metrics ablation_study rows for Kineo on EgoHumans and Human3.6M and for HSfM on Human3.6M"
    )
    parser.add_argument(
        "h36m_dataset_dir", type=str, help="Path to Human3.6M dataset directory"
    )
    parser.add_argument(
        "egohumans_dataset_dir", type=str, help="Path to EgoHumans dataset directory"
    )
    parser.add_argument(
        "kineo_eval_data_dir", type=str, help="Path to Kineo evaluation data directory"
    )
    parser.add_argument(
        "--n-decimals", type=int, help="Number of decimals to use in the output", default=3
    )
    parser.add_argument(
        "--with-std", action="store_true", help="Include standard deviation in the output"
    )
    args = parser.parse_args()

    egohumans_ablation_study_rows = generate_egohumans_ablation_study_camera_params_rows(args.egohumans_dataset_dir, args.kineo_eval_data_dir, args.n_decimals, args.with_std)
    h36m_ablation_study_rows = generate_h36m_ablation_study_camera_params_rows(args.h36m_dataset_dir, args.kineo_eval_data_dir, args.n_decimals, args.with_std)

    print(egohumans_ablation_study_rows)
    print(h36m_ablation_study_rows)

