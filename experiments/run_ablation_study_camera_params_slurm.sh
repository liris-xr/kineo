#!/bin/bash

source experiments/egohumans_sequences.sh

# Human3.6M
sbatch --job-name="h36m_cam_params_estRt_estK_estD" --output="slurm-h36m_cam_params_estRt_estK_estD.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/camera_parameters/h36m_cam_params_estRt_estK_estD.yaml
sbatch --job-name="h36m_cam_params_estRt_estK_omitD" --output="slurm-h36m_cam_params_estRt_estK_omitD.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/camera_parameters/h36m_cam_params_estRt_estK_omitD.yaml
sbatch --job-name="h36m_cam_params_estRt_gtK_gtD" --output="slurm-h36m_cam_params_estRt_gtK_gtD.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/camera_parameters/h36m_cam_params_estRt_gtK_gtD.yaml
sbatch --job-name="h36m_cam_params_gtRt_gtK_gtD" --output="slurm-h36m_cam_params_gtRt_gtK_gtD.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/camera_parameters/h36m_cam_params_gtRt_gtK_gtD.yaml

# EgoHumans
for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    sbatch --job-name="egohumans_cam_params_estRt_estK_estD" --output="slurm-egohumans_cam_params_estRt_estK_estD.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/camera_parameters/egohumans_cam_params_estRt_estK_estD.yaml "${subsequences}"
    sbatch --job-name="egohumans_cam_params_estRt_estK_omitD" --output="slurm-egohumans_cam_params_estRt_estK_omitD.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/camera_parameters/egohumans_cam_params_estRt_estK_omitD.yaml "${subsequences}"
    sbatch --job-name="egohumans_cam_params_estRt_gtK_gtD" --output="slurm-egohumans_cam_params_estRt_gtK_gtD.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/camera_parameters/egohumans_cam_params_estRt_gtK_gtD.yaml "${subsequences}"
    sbatch --job-name="egohumans_cam_params_gtRt_gtK_gtD" --output="slurm-egohumans_cam_params_gtRt_gtK_gtD.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/camera_parameters/egohumans_cam_params_gtRt_gtK_gtD.yaml "${subsequences}"
done