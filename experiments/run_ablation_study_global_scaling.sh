#!/bin/bash

source experiments/egohumans_sequences.sh

# Human3.6M
sbatch --job-name="h36m_global_scaling_moge" --output="slurm-h36m_global_scaling_moge.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/global_scaling/h36m_global_scaling_moge.yaml
sbatch --job-name="h36m_global_scaling_smpl" --output="slurm-h36m_global_scaling_smpl.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/global_scaling/h36m_global_scaling_smpl.yaml

# EgoHumans
for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    sbatch --job-name="egohumans_global_scaling_moge_${seq}" --output="slurm-egohumans_global_scaling_moge_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/global_scaling/egohumans_global_scaling_moge.yaml "${subsequences}"
    sbatch --job-name="egohumans_global_scaling_smpl_${seq}" --output="slurm-egohumans_global_scaling_smpl_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/global_scaling/egohumans_global_scaling_smpl.yaml "${subsequences}"
done