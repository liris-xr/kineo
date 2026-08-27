#!/bin/bash

source experiments/egohumans_sequences.sh

# Human3.6M
sbatch --job-name="h36m_disabled_confidence_rtmpose_estRt_estK_estD" --output="slurm-h36m_disabled_confidence_rtmpose_estRt_estK_estD.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/confidence/h36m_disabled_confidence_rtmpose_estRt_estK_estD.yaml
sbatch --job-name="h36m_enabled_confidence_rtmpose_estRt_estK_estD" --output="slurm-h36m_enabled_confidence_rtmpose_estRt_estK_estD.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/confidence/h36m_enabled_confidence_rtmpose_estRt_estK_estD.yaml

# EgoHumans
for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    sbatch --job-name="egohumans_disabled_confidence_dwpose_estRt_estK_estD_${seq}" --output="slurm-egohumans_disabled_confidence_dwpose_estRt_estK_estD_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/confidence/egohumans_disabled_confidence_dwpose_estRt_estK_estD.yaml "${subsequences}"
    sbatch --job-name="egohumans_enabled_confidence_dwpose_estRt_estK_estD_${seq}" --output="slurm-egohumans_enabled_confidence_dwpose_estRt_estK_estD_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/confidence/egohumans_enabled_confidence_dwpose_estRt_estK_estD.yaml "${subsequences}"
done