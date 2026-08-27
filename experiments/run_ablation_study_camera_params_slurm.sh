#!/bin/bash
#
# Which camera parameters are estimated rather than taken from ground truth.
# estD optimizes all five Brown-Conrady coefficients, estDk1k2 only the radial
# k1,k2; the two differ in nothing else and are the distortion-model arm.

source experiments/egohumans_sequences.sh

CONFIGS=configs/experiments/ablation_study/camera_parameters
ARMS=(
    estRt_estK_estD
    estRt_estK_estDk1k2
    estRt_estK_omitD
    estRt_gtK_gtD
    gtRt_gtK_gtD
)

# Human3.6M
for arm in "${ARMS[@]}"; do
    sbatch --job-name="h36m_cam_params_${arm}" --output="slurm-h36m_cam_params_${arm}.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ ${CONFIGS}/h36m_cam_params_${arm}.yaml
done

# EgoHumans
for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    for arm in "${ARMS[@]}"; do
        sbatch --job-name="egohumans_cam_params_${arm}_${seq}" --output="slurm-egohumans_cam_params_${arm}_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ ${CONFIGS}/egohumans_cam_params_${arm}.yaml "${subsequences}"
    done
done
