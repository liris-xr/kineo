#!/bin/bash
#
# The benchmark tables: Kineo with progressively fewer ground-truth camera
# parameters, all with the NLF detector and the SMPL global scale. estDk1k2 is
# the fully calibration-free headline configuration; the other arms hand it
# ground truth for the parameters named in the arm.

source experiments/egohumans_sequences.sh

CONFIGS=configs/experiments/benchmarks
ARMS=(
    estRt_estK_estDk1k2
    estRt_estK_omitD
    estRt_gtK_gtD
    gtRt_gtK_gtD
)

# Human3.6M
for arm in "${ARMS[@]}"; do
    sbatch --job-name="h36m_benchmark_nlf_${arm}" --output="slurm-h36m_benchmark_nlf_${arm}.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ ${CONFIGS}/h36m_benchmark_nlf_${arm}.yaml
done

# EgoHumans
for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    for arm in "${ARMS[@]}"; do
        sbatch --job-name="egohumans_benchmark_nlf_${arm}_${seq}" --output="slurm-egohumans_benchmark_nlf_${arm}_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ ${CONFIGS}/egohumans_benchmark_nlf_${arm}.yaml "${subsequences}"
    done
done
