#!/bin/bash

source experiments/egohumans_sequences.sh

# Human3.6M
sbatch --job-name="h36m_benchmark_nlf_estRt_estK_estDk1k2" --output="slurm-h36m_benchmark_nlf_estRt_estK_estDk1k2.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/benchmarks/h36m_benchmark_nlf_estRt_estK_estDk1k2.yaml

# EgoHumans
for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    sbatch --job-name="egohumans_benchmark_nlf_estRt_estK_estDk1k2_${seq}" --output="slurm-egohumans_benchmark_nlf_estRt_estK_estDk1k2_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/benchmarks/egohumans_benchmark_nlf_estRt_estK_estDk1k2.yaml "${subsequences}"
done
