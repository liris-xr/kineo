#!/bin/bash
#
# Number of views on EgoHumans: 2, 4 and 8 cameras against the full rig. The
# full-rig arm is the EgoHumans benchmark config, so only the reduced arms are
# submitted here; run run_eval_benchmarks_slurm.sh for the other side.
#
# Camera sets follow HSfM, Table S.1. Each config resolves them per sequence.

source experiments/egohumans_sequences.sh

for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    for n_views in 2 4 8; do
        sbatch --job-name="egohumans_n_views_${n_views}_${seq}" --output="slurm-egohumans_n_views_${n_views}_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/n_views/egohumans_n_views_${n_views}.yaml "${subsequences}"
    done
done
