#!/bin/bash
#
# Keypoint sampling: uniform random against farthest point. The sampler is set
# on both sampling stages, so the arms differ only in how correspondences are
# spread, never in how many are drawn or which ones are eligible.

source experiments/egohumans_sequences.sh

for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    for sampler in uniform fps; do
        sbatch --job-name="egohumans_sampling_${sampler}_${seq}" --output="slurm-egohumans_sampling_${sampler}_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/sampling/egohumans_sampling_${sampler}.yaml "${subsequences}"
    done
done
