#!/bin/bash
#
# Bundle adjustment keypoint sampling: uniform random against farthest point.
# The uniform arm is the EgoHumans benchmark config, so only the farthest point
# arm is submitted here; run run_eval_benchmarks_slurm.sh for the other side.

source experiments/egohumans_sequences.sh

# EgoHumans
for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    sbatch --job-name="egohumans_sampling_fps_${seq}" --output="slurm-egohumans_sampling_fps_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/sampling/egohumans_sampling_fps.yaml "${subsequences}"
done
