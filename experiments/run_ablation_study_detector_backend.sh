#!/bin/bash
#
# Keypoint detector backend: DWPose on EgoHumans, RTMPose on Human3.6M, and
# ground truth keypoints on both as the upper bound. The NLF arm is the
# benchmark config on each dataset, so it is not submitted here; run
# run_eval_benchmarks_slurm.sh for that side.

source experiments/egohumans_sequences.sh

# Human3.6M
sbatch --job-name="h36m_detector_rtmpose" --output="slurm-h36m_detector_rtmpose.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/detector_backend/h36m_detector_rtmpose.yaml
sbatch --job-name="h36m_detector_gtKps" --output="slurm-h36m_detector_gtKps.out" experiments/run_h36m_eval_a100.slurm $SCRATCH/datasets/H3.6M/raw/ configs/experiments/ablation_study/detector_backend/h36m_detector_gtKps.yaml

# EgoHumans
for seq in "${!EGOHUMANS_SEQUENCES[@]}"; do
    subsequences="${EGOHUMANS_SEQUENCES[${seq}]}"
    sbatch --job-name="egohumans_detector_dwpose_${seq}" --output="slurm-egohumans_detector_dwpose_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/detector_backend/egohumans_detector_dwpose.yaml "${subsequences}"
    sbatch --job-name="egohumans_detector_gtKps_${seq}" --output="slurm-egohumans_detector_gtKps_${seq}.out" experiments/run_egohumans_eval_a100.slurm $SCRATCH/datasets/EgoHumans/ configs/experiments/ablation_study/detector_backend/egohumans_detector_gtKps.yaml "${subsequences}"
done
