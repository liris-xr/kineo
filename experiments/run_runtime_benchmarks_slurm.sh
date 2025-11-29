#!/bin/bash

# Human3.6M
sbatch --job-name="h36m_runtime_nlf_long_video" --output="slurm-h36m_runtime_nlf_long_video.out" experiments/run_h36m_runtime_eval_a100.slurm $SCRATCH/datasets/H3.6M/runtime_eval_videos/long_video/ configs/experiments/runtime/h36m_runtime_nlf_long_video.yaml
sbatch --job-name="h36m_runtime_nlf_medium_video" --output="slurm-h36m_runtime_nlf_medium_video.out" experiments/run_h36m_runtime_eval_a100.slurm $SCRATCH/datasets/H3.6M/runtime_eval_videos/medium_video/ configs/experiments/runtime/h36m_runtime_nlf_medium_video.yaml
sbatch --job-name="h36m_runtime_nlf_short_video" --output="slurm-h36m_runtime_nlf_short_video.out" experiments/run_h36m_runtime_eval_a100.slurm $SCRATCH/datasets/H3.6M/runtime_eval_videos/short_video/ configs/experiments/runtime/h36m_runtime_nlf_short_video.yaml

sbatch --job-name="h36m_runtime_rtmpose_long_video" --output="slurm-h36m_runtime_rtmpose_long_video.out" experiments/run_h36m_runtime_eval_a100.slurm $SCRATCH/datasets/H3.6M/runtime_eval_videos/long_video/ configs/experiments/runtime/h36m_runtime_rtmpose_long_video.yaml
sbatch --job-name="h36m_runtime_rtmpose_medium_video" --output="slurm-h36m_runtime_rtmpose_medium_video.out" experiments/run_h36m_runtime_eval_a100.slurm $SCRATCH/datasets/H3.6M/runtime_eval_videos/medium_video/ configs/experiments/runtime/h36m_runtime_rtmpose_medium_video.yaml
sbatch --job-name="h36m_runtime_rtmpose_short_video" --output="slurm-h36m_runtime_rtmpose_short_video.out" experiments/run_h36m_runtime_eval_a100.slurm $SCRATCH/datasets/H3.6M/runtime_eval_videos/short_video/ configs/experiments/runtime/h36m_runtime_rtmpose_short_video.yaml

sbatch --job-name="h36m_runtime_rtmpose_long_video_b32_half" --output="slurm-h36m_runtime_rtmpose_long_video_b32_half.out" experiments/run_h36m_runtime_eval_a100.slurm $SCRATCH/datasets/H3.6M/runtime_eval_videos/long_video/ configs/experiments/runtime/h36m_runtime_rtmpose_long_video_b32_half.yaml
sbatch --job-name="h36m_runtime_rtmpose_medium_video_b32_half" --output="slurm-h36m_runtime_rtmpose_medium_video_b32_half.out" experiments/run_h36m_runtime_eval_a100.slurm $SCRATCH/datasets/H3.6M/runtime_eval_videos/medium_video/ configs/experiments/runtime/h36m_runtime_rtmpose_medium_video_b32_half.yaml
sbatch --job-name="h36m_runtime_rtmpose_short_video_b32_half" --output="slurm-h36m_runtime_rtmpose_short_video_b32_half.out" experiments/run_h36m_runtime_eval_a100.slurm $SCRATCH/datasets/H3.6M/runtime_eval_videos/short_video/ configs/experiments/runtime/h36m_runtime_rtmpose_short_video_b32_half.yaml
