#!/bin/bash
#SBATCH --job-name=h36m_runtime_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out

#SBATCH -C a100
#SBATCH -A rch@a100

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load arch/a100
module load cuda/12.8.0
module load miniforge/24.9.0

VIDEO_DIR=$1
CONFIG_FILE=$2

conda run --live-stream -n kineo python experiments/h36m_runtime_eval.py "$VIDEO_DIR" --config-file "$CONFIG_FILE"