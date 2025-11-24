#!/bin/bash
#SBATCH --job-name=egohumans_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00
#SBATCH --hint=nomultithread
#SBATCH --output=slurm-%x-%j.out

#SBATCH -C v100-32g
#SBATCH -A rch@v100

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load cuda/12.8.0
module load miniforge/24.9.0

DATASET_DIR=$1
CONFIG_FILE=$2

if [ -n "$3" ]; then
    SEQUENCES_FILTER="--sequences-filter $3"
fi

conda run --live-stream -n kineo python experiments/egohumans_eval.py "$DATASET_DIR" --config-file "$CONFIG_FILE" $SEQUENCES_FILTER