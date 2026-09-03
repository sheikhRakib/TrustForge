#!/bin/bash
#SBATCH --job-name=trustforge_lab3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=partition-a
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --qos=train
#
# Submit:
#   sbatch run.slurm
#
# Optional args:
#   sbatch --export=ALL,EXTRA_ARGS="--some-flag" run.slurm

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "EXTRA_ARGS=${EXTRA_ARGS:-}"

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate trustforge

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

echo "Python: $(python -V)"
echo "Which python: $(which python)"
echo "Working dir: $(pwd)"

python -u main.py ${EXTRA_ARGS:-}

echo "Job finished at $(date)"
