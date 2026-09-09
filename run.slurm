#!/bin/bash
#SBATCH --job-name=trustforge
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
#   sbatch --export=ALL,EXTRA_ARGS="--cwe cwe89 --limit 20" run.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
echo "Job started on $(hostname) at $(date); job=${SLURM_JOB_ID:-local}"

export PYTHONUNBUFFERED=1
source "${TRUSTFORGE_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate trustforge

# Positional arguments preserve spaces safely; legacy EXTRA_ARGS supports simple words.
read -r -a extra_args <<< "${EXTRA_ARGS:-}"
python -u main.py "${extra_args[@]}" "$@"
echo "Job finished at $(date)"
