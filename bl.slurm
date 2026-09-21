#!/bin/bash
#SBATCH --job-name=trustforge
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=partition-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --qos=train

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
echo "Job started on $(hostname) at $(date); job=${SLURM_JOB_ID:-local}"

source "${TRUSTFORGE_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate trustforge

python baseline.py
echo "Job finished at $(date)"
