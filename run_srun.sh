#!/bin/bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_ROOT"
mkdir -p logs
srun --pty --partition=partition-a --gres=gpu:1 --cpus-per-task=8 \
  --mem=32G --time=12:00:00 --qos=train --job-name=trustforge \
  bash -c '
set -euo pipefail
export PYTHONUNBUFFERED=1
source "${TRUSTFORGE_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate trustforge
python -u main.py "$@"
' bash "$@"
