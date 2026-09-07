#!/bin/bash
# Interactive SLURM run via srun.
#
# Usage:
#   ./run_srun.sh
#   ./run_srun.sh --cwe cwe89 --limit 20

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs

srun \
  --pty \
  --partition=partition-a \
  --gres=gpu:2 \
  --cpus-per-task=16 \
  --mem=90G \
  --time=12:00:00 \
  --qos=train \
  --job-name=trustforge \
  bash -lc "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate trustforge
    cd '$ROOT'
    echo \"Host: \$(hostname)  date: \$(date)\"
    echo \"Python: \$(python -V)  which: \$(which python)\"
    python -u main.py $*
  "
