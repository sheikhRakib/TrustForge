#!/bin/bash
# Interactive / foreground SLURM run via srun.
#
# Usage:
#   ./run_srun.sh
#
# Or call srun directly:
#   srun --partition=partition-a --gres=gpu:1 --cpus-per-task=8 --mem=90G \
#        --time=12:00:00 --qos=train --job-name=trustforge_lab3 \
#        bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate trustforge && python main.py'

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
  --job-name=trustforge_lab3 \
  bash -lc "
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    export HF_HUB_DISABLE_PROGRESS_BARS=0
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate trustforge
    cd '$ROOT'
    echo \"Host: \$(hostname)  date: \$(date)\"
    echo \"Python: \$(python -V)  which: \$(which python)\"
    python -u main.py $*
  "
