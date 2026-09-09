#!/bin/bash
# Live GPU utilization on the node of a running TrustForge SLURM job.
#
# Usage (second terminal, while the job is running):
#   ./monitor_gpus.sh              # attach to your newest running job
#   ./monitor_gpus.sh 1234567      # attach to a specific JOBID
#   INTERVAL=2 ./monitor_gpus.sh   # refresh every 2s (default 1)
#
# Needs an overlapping step on the same allocation (SLURM --overlap).

set -euo pipefail

INTERVAL="${INTERVAL:-1}"
JOBID="${1:-}"

if [[ -z "${JOBID}" ]]; then
  JOBID="$(
    squeue -hu "${USER}" -t R -o "%i %j %M" \
      | awk '/trustforge/ { print $1; exit }'
  )"

fi

if [[ -z "${JOBID}" ]]; then
  echo "No running SLURM job found for ${USER}."
  echo "Start one first (./run_srun.sh or sbatch run.slurm), then rerun."
  exit 1
fi

NODE="$(squeue -hj "${JOBID}" -o "%N" | head -n 1)"
echo "Monitoring JOBID=${JOBID} on ${NODE} (Ctrl-C to stop)"
echo "Refreshing every ${INTERVAL}s via: watch nvidia-smi"
echo

srun --overlap --jobid="${JOBID}" --pty watch -n "${INTERVAL}" nvidia-smi
