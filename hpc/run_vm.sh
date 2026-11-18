#!/bin/bash
# Run the whole CFD dataset on one Linux VM (e.g. Google Cloud), detached from the
# SSH session. Safe to re-run: finished cases are skipped.
#
#   bash hpc/run_vm.sh            # start (prepare on first run, then all cases)
#   tail -f ~/windcfd.log         # watch progress
#   bash hpc/run_vm.sh status     # summary
set -euo pipefail

PROJECT="${PROJECT:-$HOME/runs/idaho}"
WORK="${WORK:-$HOME/work}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
eval "$("$HOME/bin/micromamba" shell hook -s bash)"
micromamba activate foam
export OMP_NUM_THREADS=1
cd "$(dirname "$0")/.."

ARGS=(--config hpc/cluster.yaml --set "project_dir=$PROJECT" --set "output.work_dir=$WORK")
if [ "${1:-}" = "status" ]; then
  python -m windsurrogate "${ARGS[@]}" status
  exit 0
fi

TPC=$(lscpu | awk -F: '/Thread\(s\) per core/ {gsub(/ /,"",$2); print $2}')
CONCURRENT=$(( $(nproc) / ${TPC:-1} ))
[ "$CONCURRENT" -lt 1 ] && CONCURRENT=1
mkdir -p "$PROJECT" "$WORK"

if [ ! -f "$PROJECT/cases.jsonl" ]; then
  python -m windsurrogate "${ARGS[@]}" prepare 2>&1 | tee "$HOME/prepare.log"
fi

nohup python -m windsurrogate "${ARGS[@]}" run --concurrent "$CONCURRENT" > "$HOME/windcfd.log" 2>&1 &
echo "started: $CONCURRENT cases at a time, log in ~/windcfd.log (you can close SSH now)"
