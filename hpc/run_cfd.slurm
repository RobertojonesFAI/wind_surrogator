#!/bin/bash
# CFD dataset generation as a Slurm job array: one array task = one shard
# = one whole node running one serial OpenFOAM case per physical core.
#
#   mkdir -p logs && sbatch -p <partition> --array=0-7 hpc/run_cfd.slurm   # 8 nodes, NUM_SHARDS=8
#
# Resumable: resubmit the same command after a time limit, a failure or a preemption;
# finished cases are skipped. NUM_SHARDS must equal the number of array tasks and must
# not change while jobs of the same batch are still running.
#
# Partitions: check per-user node limits and maximum wall time with
#   scontrol show partition | grep -E "PartitionName|MaxNodes|MaxTime"
# On a preemptible partition also pass --requeue, so preempted tasks go back to the queue.
#SBATCH -J windcfd
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 7-00:00:00
#SBATCH -o logs/windcfd_%A_%a.out

set -euo pipefail

# ---- edit these ---------------------------------------------------------------
PROJECT="${PROJECT:-$HOME/windsurrogate_runs/idaho}"   # results (shared storage)
NUM_SHARDS="${NUM_SHARDS:-8}"                            # = number of array tasks
# STOP_HOURS: stop starting new cases after this many hours. Default: the job's
# remaining time minus 2 h, so the cases already running can finish.
# --------------------------------------------------------------------------------

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
eval "$("$HOME/bin/micromamba" shell hook -s bash)"
micromamba activate foam
export OMP_NUM_THREADS=1

if [ -z "${STOP_HOURS:-}" ]; then
  LEFT=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null || true)
  STOP_HOURS=$(python - "$LEFT" <<'EOF'
import sys
s = sys.argv[1].strip()
try:
    days = 0
    if "-" in s:
        days, s = s.split("-", 1)
        days = int(days)
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    hours = 24 * days + parts[0] + parts[1] / 60 + parts[2] / 3600
    print(f"{max(hours - 2.0, 0.5):.2f}")
except Exception:
    print("70")
EOF
)
fi

CODE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$CODE_DIR"
mkdir -p logs
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NCPU="${SLURM_CPUS_ON_NODE:-$(nproc)}"
# Physical cores: one case per core (hyper-threads do not help OpenFOAM much).
TPC=$(lscpu | awk -F: '/Thread\(s\) per core/ {gsub(/ /,"",$2); print $2}')
CONCURRENT=$(( NCPU / ${TPC:-1} ))
# Running cases need ~0.3 GB of disk each: use node-local disk if it has room, else $PROJECT.
LOCAL="${TMPDIR:-/tmp}"
FREE_GB=$(df -BG --output=avail "$LOCAL" | tail -1 | tr -dc '0-9')
if [ "${FREE_GB:-0}" -gt $(( CONCURRENT / 2 + 10 )) ]; then
  WORK="$LOCAL/windsurrogate_${SLURM_JOB_ID:-local}_${SHARD}"
else
  WORK="$PROJECT/_work/${SLURM_JOB_ID:-local}_${SHARD}"
fi
mkdir -p "$WORK"

echo "node $(hostname)  shard $SHARD/$NUM_SHARDS  cores $CONCURRENT  stop after ${STOP_HOURS} h  work $WORK  project $PROJECT"
free -g | head -2
df -h "$WORK" | tail -1

python -m windsurrogate --config hpc/cluster.yaml \
  --set "project_dir=$PROJECT" --set "output.work_dir=$WORK" \
  run --shard "$SHARD" --num-shards "$NUM_SHARDS" --concurrent "$CONCURRENT" --hours "$STOP_HOURS"

rm -rf "$WORK"
