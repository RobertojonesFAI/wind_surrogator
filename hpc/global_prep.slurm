#!/bin/bash
# Worldwide dataset preparation on a cluster node with internet access:
#   STEP=descriptors  terrain statistics on a ~10 km grid from Copernicus GLO-90
#   STEP=sample       stratified locations + case queue (cases.jsonl, locations.jsonl)
#   STEP=fetch        terrain + land cover crop per location (FABDEM / GLO-30 + WorldCover)
#   STEP=pilot-fetch  only the crops of the pilot cases (so the pilot can start early)
#
#   mkdir -p logs
#   PROJECT=/path/to/global STEP=descriptors sbatch -p short hpc/global_prep.slurm
#   PROJECT=/path/to/global STEP=sample      sbatch -p short -c 4 -t 01:00:00 hpc/global_prep.slurm
#   PROJECT=/path/to/global STEP=pilot-fetch sbatch -p short hpc/global_prep.slurm
#   PROJECT=/path/to/global STEP=fetch NUM_SHARDS=4 sbatch -p short --array=0-3 hpc/global_prep.slurm
#
# Every step is resumable: submit it again after a time limit or an error.
# Check first that compute nodes reach the internet:
#   srun -p short -N1 -t 00:05:00 curl -sI https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com/tileList.txt
#SBATCH -J windprep
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 48
#SBATCH --mem=120G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/windprep_%A_%a.out

set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to the global project folder}"
STEP="${STEP:?set STEP to descriptors, sample, pilot-fetch or fetch}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
# Network-bound: too many threads overwhelm the node's DNS resolver ("Could not resolve host").
THREADS="${THREADS:-12}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
eval "$("$HOME/bin/micromamba" shell hook -s bash)"
micromamba activate foam
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs "$PROJECT"
WS="python -m windsurrogate --config config/global.yaml --set project_dir=$PROJECT"

echo "node $(hostname)  step $STEP  shard $SHARD/$NUM_SHARDS  threads $THREADS  project $PROJECT"
case "$STEP" in
  descriptors) $WS global-descriptors --threads "$THREADS" ;;
  sample)      $WS global-sample ;;
  pilot-fetch) $WS global-fetch --pilot-only --threads "$THREADS" ;;
  fetch)       $WS global-fetch --shard "$SHARD" --num-shards "$NUM_SHARDS" --threads "$THREADS" ;;
  *) echo "unknown STEP=$STEP"; exit 2 ;;
esac
