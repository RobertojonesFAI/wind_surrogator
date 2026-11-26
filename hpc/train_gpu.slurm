#!/bin/bash
# Train the U-Net on one GPU from a dataset snapshot (hpc/snapshot_dataset.sh).
#
#   mkdir -p logs
#   PROJECT=/path/to/global SNAPSHOT=b1 EXTRA_CONFIG=config/global.yaml \
#     sbatch -p gpu -t 2-00:00:00 hpc/train_gpu.slurm
#
# Training is single-GPU, so one node is all it uses. The snapshot is unpacked on the node's
# local disk (thousands of small files read every epoch). Checkpoints go to
# $PROJECT/checkpoints/$RUN, one per epoch: after a time limit, submit the same command
# again and it resumes from last.pt (or chain it: sbatch --dependency=afterany:<jobid> ...).
# Never run two jobs with the same RUN at the same time.
#
# Settings (environment variables):
#   PROJECT       project folder (contains snapshots/)
#   SNAPSHOT      snapshot tag, e.g. b1
#   RUN           checkpoint folder name (default: the snapshot tag)
#   EXTRA_CONFIG  YAML merged after hpc/cluster.yaml, e.g. config/global.yaml
#   EPOCHS        default 25 (config default 120 was set for ~1,200 cases)
#   ENV_NAME      micromamba env with PyTorch (default ml, see hpc/setup_ml_env.sh)
#   EXTRA_SET     more overrides, e.g. "--set train.batch_size=16"
#SBATCH -J windtrain
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH -o logs/windtrain_%j.out

set -euo pipefail

PROJECT="${PROJECT:?set PROJECT}"
SNAPSHOT="${SNAPSHOT:?set SNAPSHOT (tag of hpc/snapshot_dataset.sh)}"
RUN="${RUN:-$SNAPSHOT}"
EPOCHS="${EPOCHS:-25}"
ENV_NAME="${ENV_NAME:-ml}"
SNAP="$PROJECT/snapshots/$SNAPSHOT"
CKDIR="$PROJECT/checkpoints/$RUN"

[ -f "$SNAP/MANIFEST.txt" ] || { echo "no finished snapshot at $SNAP" >&2; exit 1; }

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
eval "$("$HOME/bin/micromamba" shell hook -s bash)"
micromamba activate "$ENV_NAME"
export PYTHONUNBUFFERED=1

CODE_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$CODE_DIR"
mkdir -p logs "$CKDIR"
cp "$SNAP/MANIFEST.txt" "$CKDIR/snapshot_MANIFEST.txt"

# Unpack on node-local disk if it has room (snapshot size + 5 GB), else fail early.
LOCAL="${TMPDIR:-/tmp}/windtrain_${SLURM_JOB_ID:-local}"
NEED_GB=$(( $(du -sBG "$SNAP" | cut -f1 | tr -dc '0-9') + 5 ))
FREE_GB=$(df -BG --output=avail "$(dirname "$LOCAL")" | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt "$NEED_GB" ]; then
  echo "local disk has ${FREE_GB} GB free, the snapshot needs ${NEED_GB} GB" >&2
  exit 1
fi
mkdir -p "$LOCAL/dataset"
trap 'rm -rf "$LOCAL"' EXIT
t0=$(date +%s)
for t in "$SNAP"/*.tar; do tar -C "$LOCAL/dataset" -xf "$t"; done
echo "snapshot $SNAPSHOT unpacked to $LOCAL/dataset in $(( $(date +%s) - t0 )) s"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no GPU visible'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

NCPU="${SLURM_CPUS_PER_TASK:-8}"
EXTRA_ARGS=()
if [ -n "${EXTRA_CONFIG:-}" ]; then EXTRA_ARGS+=(--config "$EXTRA_CONFIG"); fi

# shellcheck disable=SC2086  # EXTRA_SET is split on purpose
python -m windsurrogate --config hpc/cluster.yaml ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  --set project_dir="$PROJECT" \
  --set output.dataset_dir="$LOCAL/dataset" \
  --set train.checkpoint_dir="$CKDIR" \
  --set train.epochs="$EPOCHS" \
  --set train.num_workers=$(( NCPU > 2 ? NCPU - 2 : 1 )) \
  ${EXTRA_SET:-} \
  train
