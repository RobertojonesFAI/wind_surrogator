#!/bin/bash
# Frozen copy of the finished samples, one uncompressed .tar per split (the .npz are already
# compressed). Training reads a snapshot, so a resumed run sees the same files even while
# the CFD jobs keep adding cases, and the same .tar files are the Google Drive backup and
# the Colab input.
#
#   PROJECT=/path/to/global TAG=b1 sbatch -p short hpc/snapshot_dataset.sh
#   PROJECT=/path/to/global TAG=b1 bash hpc/snapshot_dataset.sh          # or directly
#
# Writes $PROJECT/snapshots/$TAG/{train,val,test}.tar, the file lists and MANIFEST.txt.
#SBATCH -J windsnap
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 04:00:00
#SBATCH -o logs/windsnap_%j.out

set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to the project folder (the one that contains dataset/)}"
TAG="${TAG:-$(date +%Y%m%d)}"
SRC="$PROJECT/dataset"
OUT="$PROJECT/snapshots/$TAG"

if [ -e "$OUT/MANIFEST.txt" ]; then
  echo "snapshot $OUT already exists; choose another TAG" >&2
  exit 1
fi
mkdir -p "$OUT"

total=0
for split in train val test; do
  [ -d "$SRC/$split" ] || continue
  # Finished samples only: hidden ".*.tmp.npz" files are writes in progress.
  (cd "$SRC" && find "$split" -maxdepth 1 -type f -name '[cg]*.npz' | sort) > "$OUT/$split.list"
  n=$(wc -l < "$OUT/$split.list")
  echo "$split: $n samples"
  tar -C "$SRC" -cf "$OUT/$split.tar.part" -T "$OUT/$split.list"
  mv "$OUT/$split.tar.part" "$OUT/$split.tar"
  total=$((total + n))
done

{
  echo "snapshot: $TAG"
  echo "created: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source: $SRC"
  echo "code: $(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  for split in train val test; do
    [ -f "$OUT/$split.list" ] && echo "$split: $(wc -l < "$OUT/$split.list") samples, $(du -h "$OUT/$split.tar" | cut -f1)"
  done
  echo "total: $total samples"
} > "$OUT/MANIFEST.txt"
cat "$OUT/MANIFEST.txt"
