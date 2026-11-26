#!/bin/bash
# One-time setup of the training env "ml" (PyTorch with CUDA + this package), next to the
# OpenFOAM env "foam". Run it on the login node (it needs internet), from the repository.
#
#   bash hpc/setup_ml_env.sh
#   MAMBA_ROOT_PREFIX=/path/to/scratch/mamba bash hpc/setup_ml_env.sh   # if $HOME quota is small
#
# The PyPI torch wheels bundle the CUDA runtime; the GPU node only needs a recent NVIDIA
# driver (check with nvidia-smi on the node: "CUDA Version" must be >= torch.version.cuda).
set -euo pipefail

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
BIN="$HOME/bin"
if [ ! -x "$BIN/micromamba" ]; then
  echo "micromamba not found in $BIN: run hpc/setup_env.sh first" >&2
  exit 1
fi
eval "$("$BIN/micromamba" shell hook -s bash)"

if [ ! -d "$MAMBA_ROOT_PREFIX/envs/ml" ]; then
  micromamba create -y -n ml -c conda-forge python=3.11 numpy scipy rasterio pyproj pyyaml matplotlib pip
fi
micromamba activate ml
# ~3 GB of wheels: skip pip's cache so it does not fill $HOME.
pip install --no-cache-dir "torch>=2.1"
pip install --no-cache-dir -e ".[ml]"
python -c "import torch, windsurrogate; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"
echo "env ml ready in $MAMBA_ROOT_PREFIX/envs/ml"
