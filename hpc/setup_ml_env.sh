#!/bin/bash
# One-time setup of the training env "ml" (PyTorch with CUDA + this package), next to the
# OpenFOAM env "foam". Run it from the repository on a compute node with internet access
# (on Borah: `dev-session`), not on a login node: when the login node has a newer glibc than
# the compute nodes, pip and micromamba pick builds that do not run on the compute nodes.
#
#   bash hpc/setup_ml_env.sh
#   MAMBA_ROOT_PREFIX=/path/to/scratch/mamba bash hpc/setup_ml_env.sh   # if $HOME quota is small
#
# The PyPI torch wheels bundle the CUDA runtime; the GPU node only needs a recent enough
# NVIDIA driver: "CUDA Version" in nvidia-smi on a GPU node must be >= torch.version.cuda.
# (Borah, 2026-11: driver 550 = CUDA 12.4; from a compute node pip picks torch 2.6.0+cu124.)
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
