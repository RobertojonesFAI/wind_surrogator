#!/bin/bash
# One-time setup on a Linux VM or cluster login node. Installs micromamba and a conda env "foam"
# with OpenFOAM v2412 + the Python packages the pipeline needs.
#
#   bash hpc/setup_env.sh                 # env in $HOME/mamba
#   MAMBA_ROOT_PREFIX=/path/to/scratch/mamba bash hpc/setup_env.sh   # if $HOME quota is small
set -euo pipefail

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"
BIN="$HOME/bin"
mkdir -p "$BIN" "$MAMBA_ROOT_PREFIX"

if [ ! -x "$BIN/micromamba" ]; then
  curl -sSL -o "$BIN/micromamba" \
    https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
  chmod +x "$BIN/micromamba"
fi

if [ ! -d "$MAMBA_ROOT_PREFIX/envs/foam" ]; then
  "$BIN/micromamba" create -y -n foam -c conda-forge \
    openfoam=2412 python=3.11 numpy scipy rasterio pyproj pyyaml matplotlib
fi

eval "$("$BIN/micromamba" shell hook -s bash)"
micromamba activate foam
simpleFoam -help | head -2
python -c "import numpy, scipy, rasterio, yaml; print('python deps OK')"
echo "OpenFOAM $WM_PROJECT_VERSION ready in $MAMBA_ROOT_PREFIX/envs/foam"
echo "Remember: export MAMBA_ROOT_PREFIX=$MAMBA_ROOT_PREFIX  (the job scripts need the same value)"
