"""Regenerate the three Colab notebooks in notebooks/ (run from anywhere: python tools/make_notebooks.py)."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


def nb(cells, gpu=False):
    meta = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}, "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": meta, "cells": cells}


SETUP_CODE = r'''
# --- Project location on Google Drive -------------------------------------
# Mounts Drive only if it is not mounted yet. NEVER run `rm -rf /content/drive`:
# if Drive is mounted it deletes your whole Google Drive.
import os, subprocess
if not os.path.ismount('/content/drive'):
    from google.colab import drive
    drive.mount('/content/drive')

DRIVE_ROOT = '/content/drive/MyDrive/windsurrogate'      # change if you like
PROJECT    = f'{DRIVE_ROOT}/idaho'                        # one folder per dataset (regions in config)
CODE_ZIP   = f'{DRIVE_ROOT}/windsurrogate_code.zip'       # upload the code zip here once
assert os.path.ismount('/content/drive'), 'Drive is not mounted'
assert os.path.exists(CODE_ZIP), f'missing {CODE_ZIP}: upload the code zip there first'
os.makedirs(PROJECT, exist_ok=True)

# --- Code ---------------------------------------------------------------------
if not os.path.exists('/content/windsurrogate'):
    subprocess.run(['unzip', '-q', '-o', CODE_ZIP, '-d', '/content'], check=True)
%cd /content/windsurrogate
!pip -q install rasterio pyproj pyyaml
'''

FOAM_INSTALL = r'''
%%bash
# OpenFOAM v2412 (openfoam.com) from conda-forge, cached on Drive so later
# sessions only untar it (~1 min instead of ~5 min).
set -e
CACHE=/content/drive/MyDrive/windsurrogate/_cache/foam-env-2412.tar.gz
export MAMBA_ROOT_PREFIX=/content/mamba
if [ ! -x /content/bin/micromamba ]; then
  mkdir -p /content/bin
  curl -sSL -o /content/bin/micromamba https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
  chmod +x /content/bin/micromamba
fi
if [ ! -d /content/mamba/envs/foam ]; then
  if [ -f "$CACHE" ]; then
    mkdir -p /content/mamba/envs && tar -xzf "$CACHE" -C /content/mamba/envs
  else
    /content/bin/micromamba create -y -q -n foam -c conda-forge openfoam=2412
    mkdir -p "$(dirname "$CACHE")" && tar -czf "$CACHE" -C /content/mamba/envs foam
  fi
fi
eval "$(/content/bin/micromamba shell hook -s bash)" && micromamba activate foam
simpleFoam -help | head -2 && echo "OpenFOAM $WM_PROJECT_VERSION ready"
'''

COLAB_CFG = r'''
# Session config layered over config/default.yaml (edit region/geometry there or here).
import os
ncpu = os.cpu_count()
CONCURRENT = max(1, ncpu // 2)     # one serial case per physical core (vCPUs are hyper-threads)
cfg_text = f"""
project_dir: {PROJECT}
solver:
  concurrent_cases: {CONCURRENT}
  procs_per_case: 1
  activate: 'export MAMBA_ROOT_PREFIX=/content/mamba && eval "$(/content/bin/micromamba shell hook -s bash)" && micromamba activate foam'
output:
  work_dir: /content/work
"""
open('/content/colab.yaml', 'w').write(cfg_text)
print(cfg_text)
!nproc && free -g | head -2 && df -h /content | tail -1
'''


gen = nb([
    md("""
# 1 · Generate the OpenFOAM training set (CPU runtime)

Runs the CFD cases listed in `cases.jsonl` and saves one `.npz` per finished case to Drive.

* **Runtime:** CPU, *High-RAM* (more cores). Check *Runtime → View resources* for the actual compute-unit rate.
* **Resumable:** finished cases are skipped. If the session dies, just run all cells again.
* **Parallel sessions:** open the notebook in more sessions of the same account and give each a different `SHARD` (0..`NUM_SHARDS`-1).
* Keep the browser tab open unless your plan has background execution.
"""),
    code(SETUP_CODE),
    code(FOAM_INSTALL),
    code(COLAB_CFG),
    md("""
## Prepare (only once per project)
Downloads USGS 3DEP 1 arc-second tiles for every region in `config/default.yaml` (Cervidae Peak and the Lost River Range), reprojects each to its UTM zone (`dem/<region>.tif`), and samples the cases with a spatial train/val/test split per region.
Run it in **one** session before starting other shards. Re-running is safe: existing files are reused.
"""),
    code("!python -m windsurrogate --config /content/colab.yaml prepare"),
    md("""
## Optional: quick benchmark
Runs 1 case to measure minutes per case on this machine before committing hours.
"""),
    code("# !python -m windsurrogate --config /content/colab.yaml run --limit 1 --concurrent 1"),
    md("""
## Progress
Counts finished cases per split and the status of failed ones. Safe to run at any time, in any session.
"""),
    code(r'''
import glob, json, collections
D = f'{PROJECT}/dataset'
for s in ['train', 'val', 'test']:
    print(f'{s:5s}: {len(glob.glob(f"{D}/{s}/*.npz"))}')
print('finished:', len(glob.glob(f'{D}/*/*.npz')), '| listed:', sum(1 for l in open(f'{PROJECT}/cases.jsonl') if l.strip()))
print('status files:', dict(collections.Counter(json.load(open(f)).get('status') for f in glob.glob(f'{D}/_status/*.json'))))
'''),
    md("""
## Optional: retry failed cases
A case that fails twice is skipped. Cases interrupted by stopping a cell also count as failed.
Run this in **one** session, before starting the shards, to give them another try. It only deletes status files, never results.
"""),
    code(r'''
# !grep -l '"status": "failed"' {PROJECT}/dataset/_status/*.json | xargs -r rm
'''),
    md("""
## Run this session's shard
One shard per server (*New Colab Server*, CPU High-RAM). With N servers use `NUM_SHARDS = N` and `SHARD = 0 .. N-1`, all at the same time.
`NUM_SHARDS` may change between batches, as long as no worker from the previous batch is still running.
"""),
    code(r'''
SHARD, NUM_SHARDS = 0, 2        # e.g. 2 servers -> (0, 2) and (1, 2)
HOURS = 11                      # stop starting new cases after this many hours (Colab sessions end after ~12 h)
!python -m windsurrogate --config /content/colab.yaml run --shard {SHARD} --num-shards {NUM_SHARDS} --hours {HOURS}
'''),
    code("!python -m windsurrogate --config /content/colab.yaml status"),
    md("Quick look at a finished case (QA):"),
    code(r'''
import glob
from IPython.display import Image
f = sorted(glob.glob(f'{PROJECT}/dataset/train/*.npz'))[-1]
!python -m windsurrogate --config /content/colab.yaml plot-sample {f} /content/sample.png
Image('/content/sample.png')
'''),
])

train = nb([
    md("""
# 2 · Train the U-Net surrogate (GPU runtime)

* **Runtime:** GPU (T4 or L4 is enough). Training resumes from `checkpoints/last.pt` if the session dies.
* The dataset is copied from Drive to the local disk first; reading thousands of small files from Drive is slow.
"""),
    code(SETUP_CODE),
    code(r'''
!rsync -a --info=progress2 {PROJECT}/dataset/ /content/dataset/
cfg_text = f"""
project_dir: {PROJECT}
output:
  dataset_dir: /content/dataset
train:
  checkpoint_dir: {PROJECT}/checkpoints
  num_workers: 2
"""
open('/content/train.yaml', 'w').write(cfg_text)
!nvidia-smi --query-gpu=name,memory.total --format=csv
'''),
    code("!python -m windsurrogate --config /content/train.yaml train"),
    code(r'''
import json, pandas as pd
log = [json.loads(l) for l in open(f'{PROJECT}/checkpoints/train_log.jsonl')]
pd.DataFrame(log)[['epoch', 'train_loss', 'loss'] + [k for k in log[-1] if k.startswith('rmse_u') or k.startswith('recirc')]].tail(10)
'''),
    md("""
## Evaluate on the test split
The test blocks were never used for training or for picking the best epoch, so these are the numbers to report.
Writes `metrics_test.json`, `cases_test.csv` and CFD-vs-U-Net figures to `checkpoints/eval_test/`.
"""),
    code(r'''
!python -m windsurrogate --config /content/train.yaml evaluate --checkpoint {PROJECT}/checkpoints/best.pt --split test --figures 8
'''),
    code(r'''
import glob
from IPython.display import Image, display
for f in sorted(glob.glob(f'{PROJECT}/checkpoints/eval_test/figures/*.png'))[:4]:
    display(Image(f, width=900))
'''),
], gpu=True)

pred = nb([
    md("""
# 3 · Predict wind maps over a large DEM

Produces one multi-band GeoTIFF per wind direction (bands named `<field>_<height>m`), ready to be stacked as inputs of the thermal model:
`u_east, v_north, w, tke, speed, speedup, dir_from, deflection, ti, convergence, recirculation, shadow, compression` at each height, plus `terrain_m`.
Fields scale linearly with the reference wind (`--uref` at `flow.Zref`, k with its square), so one run per direction (and z0) is enough.
"""),
    code(SETUP_CODE),
    md("""
Maps that already exist are skipped, so the cell can be re-run after a disconnect. Each map is written under a temporary name and only renamed when complete.
At 50 m, one direction takes about a minute per region on an A100 and roughly 0.5-1 GB per file.
"""),
    code(r'''
import os
CKPT = f'{PROJECT}/checkpoints/best.pt'
OUT  = f'{PROJECT}/predictions'
Z0 = 0.1                                 # or loop over a few roughness values
REGIONS = ['cervidae', 'lost_river']
DIRECTIONS = range(0, 360, 30)           # 12 directions; interpolate between neighbours downstream
for region in REGIONS:
    DEM = f'{PROJECT}/dem/{region}.tif'
    for theta in DIRECTIONS:
        out = f'{OUT}/{region}/wind_{theta:03d}.tif'
        if os.path.exists(out):
            continue
        !python -m windsurrogate predict --checkpoint {CKPT} --dem {DEM} --out {out} --theta {theta} --z0 {Z0}
'''),
    code(r'''
import rasterio, numpy as np, matplotlib.pyplot as plt
with rasterio.open(f'{OUT}/lost_river/wind_270.tif') as ds:
    names = list(ds.descriptions)
    get = lambda n: ds.read(names.index(n) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(get('terrain_m'), cmap='terrain'); ax[0].set_title('terrain')
    im = ax[1].imshow(get('speedup_10m'), cmap='RdBu_r', vmin=0, vmax=2); ax[1].set_title('speed-up 10 m'); plt.colorbar(im, ax=ax[1])
    im = ax[2].imshow(get('w_50m'), cmap='PuOr', vmin=-2, vmax=2); ax[2].set_title('w 50 m (m/s)'); plt.colorbar(im, ax=ax[2])
    plt.show()
'''),
], gpu=True)

OUT.mkdir(parents=True, exist_ok=True)
for name, n in [("01_generate_cfd_dataset.ipynb", gen), ("02_train_unet.ipynb", train), ("03_predict_maps.ipynb", pred)]:
    (OUT / name).write_text(json.dumps(n, indent=1))
    print("wrote", OUT / name)
