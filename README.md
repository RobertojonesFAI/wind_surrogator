# wind_surrogator

Complete CFD wind surrogate: runs large-scale terrain wind simulations in minutes instead of the hours or days a CFD solver would need.

OpenFOAM is run over thousands of real terrain patches cut from a DEM. A U-Net is trained to reproduce the OpenFOAM wind field, and is then applied tile by tile over DEMs of any size. The outputs (speed-up and compression, flow deflection, recirculation and rotor zones, wind shadow, vertical velocity, turbulence, convergence) are written as georeferenced multi-band GeoTIFFs. They are meant to be used as input layers of a thermal-prediction model.

```
DEM (USGS 3DEP) ─► master DEM in UTM ─► cases (centre, wind direction, z0, spatial split)
   └─► per case: terrain rotated so the wind always blows towards +x, buffer, blend to flat
        └─► terrain-following mesh (one shared topology) ─► simpleFoam (steady RANS k-ε, neutral ABL)
             └─► fields sampled at 10/50/100/200/400 m above ground ─► one .npz per case
                  └─► U-Net (+ transformer bottleneck) ─► tiled inference over large DEMs ─► GeoTIFF
```

## Status

**v1** (neutral atmosphere, 50 m resolution) is trained on 1,217 OpenFOAM cases over two regions of Idaho: Cervidae Peak / Boise Front, and the Lost River Range. The full dataset of 2,000 cases is being completed.

Validation split (terrain blocks never used for training):

| Height above ground | RMSE of u / U_in | Recirculation IoU |
|---|---|---|
| 10 m | 0.092 | 0.64 |
| 50 m | 0.084 | 0.56 |
| 100 m | 0.081 | 0.53 |
| 200 m | 0.074 | 0.46 |
| 400 m | 0.061 | n/a (almost no recirculation) |

- **Skill vs. "flat terrain": 0.93.** The surrogate's mean squared error is 7 % of the error of assuming undisturbed flow.
- **Training:** 120 epochs, about 8 s per epoch on one A100.
- **Inference:** about 1 minute per wind direction for a ~133 × 138 km region at 50 m (A100).
- **Test-split numbers:** produced with `windsurrogate evaluate` (see below).

**v2** (stratified flow: foehn, downslope windstorms, hydraulic jumps, mountain waves, gap winds) is at the design stage; see [docs/v2-stratified-design.md](docs/v2-stratified-design.md).

## Physics and numerics (v1)

| | |
|---|---|
| Solver | OpenFOAM v2412 (openfoam.com, conda-forge build), `simpleFoam`, steady, incompressible, SIMPLEC |
| Turbulence | k-ε with ABL constants (κ = 0.40, σε = 1.11; Hargreaves & Wright 2007). RNG k-ε and k-ω SST selectable |
| Inflow | `atmBoundaryLayerInlet{Velocity,K,Epsilon}`: Richards & Hoxey log profile, Uref = 10 m/s at 100 m |
| Ground | `atmNutkWallFunction` / `atmEpsilonWallFunction` with roughness z0 (0.02–0.5 m, log-uniform per case) |
| Top / sides / outlet | constant shear stress / symmetry / fixed pressure with `inletOutlet` |
| Mesh | terrain-following hexahedra (sigma coordinate), 50 m cells in the core, ×1.15 growth to 300 m, 30 levels, 5 m first cell, ~1.0 M cells. One shared topology: per case only `points` change |
| Domain | 6.4 km labelled core; real terrain over the network input window (4.8 km upwind, 1.6 km sideways and downwind); then 1.5 km cosine blend to flat and 0.5 km flat |
| Convergence | 98 probes in the core (10 and 100 m AGL). A case stops when no probe changes by more than 0.2 % of Uref over 200 iterations. If the flow keeps oscillating (steep lee slopes), the label is the average of iterations 1000–1500 |

**Normalisation.** In neutral RANS over rough ground the flow scales with Uref, so a single speed is simulated. The network predicts u/U_in(z), v/U_in(z), w/U_in(z) and k/k_in, where U_in is the inflow profile at the same height. Wind direction is removed by rotating the terrain, so the network always sees "wind from −x".

**Network.** U-Net (base 48 channels, 4 levels) with a 2-layer transformer at the bottleneck for global context, about 16.7 M parameters.
- Input: 4 channels × 192 × 256 px at 50 m (relative elevation, dh/dx, dh/dy, log10 z0).
- Output: 20 channels × 128 × 128 px (u, v, w, k at 5 heights).
- Augmentation: mirror across the wind axis.

## Quick start

### Google Colab (no local install)

1. Zip this repository as `windsurrogate_code.zip` and upload it to `MyDrive/windsurrogate/`.
2. **`notebooks/01_generate_cfd_dataset.ipynb`** (CPU, High-RAM). Installs OpenFOAM (cached on Drive), downloads the DEMs, samples the cases and runs one shard of them. Open one session per shard to run in parallel.
3. **`notebooks/02_train_unet.ipynb`** (GPU). Copies the dataset to local disk, trains (resumable), then evaluates on the test split.
4. **`notebooks/03_predict_maps.ipynb`** (GPU). One GeoTIFF per wind direction over each region's DEM.

Practical notes, including what to do when Drive disconnects, are in [docs/colab-guide.md](docs/colab-guide.md).

### Command line

```bash
micromamba create -f environment.yml && micromamba activate foam   # OpenFOAM + Python deps
pip install -e ".[ml]"                                              # + PyTorch for train / evaluate / predict

windsurrogate --config my.yaml prepare                       # DEMs + cases.jsonl
windsurrogate --config my.yaml run --shard 0 --num-shards 2 --hours 11
windsurrogate --config my.yaml status
windsurrogate --config my.yaml train
windsurrogate --config my.yaml evaluate --checkpoint runs/idaho/checkpoints/best.pt --split test
windsurrogate predict --checkpoint best.pt --dem dem/lost_river.tif --out wind_270.tif --theta 270 --z0 0.1
```

Any setting can be overridden with `--set`, for example `--set geometry.dx=60 --set sampling.n_cases=1500`.

### Clusters and VMs

- `hpc/setup_env.sh` installs the environment.
- `hpc/run_cfd.slurm` runs one shard per node as a Slurm job array, one case per physical core.
- `hpc/run_vm.sh` does the same on a single cloud VM.

Details in [docs/hpc.md](docs/hpc.md).

## Outputs

`predict` writes one GeoTIFF per wind direction and z0. Bands are named `<field>_<height>m`:

| Field | Meaning |
|---|---|
| `u_east`, `v_north`, `w` | wind components (m/s) |
| `tke` | turbulent kinetic energy (m²/s²) |
| `speed`, `speedup` | horizontal speed and speed / inflow speed at that height (> 1 compression, < 0.6 shadow) |
| `dir_from`, `deflection` | wind direction and turning relative to the inflow (degrees, + = to the left) |
| `ti` | turbulence intensity |
| `convergence` | −∇ₕ·U (1/s) |
| `recirculation` | along-wind component < 0: rotor or separated flow |
| `shadow`, `compression` | masks from the speed-up thresholds |
| `terrain_m` | terrain elevation |

Everything scales with the reference wind (U ∝ Uref, k ∝ Uref²). In practice it is enough to precompute 12–16 directions and interpolate between the two nearest ones.

`evaluate` writes, to `<checkpoint dir>/eval_<split>/`:
- `metrics_<split>.json`: aggregate metrics, also broken down by region and by terrain relief;
- `cases_<split>.csv`: one row per case;
- `figures/`: CFD vs. surrogate side by side (speed-up, w, recirculation), spread across the best, median and worst cases.

## Repository layout

```
config/default.yaml          all settings (regions, DEM, geometry, flow, solver, sampling, training)
windsurrogate/dem.py         3DEP / Copernicus download, mosaic, reprojection, synthetic test DEM
windsurrogate/geometry.py    fixed case geometry (windows, nodes, grading)
windsurrogate/terrain.py     rotated terrain with anti-aliasing, blend to flat, sigma mesh
windsurrogate/sampling.py    case sampling and spatial block split (no train/test leakage)
windsurrogate/foamcase.py    OpenFOAM case writer and shared topology
windsurrogate/runner.py      resumable, shardable worker; probe-based convergence stop
windsurrogate/postprocess.py sampling at fixed heights above ground -> .npz
windsurrogate/derived.py     derived fields for the thermal model
windsurrogate/ml/            dataset, U-Net, training, evaluation, tiled inference -> GeoTIFF
notebooks/                   Colab notebooks (generated by tools/make_notebooks.py)
hpc/                         environment setup, Slurm array job, single-VM runner
tests/                       unit tests (pytest) and a manual physics check (needs OpenFOAM)
docs/                        design notes, operations guides, v2 design
```

## Tests

```bash
pytest tests/                                   # inference geometry oracle + evaluation smoke test
python tests/physics_check.py my.yaml           # flat-terrain and Gaussian-hill checks (needs OpenFOAM, a few minutes)
```

Physics checks done so far:
- **Flat terrain:** the inflow log profile is kept to the outlet within 2.8 % below 500 m.
- **Gaussian hill** (H = 300 m, σ = 600 m): crest speed-up of 1.62 at 10 m, decreasing to 1.14 at 400 m; lee wake of 0.45·U_in at 10 m.
- **Inference oracle:** rotation, tiling, blending and vector rotation are consistent for any direction, and chunked prediction equals single-pass prediction.

## Cost

- One case takes 20–80 minutes on one core (300–1,500 iterations), about 1.0–1.5 cases per core-hour. It uses ~1.3 GB of RAM and writes ~1.4 MB.
- A Colab CPU High-RAM session (8 vCPU = 4 physical cores) runs 4 cases at a time, about 5–7 cases per hour, at about 0.5 compute units per hour.
- Training is cheap; most of the budget goes to the CFD.

## Known limitations (v1)

- **Neutral, steady atmosphere.** v1 captures the mechanical effects: crest compression, channelling, deflection, shadow, lee separation. It does not capture thermally driven slope and valley winds, or anything that needs stratification: mountain waves and wave rotors, foehn, downslope windstorms, hydraulic jumps, blocking. That is the goal of v2.
- k-ε tends to underestimate recirculation length on steep slopes; k-ω SST can be compared on a subset.
- z0 is uniform per case. The network input already has a z0 channel, so spatial z0 from land cover only needs the CFD side.
- Trained on Idaho terrain only. Other mountain ranges and DEM types (surface models with forest canopy) need their own cases before the surrogate is used there.

## Documentation

- [docs/design-v1.md](docs/design-v1.md): v1 decisions, validation and results
- [docs/colab-guide.md](docs/colab-guide.md): running on Google Colab
- [docs/hpc.md](docs/hpc.md): Slurm clusters and cloud VMs
- [docs/v2-stratified-design.md](docs/v2-stratified-design.md): v2 design (stratified flow)
