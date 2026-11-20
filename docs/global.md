# Worldwide dataset

The regional dataset covers two areas of Idaho. The worldwide dataset samples terrain of
every continent so the surrogate generalises to places it never saw, and replaces the
uniform roughness with a roughness map from land cover. Physics, mesh and network are
the same as in v1 (neutral atmosphere, 50 m).

```
GLO-90 (10 km descriptors) ─► stratified locations + case queue (pilot first)
   └─► per location: FABDEM (or GLO-30) elevation + WorldCover land cover, local TM crop
        └─► per case: rotated terrain, z0 map ─► OpenFOAM (z0 per ground face) ─► .npz
```

## Data sources

| Source | Use | Access | License |
|---|---|---|---|
| Copernicus GLO-90 | Terrain descriptors for sampling | COG on AWS (`copernicus-dem-90m`), no account | Copernicus DEM licence |
| FABDEM V1-2 | Case elevation: Copernicus GLO-30 with forests and buildings removed (bare earth, like 3DEP) | 10° zips at data.bris.ac.uk; single tiles extracted with HTTP range requests | CC BY-NC-SA 4.0, non-commercial only |
| Copernicus GLO-30 | Fallback for tiles FABDEM cannot deliver (it is a surface model, with trees) | COG on AWS (`copernicus-dem-30m`) | Copernicus DEM licence |
| ESA WorldCover 2021 (v200) | Land cover → roughness z0 | COG on AWS (`esa-worldcover`) | CC BY 4.0 |

FABDEM is free for non-commercial use only; a commercial product needs a licence from
Fathom. Each crop records which elevation source it used (`dem_source`).

## Sampling

`global-descriptors` reads every GLO-90 tile at ~360 m (COG overviews) and computes, on
~11 km cells, relief (P98 − P2), standard deviation, mean and 90th-percentile slope, the
valid fraction and a sea-like fraction.

`global-sample` builds the case queue (`config/default.yaml`, section `global.sampling`):

| Group | Share | What |
|---|---|---|
| global | 70 % | Usable cells (≤ 50 % sea) drawn per relief class with a fixed share per class; cells in crowded 10° boxes are down-weighted so no single range dominates |
| hard | 10 % | Top 3 % of cells by slope or relief |
| site | 20 % | Discs around the sites in `config/sites.yaml` (flying sites, terrain-wind sites, validation campaigns, Idaho) |

- Two wind directions per global location and four per site location, spread around the compass.
- Minimum spacing between locations: 3 km (1.5 km inside sites).
- Roughness: the land-cover map times a random factor per case (0.6–1.6, log-uniform).
- **Splits** per 2° block (90 / 5 / 5 %). Locations closer than 12 km to a block of another
  split are rejected, so validation and test terrain never overlaps training terrain.
  Sites marked `val` or `test` are held out completely (no other case within their radius
  plus 12 km). Test sites: Owens Valley, Valle de Bravo, Tolmin, Bir Billing, Queenstown and
  Big Southern Butte.
- **Queue.** The pilot cases (~500: every relief class, hard terrain, sites, and the
  steepest locations of the list) come first, then all the other cases in random order.
  Any prefix of the list is a balanced sample, so the list can be longer than the compute
  will finish.

## Terrain crops

`global-fetch` writes one `terrain/<L000>/<L000123>.npz` per location: elevation
(uint16, 0.2 m steps) and WorldCover class (uint8) on a 30 m north-up grid in a local
Transverse Mercator projection centred on the location, ~23.7 km wide (the input window
for any wind direction plus the taper to flat). About 1 MB per location. Crops are
self-contained, so compute nodes need no internet.

- Open sea (no tile, or no data over water) is set to sea level; voids over land are interpolated.
- FABDEM tiles are cached in `dem_tiles/fabdem` while fetching. Shards split the work by
  FABDEM zip, so each zip is read by one shard. `--clean-cache` deletes the cache at the end.
- Pilot locations are fetched first. Cases whose crop does not exist yet are skipped by
  `run` (shown as `waiting_terrain` in `status`), so the pilot can start while the rest is
  still being fetched.

## Roughness in the CFD

The land-cover map becomes log10(z0) with `landcover.z0_table` (forest 0.8 m, shrubs 0.1,
grass 0.03, crops 0.05, built-up 0.8, bare 0.01, snow 0.002, water 0.0002, ...).

- **Network input:** log10(z0) averaged over each 50 m pixel (the existing 4th channel).
- **Inflow:** z0_ref = geometric mean over the upwind part of the input window; the inlet
  profile and the normalisation U_in(z), k_in use z0_ref.
- **Ground:** one z0 per ground face (`z0 nonuniform List<scalar>` in
  `atmNutkWallFunction` / `atmEpsilonWallFunction`), blended to z0_ref over the same taper
  as the terrain. The face order of the ground patch is taken from the shared topology.
- Samples store `log10_z0_input` (float16) and `z0_ref`; regional samples (no map) still
  load with their uniform z0.

## Disk

| What | Size |
|---|---|
| Descriptors | ~0.3 GB |
| Crops (~90,000 locations × ~1 MB) | ~90 GB |
| FABDEM tile cache while fetching | up to ~150–250 GB (deleted with `--clean-cache`) |
| Samples (float16, ~0.7 MB each) | ~70 GB per 100,000 cases |

## Running on a cluster

Code and conda env as in [hpc.md](hpc.md). `PROJECT` is the global project folder.

```bash
export PROJECT=/path/to/windsurrogate_runs/global
mkdir -p logs

# 0. Do compute nodes reach the internet? (otherwise run steps 1-3 elsewhere and copy the project)
srun -p short -N1 -t 00:05:00 curl -sI https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com/tileList.txt

# 1. Descriptors (~26,000 tiles, resumable)
STEP=descriptors sbatch -p short hpc/global_prep.slurm
# 2. Locations and case queue (a few minutes)
STEP=sample sbatch -p short -c 4 -t 01:00:00 hpc/global_prep.slurm
# 3a. Crops of the pilot locations
STEP=pilot-fetch sbatch -p short hpc/global_prep.slurm
# 3b. All crops, 4 nodes (can run while the pilot runs)
STEP=fetch NUM_SHARDS=4 sbatch -p short --array=0-3 hpc/global_prep.slurm

# 4. Pilot: the first ~500 cases (LIMIT per shard = ceil(pilot cases / 8))
EXTRA_CONFIG=config/global.yaml LIMIT=63 NUM_SHARDS=8 sbatch -p bsudfq --array=0-7 -t 12:00:00 hpc/run_cfd.slurm
python -m windsurrogate --config config/global.yaml --set project_dir=$PROJECT status --by stratum --failures 10

# 5. Production: the same command without LIMIT, with the full time limit
EXTRA_CONFIG=config/global.yaml NUM_SHARDS=8 sbatch -p bsudfq --array=0-7 -t 7-00:00:00 hpc/run_cfd.slurm
```

Checks after the pilot (`status --by stratum --failures 10`):
- failures per stratum, especially `hard` and `r5` (mesh quality on the steepest terrain: `max_non_orth`, `mesh_ok` in the status files);
- `converged_frac` and `mean_minutes` per stratum: they set how many cases fit in the allocation;
- a few `plot-sample` figures from different strata.

### Re-prioritising between batches

After a first model is trained, give more weight to the groups where it does worst and
reorder the pending cases (only while no job is running; ids and finished files do not change):

```bash
echo '{"hard": 2.0, "r5": 1.5, "site": 1.5}' > weights.json
python -m windsurrogate --config config/global.yaml --set project_dir=$PROJECT reorder --weights weights.json
```

`evaluate` reports metrics per stratum, group (global / hard / site) and region (site name).

## Limitations

- Neutral atmosphere only (no stability, no thermals); see [v2-stratified-design.md](v2-stratified-design.md).
- `predict` still takes a uniform z0; a z0 map for inference is a next step.
- WorldCover is 2021; roughness of forests varies with season (the per-case random factor covers part of it).
