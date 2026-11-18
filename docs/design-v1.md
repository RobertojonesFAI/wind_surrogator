# v1 design: neutral terrain wind surrogate

## Goal

Predict how terrain shapes the near-surface wind (speed-up and compression, deflection and channelling, lee separation and rotors, shadow, vertical velocity, turbulence) at a fraction of the cost of CFD, over DEMs of any size. The output fields feed a thermal-prediction model as extra input layers.

## Regions

| Region | bbox (lon/lat, WGS84) | CRS | Contents |
|---|---|---|---|
| `cervidae` | −116.42, 43.27, −115.52, 43.97 | UTM 11N | Cervidae Peak, Lucky Peak / Arrowrock, Boise Front, Bogus Basin |
| `lost_river` | −114.35, 43.55, −112.95, 44.60 | UTM 12N | Lost River Range (Borah Peak), Big Lost, Little Lost and Pahsimeroi valleys, Challis to Arco |

- **DEMs:** one master DEM per region, from USGS 3DEP 1 arc-second (~30 m) with a 13 km margin, reprojected to the region's UTM zone. `prepare` builds them automatically. A local GeoTIFF glob can be used instead (`dem.source: local`).
- **Case allocation:** 2,000 cases split between regions in proportion to area (~30 % Cervidae, ~70 % Lost River).
- **Train/val/test split:** by 20 km spatial blocks within each region, with at least one validation block and one test block per region.
- **Case order:** shuffled, so any prefix of the list (what an early batch of runs covers) mixes regions, splits and wind directions.

## Solver

- **Code:** OpenFOAM v2412 (openfoam.com), conda-forge build. The same build is used on every machine.
- **Equations:** `simpleFoam`, steady incompressible RANS, SIMPLEC.
- **Turbulence:** k-ε with κ = 0.40 and σε = κ²/((C2 − C1)√Cμ) = 1.11, which keeps the inflow profile in equilibrium (Hargreaves & Wright 2007).
- **Inflow:** Richards & Hoxey log profile (`atmBoundaryLayerInlet*`), Uref = 10 m/s at Zref = 100 m.
- **Ground:** rough wall functions (`atmNutkWallFunction`, `atmEpsilonWallFunction`), z0 log-uniform in 0.02–0.5 m.
- **Top:** constant shear stress τ = u*².
- **Sides:** symmetry.
- **Outlet:** fixed pressure with `inletOutlet`.
- **Steep terrain:** one non-orthogonal corrector is added automatically when `checkMesh` reports more than 50° non-orthogonality.
- **Stopping rule:** normalised residuals are a poor stopping signal on terrain meshes; they fall long before the near-ground flow settles. The runner instead watches 98 probes over the core (10 and 100 m AGL). It stops the case (`stopAt writeNow`) once no probe has changed by more than 0.2 % of Uref over 200 iterations, after at least 300 iterations.
- **Oscillating cases:** behind very steep crests the RANS solution can keep oscillating. A `fieldAverage` from iteration 1000 is kept, and the label of such a case is the 1000–1500 average (`averaged = true` in its metadata).

## Geometry

- **Frame:** rotated per case so the wind always enters at x-min. Wind direction is therefore not a network input.
- **Labelled core:** 128 × 128 px at 50 m (6.4 km).
- **Network input window:** 192 × 256 px (4.8 km upwind, 1.6 km sideways and downwind), with real terrain.
- **Outside the window:** 1.5 km cosine blend to flat, then 0.5 km flat.
- **Mesh:** terrain-following sigma mesh (z = z_f + h(1 − z_f/top)), ×1.15 horizontal growth up to 300 m cells, 30 levels with a 5 m first cell, about 1.0 M cells.
- **Shared topology:** one `blockMesh` topology is shared by all cases; each case only writes its own `points`.

## Labels and network

- **Sampled fields:** each case is sampled at 10/50/100/200/400 m AGL on the core grid and saved as one `.npz`.
- **Targets:** u/U_in(z), v/U_in(z), w/U_in(z) and k/k_in, normalised by the inflow profile at the same height. Neutral RANS over rough ground scales with Uref, so one speed is enough.
- **Inputs:**
  - elevation relative to the core mean, divided by 500 m;
  - dh/dx and dh/dy;
  - log10 z0, centred and scaled.
- **Model:** U-Net (base 48, depth 4) with a 2-layer transformer at the bottleneck, about 16.7 M parameters.
- **Augmentation:** mirror across the wind axis (v and dh/dy change sign).
- **Loss:** MSE over the core.
- **Inference:** overlapping tiles on a global, anchored lattice with Hann-window blending. Chunked prediction is identical to single-pass prediction. Vectors are rotated back to east/north, derived fields are computed, and one multi-band GeoTIFF is written per direction.

## Validation of the pipeline

- **Flat terrain:** the inflow profile is maintained to the outlet; maximum deviation 2.8 % below 500 m.
- **Gaussian hill** (H = 300 m, σ = 600 m): crest speed-up 1.62 at 10 m, 1.14 at 400 m; lee wake 0.45·U_in at 10 m.
- **Inference oracle:** a fake model returning the slope channels checks frame axes, tile windows, blending and vector rotation for arbitrary directions.
- **Learning check:** with 24 training cases on a coarse test mesh, the network already cut validation error by 65 % relative to the flat-terrain answer.

## Results: model v1

- **Data:** 1,217 cases (980 train, 127 val, 119 test).
- **Training:** 120 epochs on one A100, about 8 s per epoch.
- **Validation, best epoch:**
  - loss 0.0369; **skill 0.93** relative to the flat-terrain answer;
  - RMSE of u/U_in at 10/50/100/200/400 m: 0.092 / 0.084 / 0.081 / 0.074 / 0.061;
  - recirculation IoU at 10/50/100/200 m: 0.64 / 0.56 / 0.53 / 0.46. At 400 m there is almost no recirculation, so the IoU is not meaningful there.
- **Generalisation gap:** final training loss 0.0086 vs. validation 0.037. More data (the full 2,000 cases) should reduce it.
- **Maps:** 12 directions × 2 regions at 50 m (z0 = 0.1), about 1 minute per direction for the Lost River region on an A100.
- **Visual check** with westerly wind over the Lost River Range:
  - compression on west-facing crests and spurs;
  - shadow and recirculation on the lee (east) side of the crests and in deep valleys;
  - channelling along valleys;
  - no tile seams.

## Cost

- **Per case:** 20–80 minutes on one core (300–1,500 iterations), ~1.3 GB RAM, ~1.4 MB output.
- **Colab CPU High-RAM** (4 physical cores): 4 cases at a time, about 5–7 cases per hour, ~0.5 compute units per hour.
- **Total:** the full 2,000-case dataset is roughly 0.05 compute units per case.

## Next steps

- Evaluate on the test split (`windsurrogate evaluate`) and inspect the CFD-vs-surrogate figures, especially the size of predicted rotor zones.
- Retrain with the full 2,000 cases, starting from the v1 weights.
- Spatial z0 from land cover; k-ω SST comparison on a subset.
- v2: stratified flow ([v2-stratified-design.md](v2-stratified-design.md)).
