# v2 design: stratified flow

Status: **draft for discussion**. Nothing here is implemented yet; v1 ([design-v1.md](design-v1.md)) remains the baseline.

## 1. Goal

Predict the terrain-induced phenomena that depend on atmospheric stability, over terrain from anywhere in the world:

| Phenomenon | Regime | Classic locations |
|---|---|---|
| Blocking, gap winds / venturi | Fr ≪ 1 | Columbia Gorge, Tehuantepec, Cajon Pass |
| Hydraulic jump, downslope windstorm (foehn, chinook, sundowner, bora, zonda) | Fr ≈ 1, often with an inversion | Front Range (Boulder), Santa Ynez Mountains, Alps, Dinaric Alps, Andes |
| Mountain waves and wave rotors | Fr > 1 | Owens Valley (Sierra Nevada), Alps |
| The mechanical effects of v1 (crest compression, shadow, lee separation) | any | everywhere |

Here Fr = U / (N·h), where U is the wind speed, N the Brunt–Väisälä frequency (stability) and h the barrier height.

**Out of scope for v2:** moisture (moist foehn, clouds), radiation and the diurnal cycle (anabatic and katabatic winds), Coriolis. These are candidates for a v3.

## 2. Why v1 is not enough

v1 is neutral: N = 0, no buoyancy, no temperature. It captures the mechanical part of terrain flow. It cannot produce hydraulic jumps, waves or foehn warming, or the blocking that forces flow through gaps.

v1 also removes wind speed by normalisation (U ∝ Uref). With stratification the pattern changes with Fr, so that is no longer valid.

## 3. Physics and solver

- **Equations:** incompressible RANS with the Boussinesq approximation and potential temperature θ as a transported scalar.
  - The openfoam.com v2312 API documents `buoyantBoussinesqSimpleFoam` and `buoyantBoussinesqPimpleFoam`. **Check that they are still in v2412** (`which buoyantBoussinesqPimpleFoam` in the conda env).
  - Precedent: NREL's SOWFA is an OpenFOAM-based atmospheric boundary layer solver with stability.
- **Turbulence:** k-ε with buoyancy production and destruction. The openfoam.com ABL module provides the `atmBuoyancyTurbSource` fvOption. Compare with k-ω SST in the pilot.
- **Inflow:** U(z), direction and θ(z) profiles taken from ERA5 (§6), including inversions.
- **Top and outflow:** a **Rayleigh damping layer** (sponge) in the upper third of the domain and near the outlet, so waves are not reflected. If v2412 has no ready-made option, implement it as a coded fvOption.
- **Steady or transient:**
  - Clearly Fr > 1.5 or Fr < 0.5: try steady (SIMPLE), with the v1 probe-based stopping rule.
  - Fr ≈ 0.5–1.5 (jumps, rotors): transient (PIMPLE). Store the time mean plus the standard deviation or maximum, which gives rotor turbulence and gusts.
  - The pilot (§9, F1–F2) sets the actual threshold.
- **Extra physical outputs:** θ′ (potential temperature anomaly, for foehn warming) and w up to several km.

## 4. Domain and mesh (initial proposal, fixed by the pilot)

Waves and jumps need a tall domain and a long lee. 30 m detail needs fine cells. Both do not fit in one cheap case.

| | v1 | v2 (proposal) |
|---|---|---|
| Labelled core | 128² px at 50 m (6.4 km) | 256² px at 100 m (25.6 km) |
| Domain height | a few km | ~12 km, with a sponge in the top ~4 km |
| Lee distance | 1.6 km | ≥ 10 km inside the core, plus the buffer |
| Cells | ~1.0 M | ~4–6 M (to be measured) |
| Cost per case | ~0.75 core-hours | ~10–50× (steady at the low end, transient at the high end) |

For near-ground detail, v2 is combined with v1: the v1 prediction (or a 30 m version of it) is an **input channel** of v2. v2 then learns the stratification correction on top of the mechanical field, instead of relearning everything.

## 5. Network

- **Spatial inputs:**
  - relative elevation, slopes and z0, as in v1;
  - the **neutral v1 prediction** (u, v, w, k at 10–100 m) as extra channels.
- **Scalar or profile inputs**, through FiLM conditioning in the bottleneck and decoder:
  - U at the reference height and wind shear;
  - N in 3–4 layers (for example 0–1, 1–3, 3–6 and 6–12 km);
  - inversion height and strength (Δθ);
  - computed Fr.
- **Direction:** the terrain is still rotated to the low-level wind direction. Directional shear with height is a scalar input.
- **Outputs:** u, v, w, k and θ′ at extended heights above ground (for example 10, 50, 100, 200, 400, 800, 1500, 3000 and 5000 m). For transient cases, also the standard deviation of u and w and a recirculation probability.
- **Normalisation:** u, v and w by Uref; θ′ by a reference Δθ. The field no longer scales linearly with U.
- **Split:** hold out **entire regions** for testing (leave-region-out), not only blocks inside the same region.

## 6. Sampling terrain and atmosphere

### Terrain

- **DEM:** FABDEM (Copernicus GLO-30 with forests and buildings removed, 30 m; CC BY-NC-SA 4.0, non-commercial) outside the US, and 3DEP inside the US.
  - Commercial use would require a licence from Fathom.
  - Raw GLO-30 is a surface model and would put forest canopy into the CFD as if it were terrain.
- **No uniform random sampling.** Random points would be mostly plains and ocean. Instead:
  1. Compute terrain descriptors on a ~25 km global grid (for example in Google Earth Engine): relief, mean and maximum slope, dominant ridge orientation, valley width, presence of gaps and passes.
  2. Cluster them into terrain classes.
  3. Sample evenly per class.
- **Targeted sites** (known phenomenon plus validation data):
  - Owens Valley / Sierra Nevada: waves and rotors (**T-REX** field campaign, 2006).
  - Front Range, Boulder: downslope windstorms and chinook.
  - Santa Ynez Mountains, Santa Barbara: sundowners (**SBSE** field campaign, 2022).
  - Alps: foehn (**MAP** field campaign, 1999).
  - Dinaric Alps: bora.
  - Columbia Gorge, Tehuantepec, Cajon Pass: gap winds.
  - Andes (Mendoza): zonda.
  - Popular soaring and paragliding sites: Alps, central Mexico, Andes, Turkey, Korea, Himalaya.
  - Idaho (Cervidae, Lost River), for continuity with v1.
- **Mesh check:** before production, run `checkMesh` on the steepest cases (Himalaya, Andes), which have more relief per window than Idaho.

### Atmosphere

- For each site, take U, direction and θ profiles from **ERA5** pressure levels.
- Sample **stratified by Fr and by inversion strength**, so the rare regimes (Fr ≈ 1, strong inversion) are not under-represented.
- Deliberately include the dates of documented events: Boulder windstorms, sundowners documented by SBSE, T-REX intensive observation periods.

## 7. Validation

1. **Solver benchmarks**, before any production:
   - linear hydrostatic waves over a Witch of Agnesi hill, compared with the analytical solution;
   - the mountain-wave test of Schär et al. (2002);
   - downslope windstorm / hydraulic jump (Long 1955; Durran 1986);
   - blocking and the dividing streamline around a 3D hill at Fr < 1 (Hunt & Snyder 1980).
2. **Against observations:** T-REX (Owens Valley rotors), SBSE (sundowners), MAP (foehn), and surface stations during Boulder windstorms.
3. **Against flight tracks:** wind derived from glider and paraglider tracks **below ~400 m AGL**. Higher up, the comparison would mostly test ERA5, not the surrogate.
4. **Surrogate:** leave-region-out test set. Report metrics per Fr regime, recirculation IoU, w and θ′ errors, and CFD-vs-surrogate figures.

## 8. Compute and storage

- **Compute:** at 10–50× the v1 cost (about 7.5–40 core-hours per case), 129,000 core-hours (8 nodes × 48 cores × 14 days) give **~3,000–17,000 v2 cases**.
- **Storage:** with 4× more pixels and ~9 heights, ~10–20 MB per case, so **~50–300 GB** in total.
- **Training:** hours to about a day per full run on one A100-class GPU.

## 9. Phases

| Phase | Work | Exit criterion |
|---|---|---|
| F0 | Close the open decisions in §10 | This document agreed |
| F1 | Stratified solver prototype + 2D/3D benchmarks (§7.1) | Reproduces the three Fr regimes and the analytical wave solution |
| F2 | Pilot of 50–100 real cases across regimes | Measured cost per case; steady/transient threshold; fixed domain size |
| F3 | Sampling pipeline: terrain descriptors + ERA5 profiles + case list | Even coverage across terrain classes and Fr |
| F4 | Production runs | ≥ 85 % of cases converged |
| F5 | Train v2 (with v1 as input) + leave-region-out evaluation | Clear improvement over v1 for Fr ≲ 1.5 |
| F6 | Integration with the thermal model (per-cell summaries, interpolation in direction and Fr) | Agreed with the thermal-model side |

## 10. Open decisions

1. Steady, transient or mixed? The Fr threshold is set in F2.
2. Resolution and domain size (100 m / 25.6 km is the starting proposal).
3. Turbulence model: k-ε with buoyancy or k-ω SST.
4. DEM source and licence (FABDEM is non-commercial).
5. Output heights, and whether to store temporal variability.
6. Moisture and Coriolis (proposal: not in v2).
7. Where the compute comes from: HPC allocation, Colab, or paid cloud.

## 11. Risks

- **The stratified solver does not converge, or reflects waves at the top.** Mitigation: sponge layer, benchmarks in F1, transient runs where needed.
- **Cost per case is higher than expected.** Mitigation: the F2 pilot measures it before committing large allocations.
- **The surrogate does not generalise to new regions.** Mitigation: sampling by terrain class and leave-region-out validation.
- **FABDEM licence** if the product becomes commercial.
- **Transient phenomena (rotors, jumps) are poorly represented by a time mean.** Mitigation: store variability as well.
