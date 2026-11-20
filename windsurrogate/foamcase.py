"""OpenFOAM v2412 (openfoam.com) case generation for neutral ABL flow over terrain."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

from . import foamio
from .geometry import CaseGeometry
from .terrain import CaseTerrain, case_z0, mesh_points

POLY_FILES = ("faces", "owner", "neighbour", "boundary")
GROUND_ORDER = "ground_order.npy"     # cell-column index (j * nx + i) of every ground face


# --------------------------------------------------------------------------- environment
_ENV_CACHE: dict[str, dict] = {}


def foam_env(activate: str) -> dict:
    """Environment of a shell after running the OpenFOAM activation command (cached)."""
    if activate in _ENV_CACHE:
        return _ENV_CACHE[activate]
    out = subprocess.run(["bash", "-c", f"{activate} >/dev/null 2>&1 && env -0"],
                         capture_output=True, check=True)
    env = dict(item.split("=", 1) for item in out.stdout.decode().split("\0") if "=" in item)
    if "WM_PROJECT_VERSION" not in env:
        raise RuntimeError("OpenFOAM environment not found after activation command: " + activate)
    _ENV_CACHE[activate] = env
    return env


def run(cmd: list[str] | str, cwd: Path, env: dict, log: Path | None = None, timeout=None) -> None:
    shell = isinstance(cmd, str)
    with open(log, "w") if log else open(os.devnull, "w") as fh:
        r = subprocess.run(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                           shell=shell, timeout=timeout, executable="/bin/bash" if shell else None)
    if r.returncode != 0:
        tail = Path(log).read_text()[-3000:] if log else ""
        raise RuntimeError(f"{cmd} failed with code {r.returncode}\n{tail}")


# --------------------------------------------------------------------------- physics helpers
def ustar(cfg, z0: float) -> float:
    f = cfg["flow"]
    return f["kappa"] * f["Uref"] / math.log((f["Zref"] + z0) / z0)


def inflow_profile(cfg, z0: float, z):
    """Richards & Hoxey inflow (U, k, epsilon, omega) at height z above flat ground."""
    f = cfg["flow"]
    us = ustar(cfg, z0)
    z = np.maximum(np.asarray(z, float), 0.0)
    U = us / f["kappa"] * np.log((z + z0) / z0)
    k = np.full_like(z, us**2 / math.sqrt(f["Cmu"]))
    eps = us**3 / (f["kappa"] * (z + z0))
    omega = us / (f["kappa"] * math.sqrt(f["Cmu"]) * (z + z0))
    return U, k, eps, omega


# --------------------------------------------------------------------------- topology
def block_mesh_dict(nx: int, ny: int, nz: int) -> str:
    return f"""
scale 1;
vertices
(
    (0 0 0) ({nx} 0 0) ({nx} {ny} 0) (0 {ny} 0)
    (0 0 {nz}) ({nx} 0 {nz}) ({nx} {ny} {nz}) (0 {ny} {nz})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    inlet  {{ type patch;    faces ((0 4 7 3)); }}
    outlet {{ type patch;    faces ((2 6 5 1)); }}
    ground {{ type wall;     faces ((0 3 2 1)); }}
    top    {{ type patch;    faces ((4 5 6 7)); }}
    sides  {{ type symmetry; faces ((1 5 4 0) (3 7 6 2)); }}
);
mergePatchPairs ();
"""


def control_dict(app: str, end_time: int, write_interval: int, libs: bool = True,
                 functions: str = "") -> str:
    lib = "libs (atmosphericModels);\n" if libs else ""
    return f"""
{lib}application     {app};
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {end_time};
deltaT          1;
writeControl    timeStep;
writeInterval   {write_interval};
purgeWrite      1;
writeFormat     ascii;
writePrecision  7;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
{functions}
"""


def probe_locations(geom: CaseGeometry, terrain: CaseTerrain, n: int = 7, agls=(10.0, 100.0)) -> np.ndarray:
    """n x n grid of points over the core at fixed heights above ground (convergence monitor)."""
    from scipy.interpolate import RegularGridInterpolator
    h = geom.core_half * 0.9
    xs = np.linspace(-h, h, n)
    X, Y = np.meshgrid(xs, xs)
    hg = RegularGridInterpolator((geom.y_nodes, geom.x_nodes), terrain.h_nodes)(np.c_[Y.ravel(), X.ravel()])
    pts = [np.c_[X.ravel(), Y.ravel(), hg + a] for a in agls]
    return np.concatenate(pts)


def probes_function(locs: np.ndarray, every: int = 10, avg_start: int | None = None) -> str:
    """Probe monitor plus (optionally) an iteration average used when a case never settles
    (steady RANS can oscillate behind steep ridges); the mean is then used as the label."""
    loc = " ".join(f"({x:.2f} {y:.2f} {z:.2f})" for x, y, z in locs)
    avg = "" if avg_start is None else f"""
    fieldAverage1
    {{
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        timeStart       {avg_start};
        writeControl    writeTime;
        fields
        (
            U {{ mean on; prime2Mean off; base iteration; }}
            k {{ mean on; prime2Mean off; base iteration; }}
        );
    }}"""
    return f"""
functions
{{{avg}
    probes
    {{
        type            probes;
        libs            (sampling);
        writeControl    timeStep;
        writeInterval   {every};
        fields          (U);
        fixedLocations  true;
        probeLocations  ({loc});
    }}
}}
"""


def make_topology(cfg, geom: CaseGeometry, topo_dir: Path, env: dict) -> dict:
    """Create the shared integer-box polyMesh once and verify point/cell ordering."""
    meta_path = topo_dir / "topology.json"
    want = dict(nx=geom.nx, ny=geom.ny, nz=geom.nz)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if all(meta.get(k) == v for k, v in want.items()) and meta.get("ordering_ok") \
                and (topo_dir / GROUND_ORDER).exists():
            return meta
    topo_dir.mkdir(parents=True, exist_ok=True)
    foamio.write_dict(topo_dir / "system/controlDict", "controlDict",
                      control_dict("blockMesh", 1, 1, libs=False))
    foamio.write_dict(topo_dir / "system/blockMeshDict", "blockMeshDict",
                      block_mesh_dict(geom.nx, geom.ny, geom.nz))
    foamio.write_dict(topo_dir / "system/fvSchemes", "fvSchemes",
                      "ddtSchemes{default steadyState;} gradSchemes{default Gauss linear;} "
                      "divSchemes{default none;} laplacianSchemes{default Gauss linear corrected;} "
                      "interpolationSchemes{default linear;} snGradSchemes{default corrected;}")
    foamio.write_dict(topo_dir / "system/fvSolution", "fvSolution", "")
    run(["blockMesh"], topo_dir, env, topo_dir / "log.blockMesh")
    run(["postProcess", "-func", "writeCellCentres", "-constant"], topo_dir, env,
        topo_dir / "log.writeCellCentres")

    nx, ny, nz = geom.nx, geom.ny, geom.nz
    pts = _read_points(topo_dir / "constant/polyMesh/points")
    ijk = np.rint(pts).astype(np.int64)
    expect = ijk[:, 0] + (nx + 1) * (ijk[:, 1] + (ny + 1) * ijk[:, 2])
    pts_ok = bool(np.array_equal(expect, np.arange(len(pts))))
    cfile = next(p for p in (topo_dir / "constant/C", topo_dir / "0/C") if p.exists())
    C = foamio.read_internal_field(cfile, nx * ny * nz)
    cijk = np.floor(C).astype(np.int64)
    expect_c = cijk[:, 0] + nx * (cijk[:, 1] + ny * cijk[:, 2])
    cells_ok = bool(np.array_equal(expect_c, np.arange(len(C))))
    # Ground faces: their centres (i+0.5, j+0.5, 0) give the cell column of every face, so a
    # (ny, nx) map of roughness can be written as a per-face list in patch order.
    Cg = foamio.read_patch_field(cfile, "ground")
    gi, gj = np.floor(Cg[:, 0]).astype(np.int64), np.floor(Cg[:, 1]).astype(np.int64)
    ground_order = gj * nx + gi
    ground_ok = bool(len(Cg) == nx * ny and np.allclose(Cg[:, 2], 0.0)
                     and np.array_equal(np.sort(ground_order), np.arange(nx * ny)))
    np.save(topo_dir / GROUND_ORDER, ground_order)
    meta = dict(want, points_ok=pts_ok, cells_ok=cells_ok, ground_ok=ground_ok,
                ordering_ok=pts_ok and cells_ok and ground_ok, n_points=len(pts), n_cells=len(C))
    if not meta["ordering_ok"]:
        raise RuntimeError(f"unexpected blockMesh ordering: {meta}")
    for f in ("0/C", "0/Cx", "0/Cy", "0/Cz", "constant/C", "constant/Cx", "constant/Cy", "constant/Cz"):
        (topo_dir / f).unlink(missing_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


def _read_points(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    i = raw.index(b"(", raw.index(b"}"))
    # skip the header block: find count line before the list
    head, body = raw[:i], raw[i + 1: raw.rindex(b")")]
    body = body.replace(b"(", b" ").replace(b")", b" ")
    return np.array(body.split(), dtype=np.float64).reshape(-1, 3)


# --------------------------------------------------------------------------- case files
def _turbulence_properties(cfg) -> tuple[str, str]:
    f = cfg["flow"]
    model = f["turbulence_model"]
    if model == "kEpsilon":
        C1, C2 = 1.44, 1.92
        sigma_eps = f["kappa"] ** 2 / ((C2 - C1) * math.sqrt(f["Cmu"]))
        coeffs = (f"kEpsilonCoeffs {{ Cmu {f['Cmu']}; C1 {C1}; C2 {C2}; C3 0; "
                  f"sigmak 1.0; sigmaEps {sigma_eps:.4f}; }}")
    elif model == "RNGkEpsilon":
        coeffs = ""
    elif model == "kOmegaSST":
        coeffs = ""
    else:
        raise ValueError(model)
    body = f"simulationType RAS;\nRAS\n{{\n    RASModel {model};\n    turbulence on;\n" \
           f"    printCoeffs on;\n    {coeffs}\n}}\n"
    return model, body


def _fv_schemes(model: str) -> str:
    turb = "div(phi,omega) $turbulence;" if model == "kOmegaSST" else "div(phi,epsilon) $turbulence;"
    return f"""
ddtSchemes {{ default steadyState; }}
gradSchemes
{{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}}
divSchemes
{{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    turbulence      bounded Gauss limitedLinear 1;
    div(phi,k)      $turbulence;
    {turb}
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear limited corrected 0.5; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default limited corrected 0.5; }}
wallDist {{ method meshWave; }}
"""


def _fv_solution(cfg, model: str, n_corr: int = 0) -> str:
    s = cfg["solver"]
    t = "omega" if model == "kOmegaSST" else "epsilon"
    return f"""
solvers
{{
    p
    {{
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.05;
    }}
    "(U|k|epsilon|omega)"
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }}
}}
SIMPLE
{{
    nNonOrthogonalCorrectors {n_corr};
    consistent      true;
    // Convergence is decided from probes by the Python runner; this is only a
    // last-resort early exit for trivially converged (e.g. flat) cases.
    residualControl
    {{
        p               {s['residual_p']};
        U               {s['residual_U']};
        "(k|{t})"       {s['residual_turb']};
    }}
}}
relaxationFactors
{{
    equations
    {{
        U               0.8;
        "(k|epsilon|omega)" 0.6;
    }}
}}
cache {{ grad(U); }}
"""


def _abl_entries(cfg, z0: float) -> str:
    f = cfg["flow"]
    return (f"flowDir (1 0 0); zDir (0 0 1); Uref {f['Uref']}; Zref {f['Zref']}; "
            f"z0 uniform {z0}; d uniform 0; kappa {f['kappa']}; Cmu {f['Cmu']};")


def write_case(cfg, geom: CaseGeometry, terrain: CaseTerrain, case: dict, case_dir: Path,
               topo_dir: Path) -> None:
    f, s = cfg["flow"], cfg["solver"]
    z0 = case_z0(terrain, case)
    model, turb_body = _turbulence_properties(cfg)
    if terrain.z0_ground is not None:
        order = np.load(topo_dir / GROUND_ORDER)
        z0_wall = foamio.nonuniform(terrain.z0_ground.reshape(-1)[order])
    else:
        z0_wall = f"uniform {z0}"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Mesh: shared topology + case-specific points.
    poly = case_dir / "constant/polyMesh"
    poly.mkdir(parents=True, exist_ok=True)
    for name in POLY_FILES:
        dst = poly / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to((topo_dir / "constant/polyMesh" / name).resolve())
    pts = mesh_points(geom, terrain)
    foamio.write_points(poly / "points", pts)

    # Dictionaries.
    foamio.write_dict(case_dir / "system/controlDict", "controlDict",
                      control_dict("simpleFoam", s["max_iter"], s["max_iter"],
                                   functions=probes_function(probe_locations(geom, terrain),
                                                             avg_start=int(s.get("avg_start_iter", 1000)))))
    foamio.write_dict(case_dir / "system/fvSchemes", "fvSchemes", _fv_schemes(model))
    nc = s.get("non_orth_correctors", "auto")
    set_non_orth(cfg, case_dir, 0 if nc == "auto" else int(nc))
    n = int(s["procs_per_case"])
    foamio.write_dict(case_dir / "system/decomposeParDict", "decomposeParDict",
                      f"numberOfSubdomains {n};\nmethod scotch;\n")
    foamio.write_dict(case_dir / "constant/transportProperties", "transportProperties",
                      "transportModel Newtonian;\nnu 1.5e-05;\n")
    foamio.write_dict(case_dir / "constant/turbulenceProperties", "turbulenceProperties", turb_body)

    # Initial fields from the inflow profile evaluated at height above local ground.
    agl = cell_agl(geom, terrain)
    U0, k0, e0, w0 = inflow_profile(cfg, z0, agl.reshape(-1))
    us = ustar(cfg, z0)
    k_in = us**2 / math.sqrt(f["Cmu"])
    abl = _abl_entries(cfg, z0)
    shear_top = f.get("top_bc", "shear") == "shear"
    top_U = (f"type fixedShearStress; tau ({us**2:.6g} 0 0); value uniform ({f['Uref']} 0 0);"
             if shear_top else "type slip;")
    top_s = "type zeroGradient;" if shear_top else "type slip;"
    sym = "sides { type symmetry; }"
    Uvec = np.zeros((U0.size, 3))
    Uvec[:, 0] = U0
    foamio.write_field(case_dir / "0/U", "volVectorField", "U", "[0 1 -1 0 0 0 0]",
                       foamio.nonuniform(Uvec), f"""
    inlet  {{ type atmBoundaryLayerInletVelocity; {abl} value uniform ({f['Uref']} 0 0); }}
    outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({f['Uref']} 0 0); }}
    ground {{ type noSlip; }}
    top    {{ {top_U} }}
    {sym}""")
    foamio.write_field(case_dir / "0/p", "volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0", f"""
    inlet  {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform 0; }}
    ground {{ type zeroGradient; }}
    top    {{ {top_s} }}
    {sym}""")
    foamio.write_field(case_dir / "0/k", "volScalarField", "k", "[0 2 -2 0 0 0 0]", f"uniform {k_in:.6g}", f"""
    inlet  {{ type atmBoundaryLayerInletK; {abl} value uniform {k_in:.6g}; }}
    outlet {{ type inletOutlet; inletValue uniform {k_in:.6g}; value uniform {k_in:.6g}; }}
    ground {{ type kqRWallFunction; value uniform {k_in:.6g}; }}
    top    {{ {top_s} }}
    {sym}""")
    if model == "kOmegaSST":
        foamio.write_field(case_dir / "0/omega", "volScalarField", "omega", "[0 0 -1 0 0 0 0]",
                           foamio.nonuniform(w0), f"""
    inlet  {{ type atmBoundaryLayerInletOmega; {abl} value uniform 1; }}
    outlet {{ type inletOutlet; inletValue uniform {w0.min():.6g}; value uniform {w0.min():.6g}; }}
    ground {{ type atmOmegaWallFunction; z0 {z0_wall}; value uniform 1; }}
    top    {{ {top_s} }}
    {sym}""")
    else:
        foamio.write_field(case_dir / "0/epsilon", "volScalarField", "epsilon", "[0 2 -3 0 0 0 0]",
                           foamio.nonuniform(e0), f"""
    inlet  {{ type atmBoundaryLayerInletEpsilon; {abl} value uniform 1; }}
    outlet {{ type inletOutlet; inletValue uniform {e0.min():.6g}; value uniform {e0.min():.6g}; }}
    ground {{ type atmEpsilonWallFunction; z0 {z0_wall}; Cmu {f['Cmu']}; kappa {f['kappa']}; value uniform 1; }}
    top    {{ {top_s} }}
    {sym}""")
    foamio.write_field(case_dir / "0/nut", "volScalarField", "nut", "[0 2 -1 0 0 0 0]", "uniform 0", f"""
    inlet  {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
    ground {{ type atmNutkWallFunction; z0 {z0_wall}; Cmu {f['Cmu']}; kappa {f['kappa']}; boundNut true; value uniform 0; }}
    top    {{ type calculated; value uniform 0; }}
    {sym}""")


def set_non_orth(cfg, case_dir: Path, n_corr: int) -> None:
    model = cfg["flow"]["turbulence_model"]
    foamio.write_dict(case_dir / "system/fvSolution", "fvSolution", _fv_solution(cfg, model, n_corr))


def cell_agl(geom: CaseGeometry, terrain: CaseTerrain) -> np.ndarray:
    """Height above local ground of every cell centre, shape (nz, ny, nx)."""
    zf = geom.z_flat_nodes(terrain.top)
    zc = 0.5 * (zf[1:] + zf[:-1])
    h = terrain.h_nodes
    hc = 0.25 * (h[1:, 1:] + h[1:, :-1] + h[:-1, 1:] + h[:-1, :-1])
    return zc[:, None, None] * (1.0 - hc[None] / terrain.top)
