"""Turn a finished OpenFOAM case into a compact training sample."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from . import foamio
from .foamcase import cell_agl, inflow_profile, ustar
from .geometry import CaseGeometry
from .terrain import CaseTerrain


def sample_agl(field: np.ndarray, agl: np.ndarray, heights) -> np.ndarray:
    """Linear interpolation of field (nz, ny, nx[, c]) to fixed heights above ground.

    agl (nz, ny, nx) increases monotonically with k in every column.
    Returns (len(heights), ny, nx[, c]).
    """
    nz = agl.shape[0]
    out = []
    for h in heights:
        k1 = np.clip((agl < h).sum(axis=0), 1, nz - 1)          # first level above h
        k0 = k1 - 1
        a0 = np.take_along_axis(agl, k0[None], 0)[0]
        a1 = np.take_along_axis(agl, k1[None], 0)[0]
        w = np.clip((h - a0) / np.maximum(a1 - a0, 1e-9), 0.0, 1.0)
        if field.ndim == 4:
            f0 = np.take_along_axis(field, k0[None, ..., None], 0)[0]
            f1 = np.take_along_axis(field, k1[None, ..., None], 0)[0]
            out.append(f0 + (f1 - f0) * w[..., None])
        else:
            f0 = np.take_along_axis(field, k0[None], 0)[0]
            f1 = np.take_along_axis(field, k1[None], 0)[0]
            out.append(f0 + (f1 - f0) * w)
    return np.stack(out)


def parse_log(log: Path) -> dict:
    txt = log.read_text(errors="replace") if log.exists() else ""
    m = re.search(r"SIMPLE solution converged in (\d+) iterations", txt)
    times = re.findall(r"^Time = (\d+)", txt, flags=re.M)
    res = {}
    for name in ("Ux", "Uy", "Uz", "p", "k", "epsilon", "omega"):
        r = re.findall(rf"Solving for {name}, Initial residual = ([0-9.eE+-]+)", txt)
        if r:
            res[name] = float(r[-1])
    ct = re.findall(r"ExecutionTime = ([0-9.]+) s", txt)
    return dict(residual_converged=m is not None, iterations=int(m.group(1)) if m else (int(times[-1]) if times else 0),
                final_residuals=res, solver_seconds=float(ct[-1]) if ct else None,
                diverged=("nan" in txt.lower() and "Floating point exception" in txt) or "FOAM FATAL" in txt)


def probe_convergence(case_dir: Path, uref: float, window: int = 100) -> dict:
    """Max change of probed U over the last `window` iterations, relative to Uref."""
    files = sorted(Path(case_dir).glob("postProcessing/probes/*/U"))
    if not files:
        return {}
    rows = []
    for line in files[-1].read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        nums = np.array(line.replace("(", " ").replace(")", " ").split(), dtype=float)
        rows.append(nums)
    if len(rows) < 3:
        return {}
    n = min(len(r) for r in rows)
    a = np.array([r[:n] for r in rows])
    t, v = a[:, 0], a[:, 1:].reshape(len(a), -1, 3)
    j = int(np.searchsorted(t, t[-1] - window))
    j = min(j, len(t) - 2)
    change = np.abs(v[-1] - v[j]).max(axis=-1) / uref
    return dict(probe_max_change=float(change.max()), probe_p95_change=float(np.percentile(change, 95)),
                probe_window=int(t[-1] - t[j]), probe_last_iter=int(t[-1]))


def build_sample(cfg, geom: CaseGeometry, terrain: CaseTerrain, case: dict, case_dir: Path,
                 use_instant: bool = True) -> dict:
    tdir = foamio.latest_time_dir(case_dir)
    if tdir is None:
        raise RuntimeError("no result time directory")
    nx, ny, nz = geom.nx, geom.ny, geom.nz
    n = nx * ny * nz
    # Unconverged (oscillating) cases: use the iteration average if it was collected.
    use_mean = not use_instant and (tdir / "UMean").exists() and (tdir / "kMean").exists()
    uname, kname = ("UMean", "kMean") if use_mean else ("U", "k")
    U = foamio.read_internal_field(tdir / uname, n).reshape(nz, ny, nx, 3)
    k = foamio.read_internal_field(tdir / kname, n).reshape(nz, ny, nx)
    agl = cell_agl(geom, terrain)
    heights = np.asarray(cfg["output"]["heights_agl"], float)
    js, is_ = geom.core_cell_index
    Uh = sample_agl(U[:, js, is_], agl[:, js, is_], heights)      # (H, N, N, 3)
    kh = sample_agl(k[:, js, is_], agl[:, js, is_], heights)      # (H, N, N)
    # Terrain under the core cells (CFD ground, cell centres) for consistency checks.
    h = terrain.h_nodes
    hc = 0.25 * (h[1:, 1:] + h[1:, :-1] + h[:-1, 1:] + h[:-1, :-1])
    z0 = float(case["z0"])
    U_in, k_in, _, _ = inflow_profile(cfg, z0, heights)
    return dict(
        terrain_input=terrain.h_input.astype(np.float32),           # (rows, cols) m rel. base
        terrain_core=hc[js, is_].astype(np.float32),
        U=np.moveaxis(Uh, -1, 1).astype(np.float32),                # (H, 3, N, N) m/s, wind frame
        k=kh.astype(np.float32),                                    # (H, N, N) m2/s2
        heights=heights.astype(np.float32),
        U_inflow=U_in.astype(np.float32),                           # inflow speed at each height
        k_inflow=k_in.astype(np.float32),
        ustar=np.float32(ustar(cfg, z0)),
        averaged=np.bool_(use_mean),
    )
