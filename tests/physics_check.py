"""Physical sanity checks on idealised terrain (run manually: takes a few minutes).

1. Flat ground: the inflow log profile must be maintained across the domain
   (horizontal homogeneity of the ABL; Richards & Hoxey / Hargreaves & Wright).
2. Isolated Gaussian hill: speed-up at the crest, slow-down/recirculation in the lee.

usage: python tests/physics_check.py path/to/config.yaml
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from windsurrogate.config import load_config  # noqa: E402
from windsurrogate.foamcase import foam_env, make_topology, write_case, inflow_profile, cell_agl  # noqa: E402
from windsurrogate.geometry import CaseGeometry  # noqa: E402
from windsurrogate.postprocess import build_sample, probe_convergence  # noqa: E402
from windsurrogate.runner import run_monitored  # noqa: E402
from windsurrogate.terrain import CaseTerrain  # noqa: E402
from windsurrogate import foamio  # noqa: E402


def idealised(geom, kind, H=300.0, L=600.0):
    X, Y = np.meshgrid(geom.x_nodes, geom.y_nodes)
    if kind == "flat":
        h = np.zeros_like(X)
    else:
        h = H * np.exp(-(X**2 + Y**2) / (2 * L**2))
    xc, yc = geom.input_cell_centres()
    XC, YC = np.meshgrid(xc, yc)
    hin = np.zeros_like(XC) if kind == "flat" else H * np.exp(-(XC**2 + YC**2) / (2 * L**2))
    return CaseTerrain(base=0.0, h_nodes=h, h_input=hin.astype(np.float32), top=geom.domain_top(h.max()),
                       max_slope_deg=0.0)


def run_one(cfg, geom, topo, env, kind, work):
    d = work / kind
    shutil.rmtree(d, ignore_errors=True)
    case = dict(case_id=kind, z0=0.1, split="test", theta=270.0)
    terr = idealised(geom, kind)
    write_case(cfg, geom, terr, case, d, topo)
    t0 = time.time()
    run_monitored("simpleFoam", d, env, cfg, timeout=3 * 3600)
    s = build_sample(cfg, geom, terr, case, d)
    tdir = foamio.latest_time_dir(d)
    n = geom.n_cells
    U = foamio.read_internal_field(tdir / "U", n).reshape(geom.nz, geom.ny, geom.nx, 3)
    return kind, s, U, terr, time.time() - t0, int(tdir.name), probe_convergence(d, cfg["flow"]["Uref"])


def main(cfg_path):
    cfg = load_config(cfg_path)
    geom = CaseGeometry.from_config(cfg)
    env = foam_env(cfg["solver"]["activate"])
    work = Path(cfg["output"]["work_dir"]) / "_physics"
    topo = Path(cfg["output"]["work_dir"]) / "_topology"
    make_topology(cfg, geom, topo, env)
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(lambda k: run_one(cfg, geom, topo, env, k, work), ["flat", "hill"]))
    report = {}
    for kind, s, U, terr, secs, iters, conv in res:
        heights = s["heights"]
        Uin = s["U_inflow"]
        agl = cell_agl(geom, terr)
        if kind == "flat":
            # Compare the column profile near the outlet with the inflow profile.
            j = geom.ny // 2
            i_out = geom.nx - 3
            prof = U[:, j, i_out, 0]
            z = agl[:, j, i_out]
            Uth, _, _, _ = inflow_profile(cfg, 0.1, z)
            m = z < 500
            err = np.abs(prof[m] - Uth[m]) / Uth[m]
            report[kind] = dict(max_rel_dev_below_500m=float(err.max()), iterations=iters,
                                minutes=round(secs / 60, 1), **conv)
        else:
            spd = np.linalg.norm(s["U"], axis=1)          # (H, N, N)
            N = spd.shape[-1]
            c = N // 2
            crest = spd[:, c, c] / Uin
            lee = spd[:, c, min(N - 1, c + int(900 / geom.dx))] / Uin
            report[kind] = dict(crest_speedup={int(h): round(float(v), 3) for h, v in zip(heights, crest)},
                                lee_900m_ratio={int(h): round(float(v), 3) for h, v in zip(heights, lee)},
                                min_u_10m=float(s["U"][0, 0].min()), iterations=iters,
                                minutes=round(secs / 60, 1), **conv)
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main(sys.argv[1])
