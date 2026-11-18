"""Measure solver speed on one machine for different numbers of simultaneous cases.

The same case is copied n times and the copies run together for a fixed number of
iterations, with and without core pinning. The useful number is the node throughput
(iterations per minute summed over all copies): it tells how many cases to run at
once and whether pinning helps on this hardware.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

from . import affinity
from .dem import MasterDEM, region_dem_path, regions
from .foamcase import foam_env, make_topology, run, set_non_orth, write_case
from .geometry import CaseGeometry
from .runner import parse_checkmesh
from .sampling import read_cases
from .terrain import case_terrain

_ET = re.compile(r"^ExecutionTime = ([0-9.eE+-]+) s", re.M)


def _seconds_per_iter(log: Path, skip: int) -> float:
    t = [float(x) for x in _ET.findall(log.read_text(errors="replace"))]
    if len(t) <= skip + 2:
        return float("nan")
    return (t[-1] - t[skip]) / (len(t) - 1 - skip)


def bench(cfg, copies=(1, 24, 48), iters: int = 30, modes=("pin", "nopin"), case_index: int = 0,
          work_dir=None) -> list[dict]:
    geom = CaseGeometry.from_config(cfg)
    env = foam_env(cfg["solver"]["activate"])
    base = Path(work_dir or cfg["output"]["work_dir"]).expanduser() / "_bench"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    topo = base / "_topology"
    make_topology(cfg, geom, topo, env)

    case = read_cases(cfg.path(cfg["output"]["cases_file"]))[case_index]
    regs = regions(cfg)
    dem = MasterDEM.open(region_dem_path(cfg, case.get("region", regs[0]["name"])))
    terrain = case_terrain(geom, dem, case["center_e"], case["center_n"], case["theta"])
    template = base / "template"
    write_case(cfg, geom, terrain, case, template, topo)
    run(["checkMesh", "-constant", "-noTopology"], template, env, template / "log.checkMesh")
    n_corr = 1 if parse_checkmesh(template / "log.checkMesh").get("max_non_orth", 0) > 50 else 0
    set_non_orth(cfg, template, n_corr)
    cd = template / "system/controlDict"
    cd.write_text(re.sub(r"endTime\s+\d+;", f"endTime         {iters};", cd.read_text()))

    slots = affinity.core_slots()
    print(f"bench case {case['case_id']} ({case.get('region')}, relief {case.get('relief')} m), "
          f"{iters} iterations per run; {affinity.describe()}", flush=True)
    results = []
    for n in copies:
        for mode in modes:
            if mode == "pin" and n > len(slots):
                print(f"  skip n={n} pin: only {len(slots)} physical cores", flush=True)
                continue
            dirs, procs = [], []
            for i in range(n):
                d = base / f"run_{mode}_{n}_{i}"
                shutil.rmtree(d, ignore_errors=True)
                shutil.copytree(template, d, symlinks=True)
                dirs.append(d)
            t0 = time.time()
            for i, d in enumerate(dirs):
                prefix = affinity.pin_prefix(*slots[i]) if mode == "pin" else ""
                log = open(d / "log.simpleFoam", "w")
                procs.append((subprocess.Popen(f"{prefix}simpleFoam", cwd=d, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, shell=True,
                                               executable="/bin/bash"), log))
            for p, log in procs:
                p.wait()
                log.close()
            wall = time.time() - t0
            spi = [_seconds_per_iter(d / "log.simpleFoam", skip=min(5, iters // 3)) for d in dirs]
            spi = [s for s in spi if s == s]
            mean = sum(spi) / len(spi) if spi else float("nan")
            r = dict(copies=n, mode=mode, s_per_iter=round(mean, 2),
                     iters_per_min_per_case=round(60 / mean, 2) if mean == mean else None,
                     node_iters_per_min=round(n * 60 / mean, 1) if mean == mean else None,
                     wall_s=round(wall, 1))
            results.append(r)
            print(f"  n={n:3d} {mode:5s}  {r['s_per_iter']:6.2f} s/iter per case  "
                  f"{r['iters_per_min_per_case']:6.2f} iter/min per case  "
                  f"node: {r['node_iters_per_min']:7.1f} iter/min", flush=True)
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
    best = max((r for r in results if r["node_iters_per_min"]), key=lambda r: r["node_iters_per_min"], default=None)
    if best:
        print(f"best throughput: {best['copies']} cases, {best['mode']} "
              f"({best['node_iters_per_min']} iterations/min for the node)", flush=True)
    shutil.rmtree(base, ignore_errors=True)
    return results
