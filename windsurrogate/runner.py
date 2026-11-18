"""Run CFD cases end to end, resumable and shardable across machines/sessions."""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .dem import MasterDEM
from .foamcase import foam_env, make_topology, run, set_non_orth, write_case
from .geometry import CaseGeometry
from .postprocess import build_sample, parse_log, probe_convergence
from .sampling import read_cases
from .terrain import case_terrain


def dataset_paths(cfg):
    root = cfg.path(cfg["output"]["dataset_dir"])
    return root, root / "_status"


def sample_path(cfg, case: dict) -> Path:
    root, _ = dataset_paths(cfg)
    return root / case["split"] / f"{case['case_id']}.npz"


def _write_status(cfg, case_id: str, status: dict) -> None:
    _, sdir = dataset_paths(cfg)
    sdir.mkdir(parents=True, exist_ok=True)
    tmp = sdir / f".{case_id}.json.tmp"
    tmp.write_text(json.dumps(status, indent=1))
    os.replace(tmp, sdir / f"{case_id}.json")


def _read_status(cfg, case_id: str) -> dict | None:
    _, sdir = dataset_paths(cfg)
    p = sdir / f"{case_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def parse_checkmesh(log: Path) -> dict:
    txt = log.read_text(errors="replace")
    out = {"mesh_ok": "Mesh OK." in txt}
    m = re.search(r"Mesh non-orthogonality Max: ([0-9.eE+-]+)", txt)
    if m:
        out["max_non_orth"] = float(m.group(1))
    m = re.search(r"Max skewness = ([0-9.eE+-]+)", txt)
    if m:
        out["max_skewness"] = float(m.group(1))
    out["failed_checks"] = len(re.findall(r"\*\*\*", txt))
    return out


def run_monitored(cmd: str, case_dir: Path, env: dict, cfg, timeout: float) -> bool:
    """Run the solver and stop it (stopAt writeNow) once the probed velocities settle.

    Normalised residuals are a poor convergence signal on terrain meshes, so the
    stopping rule is: max |U(t) - U(t - window)| / Uref over all probes < conv_tol,
    after at least min_iter iterations. Returns True if stopped on convergence.
    """
    s = cfg["solver"]
    tol, window = float(s.get("conv_tol", 2e-3)), int(s.get("conv_window", 200))
    min_iter, poll = int(s.get("min_iter", 300)), float(s.get("conv_poll_seconds", 20))
    uref = cfg["flow"]["Uref"]
    log = open(case_dir / "log.simpleFoam", "w")
    proc = subprocess.Popen(cmd, cwd=case_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                            shell=True, executable="/bin/bash")
    t0, stopped = time.time(), False
    try:
        while proc.poll() is None:
            time.sleep(poll)
            if time.time() - t0 > timeout:
                proc.kill()
                raise TimeoutError(f"solver exceeded {timeout / 3600:.1f} h")
            if stopped:
                continue
            c = probe_convergence(case_dir, uref, window=window)
            if c.get("probe_window", 0) >= window and c.get("probe_last_iter", 0) >= min_iter \
                    and c["probe_max_change"] < tol:
                cd = case_dir / "system/controlDict"
                cd.write_text(cd.read_text().replace("stopAt          endTime;", "stopAt          writeNow;"))
                stopped = True
    finally:
        log.close()
    if proc.returncode != 0:
        tail = (case_dir / "log.simpleFoam").read_text(errors="replace")[-3000:]
        raise RuntimeError(f"{cmd} failed with code {proc.returncode}\n{tail}")
    return stopped


def run_case(cfg, geom: CaseGeometry, dem: MasterDEM, case: dict, topo_dir: Path, env: dict,
             prefix: str = "") -> dict:
    """Run one case. `prefix` is prepended to the solver command (e.g. core pinning)."""
    s = cfg["solver"]
    work = Path(cfg["output"]["work_dir"]).expanduser()
    case_dir = work / case["case_id"]
    if case_dir.exists():
        shutil.rmtree(case_dir)
    t0 = time.time()
    info = dict(case, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
    try:
        terrain = case_terrain(geom, dem, case["center_e"], case["center_n"], case["theta"])
        info.update(base=terrain.base, top=terrain.top, max_slope_deg=terrain.max_slope_deg)
        write_case(cfg, geom, terrain, case, case_dir, topo_dir)
        run(["checkMesh", "-constant", "-noTopology"], case_dir, env, case_dir / "log.checkMesh")
        info.update(parse_checkmesh(case_dir / "log.checkMesh"))
        if s.get("non_orth_correctors", "auto") == "auto":
            n_corr = 1 if info.get("max_non_orth", 0) > 50 else 0
            set_non_orth(cfg, case_dir, n_corr)
            info["non_orth_correctors"] = n_corr
        t1 = time.time()
        nproc = int(s["procs_per_case"])
        timeout = float(s.get("timeout_hours", 6)) * 3600
        if nproc > 1:
            run(["decomposePar", "-force"], case_dir, env, case_dir / "log.decomposePar")
            stopped = run_monitored(f"mpirun -np {nproc} simpleFoam -parallel", case_dir, env, cfg, timeout)
            run(["reconstructPar", "-latestTime"], case_dir, env, case_dir / "log.reconstructPar")
        else:
            stopped = run_monitored(f"{prefix}simpleFoam", case_dir, env, cfg, timeout)
        info.update(parse_log(case_dir / "log.simpleFoam"))
        conv = probe_convergence(case_dir, cfg["flow"]["Uref"], window=int(s.get("conv_window", 200)))
        info.update(conv)
        info["converged"] = bool(stopped) or conv.get("probe_max_change", 1.0) < float(s.get("conv_tol", 2e-3))
        info["wall_seconds_solver"] = round(time.time() - t1, 1)
        sample = build_sample(cfg, geom, terrain, case, case_dir, use_instant=info["converged"])
        info["averaged"] = bool(sample["averaged"])
        meta = dict(info, status="done")
        meta["wall_seconds"] = round(time.time() - t0, 1)
        out = sample_path(cfg, case)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.stem}.tmp.npz")
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **sample)
        os.replace(tmp, out)
        info = meta
    except Exception as e:  # noqa: BLE001
        prev = _read_status(cfg, case["case_id"]) or {}
        info.update(status="failed", error=f"{type(e).__name__}: {e}"[-4000:],
                    traceback=traceback.format_exc()[-4000:], attempts=prev.get("attempts", 0) + 1,
                    wall_seconds=round(time.time() - t0, 1))
        if case_dir.exists():
            for log in case_dir.glob("log.*"):
                info.setdefault("logs", {})[log.name] = log.read_text(errors="replace")[-2000:]
    finally:
        keep = info.get("status") == "failed" and s.get("keep_failed_cases", False)
        if case_dir.exists() and not keep:
            shutil.rmtree(case_dir, ignore_errors=True)
    _write_status(cfg, case["case_id"], {k: v for k, v in info.items() if k != "traceback"}
                  | ({"traceback": info["traceback"]} if "traceback" in info else {}))
    return info


def pending_cases(cfg, shard: int = 0, n_shards: int = 1, max_attempts: int = 2) -> list[dict]:
    cases = read_cases(cfg.path(cfg["output"]["cases_file"]))
    todo = []
    for i, c in enumerate(cases):
        if i % n_shards != shard:
            continue
        if sample_path(cfg, c).exists():
            continue
        st = _read_status(cfg, c["case_id"])
        if st and st.get("status") == "failed" and st.get("attempts", 0) >= max_attempts:
            continue
        todo.append(c)
    return todo


def worker(cfg, shard: int = 0, n_shards: int = 1, concurrent: int | None = None,
           limit: int | None = None, stop_after_hours: float | None = None) -> None:
    """Process this shard's pending cases with `concurrent` cases running at once."""
    import copy
    import fcntl

    geom = CaseGeometry.from_config(cfg)
    env = foam_env(cfg["solver"]["activate"])
    base = Path(cfg["output"]["work_dir"]).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    # Refuse to run the same shard twice on one machine (two notebooks attached to the
    # same server would otherwise delete each other's running cases).
    lock_f = open(base / f"shard{shard}of{n_shards}.lock", "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"[worker {shard}/{n_shards}] another worker is already running this shard "
                         f"on this machine; stop it first (e.g. !pkill -f windsurrogate)")
    # Private work dir per worker process, so workers sharing a machine never collide.
    work = base / f"shard{shard}of{n_shards}_pid{os.getpid()}"
    cfg = type(cfg)(copy.deepcopy(dict(cfg)))
    cfg["output"]["work_dir"] = str(work)
    topo_dir = work / "_topology"
    meta = make_topology(cfg, geom, topo_dir, env)
    print(f"[topology] {geom.summary()} -> {meta}", flush=True)
    from .dem import region_dem_path, regions
    regs = regions(cfg)
    todo = pending_cases(cfg, shard, n_shards)
    needed = {c.get("region", regs[0]["name"]) for c in todo}
    dems = {name: MasterDEM.open(region_dem_path(cfg, name)) for name in sorted(needed)}
    if limit:
        todo = todo[:limit]
    concurrent = int(concurrent or cfg["solver"]["concurrent_cases"])
    from . import affinity
    slots = None
    if int(cfg["solver"]["procs_per_case"]) == 1 and affinity.pinning_enabled(cfg, concurrent):
        slots = queue.Queue()
        for slot in affinity.core_slots()[:concurrent]:
            slots.put(slot)
        print(f"[worker {shard}/{n_shards}] pinning each case to one core: {affinity.describe()}", flush=True)
    print(f"[worker {shard}/{n_shards}] {len(todo)} pending cases, {concurrent} at a time", flush=True)
    deadline = time.time() + stop_after_hours * 3600 if stop_after_hours else None
    lock = threading.Lock()
    stats = dict(done=0, failed=0, secs=[])
    t_start = time.time()

    def job(c):
        if deadline and time.time() > deadline:
            return None
        dem = dems[c.get("region", regs[0]["name"])]
        if slots is None:
            return run_case(cfg, geom, dem, c, topo_dir, env)
        cpu, node = slots.get()
        try:
            return run_case(cfg, geom, dem, c, topo_dir, env, prefix=affinity.pin_prefix(cpu, node))
        finally:
            slots.put((cpu, node))

    with ThreadPoolExecutor(max_workers=concurrent) as ex:
        futs = {ex.submit(job, c): c for c in todo}
        for fut in as_completed(futs):
            info = fut.result()
            if info is None:
                continue
            with lock:
                if info["status"] == "done":
                    stats["done"] += 1
                    stats["secs"].append(info["wall_seconds"])
                else:
                    stats["failed"] += 1
                n_fin = stats["done"] + stats["failed"]
                rate = n_fin / max(time.time() - t_start, 1.0)
                eta = (len(todo) - n_fin) / rate / 3600 if rate > 0 else float("nan")
                print(f"[{n_fin}/{len(todo)}] {info['case_id']} {info['status']:6s} "
                      f"iters={info.get('iterations')} conv={info.get('converged')} "
                      f"t={info.get('wall_seconds')}s  ETA {eta:.1f} h"
                      + (f"  ERR {info.get('error', '')[:200]}" if info["status"] != "done" else ""),
                      flush=True)
    print(f"[worker {shard}] finished: {stats['done']} done, {stats['failed']} failed", flush=True)
    shutil.rmtree(work, ignore_errors=True)


def status_summary(cfg) -> dict:
    cases = read_cases(cfg.path(cfg["output"]["cases_file"]))
    out = {"total": len(cases), "done": 0, "failed": 0, "pending": 0, "by_split": {}, "converged": 0}
    secs = []
    for c in cases:
        sp = out["by_split"].setdefault(c["split"], {"total": 0, "done": 0})
        sp["total"] += 1
        st = _read_status(cfg, c["case_id"])
        if sample_path(cfg, c).exists():
            out["done"] += 1
            sp["done"] += 1
            if st:
                secs.append(st.get("wall_seconds", 0))
                out["converged"] += bool(st.get("converged"))
        elif st and st.get("status") == "failed":
            out["failed"] += 1
        else:
            out["pending"] += 1
    if secs:
        out["mean_case_minutes"] = round(float(np.mean(secs)) / 60, 1)
    return out
