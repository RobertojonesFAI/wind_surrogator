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
from .terrain import case_terrain, case_z0


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


def _diverged(case_dir: Path, err: Exception) -> bool:
    """Did the solver blow up (floating point exception, NaN fields)?"""
    txt = str(err)
    log = case_dir / "log.simpleFoam"
    if log.exists():
        txt += log.read_text(errors="replace")[-20000:]
    if any(k in txt for k in ("sigFpe", "Floating point exception", "code -8", "non-finite")):
        return True
    return re.search(r"(?<![A-Za-z])-?nan(?![A-Za-z])", txt, flags=re.IGNORECASE) is not None


def _solve(cfg, geom, terrain, case, case_dir: Path, topo_dir: Path, env, prefix, info: dict, safe: bool):
    """Write, check and run one case; returns (stopped on convergence, sample dict)."""
    s = cfg["solver"]
    write_case(cfg, geom, terrain, case, case_dir, topo_dir, safe=safe)
    run(["checkMesh", "-constant", "-noTopology"], case_dir, env, case_dir / "log.checkMesh")
    info.update(parse_checkmesh(case_dir / "log.checkMesh"))
    if s.get("non_orth_correctors", "auto") == "auto":
        n_corr = 1 if info.get("max_non_orth", 0) > 50 else 0
        if safe:
            n_corr = 2 if info.get("max_non_orth", 0) > 50 else 1
        set_non_orth(cfg, case_dir, n_corr, safe)
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
    for k in ("U", "k"):
        if not np.isfinite(sample[k].astype(np.float32)).all():
            raise RuntimeError(f"non-finite values in {k}: the solution diverged")
    return stopped, sample


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
        terrain = case_terrain(geom, dem, case["center_e"], case["center_n"], case["theta"],
                               z0=case.get("z0"), z0_scale=float(case.get("z0_scale", 1.0)))
        info.update(base=terrain.base, top=terrain.top, max_slope_deg=terrain.max_slope_deg,
                    z0=case_z0(terrain, case))
        # First the normal settings; if the solver diverges (steep terrain), once more from
        # scratch with conservative settings (solver.safe_retry).
        modes = (False, True) if s.get("safe_retry", True) else (False,)
        for safe in modes:
            try:
                stopped, sample = _solve(cfg, geom, terrain, case, case_dir, topo_dir, env, prefix, info, safe)
                break
            except RuntimeError as err:
                if safe or len(modes) == 1 or not _diverged(case_dir, err):
                    raise
                info["diverged_first_try"] = True
                info["first_try_iterations"] = parse_log(case_dir / "log.simpleFoam").get("iterations")
                shutil.rmtree(case_dir, ignore_errors=True)
        info["safe_mode"] = bool(info.get("diverged_first_try"))
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


def _listing(cfg) -> tuple[set, set]:
    """(finished sample files as (split, case_id), case ids with a status file), one directory
    listing each: much faster than a stat per case on a parallel file system."""
    root, sdir = dataset_paths(cfg)
    done = set()
    for split in ("train", "val", "test"):
        d = root / split
        if d.is_dir():
            with os.scandir(d) as it:
                done |= {(split, e.name[:-4]) for e in it if e.name.endswith(".npz") and not e.name.startswith(".")}
    status = set()
    if sdir.is_dir():
        with os.scandir(sdir) as it:
            status = {e.name[:-5] for e in it if e.name.endswith(".json") and not e.name.startswith(".")}
    return done, status


def terrain_ready(cfg, case: dict) -> bool:
    """Regional cases always; global cases once their terrain crop exists."""
    t = case.get("terrain")
    return t is None or (cfg.root / t).exists()


def pending_cases(cfg, shard: int = 0, n_shards: int = 1, max_attempts: int = 2,
                  verbose: bool = False) -> list[dict]:
    todo, waiting = pending_and_waiting(cfg, shard, n_shards, max_attempts)
    if verbose and waiting:
        print(f"[shard {shard}/{n_shards}] {waiting} cases wait for their terrain crop (run global-fetch)",
              flush=True)
    return todo


def pending_and_waiting(cfg, shard: int = 0, n_shards: int = 1,
                        max_attempts: int = 2) -> tuple[list[dict], int]:
    """(cases of this shard ready to run, number of cases still waiting for a terrain crop)."""
    cases = read_cases(cfg.path(cfg["output"]["cases_file"]))
    done, status = _listing(cfg)
    todo, waiting = [], 0
    for i, c in enumerate(cases):
        if i % n_shards != shard:
            continue
        if (c["split"], c["case_id"]) in done:
            continue
        if c["case_id"] in status:
            st = _read_status(cfg, c["case_id"])
            if st and st.get("status") == "failed" and st.get("attempts", 0) >= max_attempts:
                continue
        if not terrain_ready(cfg, c):
            waiting += 1
            continue
        todo.append(c)
    return todo, waiting


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
    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, wait

    from .crops import CropCache
    from .dem import region_dem_path, regions
    crops = CropCache(cfg)
    dems: dict[str, MasterDEM] = {}
    dem_lock = threading.Lock()

    def dem_for(c):
        if c.get("terrain"):
            return crops(c["terrain"])
        with dem_lock:
            name = c.get("region") or regions(cfg)[0]["name"]
            if name not in dems:
                dems[name] = MasterDEM.open(region_dem_path(cfg, name))
            return dems[name]

    concurrent = int(concurrent or cfg["solver"]["concurrent_cases"])
    from . import affinity
    slots = None
    if int(cfg["solver"]["procs_per_case"]) == 1 and affinity.pinning_enabled(cfg, concurrent):
        slots = queue.Queue()
        for slot in affinity.core_slots()[:concurrent]:
            slots.put(slot)
        print(f"[worker {shard}/{n_shards}] pinning each case to one core: {affinity.describe()}", flush=True)
    deadline = time.time() + stop_after_hours * 3600 if stop_after_hours else None
    stats = dict(done=0, failed=0)
    t_start = time.time()

    def job(c):
        if deadline and time.time() > deadline:
            return None
        dem = dem_for(c)
        if slots is None:
            return run_case(cfg, geom, dem, c, topo_dir, env)
        cpu, node = slots.get()
        try:
            return run_case(cfg, geom, dem, c, topo_dir, env, prefix=affinity.pin_prefix(cpu, node))
        finally:
            slots.put((cpu, node))

    # The case list is re-read whenever this worker runs out of cases (at most every
    # refresh_minutes), so cases whose terrain crop appears later (global-fetch running in
    # parallel) are picked up without restarting the job.
    refresh_s = float(cfg["solver"].get("refresh_minutes", 5)) * 60
    attempted: set[str] = set()
    backlog: deque = deque()
    state = dict(waiting=0, last=0.0)

    def refresh():
        todo, waiting = pending_and_waiting(cfg, shard, n_shards)
        backlog.clear()
        backlog.extend(c for c in todo if c["case_id"] not in attempted)
        state.update(waiting=waiting, last=time.time())
        print(f"[worker {shard}/{n_shards}] {len(backlog)} pending cases, {waiting} waiting for terrain, "
              f"{concurrent} at a time", flush=True)

    refresh()
    started = 0
    with ThreadPoolExecutor(max_workers=concurrent) as ex:
        inflight = set()
        while True:
            stop_new = bool((deadline and time.time() > deadline) or (limit and started >= limit))
            while not stop_new and backlog and len(inflight) < concurrent:
                c = backlog.popleft()
                attempted.add(c["case_id"])
                started += 1
                inflight.add(ex.submit(job, c))
                stop_new = bool(limit and started >= limit)
            if not stop_new and not backlog and time.time() - state["last"] > refresh_s:
                refresh()
                continue
            if not inflight:
                if stop_new or (not backlog and state["waiting"] == 0):
                    break
                time.sleep(min(60.0, refresh_s))
                continue
            finished, _ = wait(inflight, timeout=60, return_when=FIRST_COMPLETED)
            for fut in finished:
                inflight.discard(fut)
                info = fut.result()
                if info is None:
                    continue
                stats["done" if info["status"] == "done" else "failed"] += 1
                n_fin = stats["done"] + stats["failed"]
                rate = n_fin / max(time.time() - t_start, 1.0) * 3600
                print(f"[{n_fin} | {len(backlog)} queued] {info['case_id']} {info['status']:6s} "
                      f"iters={info.get('iterations')} conv={info.get('converged')} "
                      f"t={info.get('wall_seconds')}s  {rate:.0f} cases/h"
                      + (f"  ERR {info.get('error', '')[:200]}" if info["status"] != "done" else ""),
                      flush=True)
    print(f"[worker {shard}] finished: {stats['done']} done, {stats['failed']} failed", flush=True)
    shutil.rmtree(work, ignore_errors=True)


def status_summary(cfg, by: str | None = None) -> dict:
    """Progress of the case list. `by` (e.g. stratum, group, region, split) adds a breakdown
    with finished cases, failures, convergence rate and mean case time per group."""
    cases = read_cases(cfg.path(cfg["output"]["cases_file"]))
    done, status = _listing(cfg)
    out = {"total": len(cases), "done": 0, "failed": 0, "pending": 0, "waiting_terrain": 0,
           "by_split": {}, "converged": 0}
    secs = []
    groups: dict[str, dict] = {}
    for c in cases:
        sp = out["by_split"].setdefault(c["split"], {"total": 0, "done": 0})
        sp["total"] += 1
        g = None
        if by:
            g = groups.setdefault(str(c.get(by)), {"total": 0, "done": 0, "failed": 0, "converged": 0,
                                                    "secs": [], "slope": []})
            g["total"] += 1
        st = _read_status(cfg, c["case_id"]) if c["case_id"] in status else None
        if (c["split"], c["case_id"]) in done:
            out["done"] += 1
            sp["done"] += 1
            if st:
                secs.append(st.get("wall_seconds", 0))
                out["converged"] += bool(st.get("converged"))
                if g is not None:
                    g["secs"].append(st.get("wall_seconds", 0))
                    g["converged"] += bool(st.get("converged"))
                    if st.get("max_slope_deg") is not None:
                        g["slope"].append(st["max_slope_deg"])
            if g is not None:
                g["done"] += 1
        elif st and st.get("status") == "failed":
            out["failed"] += 1
            if g is not None:
                g["failed"] += 1
        elif not terrain_ready(cfg, c):
            out["waiting_terrain"] += 1
        else:
            out["pending"] += 1
    if secs:
        out["mean_case_minutes"] = round(float(np.mean(secs)) / 60, 1)
    if by:
        table = {}
        for k, g in sorted(groups.items()):
            row = {"total": g["total"], "done": g["done"], "failed": g["failed"]}
            if g["done"]:
                row["converged_frac"] = round(g["converged"] / g["done"], 3)
            if g["secs"]:
                row["mean_minutes"] = round(float(np.mean(g["secs"])) / 60, 1)
                row["p90_minutes"] = round(float(np.percentile(g["secs"], 90)) / 60, 1)
            if g["slope"]:
                row["max_slope_p90_deg"] = round(float(np.percentile(g["slope"], 90)), 1)
            table[k] = row
        out[f"by_{by}"] = table
    return out


def failure_summary(cfg, n: int = 10) -> list[dict]:
    """Most common error messages among failed cases."""
    _, status = _listing(cfg)
    from collections import Counter
    errs, examples = Counter(), {}
    for cid in status:
        st = _read_status(cfg, cid)
        if st and st.get("status") == "failed":
            key = (st.get("error") or "").splitlines()[0][:160]
            errs[key] += 1
            examples.setdefault(key, cid)
    return [{"error": k, "count": v, "example": examples[k]} for k, v in errs.most_common(n)]


def reorder_pending(cfg, weights: dict[str, float], key: str = "stratum", seed: int = 0) -> dict:
    """Re-prioritise the pending part of the case list.

    Finished and failed cases keep their place at the front; pending cases are put in a
    weighted random order (weight per value of `key`, default 1), so groups with a higher
    weight come up earlier. Case ids and files do not change. Run it only while no worker
    is running: shards are assigned by position in the list.
    """
    import json as _json
    path = cfg.path(cfg["output"]["cases_file"])
    cases = read_cases(path)
    done, status = _listing(cfg)
    finished, pending = [], []
    for c in cases:
        if (c["split"], c["case_id"]) in done or c["case_id"] in status:
            finished.append(c)
        else:
            pending.append(c)
    rng = np.random.default_rng(seed)
    w = np.array([max(float(weights.get(str(c.get(key)), 1.0)), 1e-9) for c in pending])
    # Exponential race: sorting by E/w gives a weighted random permutation.
    order = np.argsort(rng.exponential(size=len(pending)) / w, kind="stable")
    new = finished + [pending[i] for i in order]
    backup = path.with_name(path.name + time.strftime(".bak-%Y%m%d-%H%M%S"))
    shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for c in new:
            f.write(_json.dumps(c) + "\n")
    os.replace(tmp, path)
    return {"finished_or_failed": len(finished), "pending": len(pending), "backup": str(backup)}
