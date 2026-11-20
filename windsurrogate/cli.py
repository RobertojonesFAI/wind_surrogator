"""Command line entry point: python -m windsurrogate <command> [options]."""
from __future__ import annotations

import argparse
import json
import sys

from .config import load_config, save_config


def _cfg(args):
    return load_config(args.config, args.set)


def cmd_prepare(args):
    from .dem import (MasterDEM, bbox_in_crs, build_master_dem, expand_bbox, region_dem_path,
                      regions, synthetic_dem)
    from .geometry import CaseGeometry
    from .sampling import read_cases, sample_cases, write_cases

    cfg = _cfg(args)
    cfg.root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.root / "config_used.yaml")
    geom = CaseGeometry.from_config(cfg)
    print(geom.summary())
    regs = regions(cfg)
    dems = {}
    for reg in regs:
        p = region_dem_path(cfg, reg["name"])
        if args.synthetic and not p.exists():
            l, b, r, t = bbox_in_crs(expand_bbox(reg["bbox"], cfg["dem"].get("margin_km", 13)), reg["crs"])
            size = max(r - l, t - b)
            synthetic_dem(p, size_km=size / 1000, res=cfg["dem"]["resolution"], crs=reg["crs"],
                          origin=(0.5 * (l + r) - size / 2, 0.5 * (b + t) + size / 2),
                          seed=len(dems))
            print(f"synthetic DEM written to {p}")
        elif not args.synthetic:
            build_master_dem(cfg, reg)
        dems[reg["name"]] = MasterDEM.open(p)
    cpath = cfg.path(cfg["output"]["cases_file"])
    if cpath.exists() and not args.resample:
        cases = read_cases(cpath)
        print(f"{len(cases)} cases already listed in {cpath}")
    else:
        cases = sample_cases(cfg, regs, dems)
        write_cases(cpath, cases)
        print(f"sampled {len(cases)} cases -> {cpath}")
    counts = {}
    for c in cases:
        key = (c.get("region", regs[0]["name"]), c["split"])
        counts[key] = counts.get(key, 0) + 1
    for (reg, split), n in sorted(counts.items()):
        print(f"  {reg:12s} {split:5s} {n}")


def cmd_run(args):
    from .runner import worker
    cfg = _cfg(args)
    worker(cfg, shard=args.shard, n_shards=args.num_shards, concurrent=args.concurrent,
           limit=args.limit, stop_after_hours=args.hours)


def cmd_status(args):
    from .runner import failure_summary, status_summary
    cfg = _cfg(args)
    print(json.dumps(status_summary(cfg, by=args.by), indent=1))
    if args.failures:
        print(json.dumps(failure_summary(cfg, args.failures), indent=1))


def cmd_global_check(args):
    from .sources import check_sources
    res = check_sources(_cfg(args))
    bad = [r["source"] for r in res if not r["ok"]]
    print("all sources reachable" if not bad else f"not reachable: {', '.join(bad)}")


def cmd_global_descriptors(args):
    from .descriptors import compute, merge
    cfg = _cfg(args)
    cfg.root.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge(cfg)
    else:
        compute(cfg, threads=args.threads, limit=args.limit)


def cmd_global_sample(args):
    from .globalsample import run_sampling
    cfg = _cfg(args)
    cfg.root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, cfg.root / "config_used.yaml")
    run_sampling(cfg, resample=args.resample)


def cmd_global_fetch(args):
    from .crops import fetch
    cfg = _cfg(args)
    fetch(cfg, shard=args.shard, n_shards=args.num_shards, threads=args.threads, limit=args.limit,
          pilot_only=args.pilot_only, clean_cache=args.clean_cache)


def cmd_reorder(args):
    from .runner import reorder_pending
    cfg = _cfg(args)
    with open(args.weights) as f:
        weights = json.load(f)
    print(json.dumps(reorder_pending(cfg, weights, key=args.key, seed=args.seed), indent=1))


def cmd_train(args):
    from .ml.train import train
    train(_cfg(args), resume=not args.fresh, max_steps=args.max_steps)


def cmd_predict(args):
    from .ml.infer import predict_dem_to_geotiff
    cfg = _cfg(args)
    predict_dem_to_geotiff(cfg, args.checkpoint, args.dem, args.out, theta=args.theta,
                           z0=args.z0, uref=args.uref, bbox=args.bbox, device=args.device)


def cmd_bench(args):
    from .bench import bench
    bench(_cfg(args), copies=[int(x) for x in args.copies.split(",")], iters=args.iters,
          modes=args.modes.split(","), case_index=args.case_index, work_dir=args.work_dir)


def cmd_evaluate(args):
    from .ml.evaluate import evaluate_checkpoint
    evaluate_checkpoint(_cfg(args), args.checkpoint, split=args.split, out_dir=args.out,
                        n_figures=args.figures, device=args.device)


def cmd_plot(args):
    from .plotting import plot_sample
    cfg = _cfg(args)
    print(plot_sample(args.sample, args.out, dx=cfg["geometry"]["dx"]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="windsurrogate")
    ap.add_argument("--config", action="append", default=None,
                    help="YAML file merged over config/default.yaml (repeatable, merged in order)")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set geometry.dx=60")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download/reproject DEM and sample cases")
    p.add_argument("--synthetic", type=float, default=None, help="use a synthetic DEM of this size (km)")
    p.add_argument("--resample", action="store_true", help="overwrite an existing case list")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="run pending CFD cases (resumable)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--concurrent", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--hours", type=float, default=None, help="stop starting new cases after this many hours")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="dataset progress")
    p.add_argument("--by", default=None, help="breakdown by a case field, e.g. stratum, group, region")
    p.add_argument("--failures", type=int, default=0, help="also list the N most common errors")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("global-check", help="probe the data sources (Copernicus, WorldCover, FABDEM)")
    p.set_defaults(func=cmd_global_check)

    p = sub.add_parser("global-descriptors", help="terrain statistics on a worldwide ~10 km grid (GLO-90)")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="only this many tiles (testing)")
    p.add_argument("--merge-only", action="store_true", help="only merge the per-tile files")
    p.set_defaults(func=cmd_global_descriptors)

    p = sub.add_parser("global-sample", help="stratified worldwide locations and case queue")
    p.add_argument("--resample", action="store_true", help="overwrite an existing case list")
    p.set_defaults(func=cmd_global_sample)

    p = sub.add_parser("global-fetch", help="terrain + land cover crop for every location (resumable)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--pilot-only", action="store_true", help="only the locations of pilot cases")
    p.add_argument("--clean-cache", action="store_true", help="delete the FABDEM tile cache when done")
    p.set_defaults(func=cmd_global_fetch)

    p = sub.add_parser("reorder", help="re-prioritise pending cases with weights per group (no running jobs!)")
    p.add_argument("--weights", required=True, help='JSON file, e.g. {"r4": 2.0, "hard": 3.0}')
    p.add_argument("--key", default="stratum", help="case field the weights refer to")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_reorder)

    p = sub.add_parser("train", help="train the U-Net surrogate")
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--max-steps", type=int, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="predict wind maps over a large DEM")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dem", required=True, help="GeoTIFF in a metric CRS (e.g. the master DEM)")
    p.add_argument("--out", required=True)
    p.add_argument("--theta", type=float, required=True, help="wind direction FROM, degrees")
    p.add_argument("--z0", type=float, default=0.1)
    p.add_argument("--uref", type=float, default=None, help="m/s at flow.Zref; default = flow.Uref")
    p.add_argument("--bbox", type=float, nargs=4, default=None, help="xmin ymin xmax ymax in DEM CRS")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("bench", help="measure solver speed for n simultaneous cases, pinned or not")
    p.add_argument("--copies", default="1,24,48", help="comma-separated numbers of simultaneous cases")
    p.add_argument("--iters", type=int, default=30, help="iterations per run")
    p.add_argument("--modes", default="pin,nopin", help="pin, nopin or both")
    p.add_argument("--case-index", type=int, default=0, help="which case of cases.jsonl to copy")
    p.add_argument("--work-dir", default=None, help="scratch folder (default: output.work_dir)")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("evaluate", help="metrics and CFD-vs-surrogate figures on a split (default: test)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default=None, help="output folder (default: <checkpoint dir>/eval_<split>)")
    p.add_argument("--figures", type=int, default=6, help="number of comparison figures")
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("plot-sample", help="quick-look PNG of one CFD sample")
    p.add_argument("sample")
    p.add_argument("out")
    p.set_defaults(func=cmd_plot)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
