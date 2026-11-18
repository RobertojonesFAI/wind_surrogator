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
    from .runner import status_summary
    print(json.dumps(status_summary(_cfg(args)), indent=1))


def cmd_train(args):
    from .ml.train import train
    train(_cfg(args), resume=not args.fresh, max_steps=args.max_steps)


def cmd_predict(args):
    from .ml.infer import predict_dem_to_geotiff
    cfg = _cfg(args)
    predict_dem_to_geotiff(cfg, args.checkpoint, args.dem, args.out, theta=args.theta,
                           z0=args.z0, uref=args.uref, bbox=args.bbox, device=args.device)


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
    ap.add_argument("--config", default=None, help="YAML file merged over config/default.yaml")
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
    p.set_defaults(func=cmd_status)

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
