"""Evaluate a trained checkpoint on one dataset split (default: the held-out test blocks).

Outputs, in `out_dir`:
  metrics_<split>.json   aggregate metrics (same definitions as during training), plus
                         breakdowns by region and by terrain relief
  cases_<split>.csv      one row per case
  figures/<case>.png     CFD vs surrogate side by side for a few cases (best, median, worst)
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from ..geometry import CaseGeometry
from .data import WindDataset, list_samples

FLAT = np.array([1.0, 0.0, 0.0, 1.0], np.float32).reshape(1, 4, 1, 1)   # "flat terrain" answer
RELIEF_BINS = [(0, 500), (500, 1000), (1000, 1e9)]


def _case_errors(p: np.ndarray, t: np.ndarray):
    """p, t: (H, 4, N, N) normalised fields. Returns per-height squared errors and recirculation counts."""
    se = ((p - t) ** 2).mean(axis=(-1, -2))                 # (H, 4)
    se_flat = ((FLAT - t) ** 2).mean(axis=(-1, -2))         # (H, 4)
    pr, tr = p[:, 0] < 0, t[:, 0] < 0
    inter = (pr & tr).sum(axis=(-1, -2)).astype(float)      # (H,)
    union = (pr | tr).sum(axis=(-1, -2)).astype(float)
    return se, se_flat, inter, union


def _summary(se, se_flat, inter, union, n, heights) -> dict:
    """Aggregate metrics from summed per-case errors (identical to ml.train.evaluate)."""
    se, se_flat = se / max(n, 1), se_flat / max(n, 1)
    loss, bl = float(se.mean()), float(se_flat.mean())
    out = {"n_cases": int(n), "loss": loss, "baseline_loss": bl,
           "skill": 1.0 - loss / bl if bl > 0 else float("nan")}
    rmse = np.sqrt(se)
    for i, h in enumerate(heights):
        h = int(h)
        out[f"rmse_u_{h}m"] = float(rmse[i, 0])
        out[f"rmse_v_{h}m"] = float(rmse[i, 1])
        out[f"rmse_w_{h}m"] = float(rmse[i, 2])
        out[f"rmse_k_{h}m"] = float(rmse[i, 3])
        out[f"recirc_iou_{h}m"] = float(inter[i] / union[i]) if union[i] > 0 else float("nan")
    return out


def _predict(model, device, xs):
    import torch
    x = torch.from_numpy(np.stack(xs)).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        return model(x).float().cpu().numpy()


def evaluate_checkpoint(cfg, checkpoint, split: str = "test", out_dir=None, n_figures: int = 6,
                        device=None, batch: int = 8) -> dict:
    from .infer import load_model

    model, ck_cfg, dev = load_model(checkpoint, device)
    geom = CaseGeometry.from_config(ck_cfg)
    heights = [float(h) for h in ck_cfg["output"]["heights_agl"]]
    H = len(heights)
    for key in ("dx", "core_px", "input_upwind_px", "input_downwind_px", "input_lateral_px"):
        if float(cfg["geometry"][key]) != float(ck_cfg["geometry"][key]):
            raise ValueError(f"geometry.{key} differs between the config ({cfg['geometry'][key]}) and the "
                             f"checkpoint ({ck_cfg['geometry'][key]}); evaluate with the training geometry")

    ddir = cfg.path(cfg["output"]["dataset_dir"])
    files = list_samples(ddir, split, max_probe_change=cfg["train"].get("max_probe_change"))
    if not files:
        raise RuntimeError(f"no samples in {ddir}/{split}")
    out_dir = Path(out_dir) if out_dir else Path(checkpoint).resolve().parent / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"evaluating {Path(checkpoint).name} on {len(files)} {split} cases ({dev})", flush=True)

    ds = WindDataset(files, geom)
    rs, cs = geom.core_slice_in_input
    rows, acc = [], {}

    def add(key, se, sf, it, un):
        a = acc.setdefault(key, [np.zeros((H, 4)), np.zeros((H, 4)), np.zeros(H), np.zeros(H), 0])
        a[0] += se; a[1] += sf; a[2] += it; a[3] += un; a[4] += 1

    for b0 in range(0, len(files), batch):
        items = [ds.load(f) for f in files[b0:b0 + batch]]
        preds = _predict(model, dev, [it[0] for it in items])[:, :, rs, cs]
        for f, (_, y, meta), p in zip(files[b0:b0 + batch], items, preds):
            p = p.reshape(H, 4, *p.shape[-2:])
            t = y.reshape(H, 4, *y.shape[-2:])
            se, sf, it, un = _case_errors(p, t)
            relief = float(meta.get("relief") or 0.0)
            region = meta.get("region") or "-"
            rbin = next(f"{lo:g}-{hi:g} m" if hi < 1e8 else f">{lo:g} m" for lo, hi in RELIEF_BINS if lo <= relief < hi)
            add("all", se, sf, it, un)
            add(("region", region), se, sf, it, un)
            add(("relief", rbin), se, sf, it, un)
            row = dict(case_id=meta.get("case_id", f.stem), file=f.name, region=region, relief_m=relief,
                       theta=meta.get("theta"), z0=meta.get("z0"), converged=meta.get("converged"),
                       averaged=bool(meta.get("averaged", False)), loss=float(se.mean()),
                       baseline_loss=float(sf.mean()))
            row["skill"] = 1.0 - row["loss"] / row["baseline_loss"] if row["baseline_loss"] > 0 else float("nan")
            for i, h in enumerate(heights):
                row[f"rmse_u_{int(h)}m"] = float(math.sqrt(se[i, 0]))
                row[f"recirc_iou_{int(h)}m"] = float(it[i] / un[i]) if un[i] > 0 else float("nan")
            rows.append(row)

    metrics = {"checkpoint": str(checkpoint), "split": split, **_summary(*acc["all"][:4], acc["all"][4], heights)}
    for kind in ("region", "relief"):
        metrics[f"by_{kind}"] = {k[1]: _summary(*v[:4], v[4], heights)
                                 for k, v in sorted(acc.items(), key=lambda kv: str(kv[0])) if k[0] == kind}

    (out_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=1))
    with open(out_dir / f"cases_{split}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Figures: spread over the skill ranking (worst ... median ... best).
    if n_figures > 0:
        order = sorted(range(len(rows)), key=lambda i: rows[i]["skill"])
        k = min(n_figures, len(order))
        picks = sorted({order[round(j * (len(order) - 1) / max(k - 1, 1))] for j in range(k)})
        fig_dir = out_dir / "figures"
        for i in picks:
            x, y, meta = ds.load(files[i])
            p = _predict(model, dev, [x])[0][:, rs, cs]
            with np.load(files[i]) as z:
                terrain, U_in = z["terrain_core"], z["U_inflow"]
            plot_comparison(fig_dir / f"{rows[i]['case_id']}.png", p, y, terrain, U_in, heights,
                            geom.dx, meta, rows[i])
        print(f"figures: {fig_dir}")

    _print(metrics, heights)
    print(f"written: {out_dir / f'metrics_{split}.json'}, {out_dir / f'cases_{split}.csv'}")
    return metrics


def _print(m: dict, heights):
    print(f"{m['split']}: {m['n_cases']} cases  loss {m['loss']:.4f}  flat-terrain loss {m['baseline_loss']:.4f}  "
          f"skill {m['skill']:.3f}")
    print("  height   rmse_u   rmse_w   rmse_k   recirc_iou")
    for h in heights:
        h = int(h)
        print(f"  {h:5d} m  {m[f'rmse_u_{h}m']:.3f}    {m[f'rmse_w_{h}m']:.3f}    {m[f'rmse_k_{h}m']:.3f}    "
              f"{m[f'recirc_iou_{h}m']:.3f}")
    for kind in ("region", "relief"):
        for name, s in m[f"by_{kind}"].items():
            print(f"  {kind} {name:12s} n={s['n_cases']:4d}  skill {s['skill']:.3f}  "
                  f"rmse_u_{int(heights[0])}m {s[f'rmse_u_{int(heights[0])}m']:.3f}  "
                  f"recirc_iou_{int(heights[0])}m {s[f'recirc_iou_{int(heights[0])}m']:.3f}")


def plot_comparison(out_png, pred, target, terrain, U_in, heights, dx, meta, row):
    """CFD (target) vs surrogate (pred), both (4H, N, N) normalised, on the wind-aligned core."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    H = len(heights)
    p, t = pred.reshape(H, 4, *pred.shape[-2:]), target.reshape(H, 4, *target.shape[-2:])
    i0 = 0
    iw = heights.index(50.0) if 50.0 in heights else min(1, H - 1)
    N = t.shape[-1]
    ext = [-N * dx / 2000, N * dx / 2000, -N * dx / 2000, N * dx / 2000]
    grid = np.linspace(ext[0], ext[1], N)

    sp_t, sp_p = np.hypot(t[i0, 0], t[i0, 1]), np.hypot(p[i0, 0], p[i0, 1])
    w_t, w_p = t[iw, 2] * U_in[iw], p[iw, 2] * U_in[iw]
    rt, rp = t[i0, 0] < 0, p[i0, 0] < 0
    overlap = np.zeros((N, N))
    overlap[rt & rp], overlap[rt & ~rp], overlap[~rt & rp] = 1, 2, 3

    fig, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    panels = [
        (sp_t, "RdBu_r", (0, 2), f"CFD speed-up {int(heights[i0])} m"),
        (sp_p, "RdBu_r", (0, 2), f"U-Net speed-up {int(heights[i0])} m"),
        (sp_p - sp_t, "PiYG", (-0.5, 0.5), "U-Net − CFD"),
        (w_t, "PuOr", (-2, 2), f"CFD w {int(heights[iw])} m (m/s)"),
        (w_p, "PuOr", (-2, 2), f"U-Net w {int(heights[iw])} m (m/s)"),
        (w_p - w_t, "PiYG", (-1, 1), "U-Net − CFD (m/s)"),
        (rt.astype(float), "Reds", (0, 1), f"CFD recirculation {int(heights[i0])} m (u' < 0)"),
        (rp.astype(float), "Reds", (0, 1), f"U-Net recirculation {int(heights[i0])} m"),
    ]
    for ax, (img, cmap, (lo, hi), title) in zip(axes.ravel(), panels):
        im = ax.imshow(img, origin="lower", extent=ext, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[2, 2]
    cmap = ListedColormap(["white", "tab:green", "tab:blue", "tab:red"])
    ax.imshow(overlap, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_title("green = both, blue = CFD only (missed), red = U-Net only")
    for ax in axes.ravel():
        ax.contour(grid, grid, terrain, 12, colors="k", linewidths=0.4, alpha=0.6)
        ax.set_xlabel("x' downwind (km)")
        ax.set_ylabel("y' (km)")
    iou = row.get(f"recirc_iou_{int(heights[i0])}m", float("nan"))
    fig.suptitle(f"{row['case_id']}  {row['region']}  theta={meta.get('theta')}°  z0={meta.get('z0')} m  "
                 f"relief={row['relief_m']:.0f} m  |  skill {row['skill']:.3f}  "
                 f"rmse_u_{int(heights[i0])}m {row[f'rmse_u_{int(heights[i0])}m']:.3f}  "
                 f"recirc IoU {iou:.2f}   (wind →)")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png
