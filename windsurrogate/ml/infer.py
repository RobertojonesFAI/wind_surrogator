"""Tiled inference over arbitrarily large DEMs.

For a wind direction theta the DEM is resampled onto a wind-aligned grid,
covered with overlapping core tiles (each with its upwind/lateral context
window, exactly as in training), predictions are blended with a smooth
window, rotated back to east/north and resampled to the output grid.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..config import Config
from ..derived import derived_fields
from ..dem import MasterDEM
from ..foamcase import inflow_profile
from ..geometry import CaseGeometry
from ..terrain import SmoothedSampler, frame_axes, vectors_to_map
from .data import denormalise, input_channels


def load_model(ckpt_path, device=None):
    import torch
    from .model import build_model
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(ck["config"])
    model = build_model(cfg, len(cfg["output"]["heights_agl"]))
    model.load_state_dict(ck["model"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model.to(dev).eval(), cfg, dev


def predict_rotated(model, device, geom: CaseGeometry, h_rot: np.ndarray, z0: float,
                    core_rows: int, core_cols: int, stride: int, batch: int = 8) -> np.ndarray:
    """Predict normalised fields on the wind-aligned core grid.

    h_rot is terrain (m) on the padded rotated grid: the core region of size
    (core_rows, core_cols) starts at row input_lateral_px and col input_upwind_px.
    (core_rows - N) and (core_cols - N) must be multiples of `stride`.
    Returns (4H, core_rows, core_cols).
    """
    import torch
    N, L, U, D = geom.core_px, geom.input_lateral_px, geom.input_upwind_px, geom.input_downwind_px
    rs = list(range(0, core_rows - N + 1, stride))
    cs = list(range(0, core_cols - N + 1, stride))
    win1 = np.sin(np.pi * (np.arange(N) + 0.5) / N) ** 2 + 1e-3
    win = np.outer(win1, win1).astype(np.float32)
    acc, wsum = None, np.zeros((core_rows, core_cols), np.float32)
    core_slice = geom.core_slice_in_input
    jobs = [(r, c) for r in rs for c in cs]
    for b0 in range(0, len(jobs), batch):
        chunk = jobs[b0:b0 + batch]
        xs = [input_channels(h_rot[r: r + N + 2 * L, c: c + N + U + D], geom.dx, z0, core_slice)
              for r, c in chunk]
        x = torch.from_numpy(np.stack(xs)).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = model(x).float()[:, :, core_slice[0], core_slice[1]].cpu().numpy()
        if acc is None:
            acc = np.zeros((p.shape[1], core_rows, core_cols), np.float32)
        for (r, c), pi in zip(chunk, p):
            acc[:, r:r + N, c:c + N] += pi * win
            wsum[r:r + N, c:c + N] += win
    return acc / np.maximum(wsum, 1e-6)


def predict_map(model, cfg, device, dem: MasterDEM, theta: float, z0: float = 0.1, uref: float | None = None,
                bbox=None, overlap: float = 0.25, batch: int = 8, anchor=None, sampler=None) -> dict:
    """Predict wind fields on a north-up grid with pixel size geom.dx.

    bbox = (xmin, ymin, xmax, ymax) in the DEM CRS. By default the DEM bounds shrunk
    by the input context so every tile sees real terrain. Tiles sit on a global
    wind-aligned lattice anchored at `anchor`, so predictions do not depend on how
    a large area is split into chunks.
    """
    geom = CaseGeometry.from_config(cfg)
    dx, N = geom.dx, geom.core_px
    margin = max(geom.input_upwind_px, geom.input_lateral_px, geom.input_downwind_px) * dx
    if bbox is None:
        l, b, r, t = dem.bounds
        bbox = (l + margin, b + margin, r - margin, t - margin)
    xmin, ymin, xmax, ymax = bbox
    ncol, nrow = int(round((xmax - xmin) / dx)), int(round((ymax - ymin) / dx))
    E = xmin + dx * (np.arange(ncol) + 0.5)
    Nn = ymax - dx * (np.arange(nrow) + 0.5)
    EE, NN = np.meshgrid(E, Nn)
    ec, nc = anchor if anchor is not None else (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
    ex, ey = frame_axes(theta)
    xr = (EE - ec) * ex[0] + (NN - nc) * ex[1]
    yr = (EE - ec) * ey[0] + (NN - nc) * ey[1]
    S = max(1, int(round(N * (1 - overlap))))
    span = S * dx
    m0, m1 = math.floor((xr.min() - N * dx) / span), math.ceil(xr.max() / span)
    n0, n1 = math.floor((yr.min() - N * dx) / span), math.ceil(yr.max() / span)
    xr0, yr0 = m0 * span, n0 * span
    core_cols, core_rows = (m1 - m0) * S + N, (n1 - n0) * S + N
    # Padded rotated terrain grid (rows along +y', cols along +x').
    L, U, D = geom.input_lateral_px, geom.input_upwind_px, geom.input_downwind_px
    cols = np.arange(-U, core_cols + D)
    rows = np.arange(-L, core_rows + L)
    XR, YR = np.meshgrid(xr0 + dx * (cols + 0.5), yr0 + dx * (rows + 0.5))
    Eg = ec + XR * ex[0] + YR * ey[0]
    Ng = nc + XR * ex[1] + YR * ey[1]
    sampler = sampler or SmoothedSampler(dem, cell_sizes=(dx, 2 * dx))
    h_rot = sampler.sample(Eg, Ng, cell=dx).astype(np.float32)

    pred = predict_rotated(model, device, geom, h_rot, z0, core_rows, core_cols, S, batch)
    # Back to the north-up output grid.
    fr = (yr - yr0) / dx - 0.5
    fc = (xr - xr0) / dx - 0.5
    pm = np.stack([ndimage.map_coordinates(ch, [fr, fc], order=1, mode="nearest") for ch in pred])
    heights = np.asarray(cfg["output"]["heights_agl"], float)
    U_in, k_in, _, _ = inflow_profile(cfg, z0, heights)
    s = 1.0 if uref is None else uref / cfg["flow"]["Uref"]
    Uw, k = denormalise(pm, U_in * s, k_in * s * s)                  # (H,3,r,c), (H,r,c)
    u_e, v_n = vectors_to_map(Uw[:, 0], Uw[:, 1], theta)
    terrain = sampler.sample(EE, NN, cell=dx).astype(np.float32)
    fields = dict(u_east=u_e, v_north=v_n, **derived_fields(u_e, v_n, Uw[:, 2], k, Uw[:, 0], Uw[:, 1],
                                                            U_in * s, dx))
    return dict(fields=fields, heights=heights, transform=(xmin, dx, 0.0, ymax, 0.0, -dx), crs=dem.crs,
                terrain=terrain, theta=theta, z0=z0, uref=(uref or cfg["flow"]["Uref"]), zref=cfg["flow"]["Zref"])


FIELD_ORDER = ("u_east", "v_north", "w", "tke", "speed", "speedup", "dir_from", "deflection", "ti",
               "convergence", "recirculation", "shadow", "compression")


def band_names(heights) -> list[str]:
    return [f"{f}_{int(h)}m" for f in FIELD_ORDER for h in heights] + ["terrain_m"]


def predict_dem_to_geotiff(cfg, ckpt, dem_path, out, theta, z0=0.1, uref=None, bbox=None, device=None,
                           chunk_km: float = 40.0):
    """Predict over bbox (default: whole DEM minus context margin) in chunks, into one GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    model, mcfg, dev = load_model(ckpt, device)
    geom = CaseGeometry.from_config(mcfg)
    dx = geom.dx
    dem = MasterDEM.open(dem_path)
    margin = max(geom.input_upwind_px, geom.input_lateral_px, geom.input_downwind_px) * dx
    if bbox is None:
        l, b, r, t = dem.bounds
        bbox = (l + margin, b + margin, r - margin, t - margin)
    xmin, ymin, xmax, ymax = bbox
    ncol, nrow = int((xmax - xmin) // dx), int((ymax - ymin) // dx)
    heights = np.asarray(mcfg["output"]["heights_agl"], float)
    names = band_names(heights)
    step = max(geom.core_px, int(chunk_km * 1000 // dx))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name and rename at the end, so an interrupted run never leaves
    # a half-written file that looks finished (callers can skip outputs that exist).
    part = out.with_name(out.name + ".part")
    with rasterio.open(part, "w", driver="GTiff", height=nrow, width=ncol, count=len(names), dtype="float32",
                       crs=dem.crs, transform=from_origin(xmin, ymax, dx, dx), compress="deflate",
                       predictor=3, tiled=True, BIGTIFF="IF_SAFER") as ds:
        for i, nm in enumerate(names, start=1):
            ds.set_band_description(i, nm)
        anchor = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
        sampler = SmoothedSampler(dem, cell_sizes=(dx, 2 * dx))
        pad = 2  # extra pixels so gradients (convergence) have no seams between chunks
        for r0 in range(0, nrow, step):
            for c0 in range(0, ncol, step):
                nr, nc = min(step, nrow - r0), min(step, ncol - c0)
                cb = (xmin + (c0 - pad) * dx, ymax - (r0 + nr + pad) * dx,
                      xmin + (c0 + nc + pad) * dx, ymax - (r0 - pad) * dx)
                res = predict_map(model, mcfg, dev, dem, theta, z0=z0, uref=uref, bbox=cb,
                                  anchor=anchor, sampler=sampler)
                bands = [res["fields"][f][i] for f in FIELD_ORDER for i in range(len(heights))] + [res["terrain"]]
                for i, bnd in enumerate(bands, start=1):
                    ds.write(bnd[pad:pad + nr, pad:pad + nc].astype(np.float32), i,
                             window=Window(c0, r0, nc, nr))
                print(f"  chunk rows {r0}:{r0 + nr} cols {c0}:{c0 + nc} done", flush=True)
        ds.update_tags(theta_from_deg=theta, z0=z0, uref=uref or mcfg["flow"]["Uref"], zref=mcfg["flow"]["Zref"])
    part.replace(out)
    meta = dict(bands=names, theta=theta, z0=z0, uref=uref or mcfg["flow"]["Uref"], zref=mcfg["flow"]["Zref"],
                heights=heights.tolist(), crs=dem.crs, dx=dx)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}")
    return out
