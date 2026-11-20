"""Global terrain descriptors on a ~10 km grid, used to decide where to sample cases.

For every Copernicus GLO-90 tile (1 x 1 degree) the DEM is read at reduced resolution
(COG overviews, ~360 m with decimate=4) and summarised on cells of 1/cells_per_deg
degree in latitude; in longitude the cells widen with latitude so they stay ~11 km wide.

Per cell: valid fraction, sea-like fraction (|h| < 1 m and flat), mean elevation,
relief (P98 - P2), standard deviation, mean and 90th-percentile slope.

Resumable: one small .npz per tile in `tiles_dir`; `merge` concatenates them.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .sources import cop_tile_list_url, cop_tile_url, parse_cop_stem, setup_gdal_env, vsicurl

FIELDS = ("lat_min", "lat_max", "lon_min", "lon_max", "frac_valid", "frac_sea", "elev_mean",
          "relief", "std", "slope_mean", "slope_p90")
EARTH_M_PER_DEG = 111_320.0


def list_tiles(lat_range=(-56, 72), timeout: float = 120) -> list[tuple[int, int]]:
    """GLO-90 tiles (lat, lon of the SW corner) inside the latitude range."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(cop_tile_list_url(90), timeout=timeout) as r:
            stems = r.read().decode().split()
    except urllib.error.HTTPError as e:
        # No tile list: try every 1x1 degree tile; missing (ocean) tiles are skipped when read.
        print(f"tile list unavailable ({e.code}); trying every tile in the latitude range")
        return [(la, lo) for la in range(lat_range[0], lat_range[1]) for lo in range(-180, 180)]
    out = []
    for s in stems:
        try:
            la, lo = parse_cop_stem(s)
        except (IndexError, ValueError):
            continue
        if lat_range[0] <= la < lat_range[1]:
            out.append((la, lo))
    return sorted(set(out))


def summarise(z: np.ndarray, lat0: int, lon0: int, cells_per_deg: int = 10) -> dict:
    """Cell statistics for one 1x1 degree tile. z: (rows, cols), north-up, NaN = no data."""
    R, C = z.shape
    lat_rows = lat0 + 1 - (np.arange(R) + 0.5) / R            # row centre latitudes
    dy = EARTH_M_PER_DEG / R
    dx = EARTH_M_PER_DEG * np.cos(np.radians(lat_rows)) / C   # per row
    gy = np.gradient(z, axis=0) / dy
    gx = np.gradient(z, axis=1) / np.maximum(dx[:, None], 1.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    n_lat = cells_per_deg
    n_lon = max(1, int(round(cells_per_deg * math.cos(math.radians(lat0 + 0.5)))))
    rows = np.linspace(0, R, n_lat + 1).round().astype(int)
    cols = np.linspace(0, C, n_lon + 1).round().astype(int)
    out = {k: [] for k in FIELDS}
    for i in range(n_lat):
        for j in range(n_lon):
            zz = z[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
            ss = slope[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
            ok = np.isfinite(zz) & np.isfinite(ss)
            n = zz.size
            out["lat_max"].append(lat0 + 1 - i / n_lat)
            out["lat_min"].append(lat0 + 1 - (i + 1) / n_lat)
            out["lon_min"].append(lon0 + j / n_lon)
            out["lon_max"].append(lon0 + (j + 1) / n_lon)
            fv = ok.sum() / max(n, 1)
            out["frac_valid"].append(fv)
            if ok.sum() < 4:
                for k in ("frac_sea", "elev_mean", "relief", "std", "slope_mean", "slope_p90"):
                    out[k].append(np.nan if k != "frac_sea" else 1.0)
                continue
            zv, sv = zz[ok], ss[ok]
            sea = (np.abs(zv) < 1.0) & (sv < 0.5)
            out["frac_sea"].append(1.0 - fv + fv * sea.mean())
            p2, p98 = np.percentile(zv, [2, 98])
            out["elev_mean"].append(float(zv.mean()))
            out["relief"].append(float(p98 - p2))
            out["std"].append(float(zv.std()))
            out["slope_mean"].append(float(sv.mean()))
            out["slope_p90"].append(float(np.percentile(sv, 90)))
    return {k: np.asarray(v, np.float32 if k not in ("lat_min", "lat_max", "lon_min", "lon_max")
                          else np.float64) for k, v in out.items()}


def read_tile(lat: int, lon: int, decimate: int = 4) -> np.ndarray | None:
    """GLO-90 tile at reduced resolution (uses the COG overviews). None if it does not exist."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.errors import RasterioIOError
    url = vsicurl(cop_tile_url(lat, lon, 90))
    try:
        with rasterio.open(url) as ds:
            shape = (max(16, ds.height // decimate), max(16, ds.width // decimate))
            z = ds.read(1, out_shape=shape, resampling=Resampling.average).astype(np.float32)
            nod = ds.nodata
    except RasterioIOError as e:
        msg = str(e)
        if "404" in msg or "403" in msg or "does not exist" in msg or "No such file" in msg:
            return None
        raise
    if nod is not None:
        z[z == nod] = np.nan
    z[z < -1000] = np.nan
    return z


def _tile_file(tiles_dir: Path, lat: int, lon: int) -> Path:
    return tiles_dir / f"{lat:+03d}{lon:+04d}.npz"


def compute(cfg, threads: int | None = None, limit: int | None = None, tiles=None) -> Path:
    """Compute (or resume) per-tile descriptors, then merge them into one file."""
    d = cfg["global"]["descriptors"]
    setup_gdal_env()
    tiles_dir = cfg.path(d["tiles_dir"])
    tiles_dir.mkdir(parents=True, exist_ok=True)
    if tiles is None:
        tiles = list_tiles(tuple(d["lat_range"]))
    todo = [t for t in tiles if not _tile_file(tiles_dir, *t).exists()]
    if limit:
        todo = todo[:limit]
    print(f"{len(tiles)} GLO-90 tiles in range, {len(todo)} to do", flush=True)
    dec, cpd = int(d["decimate"]), int(d["cells_per_deg"])

    def job(t):
        la, lo = t
        z = read_tile(la, lo, dec)
        f = _tile_file(tiles_dir, la, lo)
        tmp = f.with_name(f".{f.stem}.tmp.npz")
        if z is None:
            np.savez(tmp, missing=np.bool_(True))
        else:
            np.savez(tmp, **summarise(z, la, lo, cpd))
        tmp.replace(f)
        return t

    t0, n_done, errors = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=int(threads or d.get("threads", 16))) as ex:
        futs = {ex.submit(job, t): t for t in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                errors.append((futs[fut], f"{type(e).__name__}: {e}"))
            n_done += 1
            if n_done % 200 == 0 or n_done == len(todo):
                rate = n_done / max(time.time() - t0, 1e-6)
                print(f"  {n_done}/{len(todo)} tiles  {rate:.1f}/s  errors {len(errors)}", flush=True)
    for t, e in errors[:20]:
        print(f"  error {t}: {e}")
    if errors:
        print(f"{len(errors)} tiles failed; run the command again to retry them")
    return merge(cfg)


def merge(cfg) -> Path:
    d = cfg["global"]["descriptors"]
    tiles_dir = cfg.path(d["tiles_dir"])
    cols = {k: [] for k in FIELDS}
    n_tiles = 0
    for f in sorted(tiles_dir.glob("*.npz")):
        if f.name.startswith("."):
            continue
        with np.load(f) as z:
            if "missing" in z.files:
                continue
            for k in FIELDS:
                cols[k].append(z[k])
        n_tiles += 1
    out = cfg.path(d["file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    arrs = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    np.savez_compressed(out, **arrs)
    n = len(arrs["relief"])
    print(f"merged {n_tiles} tiles -> {n} cells -> {out}")
    return out


def load(path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
