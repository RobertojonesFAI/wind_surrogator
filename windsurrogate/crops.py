"""Per-location terrain crops for the global dataset.

Each location gets one small .npz with a north-up square around it in a local
Transverse Mercator projection centred on the location (so the case centre is (0, 0)
and there is no projection distortion), large enough for any wind direction:

  dem_q (uint16), dem_offset, dem_scale   elevation = dem_q * dem_scale + dem_offset (m)
  lc (uint8)                               ESA WorldCover class (0 = unknown)
  res, half                                pixel size and half width (m)
  lat, lon, crs                            centre and PROJ string of the local grid
  dem_source                               fabdem | copernicus30 | mixed

Crops are self-contained, so compute nodes need no internet access. The runner turns
`lc` into a roughness map with the `landcover.z0_table` of the configuration.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

from .dem import MasterDEM
from .geometry import CaseGeometry
from . import sources
from .sources import FabdemTiles, cop_tile_url, fabdem_zip_name, setup_gdal_env, tiles_1deg, tiles_3deg, vsicurl

DEM_SCALE = 0.2          # m per uint16 step (13 km of range)
WATER_CLASS = 80


def local_crs(lat: float, lon: float) -> str:
    return (f"+proj=tmerc +lat_0={lat:.6f} +lon_0={lon:.6f} +k=1 +x_0=0 +y_0=0 "
            f"+ellps=WGS84 +units=m +no_defs")


def crop_half_width(geom: CaseGeometry, res: float, margin_m: float) -> float:
    """Half width (m) covering the input window for every wind direction, plus the taper
    to flat and a margin for the smoothing kernels."""
    x0, x1, y0, y1 = geom.input_bounds
    r = math.hypot(max(abs(x0), abs(x1)), max(abs(y0), abs(y1))) + geom.taper_m + margin_m
    return math.ceil(r / res) * res


def crop_lonlat_bounds(lat: float, lon: float, half: float, pad_deg: float = 0.01):
    from pyproj import Transformer
    t = Transformer.from_crs(local_crs(lat, lon), "EPSG:4326", always_xy=True)
    s = np.linspace(-half, half, 21)
    xs = np.concatenate([s, s, np.full_like(s, -half), np.full_like(s, half)])
    ys = np.concatenate([np.full_like(s, -half), np.full_like(s, half), s, s])
    lo, la = t.transform(xs, ys)
    return (float(np.min(lo)) - pad_deg, float(np.min(la)) - pad_deg,
            float(np.max(lo)) + pad_deg, float(np.max(la)) + pad_deg)


def _warp_into(dst: np.ndarray, src_path: str, dst_transform, dst_crs: str, resampling,
               dst_nodata) -> bool:
    """Reproject one raster into the empty pixels of dst. Returns False if it does not exist."""
    import rasterio
    from rasterio.errors import RasterioIOError
    from rasterio.warp import reproject
    try:
        ds = rasterio.open(src_path)
    except RasterioIOError as e:
        msg = str(e)
        if any(s in msg for s in ("404", "403", "does not exist", "No such file", "not recognized")):
            return False
        raise
    with ds:
        tmp = np.full(dst.shape, dst_nodata, dtype=dst.dtype)
        src_nodata = ds.nodata
        reproject(rasterio.band(ds, 1), tmp, src_nodata=src_nodata, dst_transform=dst_transform,
                  dst_crs=dst_crs, dst_nodata=dst_nodata, resampling=resampling)
    empty = ~np.isfinite(dst) if np.issubdtype(dst.dtype, np.floating) else (dst == dst_nodata)
    new = np.isfinite(tmp) if np.issubdtype(tmp.dtype, np.floating) else (tmp != dst_nodata)
    m = empty & new
    dst[m] = tmp[m]
    return True


def _inside_any(lat: np.ndarray, lon: np.ndarray, tiles) -> np.ndarray:
    """True where (lat, lon) falls in one of the 1x1 degree tiles (SW corners)."""
    out = np.zeros(lat.shape, bool)
    la0, lo0 = np.floor(lat).astype(int), np.floor(lon).astype(int)
    for la, lo in tiles:
        out |= (la0 == la) & (lo0 == lo)
    return out


def build_crop(lat: float, lon: float, half: float, res: float, dem_source: str,
               fabdem: FabdemTiles | None = None, fallback_copernicus: bool = True,
               landcover: bool = True, max_sea_frac: float = 0.9) -> dict:
    """Elevation and land cover around (lat, lon) on the local grid."""
    from rasterio.enums import Resampling
    from rasterio.fill import fillnodata
    from rasterio.transform import from_origin

    n = int(round(2 * half / res))
    crs = local_crs(lat, lon)
    transform = from_origin(-half, half, res, res)
    bounds = crop_lonlat_bounds(lat, lon, half)
    dem = np.full((n, n), np.nan, np.float32)
    used = set()
    absent_tiles = 0
    used_tiles: list[tuple[int, int]] = []
    for la, lo in tiles_1deg(bounds):
        path = None
        if dem_source == "fabdem":
            try:
                p = fabdem.get(la, lo)
                if p is None:
                    absent_tiles += 1          # ocean
                    continue
                path, src = str(p), "fabdem"
            except Exception:                  # noqa: BLE001
                if not fallback_copernicus:
                    raise
        if path is None:
            path, src = vsicurl(cop_tile_url(la, lo, 30)), "copernicus30"
        if _warp_into(dem, path, transform, crs, Resampling.bilinear, np.nan):
            used.add(src)
            used_tiles.append((la, lo))
        else:
            absent_tiles += 1
    missing = ~np.isfinite(dem)
    fill_frac = float(missing.mean())
    if missing.all():
        raise RuntimeError(f"no elevation data around ({lat:.4f}, {lon:.4f})")
    lc = np.zeros((n, n), np.uint8)
    if landcover:
        for t in tiles_3deg(bounds):
            _warp_into(lc, vsicurl(sources.worldcover_url(t)), transform, crs, Resampling.mode, 0)
    if missing.any():
        # No elevation over the open sea (absent tiles or nodata over water): sea level.
        sea = missing & ((lc == 0) | (lc == WATER_CLASS)) if landcover else np.zeros_like(missing)
        if absent_tiles:
            from pyproj import Transformer
            c = -half + res * (np.arange(n) + 0.5)
            X, Y = np.meshgrid(c, -c)
            plon, plat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(X, Y)
            sea |= missing & ~_inside_any(plat, plon, used_tiles)
        dem[sea] = 0.0
        holes = ~np.isfinite(dem)
        if holes.any():          # voids inside land tiles: interpolate
            dem = fillnodata(np.where(holes, 0, dem), mask=(~holes).astype(np.uint8),
                             max_search_distance=100)
            dem[~np.isfinite(dem)] = 0.0
        lc[(lc == 0) & sea] = WATER_CLASS
    if fill_frac > max_sea_frac:
        raise RuntimeError(f"{fill_frac:.0%} of the crop has no elevation (sources used: {sorted(used)}); "
                           "locations are on land, so this points to a data source problem")
    off = math.floor(float(np.min(dem)))
    q = np.round((dem - off) / DEM_SCALE)
    if q.max() > 65535:
        raise RuntimeError("elevation range too large for uint16 encoding")
    return dict(dem_q=q.astype(np.uint16), dem_offset=np.float64(off), dem_scale=np.float64(DEM_SCALE),
                lc=lc, res=np.float64(res), half=np.float64(half), lat=np.float64(lat), lon=np.float64(lon),
                crs=np.array(crs), dem_source=np.array("mixed" if len(used) > 1 else (used.pop() if used else "none")),
                fill_frac=np.float32(fill_frac))


def save_crop(path: Path, crop: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp.npz")
    np.savez_compressed(tmp, **crop)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- roughness
def z0_lookup(cfg) -> np.ndarray:
    """256-entry table: WorldCover class -> z0 (m)."""
    lc = cfg.get("landcover", {})
    table = np.full(256, float(lc.get("default_z0", 0.03)))
    for k, v in (lc.get("z0_table") or {}).items():
        table[int(k)] = float(v)
    return np.clip(table, float(lc.get("z0_min", 2e-4)), float(lc.get("z0_max", 1.5)))


def load_crop(path, z0_table: np.ndarray | None = None) -> MasterDEM:
    """Crop as a MasterDEM (centre at (0, 0)) with a log10(z0) map when a table is given."""
    with np.load(path) as z:
        dem = z["dem_q"].astype(np.float32) * np.float32(z["dem_scale"]) + np.float32(z["dem_offset"])
        half, res = float(z["half"]), float(z["res"])
        crs = str(z["crs"])
        logz0 = None
        if z0_table is not None:
            logz0 = np.log10(z0_table[z["lc"]]).astype(np.float32)
    return MasterDEM(data=dem, x0=-half, y0=half, res=res, crs=crs, logz0=logz0)


class CropCache:
    """Small thread-safe LRU of loaded crops (a node runs many cases of the same location)."""

    def __init__(self, cfg, maxsize: int = 96):
        self.root = cfg.root
        self.table = z0_lookup(cfg)
        self._get = lru_cache(maxsize=maxsize)(self._load)

    def _load(self, rel: str) -> MasterDEM:
        return load_crop(self.root / rel, self.table)

    def __call__(self, rel: str) -> MasterDEM:
        return self._get(rel)


# --------------------------------------------------------------------------- fetch command
def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def fetch(cfg, shard: int = 0, n_shards: int = 1, threads: int | None = None,
          limit: int | None = None, pilot_only: bool = False, clean_cache: bool = False) -> dict:
    """Build the missing crops of this shard (pilot locations first). Resumable."""
    t = cfg["global"]["terrain"]
    setup_gdal_env()
    geom = CaseGeometry.from_config(cfg)
    res = float(t["crop_res"])
    half = crop_half_width(geom, res, float(t["crop_margin_m"]))
    locs = read_jsonl(cfg.path(t["locations_file"]))
    # Shard by FABDEM zip (10x10 degree box), so each zip is read by one shard only, and
    # process a zip's locations together (warm tile cache). Pilot locations go first.
    zkey = {id(loc): fabdem_zip_name(math.floor(loc["lat"]), math.floor(loc["lon"])) for loc in locs}
    locs = [loc for loc in locs if zlib.crc32(zkey[id(loc)].encode()) % n_shards == shard]
    if pilot_only:
        locs = [loc for loc in locs if loc.get("pilot")]
    locs.sort(key=lambda loc: (not loc.get("pilot"), zkey[id(loc)]))
    todo = [loc for loc in locs if not (cfg.root / loc["terrain"]).exists()]
    if limit:
        todo = todo[:limit]
    src = t["dem_source"]
    fab = None
    if src == "fabdem":
        fab = FabdemTiles(cfg.path(t["tile_cache"]), order=t["fabdem_order"],
                          mirror_template=t.get("fabdem_mirror"),
                          scratch=cfg.path(t["tile_cache"]) / "_zips")
    print(f"[fetch {shard}/{n_shards}] {len(locs)} locations, {len(todo)} to do, crop "
          f"{2 * half / 1000:.1f} km at {res:.0f} m, source {src}", flush=True)
    fail_path = cfg.path(t["dir"]) / f"_failed_shard{shard}of{n_shards}.jsonl"
    stats = dict(done=0, failed=0)
    lock = threading.Lock()
    t0 = time.time()

    def job(loc):
        crop = build_crop(loc["lat"], loc["lon"], half, res, src, fab,
                          fallback_copernicus=bool(t.get("fallback_copernicus", True)))
        save_crop(cfg.root / loc["terrain"], crop)
        return str(crop["dem_source"])

    sources: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=int(threads or t.get("threads", 8))) as ex:
        futs = {ex.submit(job, loc): loc for loc in todo}
        for fut in as_completed(futs):
            loc = futs[fut]
            with lock:
                try:
                    s = fut.result()
                    stats["done"] += 1
                    sources[s] = sources.get(s, 0) + 1
                except Exception as e:  # noqa: BLE001
                    stats["failed"] += 1
                    fail_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(fail_path, "a") as f:
                        f.write(json.dumps(dict(loc_id=loc["loc_id"], error=f"{type(e).__name__}: {e}"[:2000])) + "\n")
                n = stats["done"] + stats["failed"]
                if n % 100 == 0 or n == len(todo):
                    rate = n / max(time.time() - t0, 1e-6)
                    eta = (len(todo) - n) / rate / 3600 if rate > 0 else float("nan")
                    print(f"  {n}/{len(todo)}  {rate * 60:.1f}/min  ETA {eta:.1f} h  failed {stats['failed']}  "
                          f"sources {sources}", flush=True)
    print(f"[fetch {shard}/{n_shards}] done {stats['done']}, failed {stats['failed']}"
          + (f" (see {fail_path})" if stats["failed"] else ""), flush=True)
    if clean_cache and fab is not None and not stats["failed"]:
        import shutil
        shutil.rmtree(fab.cache, ignore_errors=True)
        print(f"[fetch {shard}/{n_shards}] removed tile cache {fab.cache}", flush=True)
    return stats
