"""DEM download, mosaicking and reprojection to a metric master grid."""
from __future__ import annotations

import math
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

USGS_URL = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/{prod}/TIFF/current/"
            "{name}/USGS_{prod}_{name}.tif")
COP_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/{stem}/{stem}.tif")


def tile_urls(source: str, bbox) -> list[str]:
    """1x1 degree tiles covering bbox=(lon_min, lat_min, lon_max, lat_max)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    urls = []
    for lat in range(math.floor(lat_min), math.ceil(lat_max)):
        for lon in range(math.floor(lon_min), math.ceil(lon_max)):
            if source.startswith("usgs3dep"):
                prod = "13" if source.endswith("13") else "1"
                # USGS names tiles by their NW corner, e.g. n44w117 covers 43-44N, 117-116W.
                ns = f"n{lat + 1:02d}" if lat + 1 >= 0 else f"s{-(lat + 1):02d}"
                ew = f"w{-lon:03d}" if lon < 0 else f"e{lon:03d}"
                urls.append(USGS_URL.format(prod=prod, name=ns + ew))
            elif source == "copernicus30":
                # Copernicus names tiles by their SW corner.
                ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
                ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
                stem = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
                urls.append(COP_URL.format(stem=stem))
            else:
                raise ValueError(f"unknown DEM source {source!r}")
    return urls


def download(url: str, dest: Path, retries: int = 3) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.rename(dest)
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:  # tile does not exist (ocean / outside coverage)
                print(f"  missing tile (404): {url}")
                return None
            print(f"  attempt {attempt + 1} failed for {url}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {attempt + 1} failed for {url}: {e}")
    raise RuntimeError(f"could not download {url}")


def regions(cfg) -> list[dict]:
    """Configured regions (new `regions:` list, or the legacy single `region:` block)."""
    regs = cfg.get("regions") or [cfg["region"]]
    names = [r["name"] for r in regs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate region names: {names}")
    return regs


def region_dem_path(cfg, name: str) -> Path:
    return cfg.path(cfg["dem"].get("dir", "dem")) / f"{name}.tif"


def expand_bbox(bbox, km: float):
    lon0, lat0, lon1, lat1 = bbox
    dlat = km / 111.0
    dlon = km / (111.0 * math.cos(math.radians(0.5 * (lat0 + lat1))))
    return (lon0 - dlon, lat0 - dlat, lon1 + dlon, lat1 + dlat)


def bbox_in_crs(bbox, crs):
    """(left, bottom, right, top) of a lon/lat bbox in a projected CRS."""
    from rasterio.warp import transform_bounds
    return transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)


def _local_paths(spec) -> list[Path]:
    import glob
    specs = spec if isinstance(spec, (list, tuple)) else [spec]
    out = []
    for sp in specs:
        hits = sorted(glob.glob(str(Path(str(sp)).expanduser())))
        out += [Path(h) for h in hits]
    return out


def build_master_dem(cfg, region: dict) -> Path:
    """Download tiles (or use local files) and write the region's master DEM in region.crs.

    The DEM covers the region bbox plus `dem.margin_km` on every side, so a case can be
    centred anywhere inside the bbox and still have its full CFD domain on real terrain.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    from rasterio.warp import reproject, transform_bounds
    from rasterio.transform import from_origin
    from rasterio.fill import fillnodata

    out = region_dem_path(cfg, region["name"])
    if out.exists():
        print(f"master DEM exists: {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    src_kind = cfg["dem"]["source"]
    dst_crs = region["crs"]
    res = float(cfg["dem"]["resolution"])
    bbox = expand_bbox(region["bbox"], float(cfg["dem"].get("margin_km", 13)))
    # Read a slightly larger lon/lat box so the enclosing UTM rectangle is fully covered.
    big = expand_bbox(bbox, 15.0)
    if src_kind == "local":
        paths = _local_paths(cfg["dem"]["local_path"])
    else:
        # Raw tiles go to fast local scratch; only the reprojected master DEM is kept.
        tiles_dir = Path(cfg["output"]["work_dir"]).expanduser() / "dem_tiles"
        paths = []
        for url in tile_urls(src_kind, big):
            print(f"downloading {url}")
            p = download(url, tiles_dir / url.rsplit("/", 1)[-1])
            if p is not None:
                paths.append(p)
    if not paths:
        raise RuntimeError(f"no DEM files for region {region['name']}")

    srcs = [rasterio.open(p) for p in paths]
    src_crs = srcs[0].crs
    if any(s.crs != src_crs for s in srcs):
        raise ValueError("all local DEM files must share one CRS")
    mb = big if src_crs.to_epsg() == 4326 else transform_bounds("EPSG:4326", src_crs, *big, densify_pts=21)
    mosaic, mtrans = merge(srcs, bounds=mb, nodata=np.nan, dtype="float32")
    mosaic = mosaic[0]
    for s in srcs:
        s.close()

    left, bottom, right, top = bbox_in_crs(bbox, dst_crs)
    left, bottom = math.floor(left / res) * res, math.floor(bottom / res) * res
    right, top = math.ceil(right / res) * res, math.ceil(top / res) * res
    width, height = int(round((right - left) / res)), int(round((top - bottom) / res))
    dst_transform = from_origin(left, top, res, res)
    dst = np.full((height, width), np.nan, dtype=np.float32)
    reproject(mosaic, dst, src_transform=mtrans, src_crs=src_crs, src_nodata=np.nan,
              dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    bad = ~np.isfinite(dst)
    if bad.mean() > 0.2:
        raise RuntimeError(f"{bad.mean() * 100:.0f}% of the DEM is empty: the files do not cover "
                           f"region {region['name']} {region['bbox']} plus its margin")
    if bad.any():
        print(f"filling {bad.mean() * 100:.2f}% nodata pixels")
        dst = fillnodata(np.where(bad, 0, dst), mask=(~bad).astype(np.uint8), max_search_distance=200)
    profile = dict(driver="GTiff", height=height, width=width, count=1, dtype="float32",
                   crs=dst_crs, transform=dst_transform, nodata=np.nan, tiled=True,
                   blockxsize=512, blockysize=512, compress="deflate", predictor=3, BIGTIFF="IF_SAFER")
    with rasterio.open(out, "w", **profile) as ds:
        ds.write(dst, 1)
    print(f"wrote {out} ({width}x{height} px at {res} m, "
          f"{np.nanmin(dst):.0f}-{np.nanmax(dst):.0f} m)")
    return out


@dataclass
class MasterDEM:
    """In-memory master DEM in a metric CRS (north-up, square pixels)."""
    data: np.ndarray        # (rows, cols) float32
    x0: float               # west edge
    y0: float               # north edge
    res: float
    crs: str
    logz0: np.ndarray | None = None   # optional log10(z0) map on the same grid

    @classmethod
    def open(cls, path) -> "MasterDEM":
        import rasterio
        with rasterio.open(path) as ds:
            t = ds.transform
            if abs(t.a + t.e) > 1e-6 or t.b != 0 or t.d != 0:
                raise ValueError("master DEM must be north-up with square pixels")
            data = ds.read(1).astype(np.float32)
            return cls(data=data, x0=t.c, y0=t.f, res=t.a, crs=str(ds.crs))

    @property
    def shape(self):
        return self.data.shape

    @property
    def bounds(self):
        rows, cols = self.shape
        return (self.x0, self.y0 - rows * self.res, self.x0 + cols * self.res, self.y0)

    def to_pixel(self, e, n):
        """Fractional (row, col) of pixel-centre coordinates."""
        return (self.y0 - n) / self.res - 0.5, (e - self.x0) / self.res - 0.5

    def patch(self, e_c: float, n_c: float, radius: float) -> "MasterDEM":
        """Square sub-DEM around (e_c, n_c) with half-width `radius` (clipped to the DEM)."""
        rows, cols = self.shape
        r_c, c_c = self.to_pixel(e_c, n_c)
        rpx = int(math.ceil(radius / self.res)) + 2
        r0, r1 = max(0, int(r_c) - rpx), min(rows, int(r_c) + rpx + 1)
        c0, c1 = max(0, int(c_c) - rpx), min(cols, int(c_c) + rpx + 1)
        lz = None if self.logz0 is None else self.logz0[r0:r1, c0:c1]
        return MasterDEM(self.data[r0:r1, c0:c1], self.x0 + c0 * self.res,
                         self.y0 - r0 * self.res, self.res, self.crs, lz)


def synthetic_dem(path, size_km: float = 120.0, res: float = 30.0, seed: int = 0,
                  crs: str = "EPSG:32611", origin=(500000.0, 4900000.0)) -> Path:
    """Fractal mountains + ridges + valleys, for testing without network access.

    origin = (west, north) edge in `crs`.
    """
    import rasterio
    from rasterio.transform import from_origin
    from scipy import ndimage

    rng = np.random.default_rng(seed)
    n = int(size_km * 1000 / res)
    k = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2
    k[0, 0] = 1.0
    spec = (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))) * k ** (-1.9 / 2)
    spec[0, 0] = 0
    h = np.real(np.fft.ifft2(spec))
    h = ndimage.gaussian_filter(h, 1.0)
    h = (h - h.mean()) / h.std()
    # Ridged component: sharp crests like real mountains.
    ridged = 1.0 - np.abs(ndimage.gaussian_filter(rng.normal(size=(n, n)), 40))
    ridged = (ridged - ridged.mean()) / ridged.std()
    yy, xx = np.mgrid[0:n, 0:n] / n
    plain = 1.0 / (1.0 + np.exp(-(xx - 0.35) * 12))  # flat plain in the west, mountains east
    z = 1200 + plain * (420 * h + 160 * ridged) + (1 - plain) * 40 * h
    z = z.astype(np.float32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs=crs, transform=from_origin(origin[0], origin[1], res, res),
                       nodata=np.nan, tiled=True, compress="deflate") as ds:
        ds.write(z, 1)
    return path
