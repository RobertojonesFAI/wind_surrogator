"""Global raster sources: Copernicus DEM (GLO-30 / GLO-90), FABDEM V1-2 and ESA WorldCover.

All of them are 1x1 (DEM) or 3x3 degree (WorldCover) GeoTIFF tiles named by their
south-west corner. Copernicus and WorldCover are Cloud Optimized GeoTIFFs in public S3
buckets, read in place with HTTP range requests (no account needed). FABDEM is
distributed as 10x10 degree zip files; single tiles are pulled out of the remote zip with
range requests when the server allows it, otherwise the whole zip is downloaded once.
"""
from __future__ import annotations

import math
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

COP_BUCKET = {30: "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com",
              90: "https://copernicus-dem-90m.s3.eu-central-1.amazonaws.com"}
COP_RES_CODE = {30: "10", 90: "30"}           # arc-second code used in the file names
WORLDCOVER_URL = ("https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
                  "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif")
FABDEM_BRIS = "https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn"

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
    GDAL_HTTP_MAX_RETRY="6",
    GDAL_HTTP_RETRY_DELAY="3",
    GDAL_HTTP_TIMEOUT="120",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    VSI_CACHE="TRUE",
    VSI_CACHE_SIZE="67108864",
    GDAL_CACHEMAX="256",
)


def setup_gdal_env() -> None:
    """GDAL settings for reading remote COGs, set process-wide (GDAL reads them from the
    environment, which also covers worker threads). Existing values are kept."""
    for k, v in GDAL_ENV.items():
        os.environ.setdefault(k, v)


def vsicurl(url: str) -> str:
    return url if url.startswith("/vsi") or not url.startswith("http") else "/vsicurl/" + url


# --------------------------------------------------------------------------- tile names
def _ns(lat: int, width: int = 2) -> str:
    return f"N{lat:0{width}d}" if lat >= 0 else f"S{-lat:0{width}d}"


def _ew(lon: int, width: int = 3) -> str:
    return f"E{lon:0{width}d}" if lon >= 0 else f"W{-lon:0{width}d}"


def cop_tile_stem(lat: int, lon: int, res: int = 30) -> str:
    """Copernicus tile covering [lat, lat+1) x [lon, lon+1), e.g. Copernicus_DSM_COG_10_N44_00_W115_00_DEM."""
    return f"Copernicus_DSM_COG_{COP_RES_CODE[res]}_{_ns(lat)}_00_{_ew(lon)}_00_DEM"


def cop_tile_url(lat: int, lon: int, res: int = 30) -> str:
    stem = cop_tile_stem(lat, lon, res)
    return f"{COP_BUCKET[res]}/{stem}/{stem}.tif"


def cop_tile_list_url(res: int = 30) -> str:
    return f"{COP_BUCKET[res]}/tileList.txt"


def parse_cop_stem(stem: str) -> tuple[int, int]:
    """'Copernicus_DSM_COG_30_S12_00_W078_00_DEM' -> (-12, -78)."""
    parts = stem.strip().split("_")
    la, lo = parts[4], parts[6]
    lat = int(la[1:]) * (1 if la[0] == "N" else -1)
    lon = int(lo[1:]) * (1 if lo[0] == "E" else -1)
    return lat, lon


def fabdem_tile_name(lat: int, lon: int) -> str:
    """FABDEM V1-2 tile, e.g. N44W115_FABDEM_V1-2.tif (south-west corner)."""
    return f"{_ns(lat)}{_ew(lon)}_FABDEM_V1-2.tif"


def fabdem_zip_name(lat: int, lon: int) -> str:
    """10x10 degree zip holding a tile, e.g. N40W120-N50W110_FABDEM_V1-2.zip.

    The north-east corner at 180 degrees is written W180 (as in S20E170-S10W180).
    """
    la0 = 10 * math.floor(lat / 10)
    lo0 = 10 * math.floor(lon / 10)
    la1, lo1 = la0 + 10, lo0 + 10
    lo1_txt = "W180" if lo1 == 180 else _ew(lo1)
    return f"{_ns(la0)}{_ew(lo0)}-{_ns(la1)}{lo1_txt}_FABDEM_V1-2.zip"


def worldcover_tile(lat: float, lon: float) -> str:
    """3x3 degree WorldCover tile name holding a point, e.g. N42W117."""
    la0 = 3 * math.floor(lat / 3)
    lo0 = 3 * math.floor(lon / 3)
    return f"{_ns(la0)}{_ew(lo0)}"


def worldcover_url(tile: str) -> str:
    return WORLDCOVER_URL.format(tile=tile)


def tiles_1deg(bounds) -> list[tuple[int, int]]:
    """1x1 degree tiles (lat, lon of the SW corner) touching a lon/lat bbox."""
    lon0, lat0, lon1, lat1 = bounds
    return [(la, lo) for la in range(math.floor(lat0), math.ceil(lat1))
            for lo in range(math.floor(lon0), math.ceil(lon1))]


def tiles_3deg(bounds) -> list[str]:
    lon0, lat0, lon1, lat1 = bounds
    out = []
    for la in range(3 * math.floor(lat0 / 3), math.ceil(lat1), 3):
        for lo in range(3 * math.floor(lon0 / 3), math.ceil(lon1), 3):
            out.append(f"{_ns(la)}{_ew(lo)}")
    return out


# --------------------------------------------------------------------------- HTTP helpers
def _request(url: str, headers: dict | None = None, timeout: float = 120, method: str = "GET"):
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def url_exists(url: str, timeout: float = 60) -> bool:
    try:
        with _request(url, {"Range": "bytes=0-0"}, timeout=timeout):
            return True
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return False
        raise


def download(url: str, dest: Path, retries: int = 4, timeout: float = 300) -> Path | None:
    """Download to dest (atomic). Returns None on 404/403 (tile does not exist)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}.part")
    for attempt in range(retries):
        try:
            with _request(url, timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 22)
            os.replace(tmp, dest)
            return dest
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            err = e
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(3 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {url}: {err}")


class HTTPRangeFile:
    """Read-only, seekable file over HTTP range requests (enough for zipfile.ZipFile)."""

    def __init__(self, url: str, block: int = 1 << 22, timeout: float = 120, retries: int = 4):
        self.url, self.block, self.timeout, self.retries = url, block, timeout, retries
        self.pos = 0
        self._buf_start, self._buf = 0, b""
        with self._get(0, 0) as r:
            if r.status != 206:
                raise OSError(f"server ignores range requests: {url}")
            cr = r.headers.get("Content-Range", "")
            self.size = int(cr.rsplit("/", 1)[-1])

    def _get(self, start: int, end: int):
        last = None
        for attempt in range(self.retries):
            try:
                return _request(self.url, {"Range": f"bytes={start}-{end}"}, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                if e.code in (403, 404, 416):
                    raise
                last = e
            except Exception as e:  # noqa: BLE001
                last = e
            time.sleep(2 * (attempt + 1))
        raise OSError(f"range request failed for {self.url}: {last}")

    def _fetch(self, start: int, n: int) -> bytes:
        end = min(start + n, self.size) - 1
        if end < start:
            return b""
        with self._get(start, end) as r:
            return r.read()

    # file protocol --------------------------------------------------------
    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        if whence == 0:
            self.pos = off
        elif whence == 1:
            self.pos += off
        else:
            self.pos = self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        b0, b1 = self._buf_start, self._buf_start + len(self._buf)
        if b0 <= self.pos and self.pos + n <= b1:
            out = self._buf[self.pos - b0: self.pos - b0 + n]
        elif n >= self.block:
            out = self._fetch(self.pos, n)
        else:
            self._buf_start, self._buf = self.pos, self._fetch(self.pos, self.block)
            out = self._buf[:n]
        self.pos += len(out)
        return out

    def close(self):
        self._buf = b""


def extract_from_remote_zip(zip_url: str, members: list[str], out_dir: Path,
                            allow_full_download: bool = False, scratch: Path | None = None) -> dict:
    """Extract members (matched by base name) of a remote zip into out_dir.

    Returns {member: path or None if not in the zip}. Uses range requests; if the server
    does not support them and allow_full_download is set, downloads the whole zip into
    `scratch` first and deletes it afterwards.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path | None] = {}
    todo = [m for m in members if not (out_dir / m).exists()]
    for m in members:
        if (out_dir / m).exists():
            result[m] = out_dir / m
    if not todo:
        return result
    local_zip = None
    try:
        fobj = HTTPRangeFile(zip_url)
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            raise FileNotFoundError(zip_url) from e
        raise
    except OSError:
        if not allow_full_download:
            raise
        scratch = Path(scratch or out_dir)
        local_zip = download(zip_url, scratch / zip_url.rsplit("/", 1)[-1])
        if local_zip is None:
            raise FileNotFoundError(zip_url)
        fobj = open(local_zip, "rb")
    try:
        with zipfile.ZipFile(fobj) as zf:
            by_base = {Path(i.filename).name: i for i in zf.infolist() if not i.is_dir()}
            for m in todo:
                info = by_base.get(m)
                if info is None:
                    result[m] = None
                    continue
                dest = out_dir / m
                tmp = dest.with_name(f".{m}.{os.getpid()}.{threading.get_ident()}.part")
                with zf.open(info) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=1 << 24)
                os.replace(tmp, dest)
                result[m] = dest
    finally:
        fobj.close()
        if local_zip is not None:
            Path(local_zip).unlink(missing_ok=True)
    return result


class FabdemTiles:
    """Local cache of FABDEM tiles, filled on demand from the Bristol zips or a mirror.

    `order` lists the sources to try per tile: "bris" (range reads into the official
    zips), "bris_full" (download whole zips if range reads are refused), "mirror" (a
    per-tile URL template, e.g. a copy of the dataset with one file per tile).
    Returns None for tiles that exist in no source (open ocean).
    """

    def __init__(self, cache_dir: Path, order=("bris",), base_url: str = FABDEM_BRIS,
                 mirror_template: str | None = None, scratch: Path | None = None):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.order = list(order)
        self.base_url = base_url.rstrip("/")
        self.mirror_template = mirror_template
        self.scratch = Path(scratch) if scratch else self.cache / "_zips"
        self._lock = threading.Lock()
        self._zip_locks: dict[str, threading.Lock] = {}
        self._missing_zips: set[str] = set()
        self._no_range = False

    def _zip_lock(self, name: str) -> threading.Lock:
        with self._lock:
            return self._zip_locks.setdefault(name, threading.Lock())

    def path(self, lat: int, lon: int) -> Path:
        return self.cache / fabdem_tile_name(lat, lon)

    def _absent_marker(self, lat: int, lon: int) -> Path:
        return self.cache / (fabdem_tile_name(lat, lon) + ".absent")

    def get(self, lat: int, lon: int) -> Path | None:
        p = self.path(lat, lon)
        if p.exists():
            return p
        if self._absent_marker(lat, lon).exists():
            return None
        errors = []
        for src in self.order:
            try:
                got = self._get_from(src, lat, lon)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{src}: {type(e).__name__}: {e}")
                continue
            if got is not None:
                return got
            # Source answered but has no such tile: ocean. Remember that.
            self._absent_marker(lat, lon).write_text(src)
            return None
        raise RuntimeError(f"FABDEM tile {fabdem_tile_name(lat, lon)} unavailable: " + " | ".join(errors))

    def _get_from(self, src: str, lat: int, lon: int) -> Path | None:
        name = fabdem_tile_name(lat, lon)
        if src in ("bris", "bris_full"):
            zname = fabdem_zip_name(lat, lon)
            if zname in self._missing_zips:
                return None
            if src == "bris" and self._no_range:
                raise OSError("range requests refused earlier")
            with self._zip_lock(zname):
                if self.path(lat, lon).exists():
                    return self.path(lat, lon)
                try:
                    res = extract_from_remote_zip(f"{self.base_url}/{zname}", [name], self.cache,
                                                  allow_full_download=(src == "bris_full"),
                                                  scratch=self.scratch)
                except FileNotFoundError:
                    self._missing_zips.add(zname)       # whole 10-degree box is ocean
                    return None
                except OSError as e:
                    if "range" in str(e):
                        self._no_range = True
                    raise
            return res.get(name)
        if src == "mirror":
            if not self.mirror_template:
                raise ValueError("no mirror template configured")
            url = self.mirror_template.format(name=name, zip=fabdem_zip_name(lat, lon)[:-4])
            return download(url, self.path(lat, lon))
        raise ValueError(f"unknown FABDEM source {src!r}")


def check_sources(cfg, lat: int = 43, lon: int = -116) -> list[dict]:
    """Probe every data source once (tile list, one tile of each, FABDEM range reads)."""
    import rasterio
    setup_gdal_env()
    t = cfg["global"]["terrain"]
    out = []

    def probe(name, fn):
        t0 = time.time()
        try:
            detail = fn()
            out.append(dict(source=name, ok=True, seconds=round(time.time() - t0, 1), detail=detail))
        except Exception as e:  # noqa: BLE001
            out.append(dict(source=name, ok=False, seconds=round(time.time() - t0, 1),
                            detail=f"{type(e).__name__}: {e}"[:300]))
        print(f"  {'OK  ' if out[-1]['ok'] else 'FAIL'} {name:22s} {out[-1]['seconds']:6.1f} s  {out[-1]['detail']}",
              flush=True)

    def tile_list():
        with _request(cop_tile_list_url(90), timeout=60) as r:
            n = len(r.read().split())
        return f"{n} GLO-90 tiles listed"

    def cog(url):
        def f():
            with rasterio.open(vsicurl(url)) as ds:
                a = ds.read(1, out_shape=(64, 64))
            return f"{ds.width}x{ds.height}, min {a.min()}, max {a.max()}"
        return f

    def fab_range():
        url = f"{FABDEM_BRIS}/{fabdem_zip_name(lat, lon)}"
        f = HTTPRangeFile(url)
        with zipfile.ZipFile(f) as zf:
            names = [Path(i.filename).name for i in zf.infolist()]
        has = fabdem_tile_name(lat, lon) in names
        return f"{f.size / 1e9:.2f} GB zip, {len(names)} files, {fabdem_tile_name(lat, lon)} {'found' if has else 'NOT found'}" \
               + ("" if has else f" (e.g. {names[:3]})")

    def fab_mirror():
        tmpl = t.get("fabdem_mirror")
        if not tmpl:
            return "not configured"
        url = tmpl.format(name=fabdem_tile_name(lat, lon), zip=fabdem_zip_name(lat, lon)[:-4])
        return cog(url)()

    probe("glo90 tile list", tile_list)
    probe("glo90 tile", cog(cop_tile_url(lat, lon, 90)))
    probe("glo30 tile", cog(cop_tile_url(lat, lon, 30)))
    probe("worldcover tile", cog(worldcover_url(worldcover_tile(lat + 0.5, lon + 0.5))))
    probe("fabdem bristol (range)", fab_range)
    probe("fabdem mirror", fab_mirror)
    return out
