"""Worldwide pipeline without internet: in-memory HTTP responses stand in for the FABDEM
zips, and local GeoTIFF tiles for the Copernicus / WorldCover buckets."""
from __future__ import annotations

import io
import json
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from windsurrogate import sources  # noqa: E402
from windsurrogate.config import load_config  # noqa: E402
from windsurrogate.crops import build_crop, crop_half_width, load_crop, z0_lookup  # noqa: E402
from windsurrogate.geometry import CaseGeometry  # noqa: E402
from windsurrogate.globalsample import BlockSplit, km_between, sample_global  # noqa: E402
from windsurrogate.terrain import case_terrain  # noqa: E402


# --------------------------------------------------------------------------- helpers
class FakeResponse:
    def __init__(self, status, body, headers):
        self.status, self._body, self.headers = status, body, headers

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_http(files: dict[str, bytes]):
    """Stand-in for sources._request: serves `files` {url: bytes} with range support."""
    import urllib.error

    def _request(url, headers=None, timeout=120, method="GET"):
        if url not in files:
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        data = files[url]
        rng = (headers or {}).get("Range")
        if rng:
            a, b = rng[6:].split("-")
            start, end = int(a), min(int(b), len(data) - 1)
            return FakeResponse(206, data[start:end + 1], {"Content-Range": f"bytes {start}-{end}/{len(data)}"})
        return FakeResponse(200, data, {})
    return _request


def write_tile(path: Path, lat: int, lon: int, n: int, fn, dtype="float32", nodata=None, deg=1.0):
    from rasterio.transform import from_origin
    path.parent.mkdir(parents=True, exist_ok=True)
    lats = lat + deg - (np.arange(n) + 0.5) * deg / n
    lons = lon + (np.arange(n) + 0.5) * deg / n
    LON, LAT = np.meshgrid(lons, lats)
    data = fn(LAT, LON).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype=dtype,
                       crs="EPSG:4326", transform=from_origin(lon, lat + deg, deg / n, deg / n),
                       nodata=nodata, tiled=True, blockxsize=256, blockysize=256, compress="deflate") as ds:
        ds.write(data, 1)


def hills(LAT, LON):
    return 1500 + 600 * np.sin(LAT * 40) * np.cos(LON * 33) + 200 * np.cos(LAT * 120)


def classes(LAT, LON):
    return np.where(np.sin(LON * 60) > 0, 10, 30)      # forest / grassland stripes


# --------------------------------------------------------------------------- names
def test_tile_names():
    assert sources.fabdem_tile_name(44, -115) == "N44W115_FABDEM_V1-2.tif"
    assert sources.fabdem_tile_name(-1, 5) == "S01E005_FABDEM_V1-2.tif"
    assert sources.fabdem_zip_name(44, -115) == "N40W120-N50W110_FABDEM_V1-2.zip"
    assert sources.fabdem_zip_name(-15, 175) == "S20E170-S10W180_FABDEM_V1-2.zip"
    assert sources.fabdem_zip_name(5, 0) == "N00E000-N10E010_FABDEM_V1-2.zip"
    assert sources.cop_tile_stem(44, -115, 90) == "Copernicus_DSM_COG_30_N44_00_W115_00_DEM"
    assert sources.parse_cop_stem("Copernicus_DSM_COG_30_S12_00_W078_00_DEM") == (-12, -78)
    assert sources.worldcover_tile(43.6, -115.9) == "N42W117"
    assert sources.worldcover_tile(-0.5, -0.5) == "S03W003"
    assert sources.tiles_3deg((-116.2, 43.1, -114.5, 45.2)) == ["N42W117", "N45W117"]
    assert sources.tiles_3deg((-116.2, 43.1, -113.5, 45.2)) == ["N42W117", "N42W114", "N45W117", "N45W114"]


def test_range_zip(tmp_path, monkeypatch):
    payload = {f"t{i}.tif": os.urandom(300_000 + i) for i in range(5)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k, v in payload.items():
            zf.writestr(f"inner/{k}", v)
    url = "https://example.org/A.zip"
    monkeypatch.setattr(sources, "_request", fake_http({url: buf.getvalue()}))
    res = sources.extract_from_remote_zip(url, ["t3.tif", "nope.tif"], tmp_path)
    assert res["nope.tif"] is None
    assert res["t3.tif"].read_bytes() == payload["t3.tif"]
    with pytest.raises(FileNotFoundError):
        sources.extract_from_remote_zip("https://example.org/missing.zip", ["t3.tif"], tmp_path / "other")
    # FabdemTiles: tile from the zip, missing tile in an existing zip = ocean (None),
    # missing zip = ocean too.
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        zf.writestr("N43W116_FABDEM_V1-2.tif", b"tile-bytes")
    base = "https://example.org/fab"
    monkeypatch.setattr(sources, "_request",
                        fake_http({f"{base}/N40W120-N50W110_FABDEM_V1-2.zip": zbuf.getvalue()}))
    fab = sources.FabdemTiles(tmp_path / "cache", order=["bris"], base_url=base)
    assert fab.get(43, -116).read_bytes() == b"tile-bytes"
    assert fab.get(44, -116) is None                 # zip exists, tile does not: ocean
    with pytest.raises(RuntimeError):                # missing zip: an error, not ocean
        fab.get(-30, 20)


# --------------------------------------------------------------------------- crops
@pytest.fixture(scope="module")
def fake_world(tmp_path_factory):
    """Local stand-ins: FABDEM tile N43W116 in the tile cache, Copernicus tiles N43W116 and
    N44W116, WorldCover N42W117 (paths instead of URLs: GDAL reads both the same way)."""
    root = tmp_path_factory.mktemp("world")
    n = 600
    write_tile(root / "cache" / "N43W116_FABDEM_V1-2.tif", 43, -116, n, hills)
    for la in (43, 44):
        stem = sources.cop_tile_stem(la, -116, 30)
        write_tile(root / "cop30" / stem / f"{stem}.tif", la, -116, n, hills)
    write_tile(root / "wc" / "ESA_WorldCover_10m_2021_v200_N42W117_Map.tif", 42, -117, 900, classes,
               dtype="uint8", nodata=0, deg=3.0)
    old = (dict(sources.COP_BUCKET), sources.WORLDCOVER_URL)
    sources.COP_BUCKET[30] = str(root / "cop30")
    sources.WORLDCOVER_URL = str(root / "wc") + "/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    yield root
    sources.COP_BUCKET.update(old[0])
    sources.WORLDCOVER_URL = old[1]


def test_build_crop_fabdem_with_fallback(fake_world, tmp_path):
    root = fake_world
    cfg = load_config()
    geom = CaseGeometry.from_config(cfg)
    half = crop_half_width(geom, 30.0, 1000.0)
    # Straddles 44N. N43W116 comes from the FABDEM cache; N44W116 is not cached and the
    # only FABDEM source configured fails, so it falls back to Copernicus.
    fab = sources.FabdemTiles(root / "cache", order=["mirror"], mirror_template=None)
    lat, lon = 43.98, -115.6
    crop = build_crop(lat, lon, half, 30.0, "fabdem", fab, fallback_copernicus=True)
    assert str(crop["dem_source"]) == "mixed"
    dem = crop["dem_q"] * crop["dem_scale"] + crop["dem_offset"]
    assert crop["fill_frac"] < 0.01
    # Centre pixel matches the analytic surface.
    n = dem.shape[0]
    assert abs(dem[n // 2, n // 2] - hills(np.array(lat), np.array(lon))) < 15
    assert set(np.unique(crop["lc"])) <= {10, 30}
    np.savez_compressed(tmp_path / "c.npz", **crop)
    md = load_crop(tmp_path / "c.npz", z0_lookup(cfg))
    assert md.logz0 is not None and md.logz0.shape == md.data.shape
    assert np.isclose(10 ** md.logz0.max(), 0.8) and np.isclose(10 ** md.logz0.min(), 0.03)
    # Spatial z0 through the case terrain.
    ter = case_terrain(geom, md, 0.0, 0.0, 270.0, z0_scale=1.0)
    assert ter.logz0_input.shape == geom.input_shape
    assert 0.03 <= ter.z0_ref <= 0.8
    assert ter.z0_ground.shape == (geom.ny, geom.nx)
    # Far from the input window the ground roughness is the inflow roughness.
    assert np.allclose(ter.z0_ground[:, 0], ter.z0_ref) and np.allclose(ter.z0_ground[0, :], ter.z0_ref)


def test_ocean_is_sea_level(fake_world, tmp_path):
    cfg = load_config()
    geom = CaseGeometry.from_config(cfg)
    half = crop_half_width(geom, 30.0, 1000.0)
    # West edge of the fake world: everything west of 116W has no tile ("ocean").
    crop = build_crop(43.5, -115.98, half, 30.0, "copernicus30", None)
    dem = crop["dem_q"] * crop["dem_scale"] + crop["dem_offset"]
    n = dem.shape[0]
    assert np.allclose(dem[n // 2, :5], 0.0)
    assert abs(dem[n // 2, n // 2] - hills(np.array(43.5), np.array(-115.98))) < 15


# --------------------------------------------------------------------------- sampler
def fake_descriptors(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for la in range(30, 50):
        for lo in range(-125, -100):
            n_lon = max(1, round(10 * math.cos(math.radians(la + 0.5))))
            for i in range(10):
                for j in range(n_lon):
                    rows.append((la + i / 10, la + (i + 1) / 10, lo + j / n_lon, lo + (j + 1) / n_lon))
    a = np.array(rows)
    n = len(a)
    relief = rng.gamma(1.2, 500, n)
    return dict(lat_min=a[:, 0], lat_max=a[:, 1], lon_min=a[:, 2], lon_max=a[:, 3],
                frac_valid=np.ones(n, np.float32), frac_sea=rng.uniform(0, 0.6, n).astype(np.float32),
                elev_mean=np.full(n, 1000, np.float32), relief=relief.astype(np.float32),
                std=(relief / 4).astype(np.float32), slope_mean=(relief / 100).astype(np.float32),
                slope_p90=(relief / 60).astype(np.float32))


def test_global_sampler():
    cfg = load_config(overrides=["global.sampling.n_cases=3000", "global.sampling.n_pilot=100"])
    sites = [dict(name="here", lat=43.6, lon=-116.0, radius_km=20, weight=1, split="test"),
             dict(name="there", lat=39.0, lon=-106.0, radius_km=15, weight=1, split="train")]
    locs, cases = sample_global(cfg, fake_descriptors(), sites)
    assert len(cases) == 3000
    assert len({c["case_id"] for c in cases}) == 3000
    groups = {g: sum(c["group"] == g for c in cases) for g in ("global", "hard", "site")}
    assert groups == {"global": 2100, "hard": 300, "site": 600}
    # Pilot first.
    pil = [c["pilot"] for c in cases]
    k = sum(pil)
    assert 100 <= k <= 110 and all(pil[:k]) and not any(pil[k:])
    # Spacing between global locations.
    g = [l for l in locs if l["group"] != "site"]
    la = np.array([l["lat"] for l in g])
    lo = np.array([l["lon"] for l in g])
    for i in range(0, len(g), 7):
        d = km_between(la[i], lo[i], la, lo)
        d[i] = 1e9
        assert d.min() >= 3.0 - 1e-6
    # Held-out site: nothing else within radius + buffer, and all its cases are test.
    d = km_between(43.6, -116.0, la, lo)
    assert (d >= 20 + 12 - 1e-6).all()
    assert all(c["split"] == "test" for c in cases if c["region"] == "here")
    # Global val/test locations keep clear of other-split blocks.
    bs = BlockSplit(2, [0.9, 0.05, 0.05], cfg["global"]["sampling"]["seed"])
    for l in g:
        assert bs.clean(l["lat"], l["lon"], 12)
    # Directions of one location are spread out.
    by_loc = {}
    for c in cases:
        by_loc.setdefault(c["loc_id"], []).append(c["theta"])
    spreads = [abs(((t[1] - t[0]) + 180) % 360 - 180) for t in by_loc.values() if len(t) == 2]
    assert np.median(spreads) > 60
    json.dumps(cases[:3])


def test_descriptors(tmp_path):
    from windsurrogate.descriptors import load, merge, summarise
    R = C = 300                                    # ~360 m pixels over 1 degree
    lat = 46 - (np.arange(R) + 0.5) / R
    lon = 8 + (np.arange(C) + 0.5) / C
    LON, LAT = np.meshgrid(lon, lat)
    z = 1000 + 800 * np.sin((LON - 8) * 2 * np.pi * 3)    # ridges ~ 26 km apart in x
    z[:30, :] = 0.0                                       # a strip of "sea" in the north
    out = summarise(z.astype(np.float32), 45, 8, cells_per_deg=10)
    assert len(out["relief"]) == 10 * round(10 * math.cos(math.radians(45.5)))
    north = out["lat_max"] > 45.95
    assert np.all(out["frac_sea"][north] > 0.9)
    south = ~north
    assert np.all(out["relief"][south] > 300) and np.all(out["slope_p90"][south] > 3)
    cfg = load_config(overrides=[f"project_dir={tmp_path}"])
    tiles = cfg.path(cfg["global"]["descriptors"]["tiles_dir"])
    tiles.mkdir(parents=True)
    np.savez(tiles / "+45+008.npz", **out)
    np.savez(tiles / "+00+000.npz", missing=np.bool_(True))
    d = load(merge(cfg))
    assert len(d["relief"]) == len(out["relief"])


def test_queue_waiting_and_reorder(tmp_path):
    from windsurrogate.runner import pending_cases, reorder_pending, status_summary
    from windsurrogate.sampling import write_cases
    cfg = load_config(overrides=[f"project_dir={tmp_path}"])
    cases = [dict(case_id=f"g{i:06d}", split="train", stratum="hard" if i % 3 == 0 else "r1",
                  terrain=f"terrain/L000/L{i:06d}.npz", center_e=0.0, center_n=0.0, theta=0.0)
             for i in range(30)]
    write_cases(cfg.path("cases.jsonl"), cases)
    for i in range(10):                            # crops exist for the first 10 only
        p = tmp_path / "terrain" / "L000" / f"L{i:06d}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    (tmp_path / "dataset" / "train").mkdir(parents=True)
    (tmp_path / "dataset" / "train" / "g000001.npz").write_bytes(b"x")
    todo = pending_cases(cfg, 0, 1)
    assert [c["case_id"] for c in todo] == [f"g{i:06d}" for i in range(10) if i != 1]
    st = status_summary(cfg, by="stratum")
    assert st["done"] == 1 and st["waiting_terrain"] == 20 and st["pending"] == 9
    assert st["by_stratum"]["r1"]["done"] == 1
    reorder_pending(cfg, {"hard": 1e6}, seed=1)
    from windsurrogate.sampling import read_cases
    new = read_cases(cfg.path("cases.jsonl"))
    assert new[0]["case_id"] == "g000001"                       # finished stays in front
    assert all(c["stratum"] == "hard" for c in new[1:11])       # then the heavy group
    assert sorted(c["case_id"] for c in new) == sorted(c["case_id"] for c in cases)


def test_worker_picks_up_late_terrain(tmp_path, monkeypatch):
    """Cases whose crop appears while the worker runs are processed without a restart."""
    import threading
    import time as _time

    import windsurrogate.crops as crops_mod
    from windsurrogate import runner
    from windsurrogate.sampling import write_cases
    cfg = load_config(overrides=[f"project_dir={tmp_path}", f"output.work_dir={tmp_path / 'work'}",
                                 "solver.refresh_minutes=0.01", "solver.pin_cores=false"])
    cases = [dict(case_id=f"g{i:06d}", split="train", stratum="r1", terrain=f"terrain/L{i:06d}.npz",
                  center_e=0.0, center_n=0.0, theta=0.0) for i in range(6)]
    write_cases(cfg.path("cases.jsonl"), cases)
    (tmp_path / "terrain").mkdir()

    def make(i):
        (tmp_path / "terrain" / f"L{i:06d}.npz").write_bytes(b"x")
    for i in range(2):
        make(i)

    def fake_run_case(cfg_, geom, dem, case, topo, env, prefix=""):
        out = runner.sample_path(cfg_, case)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        return dict(case, status="done", wall_seconds=0.1)

    monkeypatch.setattr(runner, "foam_env", lambda a: {})
    monkeypatch.setattr(runner, "make_topology", lambda *a, **k: {})
    monkeypatch.setattr(runner, "run_case", fake_run_case)
    monkeypatch.setattr(crops_mod, "CropCache", lambda cfg_: (lambda rel: None))

    def later():
        _time.sleep(1.5)
        for i in range(2, 6):
            make(i)
    th = threading.Thread(target=later)
    th.start()
    runner.worker(cfg, concurrent=2)
    th.join()
    done = sorted(p.stem for p in (tmp_path / "dataset" / "train").glob("*.npz"))
    assert done == [c["case_id"] for c in cases]
