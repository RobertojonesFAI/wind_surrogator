"""Stratified worldwide case sampling.

Three groups of cases:
  global  cells of the descriptor grid drawn per relief class (fixed share per class),
          spread across the world (cells in crowded 10-degree boxes are down-weighted)
  hard    the steepest / highest-relief cells (deep narrow valleys, big walls)
  site    discs around the targeted sites of config/sites.yaml

Each location gets `dirs_per_location` wind directions spread around the compass.
Splits are assigned per geographic block (split_block_deg); locations closer than
split_buffer_km to a block of another split are rejected, so validation and test
terrain never overlaps training terrain. Val/test sites are held out the same way.

The output list is a queue: the pilot cases come first, then everything else in random
order, so any prefix of the list (whatever the compute allows) is a balanced sample.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml

from .config import DEFAULT_CONFIG
from .geometry import CaseGeometry

SPLITS = ("train", "val", "test")
KM_PER_DEG = 111.32


def resolve_path(cfg, p) -> Path:
    """Absolute path; relative paths are tried in the project, the working dir, the repo."""
    p = Path(str(p)).expanduser()
    if p.is_absolute():
        return p
    for base in (cfg.root, Path.cwd(), DEFAULT_CONFIG.parent.parent):
        if (base / p).exists():
            return base / p
    return DEFAULT_CONFIG.parent.parent / p


def load_sites(path) -> list[dict]:
    with open(path) as f:
        doc = yaml.safe_load(f)
    dflt = doc.get("defaults", {})
    out = []
    for s in doc["sites"]:
        site = dict(dflt, **s)
        if site.get("split", "train") not in SPLITS:
            raise ValueError(f"site {site['name']}: bad split {site['split']}")
        out.append(site)
    names = [s["name"] for s in out]
    if len(set(names)) != len(names):
        raise ValueError("duplicate site names")
    return out


def km_between(lat1, lon1, lat2, lon2):
    """Equirectangular distance (km); fine below a few hundred km."""
    lat1, lon1, lat2, lon2 = map(np.asarray, (lat1, lon1, lat2, lon2))
    dlon = (lon2 - lon1 + 180.0) % 360.0 - 180.0
    x = np.radians(dlon) * np.cos(np.radians(0.5 * (lat1 + lat2)))
    y = np.radians(lat2 - lat1)
    return 6371.0 * np.hypot(x, y)


class SpacingIndex:
    """Reject points closer than `km` to an already accepted point (hash grid in degrees)."""

    def __init__(self):
        self.cells: dict[tuple[int, int], list[tuple[float, float, float]]] = {}
        self.g = 0.1                      # grid size in degrees of latitude (~11 km)

    def _key(self, lat, lon):
        return int(math.floor(lat / self.g)), int(math.floor(((lon + 180.0) % 360.0) / self.g))

    def ok(self, lat, lon, km):
        ki, kj = self._key(lat, lon)
        nlat = int(math.ceil(km / (self.g * KM_PER_DEG))) + 1
        coslat = max(math.cos(math.radians(lat)), 0.05)
        nlon = int(math.ceil(km / (self.g * KM_PER_DEG * coslat))) + 1
        nj = int(round(360.0 / self.g))
        for di in range(-nlat, nlat + 1):
            for dj in range(-nlon, nlon + 1):
                for (la, lo, r) in self.cells.get((ki + di, (kj + dj) % nj), ()):
                    dlon = (lo - lon + 180.0) % 360.0 - 180.0
                    x = math.radians(dlon) * math.cos(math.radians(0.5 * (lat + la)))
                    if 6371.0 * math.hypot(x, math.radians(la - lat)) < max(km, r):
                        return False
        return True

    def add(self, lat, lon, km):
        self.cells.setdefault(self._key(lat, lon), []).append((lat, lon, km))


class BlockSplit:
    """Deterministic split per geographic block of `deg` degrees."""

    def __init__(self, deg: float, fractions, seed: int):
        self.deg = float(deg)
        fr = np.asarray(fractions, float)
        self.cum = np.cumsum(fr / fr.sum())
        self.seed = int(seed)
        self._cache: dict[tuple[int, int], int] = {}

    def block(self, lat, lon):
        return int(math.floor(lat / self.deg)), int(math.floor(((lon + 180.0) % 360.0) / self.deg))

    def split_of_block(self, b) -> int:
        if b not in self._cache:
            u = np.random.default_rng([self.seed, b[0] + 1000, b[1] + 1000]).random()
            self._cache[b] = int(np.searchsorted(self.cum, u, side="right"))
        return min(self._cache[b], len(self.cum) - 1)

    def split(self, lat, lon) -> int:
        return self.split_of_block(self.block(lat, lon))

    def clean(self, lat, lon, buffer_km) -> bool:
        """True if no block of another split lies within buffer_km."""
        s = self.split(lat, lon)
        dlat = buffer_km / KM_PER_DEG
        dlon = buffer_km / (KM_PER_DEG * max(math.cos(math.radians(lat)), 0.05))
        for la in (lat - dlat, lat, lat + dlat):
            for lo in (lon - dlon, lon, lon + dlon):
                if self.split(la, lo) != s:
                    return False
        return True


def _directions(rng, k: int) -> list[float]:
    """k directions spread around the compass (one per 360/k sector, random within it)."""
    base = rng.uniform(0, 360)
    return [round(float((base + (i + rng.uniform(0, 1)) * 360.0 / k) % 360.0), 2) for i in range(k)]


def _spread_weights(lat, lon, box_deg, power):
    """Weight ~ 1 / n_box^power, n_box = number of candidate cells in the same box."""
    keys = (np.floor(lat / box_deg).astype(np.int64) * 10000
            + np.floor(((lon + 180.0) % 360.0) / box_deg).astype(np.int64))
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv].astype(float) ** power


class CellLookup:
    """Find the descriptor cell containing a point."""

    def __init__(self, desc: dict):
        self.d = desc
        self.index: dict[tuple[int, int], list[int]] = {}
        for i, (la, lo) in enumerate(zip(desc["lat_min"], desc["lon_min"])):
            self.index.setdefault((int(math.floor(la)), int(math.floor(lo))), []).append(i)

    def find(self, lat, lon) -> int | None:
        for i in self.index.get((int(math.floor(lat)), int(math.floor(lon))), ()):
            d = self.d
            if d["lat_min"][i] <= lat < d["lat_max"][i] and d["lon_min"][i] <= lon < d["lon_max"][i]:
                return i
        return None


def sample_global(cfg, desc: dict, sites: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (locations, cases)."""
    g = cfg["global"]["sampling"]
    tdir = cfg["global"]["terrain"]["dir"]
    rng = np.random.default_rng(int(g["seed"]))
    n_total = int(g["n_cases"])
    fr = g["fractions"]
    n_site_cases = int(round(n_total * fr["sites"])) if sites else 0
    n_hard_cases = int(round(n_total * fr["hard"]))
    n_global_cases = n_total - n_site_cases - n_hard_cases
    k_glob, k_site = int(g["dirs_per_location"]), int(g["site_dirs_per_location"])
    splitter = BlockSplit(g["split_block_deg"], g["split_fractions"], int(g["seed"]))
    buf = float(g["split_buffer_km"])
    spacing = SpacingIndex()
    locations: list[dict] = []

    # Exclusion discs around sites (no global/hard case there; val/test sites stay clean).
    site_lat = np.array([s["lat"] for s in sites]) if sites else np.zeros(0)
    site_lon = np.array([s["lon"] for s in sites]) if sites else np.zeros(0)
    site_excl = np.array([float(s["radius_km"]) + buf for s in sites]) if sites else np.zeros(0)

    def in_site_zone(lat, lon):
        if not len(site_lat):
            return False
        return bool((km_between(lat, lon, site_lat, site_lon) < site_excl).any())

    lookup = CellLookup(desc)

    def add_location(lat, lon, group, stratum, region, split, ndirs, cell=None):
        if cell is None:
            cell = lookup.find(lat, lon)
        rel = float(desc["relief"][cell]) if cell is not None else float("nan")
        slp = float(desc["slope_p90"][cell]) if cell is not None else float("nan")
        loc_id = f"L{len(locations):06d}"
        locations.append(dict(loc_id=loc_id, lat=round(float(lat), 6), lon=round(float(lon), 6),
                              group=group, stratum=stratum, region=region, split=SPLITS[split],
                              relief=None if math.isnan(rel) else round(rel, 1),
                              slope_p90=None if math.isnan(slp) else round(slp, 2),
                              terrain=f"{tdir}/{loc_id[:-3]}/{loc_id}.npz", ndirs=int(ndirs)))

    # ---- candidate cells ------------------------------------------------------------
    lat_c = 0.5 * (desc["lat_min"] + desc["lat_max"])
    lon_c = 0.5 * (desc["lon_min"] + desc["lon_max"])
    relief = desc["relief"].astype(float)
    slope90 = desc["slope_p90"].astype(float)
    ok = ((desc["frac_valid"] >= g["min_valid_frac"]) & (desc["frac_sea"] <= g["max_sea_frac"])
          & np.isfinite(relief) & (lat_c >= g["lat_range"][0]) & (lat_c < g["lat_range"][1]))
    cand = np.flatnonzero(ok)
    print(f"{len(relief)} descriptor cells, {len(cand)} usable")

    def draw_from(pool: np.ndarray, n_locs: int, group: str, stratum: str, ndirs: int) -> int:
        """Draw n_locs locations from cells in `pool` (with replacement of cells)."""
        if n_locs <= 0 or len(pool) == 0:
            return 0
        w = _spread_weights(lat_c[pool], lon_c[pool], g["box_deg"], g["spread_power"])
        p = w / w.sum()
        got, tries = 0, 0
        while got < n_locs and tries < 60 * n_locs + 1000:
            batch = rng.choice(pool, size=min(4 * (n_locs - got) + 16, 200000), p=p)
            for c in batch:
                tries += 1
                lat = rng.uniform(desc["lat_min"][c], desc["lat_max"][c])
                lon = rng.uniform(desc["lon_min"][c], desc["lon_max"][c])
                lon = (lon + 180.0) % 360.0 - 180.0
                if in_site_zone(lat, lon) or not splitter.clean(lat, lon, buf):
                    continue
                if not spacing.ok(lat, lon, g["min_spacing_km"]):
                    continue
                spacing.add(lat, lon, g["min_spacing_km"])
                add_location(lat, lon, group, stratum, group, splitter.split(lat, lon), ndirs, cell=int(c))
                got += 1
                if got >= n_locs:
                    break
        if got < n_locs:
            print(f"  warning: {group}/{stratum}: only {got} of {n_locs} locations (pool too small "
                  f"for min_spacing_km)")
        return got

    # ---- global, per relief class ---------------------------------------------------
    bins = np.asarray(g["relief_bins"], float)
    rw = np.asarray(g["relief_weights"], float)
    rw = rw / rw.sum()
    n_glob_locs = int(math.ceil(n_global_cases / k_glob))
    rbin = np.digitize(relief, bins) - 1
    for b in range(len(rw)):
        pool = cand[rbin[cand] == b]
        n_b = int(round(n_glob_locs * rw[b]))
        got = draw_from(pool, n_b, "global", f"r{b}", k_glob)
        print(f"  global r{b} relief {bins[b]:.0f}-{bins[b + 1]:.0f} m: {len(pool)} cells, {got} locations")

    # ---- hard -----------------------------------------------------------------------
    q = float(g["hard_quantile"])
    s_thr = np.nanquantile(slope90[cand], q)
    r_thr = np.nanquantile(relief[cand], q)
    hard_pool = cand[(slope90[cand] >= s_thr) | (relief[cand] >= r_thr)]
    n_hard_locs = int(math.ceil(n_hard_cases / k_glob))
    got = draw_from(hard_pool, n_hard_locs, "hard", "hard", k_glob)
    print(f"  hard (slope_p90 >= {s_thr:.1f} deg or relief >= {r_thr:.0f} m): "
          f"{len(hard_pool)} cells, {got} locations")

    # ---- sites ----------------------------------------------------------------------
    if sites:
        wsum = sum(float(s["weight"]) for s in sites)
        for s in sites:
            n_cases_s = n_site_cases * float(s["weight"]) / wsum
            n_locs = int(math.ceil(n_cases_s / k_site))
            r = float(s["radius_km"])
            sp = float(g["site_min_spacing_km"])
            got, tries = 0, 0
            while got < n_locs and tries < 200 * n_locs:
                tries += 1
                rr, aa = r * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
                lat = s["lat"] + rr * math.cos(aa) / KM_PER_DEG
                lon = s["lon"] + rr * math.sin(aa) / (KM_PER_DEG * math.cos(math.radians(s["lat"])))
                lon = (lon + 180.0) % 360.0 - 180.0
                if not spacing.ok(lat, lon, sp):
                    continue
                spacing.add(lat, lon, sp)
                add_location(lat, lon, "site", "site", s["name"], SPLITS.index(s["split"]), k_site)
                got += 1
            if got < n_locs:
                print(f"  warning: site {s['name']}: {got} of {n_locs} locations")
        print(f"  sites: {len(sites)} sites")

    # ---- cases ----------------------------------------------------------------------
    zlo, zhi = np.log(g["z0_scale_range"])
    cases = []
    for loc in locations:
        for th in _directions(rng, loc["ndirs"]):
            cases.append(dict(loc_id=loc["loc_id"], region=loc["region"], group=loc["group"],
                              stratum=loc["stratum"], lat=loc["lat"], lon=loc["lon"],
                              terrain=loc["terrain"], center_e=0.0, center_n=0.0, theta=th,
                              z0_scale=round(float(np.exp(rng.uniform(zlo, zhi))), 4),
                              split=loc["split"], relief=loc["relief"], pilot=False))
    # Trim each group to its target (directions overshoot by up to k-1 per location).
    targets = {"global": n_global_cases, "hard": n_hard_cases, "site": n_site_cases}
    by_group: dict[str, list[dict]] = {}
    for c in cases:
        by_group.setdefault(c["group"], []).append(c)
    cases = []
    for grp, lst in by_group.items():
        idx = rng.permutation(len(lst))[: targets.get(grp, len(lst))]
        cases += [lst[i] for i in sorted(idx)]

    # ---- pilot first, then a random queue -------------------------------------------
    n_pilot = int(g.get("n_pilot", 0))
    order = rng.permutation(len(cases))
    pilot_idx: list[int] = []
    if n_pilot:
        share = {"global": 0.60, "hard": 0.25, "site": 0.15}
        strata = sorted({c["stratum"] for c in cases if c["group"] == "global"})
        quota: dict[tuple[str, str], int] = {}
        for grp, sh in share.items():
            if grp == "global" and strata:
                for st in strata:
                    quota[(grp, st)] = int(round(n_pilot * sh / len(strata)))
            else:
                quota[(grp, grp if grp == "hard" else "site")] = int(round(n_pilot * sh))
        used_locs = set()
        for i in order:
            c = cases[i]
            key = (c["group"], c["stratum"])
            if quota.get(key, 0) > 0 and c["loc_id"] not in used_locs:
                quota[key] -= 1
                used_locs.add(c["loc_id"])
                pilot_idx.append(int(i))
        # The steepest locations of the whole list are in the pilot too (worst-case mesh check).
        steep = sorted((i for i in range(len(cases)) if cases[i]["relief"] is not None),
                       key=lambda i: -cases[i]["relief"])
        for i in steep[: max(1, n_pilot // 20)]:
            if cases[i]["loc_id"] not in used_locs:
                used_locs.add(cases[i]["loc_id"])
                pilot_idx.append(int(i))
    pset = set(pilot_idx)
    queue = [cases[i] for i in rng.permutation(pilot_idx)] + [cases[i] for i in order if i not in pset]
    for i, c in enumerate(queue):
        c["pilot"] = i < len(pilot_idx)
    queue = [dict(case_id=f"g{i:06d}", **c) for i, c in enumerate(queue)]
    used = {c["loc_id"] for c in queue}
    locations = [loc for loc in locations if loc["loc_id"] in used]
    pilot_locs = {c["loc_id"] for c in queue if c["pilot"]}
    for loc in locations:
        loc["pilot"] = loc["loc_id"] in pilot_locs
    return locations, queue


def summary(cases: list[dict]) -> str:
    from collections import Counter
    n_pilot = sum(c["pilot"] for c in cases)
    lines = [f"{len(cases)} cases, {len({c['loc_id'] for c in cases})} locations, "
             f"{n_pilot} pilot cases (pilot run with 8 shards: LIMIT={math.ceil(n_pilot / 8)})"]
    for key in ("group", "stratum", "split"):
        cnt = Counter(c[key] for c in cases)
        lines.append(f"  by {key}: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


def run_sampling(cfg, resample: bool = False) -> tuple[Path, Path]:
    from .descriptors import load
    g = cfg["global"]
    cpath = cfg.path(cfg["output"]["cases_file"])
    lpath = cfg.path(g["terrain"]["locations_file"])
    if cpath.exists() and not resample:
        raise SystemExit(f"{cpath} exists; pass --resample to overwrite it (finished cases keep "
                         f"their files, but case ids would change)")
    desc = load(cfg.path(g["descriptors"]["file"]))
    sf = g["sampling"].get("sites_file")
    sites = load_sites(resolve_path(cfg, sf)) if sf else []
    locations, cases = sample_global(cfg, desc, sites)
    write_jsonl(lpath, locations)
    write_jsonl(cpath, cases)
    print(summary(cases))
    geom = CaseGeometry.from_config(cfg)
    print(f"geometry: {geom.summary()}")
    print(f"wrote {lpath} and {cpath}")
    return lpath, cpath
