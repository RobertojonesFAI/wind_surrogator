"""Case sampling with a spatial-block train/val/test split, over one or more regions."""
from __future__ import annotations

import json
import math

import numpy as np

from .dem import MasterDEM, bbox_in_crs
from .geometry import CaseGeometry

SPLITS = ("train", "val", "test")


def _sampling_area(geom: CaseGeometry, dem: MasterDEM, region: dict):
    """Rectangle (in the region CRS) where case centres may fall: inside the region bbox and
    at least one CFD-domain radius away from the DEM edges."""
    R = geom.radius + 200.0
    left, bottom, right, top = dem.bounds
    bl, bb, br, bt = bbox_in_crs(region["bbox"], dem.crs)
    a = (max(left + R, bl), max(bottom + R, bb), min(right - R, br), min(top - R, bt))
    if a[2] <= a[0] or a[3] <= a[1]:
        raise ValueError(f"region {region['name']}: DEM too small, each case needs a "
                         f"{2 * R / 1000:.1f} km square of DEM around its centre")
    return a


def block_split(cfg, area, region_index: int):
    """Square blocks over the sampling area, each assigned to one split.

    Every split gets at least one block per region, so small regions still have
    validation and test terrain the network never saw.
    """
    s = cfg["sampling"]
    B = float(s["split_block_km"]) * 1000.0
    x0, y0, x1, y1 = area
    nbx, nby = max(1, math.ceil((x1 - x0) / B)), max(1, math.ceil((y1 - y0) / B))
    nb = nbx * nby
    rng = np.random.default_rng(s["seed"] + 1000 * region_index)
    fr = np.asarray(s["split_fractions"], float)
    fr = fr / fr.sum()
    order = rng.permutation(nb)
    table = np.zeros(nb, dtype=int)
    if nb >= 3:
        n_val = max(1, int(round(fr[1] * nb)))
        n_test = max(1, int(round(fr[2] * nb)))
        table[order[:n_val]] = 1
        table[order[n_val:n_val + n_test]] = 2
    else:
        print(f"warning: only {nb} split block(s); val/test will share terrain with train")
    return B, x0, y0, table.reshape(nby, nbx)


def _allocate(cfg, regs, areas) -> list[int]:
    n = int(cfg["sampling"]["n_cases"])
    w = np.array([r.get("weight") or (a[2] - a[0]) * (a[3] - a[1]) for r, a in zip(regs, areas)], float)
    w = w / w.sum()
    counts = np.floor(w * n).astype(int)
    counts[np.argmax(w)] += n - counts.sum()
    return counts.tolist()


def sample_cases(cfg, regs: list[dict], dems: dict[str, MasterDEM]) -> list[dict]:
    s = cfg["sampling"]
    geom = CaseGeometry.from_config(cfg)
    areas = [_sampling_area(geom, dems[r["name"]], r) for r in regs]
    counts = _allocate(cfg, regs, areas)
    core_r = geom.core_half * math.sqrt(2)  # the core must lie inside a single block
    zlo, zhi = np.log(cfg["flow"]["z0_range"])
    cases = []
    for ri, (reg, area, n_reg) in enumerate(zip(regs, areas, counts)):
        dem = dems[reg["name"]]
        B, bx0, by0, table = block_split(cfg, area, ri)
        rng = np.random.default_rng(s["seed"] + 1 + 7919 * ri)
        got, tries = 0, 0
        while got < n_reg:
            tries += 1
            if tries > 500 * max(n_reg, 1):
                raise RuntimeError(f"region {reg['name']}: could not sample enough cases; "
                                   "enlarge the bbox or lower sampling.min_relief_m")
            e = rng.uniform(area[0], area[2])
            nn = rng.uniform(area[1], area[3])
            bi, bj = int((e - bx0) // B), int((nn - by0) // B)
            lx, ly = e - bx0 - bi * B, nn - by0 - bj * B
            # Keep the core inside its block (where the block is big enough for that).
            bw = min(B, area[2] - bx0 - bi * B)
            bh = min(B, area[3] - by0 - bj * B)
            if bw > 2 * core_r and min(lx, bw - lx) < core_r:
                continue
            if bh > 2 * core_r and min(ly, bh - ly) < core_r:
                continue
            p = dem.patch(e, nn, geom.core_half)
            relief = float(np.nanmax(p.data) - np.nanmin(p.data))
            if not np.isfinite(relief) or relief < s["min_relief_m"]:
                continue
            if rng.random() > min(1.0, s["relief_floor"] + relief / s["relief_ref_m"]):
                continue
            cases.append(dict(
                case_id=f"c{len(cases):05d}",
                region=reg["name"],
                center_e=round(e, 1), center_n=round(nn, 1),
                theta=round(float(rng.uniform(0, 360)), 2),
                z0=round(float(np.exp(rng.uniform(zlo, zhi))), 4),
                split=SPLITS[table[min(bj, table.shape[0] - 1), min(bi, table.shape[1] - 1)]],
                relief=round(relief, 1),
            ))
            got += 1
    # Shuffle so that any prefix of the list (what a first batch of runs covers) mixes
    # regions, splits and directions; ids follow the shuffled order.
    order = np.random.default_rng(s["seed"] + 99).permutation(len(cases))
    cases = [cases[i] for i in order]
    for i, c in enumerate(cases):
        c["case_id"] = f"c{i:05d}"
    return cases


def write_cases(path, cases):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")


def read_cases(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
