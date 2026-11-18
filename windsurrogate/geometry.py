"""Fixed case geometry in the rotated (wind-aligned) frame.

Coordinates are metres relative to the centre of the labelled core tile.
Wind always blows towards +x. Every case shares the same horizontal node
layout and the same (nx, ny, nz) cell counts, so one blockMesh topology is
reused for all cases and only the point coordinates change.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


def _stretched_sizes(length: float, d0: float, growth: float, dmax: float) -> np.ndarray:
    """Cell sizes growing geometrically from d0 (exclusive) up to dmax, scaled to fit `length`."""
    if length <= 0:
        return np.zeros(0)
    sizes = []
    d = d0
    while sum(sizes) < length:
        d = min(d * growth, dmax)
        sizes.append(d)
    sizes = np.asarray(sizes)
    # Drop a last tiny cell if overshoot is large, then rescale exactly.
    if len(sizes) > 1 and sizes[:-1].sum() > length - 0.5 * sizes[-1]:
        sizes = sizes[:-1]
    return sizes * (length / sizes.sum())


def vertical_ratio(first: float, total: float, n: int) -> float:
    """Geometric ratio r so that first*(r^n-1)/(r-1) == total."""
    if abs(first * n - total) < 1e-9:
        return 1.0
    lo, hi = 1.0 + 1e-9, 2.0
    while first * (hi**n - 1) / (hi - 1) < total:
        hi *= 1.5
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if first * (mid**n - 1) / (mid - 1) < total:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class CaseGeometry:
    dx: float
    core_px: int
    input_upwind_px: int
    input_downwind_px: int
    input_lateral_px: int
    fine_margin_cells: int
    taper_m: float
    flat_m: float
    growth: float
    max_cell_m: float
    nz: int
    first_cell_m: float
    top_above_max_m: float
    min_top_m: float

    @classmethod
    def from_config(cls, cfg) -> "CaseGeometry":
        g = cfg["geometry"]
        return cls(**{k: g[k] for k in cls.__dataclass_fields__})

    # ---- windows (metres) -------------------------------------------------
    @property
    def core_half(self) -> float:
        return 0.5 * self.core_px * self.dx

    @property
    def core_bounds(self) -> tuple[float, float, float, float]:
        h = self.core_half
        return (-h, h, -h, h)

    @property
    def input_bounds(self) -> tuple[float, float, float, float]:
        """Network input window == region where CFD terrain is the real terrain."""
        h, d = self.core_half, self.dx
        return (-h - self.input_upwind_px * d, h + self.input_downwind_px * d,
                -h - self.input_lateral_px * d, h + self.input_lateral_px * d)

    @property
    def input_shape(self) -> tuple[int, int]:
        """(rows=y, cols=x) of the network input grid."""
        return (self.core_px + 2 * self.input_lateral_px,
                self.core_px + self.input_upwind_px + self.input_downwind_px)

    @property
    def core_slice_in_input(self) -> tuple[slice, slice]:
        r0 = self.input_lateral_px
        c0 = self.input_upwind_px
        return slice(r0, r0 + self.core_px), slice(c0, c0 + self.core_px)

    @property
    def domain_bounds(self) -> tuple[float, float, float, float]:
        x0, x1, y0, y1 = self.input_bounds
        b = self.taper_m + self.flat_m
        return (x0 - b, x1 + b, y0 - b, y1 + b)

    @property
    def fine_bounds(self) -> tuple[float, float, float, float]:
        h = self.core_half + self.fine_margin_cells * self.dx
        return (-h, h, -h, h)

    @property
    def radius(self) -> float:
        """Max distance from the core centre to any domain corner (any wind direction)."""
        x0, x1, y0, y1 = self.domain_bounds
        return float(np.hypot(max(abs(x0), abs(x1)), max(abs(y0), abs(y1))))

    # ---- node arrays ---------------------------------------------------------
    def _axis_nodes(self, lo: float, hi: float, flo: float, fhi: float) -> np.ndarray:
        nfine = int(round((fhi - flo) / self.dx))
        fine = flo + self.dx * np.arange(nfine + 1)
        left = _stretched_sizes(flo - lo, self.dx, self.growth, self.max_cell_m)
        right = _stretched_sizes(hi - fhi, self.dx, self.growth, self.max_cell_m)
        left_nodes = flo - np.cumsum(left)[::-1]
        right_nodes = fhi + np.cumsum(right)
        nodes = np.concatenate([left_nodes, fine, right_nodes])
        nodes[0], nodes[-1] = lo, hi
        return nodes

    @cached_property
    def x_nodes(self) -> np.ndarray:
        x0, x1, _, _ = self.domain_bounds
        f0, f1, _, _ = self.fine_bounds
        return self._axis_nodes(x0, x1, f0, f1)

    @cached_property
    def y_nodes(self) -> np.ndarray:
        _, _, y0, y1 = self.domain_bounds
        _, _, f0, f1 = self.fine_bounds
        return self._axis_nodes(y0, y1, f0, f1)

    @property
    def nx(self) -> int:
        return len(self.x_nodes) - 1

    @property
    def ny(self) -> int:
        return len(self.y_nodes) - 1

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz

    @cached_property
    def core_cell_index(self) -> tuple[slice, slice]:
        """(j-slice, i-slice) of core cells in the CFD cell grid."""
        xc = 0.5 * (self.x_nodes[1:] + self.x_nodes[:-1])
        yc = 0.5 * (self.y_nodes[1:] + self.y_nodes[:-1])
        i0 = int(np.argmin(np.abs(xc - (-self.core_half + 0.5 * self.dx))))
        j0 = int(np.argmin(np.abs(yc - (-self.core_half + 0.5 * self.dx))))
        return slice(j0, j0 + self.core_px), slice(i0, i0 + self.core_px)

    def z_flat_nodes(self, top: float) -> np.ndarray:
        r = vertical_ratio(self.first_cell_m, top, self.nz)
        sizes = self.first_cell_m * r ** np.arange(self.nz)
        z = np.concatenate([[0.0], np.cumsum(sizes)])
        z[-1] = top
        return z

    def domain_top(self, h_max_rel: float) -> float:
        return float(max(self.min_top_m, h_max_rel + self.top_above_max_m))

    # ---- grids for sampling ---------------------------------------------------
    def input_cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        x0, _, y0, _ = self.input_bounds
        rows, cols = self.input_shape
        xc = x0 + self.dx * (np.arange(cols) + 0.5)
        yc = y0 + self.dx * (np.arange(rows) + 0.5)
        return xc, yc

    def taper_weight(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """1 inside the input window, cosine ramp to 0 over taper_m, 0 beyond."""
        x0, x1, y0, y1 = self.input_bounds

        def ramp(v, a, b):
            d = np.maximum(np.maximum(a - v, v - b), 0.0)
            t = np.clip(d / self.taper_m, 0.0, 1.0)
            return 0.5 * (1.0 + np.cos(np.pi * t))

        return ramp(x, x0, x1) * ramp(y, y0, y1)

    def summary(self) -> str:
        x0, x1, y0, y1 = self.domain_bounds
        return (f"domain {x1 - x0:.0f} x {y1 - y0:.0f} m, cells {self.nx} x {self.ny} x {self.nz} = "
                f"{self.n_cells / 1e6:.2f} M, core {self.core_px}^2 px at {self.dx:.0f} m, "
                f"input {self.input_shape[0]}x{self.input_shape[1]} px")
