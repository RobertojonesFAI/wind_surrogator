"""Terrain extraction in the rotated, wind-aligned frame."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .dem import MasterDEM
from .geometry import CaseGeometry


def frame_axes(theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors (east, north) of the rotated x' (downwind) and y' (left of x') axes.

    theta is the meteorological wind direction: where the wind comes FROM,
    degrees clockwise from north (270 = westerly wind blowing towards the east).
    """
    t = np.deg2rad(theta_deg)
    ex = np.array([-np.sin(t), -np.cos(t)])
    ey = np.array([np.cos(t), -np.sin(t)])  # ex rotated +90 deg
    return ex, ey


def rotated_to_map(xr, yr, e_c, n_c, theta_deg):
    ex, ey = frame_axes(theta_deg)
    return e_c + xr * ex[0] + yr * ey[0], n_c + xr * ex[1] + yr * ey[1]


def vectors_to_map(u, v, theta_deg):
    """Rotate (u', v') components in the wind frame to (east, north)."""
    ex, ey = frame_axes(theta_deg)
    return u * ex[0] + v * ey[0], u * ex[1] + v * ey[1]


class SmoothedSampler:
    """Bilinear DEM sampling with anti-aliasing matched to the local target cell size."""

    def __init__(self, dem: MasterDEM, cell_sizes=(50.0, 100.0, 200.0, 400.0)):
        self.dem = dem
        self.cell_sizes = np.asarray(cell_sizes, dtype=float)
        self.levels = []
        for c in self.cell_sizes:
            sigma = 0.5 * c / dem.res
            self.levels.append(ndimage.gaussian_filter(dem.data, sigma, mode="nearest")
                               if sigma > 0.6 else dem.data)

    def sample(self, e, n, cell=None):
        e = np.asarray(e, float)
        n = np.asarray(n, float)
        r, c = self.dem.to_pixel(e, n)
        if cell is None:
            return ndimage.map_coordinates(self.levels[0], [r, c], order=1, mode="nearest")
        cell = np.broadcast_to(np.asarray(cell, float), e.shape)
        # Linear blend between the two bracketing smoothing levels (in log space).
        lc = np.log(np.clip(cell, self.cell_sizes[0], self.cell_sizes[-1]))
        ls = np.log(self.cell_sizes)
        idx = np.clip(np.searchsorted(ls, lc) - 1, 0, len(ls) - 2)
        w = (lc - ls[idx]) / (ls[idx + 1] - ls[idx])
        out = np.zeros(e.shape)
        for lev in range(len(ls) - 1):
            m = idx == lev
            if not m.any():
                continue
            a = ndimage.map_coordinates(self.levels[lev], [r[m], c[m]], order=1, mode="nearest")
            b = ndimage.map_coordinates(self.levels[lev + 1], [r[m], c[m]], order=1, mode="nearest")
            out[m] = (1 - w[m]) * a + w[m] * b
        return out


@dataclass
class CaseTerrain:
    base: float                 # absolute elevation mapped to z=0 (flat inflow ground)
    h_nodes: np.ndarray         # (ny+1, nx+1) CFD ground node heights relative to base (tapered)
    h_input: np.ndarray         # (rows, cols) network input terrain relative to base (real)
    top: float                  # domain top (m above base)
    max_slope_deg: float


def case_terrain(geom: CaseGeometry, dem: MasterDEM, e_c: float, n_c: float, theta: float) -> CaseTerrain:
    patch = dem.patch(e_c, n_c, geom.radius + 1000.0)
    sampler = SmoothedSampler(patch, cell_sizes=(geom.dx, 2 * geom.dx, 4 * geom.dx,
                                                 max(8 * geom.dx, geom.max_cell_m)))
    # Network input grid (cell centres, real terrain).
    xc, yc = geom.input_cell_centres()
    XC, YC = np.meshgrid(xc, yc)
    E, N = rotated_to_map(XC, YC, e_c, n_c, theta)
    h_in = sampler.sample(E, N, cell=geom.dx)

    # Base: mean elevation on the input-window perimeter, so the taper is as gentle as possible.
    perim = np.concatenate([h_in[0], h_in[-1], h_in[:, 0], h_in[:, -1]])
    base = float(np.mean(perim))

    # CFD node grid with anti-aliasing matched to the local cell size.
    xn, yn = geom.x_nodes, geom.y_nodes
    dxn = np.maximum(np.gradient(xn), geom.dx)
    dyn = np.maximum(np.gradient(yn), geom.dx)
    XN, YN = np.meshgrid(xn, yn)
    cell = np.maximum(dxn[None, :], dyn[:, None])
    E, N = rotated_to_map(XN, YN, e_c, n_c, theta)
    h_nodes = sampler.sample(E, N, cell=cell) - base
    h_nodes *= geom.taper_weight(XN, YN)

    gy, gx = np.gradient(h_nodes, yn, xn)
    slope = float(np.degrees(np.arctan(np.hypot(gx, gy).max())))
    top = geom.domain_top(float(h_nodes.max()))
    return CaseTerrain(base=base, h_nodes=h_nodes, h_input=(h_in - base).astype(np.float32),
                       top=top, max_slope_deg=slope)


def mesh_points(geom: CaseGeometry, terrain: CaseTerrain) -> np.ndarray:
    """Terrain-following node coordinates, shape (nz+1, ny+1, nx+1, 3).

    Linear sigma transform: z = zf + h * (1 - zf / top); the top stays flat and
    every face on x = const / y = const stays planar (only z is moved).
    """
    zf = geom.z_flat_nodes(terrain.top)
    h = terrain.h_nodes
    X = np.broadcast_to(geom.x_nodes[None, None, :], (len(zf), *h.shape))
    Y = np.broadcast_to(geom.y_nodes[None, :, None], (len(zf), *h.shape))
    Z = zf[:, None, None] + h[None] * (1.0 - zf[:, None, None] / terrain.top)
    return np.stack([X, Y, Z], axis=-1)
