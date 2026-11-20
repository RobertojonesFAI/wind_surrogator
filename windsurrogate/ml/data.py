"""Dataset: terrain window -> normalised wind at fixed heights above ground."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

H_SCALE = 500.0          # m, terrain normalisation
Z0_CENTER, Z0_SCALE = -1.0, 0.7   # log10(z0) normalisation
CHANNELS_PER_HEIGHT = ("u", "v", "w", "k")


def input_channels(h: np.ndarray, dx: float, z0: float | np.ndarray | None, core_slice,
                   log10_z0: np.ndarray | None = None) -> np.ndarray:
    """Build (4, rows, cols) network input from terrain h (m) on the wind-aligned grid.

    Channels: elevation relative to the core mean / H_SCALE, dh/dx, dh/dy, log10(z0).
    Roughness is a scalar or map `z0` (m), or directly a log10(z0) map `log10_z0`.
    """
    h = np.asarray(h, np.float32)
    ref = float(h[core_slice].mean())
    gy, gx = np.gradient(h, dx)
    lz = (np.asarray(log10_z0, np.float32) if log10_z0 is not None
          else np.log10(np.asarray(z0, np.float32)))
    z0c = (np.broadcast_to(lz, h.shape) - Z0_CENTER) / Z0_SCALE
    return np.stack([(h - ref) / H_SCALE, gx, gy, z0c]).astype(np.float32)


def normalise_targets(U: np.ndarray, k: np.ndarray, U_in: np.ndarray, k_in: np.ndarray) -> np.ndarray:
    """U (H,3,N,N), k (H,N,N) -> (4H, N, N): u,v,w / U_in(z) and k / k_in, per height."""
    out = []
    for i in range(U.shape[0]):
        out += [U[i, 0] / U_in[i], U[i, 1] / U_in[i], U[i, 2] / U_in[i], k[i] / k_in[i]]
    return np.stack(out).astype(np.float32)


def denormalise(pred: np.ndarray, U_in: np.ndarray, k_in: np.ndarray):
    """(4H, ...) -> U (H,3,...), k (H,...)."""
    H = len(U_in)
    p = pred.reshape(H, 4, *pred.shape[1:])
    U = p[:, :3] * np.asarray(U_in).reshape(H, 1, *([1] * (p.ndim - 2)))
    k = p[:, 3] * np.asarray(k_in).reshape(H, *([1] * (p.ndim - 2)))
    return U, k


def list_samples(dataset_dir: Path, split: str, max_probe_change: float | None = None) -> list[Path]:
    """Samples of a split; optionally drop cases whose probes were still moving at the end
    (max change over the last window, relative to Uref, above `max_probe_change`) unless
    their label is the iteration average of an oscillating solution."""
    # c*: regional cases (v1), g*: global cases. Hidden ".*.tmp.npz" files are in-progress writes.
    files = sorted(f for f in (Path(dataset_dir) / split).glob("[cg]*.npz"))
    if not max_probe_change:
        return files
    keep = []
    for f in files:
        with np.load(f) as z:
            meta = json.loads(str(z["meta"]))
        if meta.get("probe_max_change", 0.0) <= max_probe_change or meta.get("averaged"):
            keep.append(f)
    return keep


class WindDataset:
    """Torch-style dataset (no torch import needed to use it)."""

    def __init__(self, files, geom, flip_augment: bool = False, seed: int = 0):
        self.files = list(files)
        self.geom = geom
        self.flip = flip_augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def load(self, f):
        with np.load(f) as z:
            meta = json.loads(str(z["meta"]))
            lz = z["log10_z0_input"].astype(np.float32) if "log10_z0_input" in z.files else None
            x = input_channels(z["terrain_input"], self.geom.dx, meta["z0"], self.geom.core_slice_in_input,
                               log10_z0=lz)
            y = normalise_targets(z["U"].astype(np.float32), z["k"].astype(np.float32),
                                  z["U_inflow"], z["k_inflow"])
        return x, y, meta

    def __getitem__(self, i):
        x, y, _ = self.load(self.files[i])
        if self.flip and self.rng.random() < 0.5:
            # Mirror across the wind axis: y' -> -y'. Valid for neutral flow without Coriolis.
            x = x[:, ::-1].copy()
            x[2] *= -1.0                       # dh/dy
            y = y[:, ::-1].copy()
            y[1::4] *= -1.0                    # v at every height
        return x, y
