"""Quick-look figures for CFD samples (QA of the dataset)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def plot_sample(npz_path, out_png, dx: float = 50.0, heights=(10, 100)) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(npz_path)
    meta = json.loads(str(z["meta"]))
    H = list(z["heights"].astype(int))
    hc = z["terrain_core"]
    N = hc.shape[0]
    ext = [-N * dx / 2000, N * dx / 2000, -N * dx / 2000, N * dx / 2000]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    ax = axes[0, 0]
    im = ax.imshow(z["terrain_input"], origin="lower", cmap="terrain")
    r0, c0 = (z["terrain_input"].shape[0] - N) // 2, z["terrain_input"].shape[1] - N - (z["terrain_input"].shape[0] - N) // 2
    ax.add_patch(plt.Rectangle((c0, r0), N, N, fill=False, ec="k", lw=1.5))
    ax.set_title("input window terrain (m rel. base); box = core; wind →")
    fig.colorbar(im, ax=ax, shrink=0.8)
    for col, h in enumerate(heights[:2], start=1):
        i = H.index(h)
        U = z["U"][i]
        spd = np.hypot(U[0], U[1]) / z["U_inflow"][i]
        ax = axes[0, col]
        im = ax.imshow(spd, origin="lower", extent=ext, cmap="RdBu_r", vmin=0, vmax=2)
        s = max(1, N // 16)
        yy, xx = np.mgrid[0:N:s, 0:N:s]
        ax.quiver((xx - N / 2 + 0.5) * dx / 1000, (yy - N / 2 + 0.5) * dx / 1000, U[0][::s, ::s], U[1][::s, ::s],
                  scale=25 * float(z["U_inflow"][i]), width=0.003)
        ax.contour(np.linspace(ext[0], ext[1], N), np.linspace(ext[2], ext[3], N), hc, 10, colors="k",
                   linewidths=0.4, alpha=0.5)
        ax.set_title(f"speed-up |U_h|/U_in at {h} m AGL")
        fig.colorbar(im, ax=ax, shrink=0.8)
    i50 = H.index(50) if 50 in H else 1
    ax = axes[1, 0]
    im = ax.imshow(z["U"][i50, 2], origin="lower", extent=ext, cmap="PuOr", vmin=-2, vmax=2)
    ax.set_title(f"w at {H[i50]} m AGL (m/s)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[1, 1]
    im = ax.imshow(z["k"][0] / z["k_inflow"][0], origin="lower", extent=ext, cmap="magma", vmin=0, vmax=4)
    ax.contour(np.linspace(ext[0], ext[1], N), np.linspace(ext[2], ext[3], N), z["U"][0, 0], [0], colors="cyan")
    ax.set_title(f"k/k_in at {H[0]} m; cyan = reversed flow (u<0)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axes[1, 2]
    ang = np.degrees(np.arctan2(z["U"][0, 1], z["U"][0, 0]))
    im = ax.imshow(ang, origin="lower", extent=ext, cmap="twilight_shifted", vmin=-90, vmax=90)
    ax.set_title(f"deflection at {H[0]} m (deg, + = to the left)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    for a in axes.ravel()[1:]:
        a.set_xlabel("x' downwind (km)")
        a.set_ylabel("y' (km)")
    fig.suptitle(f"{meta['case_id']}  theta={meta['theta']}°  z0={meta['z0']} m  relief={meta.get('relief')} m  "
                 f"iters={meta.get('iterations')}  converged={meta.get('converged')}")
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
