"""Oracle test for the rotate -> tile -> blend -> un-rotate inference path.

A fake 'model' returns the input slope channels as (u', v'). After inference
the east/north components divided by the inflow speed must equal the terrain
gradient (dh/dE, dh/dN) for ANY wind direction, which checks frame axes, tile
windows, blending and vector rotation together.

usage: python tests/test_inference_geometry.py [config.yaml]   (or: pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windsurrogate.config import load_config  # noqa: E402
from windsurrogate.dem import MasterDEM, region_dem_path, regions, synthetic_dem  # noqa: E402
from windsurrogate.ml.infer import predict_map  # noqa: E402
from windsurrogate.terrain import SmoothedSampler  # noqa: E402


class SlopeOracle(torch.nn.Module):
    def __init__(self, nh):
        super().__init__()
        self.nh = nh

    def forward(self, x):
        b, _, r, c = x.shape
        out = torch.zeros(b, 4 * self.nh, r, c)
        out[:, 0::4] = x[:, 1:2]      # u' / U_in = dh/dx'
        out[:, 1::4] = x[:, 2:3]      # v' / U_in = dh/dy'
        out[:, 3::4] = 1.0
        return out


def check(cfg, dem):
    nh = len(cfg["output"]["heights_agl"])
    model = SlopeOracle(nh)
    l, b, r, t = dem.bounds
    bbox = (l + 15000, b + 15000, l + 35000, b + 33000)
    worst = 0.0
    for theta in (0.0, 37.0, 135.0, 270.0, 301.5):
        res = predict_map(model, cfg, torch.device("cpu"), dem, theta, z0=0.1, bbox=bbox)
        f = res["fields"]
        Uin = res["fields"]["speed"][0] / np.maximum(res["fields"]["speedup"][0], 1e-9)
        ge = f["u_east"][0] / Uin
        gn = f["v_north"][0] / Uin
        dx = cfg["geometry"]["dx"]
        h = res["terrain"]
        gy, gx = np.gradient(h, dx)
        tn = -gy                     # rows run north -> south
        inner = (slice(5, -5), slice(5, -5))
        err = max(np.abs(ge - gx)[inner].max(), np.abs(gn - tn)[inner].max())
        scale = np.abs(np.r_[gx[inner].ravel(), tn[inner].ravel()]).max()
        worst = max(worst, err / scale)
        print(f"theta {theta:6.1f}: max |grad error| = {err:.4f} (max slope {scale:.3f}, rel {err / scale:.3f})")
    assert worst < 0.15, worst
    print("OK: inference frame, tiling and vector rotation are consistent")


def test_inference_geometry(tmp_path):
    """Self-contained version (pytest / CI): default geometry over a synthetic DEM."""
    cfg = load_config()
    path = synthetic_dem(tmp_path / "dem.tif", size_km=50.0, res=30.0, seed=1)
    check(cfg, MasterDEM.open(path))


def main(cfg_path=None):
    if cfg_path is None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            test_inference_geometry(Path(d))
        return
    cfg = load_config(cfg_path)
    check(cfg, MasterDEM.open(region_dem_path(cfg, regions(cfg)[0]["name"])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
