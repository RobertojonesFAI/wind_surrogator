"""Smoke test for `windsurrogate evaluate`: a tiny untrained model on a few fake samples.

Checks that metrics, the per-case table and the comparison figures are written and that
the aggregate numbers are consistent. Runs on CPU in a few seconds (pytest tests/).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windsurrogate.config import load_config  # noqa: E402
from windsurrogate.geometry import CaseGeometry  # noqa: E402
from windsurrogate.ml.evaluate import evaluate_checkpoint  # noqa: E402
from windsurrogate.ml.model import build_model  # noqa: E402

SMALL = [
    "geometry.dx=100", "geometry.core_px=32", "geometry.input_upwind_px=24",
    "geometry.input_downwind_px=8", "geometry.input_lateral_px=8",
    "train.base_channels=8", "train.depth=2", "train.attn_layers=1",
]


def _fake_sample(path, geom, heights, case_id, region, seed):
    rng = np.random.default_rng(seed)
    rows, cols = geom.input_shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    terrain = 300.0 * np.exp(-((xx - cols / 2) ** 2 + (yy - rows / 2) ** 2) / (2 * 8.0 ** 2))
    terrain += rng.normal(0, 5, terrain.shape)
    N, H = geom.core_px, len(heights)
    U_in = np.linspace(6.0, 12.0, H)
    k_in = np.full(H, 0.5)
    U = np.zeros((H, 3, N, N))
    U[:, 0] = U_in[:, None, None] * (1 + 0.3 * rng.standard_normal((H, N, N)))   # some u' < 0 too
    U[:, 1] = rng.normal(0, 1, (H, N, N))
    U[:, 2] = rng.normal(0, 0.5, (H, N, N))
    k = k_in[:, None, None] * (1 + 0.2 * rng.random((H, N, N)))
    meta = dict(case_id=case_id, region=region, theta=270.0, z0=0.1, relief=300.0 + 400 * seed,
                converged=True, probe_max_change=0.001)
    np.savez_compressed(path, terrain_input=terrain, terrain_core=terrain[geom.core_slice_in_input],
                        U=U, k=k, U_inflow=U_in, k_inflow=k_in, heights=np.asarray(heights),
                        meta=json.dumps(meta))


def test_evaluate(tmp_path):
    cfg = load_config(None, SMALL + [f"project_dir={tmp_path}"])
    geom = CaseGeometry.from_config(cfg)
    heights = [float(h) for h in cfg["output"]["heights_agl"]]
    test_dir = tmp_path / "dataset" / "test"
    test_dir.mkdir(parents=True)
    for i in range(5):
        _fake_sample(test_dir / f"c{i:05d}.npz", geom, heights, f"c{i:05d}", ["a", "b"][i % 2], i)

    torch.manual_seed(0)
    model = build_model(cfg, len(heights))
    ckpt = tmp_path / "best.pt"
    torch.save(dict(model=model.state_dict(), config=dict(cfg), epoch=0, val={}), ckpt)

    m = evaluate_checkpoint(cfg, ckpt, split="test", out_dir=tmp_path / "eval", n_figures=3, device="cpu")

    assert m["n_cases"] == 5
    assert set(m["by_region"]) == {"a", "b"}
    assert sum(r["n_cases"] for r in m["by_region"].values()) == 5
    assert np.isfinite(m["loss"]) and m["baseline_loss"] > 0
    assert abs(m["skill"] - (1 - m["loss"] / m["baseline_loss"])) < 1e-9
    assert (tmp_path / "eval" / "metrics_test.json").exists()
    with open(tmp_path / "eval" / "cases_test.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    # The aggregate loss is the mean of the per-case losses (all cores have the same size).
    assert abs(np.mean([float(r["loss"]) for r in rows]) - m["loss"]) < 1e-6
    assert len(list((tmp_path / "eval" / "figures").glob("*.png"))) == 3


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_evaluate(Path(d))
    print("OK")
