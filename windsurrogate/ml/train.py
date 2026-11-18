"""Training loop for the wind surrogate."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..geometry import CaseGeometry
from .data import WindDataset, list_samples
from .model import build_model


class _TorchDS(Dataset):
    def __init__(self, ds: WindDataset):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return torch.from_numpy(x), torch.from_numpy(y)


def _worker_init(wid):
    info = torch.utils.data.get_worker_info()
    info.dataset.ds.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def masked_loss(pred, target, core):
    """MSE on the core tile; channels are already O(1)."""
    rs, cs = core
    p = pred[:, :, rs, cs]
    return torch.mean((p - target) ** 2)


@torch.no_grad()
def evaluate(model, loader, core, heights, device, amp) -> dict:
    model.eval()
    rs, cs = core
    H = len(heights)
    se = torch.zeros(H, 4, device=device)
    se_base = torch.zeros(H, 4, device=device)
    base = torch.tensor([1.0, 0.0, 0.0, 1.0], device=device).view(1, 1, 4, 1, 1)  # "flat terrain" answer
    n = 0
    recirc_inter = torch.zeros(H, device=device)
    recirc_union = torch.zeros(H, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            p = model(x)[:, :, rs, cs].float()
        b = x.shape[0]
        p = p.view(b, H, 4, *p.shape[-2:])
        t = y.view(b, H, 4, *y.shape[-2:])
        se += ((p - t) ** 2).mean(dim=(-1, -2)).sum(0)
        se_base += ((base - t) ** 2).mean(dim=(-1, -2)).sum(0)
        pr, tr = p[:, :, 0] < 0, t[:, :, 0] < 0
        recirc_inter += (pr & tr).sum(dim=(0, 2, 3)).float()
        recirc_union += (pr | tr).sum(dim=(0, 2, 3)).float()
        n += b
    rmse = (se / max(n, 1)).sqrt().cpu().numpy()
    loss, bl = float((se / max(n, 1)).mean()), float((se_base / max(n, 1)).mean())
    out = {"loss": loss, "baseline_loss": bl, "skill": 1.0 - loss / bl if bl > 0 else float("nan")}
    for i, h in enumerate(heights):
        out[f"rmse_u_{int(h)}m"] = float(rmse[i, 0])
        out[f"rmse_w_{int(h)}m"] = float(rmse[i, 2])
        out[f"rmse_k_{int(h)}m"] = float(rmse[i, 3])
        u = recirc_union[i].item()
        out[f"recirc_iou_{int(h)}m"] = float(recirc_inter[i].item() / u) if u > 0 else float("nan")
    model.train()
    return out


def train(cfg, resume: bool = True, max_steps: int | None = None):
    t = cfg["train"]
    geom = CaseGeometry.from_config(cfg)
    heights = cfg["output"]["heights_agl"]
    ddir = cfg.path(cfg["output"]["dataset_dir"])
    tr_files = list_samples(ddir, "train", max_probe_change=t.get("max_probe_change"))
    va_files = list_samples(ddir, "val", max_probe_change=t.get("max_probe_change"))
    if not tr_files:
        raise RuntimeError(f"no training samples in {ddir}/train")
    print(f"train {len(tr_files)} samples, val {len(va_files)} samples")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(t["amp"]) and device.type == "cuda"

    tr_ds = _TorchDS(WindDataset(tr_files, geom, flip_augment=t["flip_augment"]))
    va_ds = _TorchDS(WindDataset(va_files or tr_files[:8], geom))
    nw = int(t["num_workers"])
    tr_dl = DataLoader(tr_ds, batch_size=t["batch_size"], shuffle=True, num_workers=nw, drop_last=len(tr_ds) > t["batch_size"],
                       pin_memory=device.type == "cuda", worker_init_fn=_worker_init if nw else None,
                       persistent_workers=nw > 0)
    va_dl = DataLoader(va_ds, batch_size=t["batch_size"], shuffle=False, num_workers=nw)

    model = build_model(cfg, len(heights)).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    steps_per_epoch = max(1, len(tr_dl))
    total = t["epochs"] * steps_per_epoch
    warm = min(500, total // 20 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckdir = cfg.path(t["checkpoint_dir"])
    ckdir.mkdir(parents=True, exist_ok=True)
    start_epoch, best, step = 0, float("inf"), 0
    last = ckdir / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best, step = ck["epoch"] + 1, ck["best"], ck["step"]
        print(f"resumed from epoch {start_epoch}, best val {best:.4f}")

    core = geom.core_slice_in_input
    log_path = ckdir / "train_log.jsonl"
    for epoch in range(start_epoch, t["epochs"]):
        t0, run_loss, nb = time.time(), 0.0, 0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                pred = model(x)
            loss = masked_loss(pred.float(), y, core)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item()
            nb += 1
            step += 1
            if max_steps and step >= max_steps:
                break
        val = evaluate(model, va_dl, core, heights, device, amp)
        rec = dict(epoch=epoch, step=step, train_loss=run_loss / max(nb, 1), lr=sched.get_last_lr()[0],
                   seconds=round(time.time() - t0, 1), **val)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch:3d} train {rec['train_loss']:.4f} val {val['loss']:.4f} skill {val['skill']:.2f} "
              f"u10 {val.get(f'rmse_u_{int(heights[0])}m', float('nan')):.3f} ({rec['seconds']}s)", flush=True)
        state = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch,
                     best=min(best, val["loss"]), step=step, config=dict(cfg))
        torch.save(state, ckdir / "last.tmp")
        Path(ckdir / "last.tmp").replace(last)
        if val["loss"] < best:
            best = val["loss"]
            torch.save(dict(model=model.state_dict(), config=dict(cfg), epoch=epoch, val=val), ckdir / "best.pt")
        if max_steps and step >= max_steps:
            break
    print(f"done. best val loss {best:.4f}; checkpoints in {ckdir}")
