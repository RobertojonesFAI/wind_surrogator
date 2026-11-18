"""U-Net with a small self-attention bottleneck for global (upwind) context."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = _gn(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = _gn(cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.silu(h + self.skip(x))


class Bottleneck(nn.Module):
    """Transformer encoder over the coarsest feature map (tokens = pixels)."""

    def __init__(self, ch: int, layers: int = 2, heads: int = 4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(ch, heads, dim_feedforward=2 * ch, dropout=0.0,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pos = None

    def forward(self, x):
        b, c, h, w = x.shape
        if self.pos is None or self.pos.shape[1] != h * w or self.pos.device != x.device:
            yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                    torch.linspace(-1, 1, w, device=x.device), indexing="ij")
            fmax = max(h, w) / 2.0
            freqs = torch.logspace(0, torch.log10(torch.tensor(fmax)).item(), c // 4,
                                   device=x.device, dtype=torch.float32)
            pe = torch.cat([torch.sin(xx[..., None] * freqs * 3.14159), torch.cos(xx[..., None] * freqs * 3.14159),
                            torch.sin(yy[..., None] * freqs * 3.14159), torch.cos(yy[..., None] * freqs * 3.14159)], -1)
            pe = F.pad(pe, (0, c - pe.shape[-1]))
            self.pos = pe.reshape(1, h * w, c)
        t = x.flatten(2).transpose(1, 2) + self.pos.to(x.dtype)
        t = self.enc(t)
        return t.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int = 48, depth: int = 4, attn_layers: int = 2):
        super().__init__()
        chs = [base * min(2 ** i, 8) for i in range(depth + 1)]
        self.stem = ResBlock(in_ch, chs[0])
        self.down = nn.ModuleList()
        self.enc = nn.ModuleList()
        for i in range(depth):
            self.down.append(nn.Conv2d(chs[i], chs[i + 1], 3, stride=2, padding=1))
            self.enc.append(ResBlock(chs[i + 1], chs[i + 1]))
        self.mid = Bottleneck(chs[-1], attn_layers) if attn_layers > 0 else nn.Identity()
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up.append(nn.Conv2d(chs[i + 1], chs[i], 1))
            self.dec.append(ResBlock(2 * chs[i], chs[i]))
        self.head = nn.Conv2d(chs[0], out_ch, 1)
        self.depth = depth

    def forward(self, x):
        skips = [self.stem(x)]
        h = skips[0]
        for d, e in zip(self.down, self.enc):
            h = e(d(h))
            skips.append(h)
        h = self.mid(h)
        for i, (u, dblock) in enumerate(zip(self.up, self.dec)):
            skip = skips[-(i + 2)]
            h = F.interpolate(u(h), size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = dblock(torch.cat([h, skip], 1))
        return self.head(h)


def build_model(cfg, n_heights: int) -> UNet:
    t = cfg["train"]
    return UNet(4, 4 * n_heights, base=t["base_channels"], depth=t["depth"],
                attn_layers=t.get("attn_layers", 2))
