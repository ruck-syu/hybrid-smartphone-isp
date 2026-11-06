"""Learned ISP components (PyTorch).

* `RawUNet` — a compact U-Net that runs at half resolution on the packed RGGB mosaic and
  reconstructs a full-resolution 3-channel image through PixelShuffle. It is used in two roles:
    - **hybrid** (`mode="hybrid"`): input = white-balanced packed raw (4) + noise-std map (4),
      output = denoised + demosaiced *linear camera RGB*. Colour (CCM) and gamma are then applied
      by the classical, metadata-driven stages, which are differentiable so we train end-to-end
      on the sRGB ground truth.
    - **learned** (`mode="learned"`): input = raw packed mosaic only (no metadata at all),
      output = sRGB directly (the network must also learn WB, CCM and tone) — a PyNET/U-Net style
      fully-learned ISP baseline.
* Differentiable torch versions of the classical colour stages (`TorchColorBack`).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, dw: bool = False):
        super().__init__()
        if dw:  # depthwise-separable (MobileNet style) for the efficient student
            self.body = nn.Sequential(
                nn.Conv2d(cin, cin, 3, padding=1, groups=cin), nn.Conv2d(cin, cout, 1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(cout, cout, 3, padding=1, groups=cout), nn.Conv2d(cout, cout, 1), nn.LeakyReLU(0.2, True))
        else:
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.LeakyReLU(0.2, True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.LeakyReLU(0.2, True))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)


class RawUNet(nn.Module):
    def __init__(self, in_ch: int = 8, base: int = 32, depth: int = 4, out_ch: int = 3,
                 dw: bool = False, mode: str = "hybrid"):
        super().__init__()
        self.mode, self.in_ch, self.base, self.depth, self.dw = mode, in_ch, base, depth, dw
        chs = [base * 2 ** i for i in range(depth)]
        self.head = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(depth):
            self.enc.append(ConvBlock(chs[i], chs[i], dw))
            if i < depth - 1:
                self.down.append(nn.Conv2d(chs[i], chs[i + 1], 2, stride=2))
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.dec.append(ConvBlock(chs[i - 1] * 2, chs[i - 1], dw))
        # Full-resolution reconstruction: 4*out_ch channels at half-res -> PixelShuffle(2)
        self.tail = nn.Conv2d(chs[0], out_ch * 4, 3, padding=1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        packed = x[:, :4]
        f = self.head(x)
        skips = []
        for i, blk in enumerate(self.enc):
            f = blk(f)
            if i < self.depth - 1:
                skips.append(f)
                f = self.down[i](f)
        for up, dec in zip(self.up, self.dec):
            f = up(f)
            f = dec(torch.cat([f, skips.pop()], 1))
        out = self.ps(self.tail(f))
        if self.mode == "hybrid":
            # Residual on a bilinear-like demosaic of the input: network learns the correction.
            base = naive_demosaic(packed)
            return base + out
        return out

    def config(self) -> dict:
        return {"in_ch": self.in_ch, "base": self.base, "depth": self.depth, "dw": self.dw, "mode": self.mode}


def naive_demosaic(packed: torch.Tensor) -> torch.Tensor:
    """Cheap differentiable demosaic: packed [R,Gr,Gb,B] (half-res) -> RGB upsampled 2x (bilinear)."""
    r = packed[:, 0:1]
    g = 0.5 * (packed[:, 1:2] + packed[:, 2:3])
    b = packed[:, 3:4]
    rgb = torch.cat([r, g, b], 1)
    return F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False)


def srgb_oetf_t(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * x.clamp_min(1e-5).pow(1 / 2.4) - 0.055)


class TorchColorBack(nn.Module):
    """Differentiable classical back-end: per-image CCM (B,3,3) then sRGB OETF."""

    def forward(self, lin_cam: torch.Tensor, ccm: torch.Tensor) -> torch.Tensor:
        srgb_lin = torch.einsum("bij,bjhw->bihw", ccm, lin_cam)
        return srgb_oetf_t(srgb_lin)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def build_model(name: str) -> RawUNet:
    presets = {
        "hybrid_teacher": dict(in_ch=8, base=32, depth=4, mode="hybrid"),
        "hybrid_student": dict(in_ch=8, base=16, depth=4, mode="hybrid", dw=False),
        "hybrid_tiny": dict(in_ch=8, base=16, depth=3, mode="hybrid", dw=True),
        "learned": dict(in_ch=4, base=32, depth=4, mode="learned"),
        "learned_student": dict(in_ch=4, base=16, depth=4, mode="learned"),
    }
    return RawUNet(**presets[name])


def load_checkpoint(path, map_location="cpu") -> RawUNet:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    m = RawUNet(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m
