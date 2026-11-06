"""Image quality metrics: PSNR, SSIM, CIEDE2000 colour error, LPIPS."""
from __future__ import annotations

import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

_lpips = None


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.clip(a, 0, 1).astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(np.clip(a, 0, 1), b, channel_axis=-1, data_range=1.0))


def delta_e2000(a: np.ndarray, b: np.ndarray) -> float:
    """Mean CIEDE2000 between two sRGB images (D65)."""
    la = rgb2lab(np.clip(a, 0, 1))
    lb = rgb2lab(np.clip(b, 0, 1))
    return float(np.mean(deltaE_ciede2000(la, lb)))


def lpips_dist(a: np.ndarray, b: np.ndarray, device: str = "cpu") -> float:
    global _lpips
    import torch
    if _lpips is None:
        import lpips
        _lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    ta = torch.from_numpy(np.clip(a, 0, 1).transpose(2, 0, 1)[None]).float().to(device) * 2 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1)[None]).float().to(device) * 2 - 1
    with torch.no_grad():
        return float(_lpips(ta, tb).item())


def all_metrics(pred: np.ndarray, gt: np.ndarray, use_lpips: bool = True, device: str = "cpu") -> dict:
    m = {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt), "delta_e": delta_e2000(pred, gt)}
    if use_lpips:
        m["lpips"] = lpips_dist(pred, gt, device)
    return m
