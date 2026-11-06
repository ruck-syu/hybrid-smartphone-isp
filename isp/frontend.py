"""Classical, adaptive front-end used by the hybrid ISP (numpy only, no torch)."""
from __future__ import annotations

import numpy as np


def noise_std_map(packed: np.ndarray, profile) -> np.ndarray:
    """Per-pixel noise std from the heteroscedastic noise level function var = a*x + b."""
    if profile is None:
        a, b = 1e-4, 1e-6
    else:
        a, b = profile
    return np.sqrt(np.maximum(a * packed + b, 1e-10)).astype(np.float32)


def exposure_normalisation(packed_wb: np.ndarray, target: float = 0.2, max_gain: float = 32.0) -> float:
    """Adaptive illumination normalisation gain k (classical): brings the mean signal to `target`.
    The network sees k*x (and k*sigma); its output is divided by k (scale-equivariant pipeline)."""
    m = float(np.mean(packed_wb))
    return float(np.clip(target / max(m, 1e-5), 1.0, max_gain))


def make_hybrid_input(packed: np.ndarray, wb_gains, profile, k: float | None = None,
                      extra_noise_factor: float = 0.0) -> tuple[np.ndarray, float]:
    """packed: (h, w, 4) [R, Gr, Gb, B] in [0,1], no WB. Returns (h, w, 8) input and gain k."""
    g = np.array([wb_gains[0], wb_gains[1], wb_gains[1], wb_gains[2]], np.float32)
    std = noise_std_map(packed, profile) * np.sqrt(1.0 + extra_noise_factor)
    xw, sw = np.minimum(packed * g, 1.0), std * g  # clip: saturated -> neutral white
    if k is None:
        k = exposure_normalisation(xw)
    return np.concatenate([xw * k, sw * k], -1).astype(np.float32), k
