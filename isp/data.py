"""SIDD patch dataset over memory-mapped arrays produced by scripts/prep_sidd.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .frontend import exposure_normalisation, make_hybrid_input, noise_std_map  # noqa: F401


class SIDDPatches(Dataset):
    def __init__(self, root: str, split: str = "train", patch: int = 256, mode: str = "hybrid",
                 length: int = 100000, augment: bool = True, use_011: bool = True, seed: int = 0,
                 scale_prob: float = 0.0, scale: int = 2):
        self.root = Path(root)
        recs = json.load(open(self.root / "index.json"))
        self.recs = [r for r in recs if r["split"] == split and (use_011 or r["k"] == "010")]
        self.patch, self.mode, self.length, self.augment = patch, mode, length, augment
        self._noisy, self._gt = {}, {}
        self.rng = np.random.default_rng(seed)
        # scale augmentation: bin the mosaic s x s (and downscale GT in linear light) so the network
        # also sees images that are as sharp per pixel as real phone photos (see FAILURE_ANALYSIS §9)
        self.scale_prob, self.scale = scale_prob, scale

    def __len__(self):
        return self.length

    def _arrays(self, r):
        key = (r["id"], r["k"])
        if key not in self._noisy:
            self._noisy[key] = np.load(self.root / f"{r['id']}_{r['k']}_noisy.npy", mmap_mode="r")
        if r["id"] not in self._gt:
            self._gt[r["id"]] = np.load(self.root / f"{r['id']}_gt.npy", mmap_mode="r")
        return self._noisy[key], self._gt[r["id"]]

    def sample(self, r, y=None, x=None, rng=None, scaled: bool | None = None):
        rng = rng or np.random.default_rng()
        noisy, gt = self._arrays(r)
        if scaled is None:
            scaled = self.scale_prob > 0 and rng.random() < self.scale_prob
        s = self.scale if scaled else 1
        ph = self.patch // 2
        h, w = noisy.shape[:2]
        if y is None:
            y = int(rng.integers(0, h - ph * s))
            x = int(rng.integers(0, w - ph * s))
        p = noisy[y:y + ph * s, x:x + ph * s].astype(np.float32) / 65535.0
        t = gt[2 * y:2 * y + self.patch * s, 2 * x:2 * x + self.patch * s].astype(np.float32) / 255.0
        prof = r["noise_profile"]
        if s > 1:
            p = p.reshape(ph, s, ph, s, 4).mean((1, 3))
            lin = np.where(t <= 0.04045, t / 12.92, ((t + 0.055) / 1.055) ** 2.4)
            lin = lin.reshape(self.patch, s, self.patch, s, 3).mean((1, 3))
            t = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055).astype(np.float32)
            if prof is not None:  # averaging s*s samples divides the noise variance by s*s
                prof = (prof[0] / (s * s), prof[1] / (s * s))
        extra = 0.0
        if self.augment:
            if rng.random() < 0.5:  # transpose keeps RGGB, swaps Gr/Gb
                p = p.transpose(1, 0, 2)[..., [0, 2, 1, 3]]
                t = t.transpose(1, 0, 2)
            if rng.random() < 0.3 and prof is not None:  # synthetic extra sensor noise
                extra = float(rng.uniform(0.5, 3.0))
                a, b = prof
                p = np.clip(p + rng.standard_normal(p.shape).astype(np.float32)
                            * np.sqrt(np.maximum(extra * (a * p + b), 0)), 0, 1)
        out = {"gt": torch.from_numpy(np.ascontiguousarray(t.transpose(2, 0, 1))),
               "ccm": torch.tensor(r["cam2srgb"], dtype=torch.float32)}
        if self.mode == "hybrid":
            inp, k = make_hybrid_input(p, r["wb_gains"], prof, extra_noise_factor=extra)
            out["k"] = torch.tensor(k, dtype=torch.float32)
        else:
            inp = p
            out["k"] = torch.tensor(1.0)
        out["inp"] = torch.from_numpy(np.ascontiguousarray(inp.transpose(2, 0, 1)))
        return out

    def __getitem__(self, i):
        rng = np.random.default_rng((i * 7919 + int(self.rng.integers(1 << 30))) % (1 << 32))
        r = self.recs[int(rng.integers(len(self.recs)))]
        return self.sample(r, rng=rng)


def fixed_val_crops(ds: SIDDPatches, per_image: int = 4, seed: int = 123, scaled: bool = False):
    rng = np.random.default_rng(seed)
    s = ds.scale if scaled else 1
    items = []
    for r in ds.recs:
        if r["k"] != "010":
            continue
        noisy, _ = ds._arrays(r)
        h, w = noisy.shape[:2]
        for _ in range(per_image):
            y = int(rng.integers(0, h - ds.patch // 2 * s))
            x = int(rng.integers(0, w - ds.patch // 2 * s))
            items.append(ds.sample(r, y, x, rng, scaled=scaled))
    return items
