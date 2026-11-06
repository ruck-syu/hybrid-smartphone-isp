"""Inference pipelines: Hybrid ISP (classical + learned) and fully-Learned ISP, torch or ONNX backends.

Hybrid ISP processing sequence (see docs/ARCHITECTURE.md):
  1. [classical] CFA canonicalisation to RGGB + packing (H/2, W/2, 4)
  2. [classical, adaptive] noise level function: DNG NoiseProfile if present, else estimated
     from the image itself (robust local mean/variance regression)
  3. [classical, adaptive] white balance: DNG AsShotNeutral if present, else AWB (shades-of-grey)
  4. [classical, adaptive] exposure normalisation gain k (illumination adaptive)
  5. [learned]   noise-conditioned U-Net: joint denoise + demosaic -> linear camera RGB
  6. [classical] colour correction matrix (DNG ColorMatrix, CCT interpolated) -> linear sRGB
  7. [classical] sRGB OETF (gamma); optional tone mapping for dark scenes; optional sharpening
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import classical as C
from .frontend import make_hybrid_input
from .rawio import RawImage


def estimate_nlf(packed: np.ndarray, block: int = 8) -> tuple[float, float]:
    """Estimate a heteroscedastic noise model var = a*x + b from a single packed raw image.

    Signal-independent local estimates: per 2x2 block on each plane, the Haar HH coefficient has
    variance sigma^2(x) (structure largely cancels); the block mean gives x. We bin by intensity,
    take a low quantile of the variance per bin (robust to texture) and fit a line.
    """
    xs, vs = [], []
    for c in range(4):
        p = packed[..., c]
        h, w = (p.shape[0] // 2) * 2, (p.shape[1] // 2) * 2
        p = p[:h, :w]
        hh = (p[0::2, 0::2] - p[0::2, 1::2] - p[1::2, 0::2] + p[1::2, 1::2]) / 2
        mu = (p[0::2, 0::2] + p[0::2, 1::2] + p[1::2, 0::2] + p[1::2, 1::2]) / 4
        # aggregate over block x block neighbourhoods
        H, W = (hh.shape[0] // block) * block, (hh.shape[1] // block) * block
        v = (hh[:H, :W] ** 2).reshape(H // block, block, W // block, block).mean((1, 3))
        m = mu[:H, :W].reshape(H // block, block, W // block, block).mean((1, 3))
        xs.append(m.ravel())
        vs.append(v.ravel())
    x, v = np.concatenate(xs), np.concatenate(vs)
    ok = (x > 0.002) & (x < 0.9)
    x, v = x[ok], v[ok]
    if x.size < 100:
        s = C.estimate_noise_sigma(C.unpack_rggb(packed))
        return (s ** 2 * 0.5 / max(float(packed.mean()), 1e-3), s ** 2 * 0.5)
    edges = np.quantile(x, np.linspace(0, 1, 21))
    bx, bv = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() > 50:
            bx.append(np.median(x[sel]))
            # low quantile rejects texture; 1.3 ~ corrects the 10% quantile of a chi2(64)/64 variance
            bv.append(np.quantile(v[sel], 0.1) * 1.3)
    bx, bv = np.array(bx), np.array(bv)
    A = np.stack([bx, np.ones_like(bx)], 1)
    a, b = np.linalg.lstsq(A, bv, rcond=None)[0]
    a = float(np.clip(a, 1e-7, 4e-3))  # plausibility clamp (SIDD ISO 10000 is ~3e-3)
    b = float(max(b, 1e-9))
    return a, b


class Backend:
    """Wraps a torch module or an ONNX Runtime session with a common __call__(np NCHW)->np NCHW."""

    def __init__(self, path: str | Path, device: str = "cpu", threads: int | None = None):
        self.path = str(path)
        if self.path.endswith(".onnx"):
            import onnxruntime as ort
            so = ort.SessionOptions()
            if threads:
                so.intra_op_num_threads = threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
            self.kind = "onnx"
            self.inp_name = self.sess.get_inputs()[0].name
            self.fp16 = "float16" in self.sess.get_inputs()[0].type
            meta = self.sess.get_modelmeta().custom_metadata_map
            self.mode = meta.get("mode", "hybrid")
        else:
            import torch
            from .models import load_checkpoint
            self.torch = torch
            self.model = load_checkpoint(self.path).to(device)
            self.device = device
            self.kind = "torch"
            self.mode = self.model.mode
            self.fp16 = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.kind == "onnx":
            xin = x.astype(np.float16 if self.fp16 else np.float32)
            return self.sess.run(None, {self.inp_name: xin})[0].astype(np.float32)
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x).to(self.device)).float().cpu().numpy()


def run_tiled(fn, inp: np.ndarray, tile: int = 512, overlap: int = 32, mult: int = 8) -> np.ndarray:
    """inp: (h, w, C) packed half-res. Returns (2h, 2w, 3). Tiles bound peak memory."""
    h, w, _ = inp.shape
    ph, pw = (-h) % mult, (-w) % mult
    x = np.pad(inp, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    H, W = x.shape[:2]
    if H <= tile and W <= tile:
        out = fn(x.transpose(2, 0, 1)[None])[0].transpose(1, 2, 0)
        return out[: 2 * h, : 2 * w]
    out = np.zeros((2 * H, 2 * W, 3), np.float32)
    wsum = np.zeros((2 * H, 2 * W, 1), np.float32)
    step = tile - 2 * overlap
    def starts(n):  # evenly spaced tile origins covering [0, n) with overlap >= 2*overlap
        if n <= tile:
            return [0]
        k = int(np.ceil((n - tile) / step)) + 1
        return sorted({int(round(v)) // 2 * 2 for v in np.linspace(0, n - tile, k)})
    ys, xs = starts(H), starts(W)
    th, tw = min(tile, H), min(tile, W)
    ramp_y = np.minimum(np.arange(2 * th) + 1, np.arange(2 * th)[::-1] + 1).astype(np.float32)
    ramp_x = np.minimum(np.arange(2 * tw) + 1, np.arange(2 * tw)[::-1] + 1).astype(np.float32)
    wt = np.minimum(np.minimum(ramp_y[:, None], 4 * overlap), np.minimum(ramp_x[None], 4 * overlap))[..., None]
    for y in ys:
        for xx in xs:
            t = x[y:y + th, xx:xx + tw]
            o = fn(np.ascontiguousarray(t.transpose(2, 0, 1)[None]))[0].transpose(1, 2, 0)
            out[2 * y:2 * y + 2 * th, 2 * xx:2 * xx + 2 * tw] += o * wt
            wsum[2 * y:2 * y + 2 * th, 2 * xx:2 * xx + 2 * tw] += wt
    return (out / np.maximum(wsum, 1e-8))[: 2 * h, : 2 * w]


class HybridISP:
    def __init__(self, model_path: str | Path, device: str = "cpu", tile: int = 512, threads: int | None = None):
        self.net = Backend(model_path, device, threads)
        self.tile = tile

    def run(self, raw: RawImage, wb: str = "auto", ccm: str = "auto", nlf: str = "auto",
            exposure_norm: bool = True, noise_scale: float = 1.0, tone: str = "none",
            sharpen: float = 0.0, keep_intermediates: bool = False) -> tuple[np.ndarray, dict]:
        t_all = time.perf_counter()
        timings, inter = {}, {}
        t = time.perf_counter()

        def tick(name, img=None):
            nonlocal t
            now = time.perf_counter()
            timings[name] = (now - t) * 1000
            t = now
            if keep_intermediates and img is not None:
                inter[name] = img

        rggb, shift = C.to_rggb(raw.bayer, raw.cfa)
        if min(rggb.shape) < 8:
            raise ValueError("Image too small (need at least 8x8 Bayer pixels).")
        # odd sizes: reflect-pad one row/col; numpy 'reflect' copies index n-2, which has the same
        # CFA parity as the new index n, so the mosaic stays valid (no edge rows are dropped)
        ph, pw = rggb.shape[0] % 2, rggb.shape[1] % 2
        if ph or pw:
            rggb = np.pad(rggb, ((0, ph), (0, pw)), mode="reflect")
        packed = C.pack_rggb(rggb)
        tick("pack")

        # adaptive noise model
        if nlf == "metadata" or (nlf == "auto" and raw.noise_profile is not None):
            profile, nlf_src = raw.noise_profile, "metadata"
        else:
            profile, nlf_src = estimate_nlf(packed), "estimated"
        profile = (profile[0] * noise_scale, profile[1] * noise_scale)
        tick("noise_model")

        # adaptive white balance
        if wb == "metadata" or (wb == "auto" and raw.wb_gains is not None):
            gains, wb_src = np.asarray(raw.wb_gains, np.float32), "metadata"
        elif wb == "grayworld":
            gains, wb_src = C.estimate_wb_grayworld(rggb, "RGGB"), "grayworld"
        else:
            small = np.stack([packed[..., 0], 0.5 * (packed[..., 1] + packed[..., 2]), packed[..., 3]], -1)
            small = small[::4, ::4]
            gains, wb_src = C.estimate_wb_shades_of_gray(small), "shades_of_gray"
        tick("white_balance")

        if self.net.mode == "hybrid":
            inp, k = make_hybrid_input(packed, gains, profile, k=None if exposure_norm else 1.0)
            tick("normalise")
            lin = run_tiled(self.net, inp, self.tile) / k
            tick("network")
            if keep_intermediates:
                inter["network_linear_cam"] = np.clip(C.srgb_oetf(np.clip(lin, 0, 1)), 0, 1)
            if ccm == "metadata" or (ccm == "auto" and raw.cam2srgb is not None):
                M, ccm_src = raw.cam2srgb, "metadata"
            else:
                M, ccm_src = C.GENERIC_CAM2SRGB, "generic"
            lin = np.clip(C.apply_ccm(np.clip(lin, 0, None), M), 0, None)
            tick("ccm")
            if tone == "auto":
                g = C.auto_exposure_gain(lin)
                lin = C.tone_reinhard(lin * g, white=4.0) if g > 1.5 else lin
            out = C.srgb_oetf(np.clip(lin, 0, 1))
            tick("gamma_tone", out)
        else:
            k, ccm_src = 1.0, "learned"
            out = np.clip(run_tiled(self.net, packed.astype(np.float32), self.tile), 0, 1)
            tick("network", out)
        if sharpen > 0:
            out = C.sharpen_unsharp(out, amount=sharpen, sigma=1.0, threshold=0.01)
            tick("sharpen", out)
        out = np.clip(out, 0, 1)[: out.shape[0] - ph, : out.shape[1] - pw]
        out = C.from_rggb(out, shift)
        info = {"timings_ms": timings, "total_ms": (time.perf_counter() - t_all) * 1000,
                "wb_gains": [float(v) for v in gains], "wb_source": wb_src, "ccm_source": ccm_src,
                "noise_profile": [float(profile[0]), float(profile[1])], "noise_source": nlf_src,
                "exposure_gain_k": float(k), "intermediates": inter}
        return out.astype(np.float32), info
