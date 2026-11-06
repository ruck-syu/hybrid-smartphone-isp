"""Framework-free application logic: load inputs, simulate conditions, run pipelines, encode results."""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isp import classical as C  # noqa: E402
from isp.hybrid import HybridISP  # noqa: E402
from isp.metrics import all_metrics  # noqa: E402
from isp.rawio import RawImage, load_any  # noqa: E402

MODELS = Path(os.environ.get("ISP_MODELS", ROOT / "models"))
SAMPLES = Path(os.environ.get("ISP_SAMPLES", ROOT / "samples"))
MAX_MP = float(os.environ.get("ISP_MAX_MP", "1.2"))  # processed megapixels (Bayer binning above this)
THREADS = int(os.environ.get("ISP_THREADS", "0")) or None
cv2.setNumThreads(THREADS or 0)

# id -> (display name, kind, model files in order of preference, short description)
PIPELINES = {
    "traditional": ("Traditional", "classical", None,
                    "Hand-tuned stages, no learning."),
    "learned": ("Learned", "net", "learned_int8.onnx|learned_fp32.onnx|learned.pt",
                "One network from RAW to photo, no camera metadata."),
    "hybrid": ("Hybrid", "net", "hybrid_student_int8.onnx|hybrid_student.pt",
               "Classical colour stages around a 0.75 MB network."),
    "hybrid_teacher": ("Hybrid teacher", "net", "hybrid_teacher_fp32.onnx|hybrid_teacher.pt",
                       "The 4× larger network it was distilled from."),
}
_nets: dict[str, HybridISP] = {}


def get_net(spec: str) -> HybridISP | None:
    for name in spec.split("|"):
        p = MODELS / name
        if p.exists():
            if name not in _nets:
                _nets[name] = HybridISP(p, device="cpu", tile=384, threads=THREADS)
            return _nets[name]
    return None


def available() -> list[dict]:
    out = []
    for pid, (name, kind, spec, desc) in PIPELINES.items():
        if kind == "classical" or get_net(spec) is not None:
            model = None if kind == "classical" else Path(get_net(spec).net.path).name
            out.append({"id": pid, "name": name, "description": desc, "model": model})
    return out


# ------------------------------------------------------------------------------------------ inputs

def bayer_bin(raw: RawImage, f: int) -> RawImage:
    """Bayer-preserving f x f binning of each colour plane (reduces resolution and noise)."""
    if f <= 1:
        return raw
    rggb, _ = C.to_rggb(raw.bayer, raw.cfa)
    p = C.pack_rggb(rggb[: (rggb.shape[0] // (2 * f)) * 2 * f, : (rggb.shape[1] // (2 * f)) * 2 * f])
    h, w, _ = p.shape
    p = p.reshape(h // f, f, w // f, f, 4).mean((1, 3))
    return RawImage(bayer=C.unpack_rggb(p.astype(np.float32)), cfa="RGGB", wb_gains=raw.wb_gains,
                    cam2srgb=raw.cam2srgb, iso=raw.iso, orientation=raw.orientation, make=raw.make,
                    model=raw.model, noise_profile=None if raw.noise_profile is None else
                    (raw.noise_profile[0] / f ** 2, raw.noise_profile[1] / f ** 2))


def list_samples() -> list[dict]:
    out = []
    if not SAMPLES.exists():
        return out
    cams = {"GP": "Google Pixel", "IP": "iPhone 7", "S6": "Galaxy S6", "N6": "Nexus 6", "G4": "LG G4"}
    for p in sorted(SAMPLES.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        iso = meta.get("iso") or (meta.get("name", "").split("_")[3] if meta.get("name") else None)
        gt = d["gt"]
        thumb = cv2.resize(gt, (160, 160), interpolation=cv2.INTER_AREA)
        out.append({"id": p.stem, "kind": "sidd", "camera": cams.get(meta.get("camera"), meta.get("camera")),
                    "iso": int(float(iso)) if iso else None, "has_gt": True, "thumb": encode(thumb / 255.0, 80)})
    for p in sorted((SAMPLES / "dng").glob("*.dng")) if (SAMPLES / "dng").exists() else []:
        label = {"pixel4a_cc0": "Pixel 4a", "pixel7pro_cc0": "Pixel 7 Pro", "iphonexs_cc0": "iPhone XS"}.get(p.stem, p.stem)
        thumb_path = p.with_suffix(".thumb.jpg")
        if not thumb_path.exists():
            try:
                raw = bayer_bin(load_any(p), 8)
                img = C.ClassicalISP(C.ClassicalConfig()).run(raw)[0]
                img = C.apply_orientation(img, raw.orientation)
                h, w = img.shape[:2]
                s = min(h, w)
                img = img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
                cv2.imwrite(str(thumb_path), (cv2.resize(img, (160, 160)) * 255)[..., ::-1].astype(np.uint8))
            except Exception:
                continue
        thumb = cv2.imread(str(thumb_path))[..., ::-1] / 255.0
        out.append({"id": "dng:" + p.name, "kind": "dng", "camera": label, "iso": None, "has_gt": False,
                    "thumb": encode(thumb, 80)})
    return out


def load_input(path: str | None = None, sample: str | None = None):
    """Returns raw, ground truth (or None), human-readable source label."""
    if path:
        return load_any(path), None, Path(path).name
    if sample and sample.startswith("dng:"):
        p = SAMPLES / "dng" / Path(sample[4:]).name
        if not p.exists():
            raise ValueError("That example file is missing on the server.")
        names = {"pixel4a_cc0": "Pixel 4a", "pixel7pro_cc0": "Pixel 7 Pro", "iphonexs_cc0": "iPhone XS"}
        return load_any(p), None, f"{names.get(p.stem, p.stem)} DNG"
    if sample:
        p = SAMPLES / f"{Path(sample).name}.npz"
        if not p.exists():
            raise ValueError("Unknown sample.")
        d = np.load(p, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        raw = RawImage(bayer=d["bayer"].astype(np.float32), cfa=meta.get("cfa", "RGGB"),
                       wb_gains=np.array(meta["wb_gains"], np.float32) if meta.get("wb_gains") else None,
                       cam2srgb=np.array(meta["cam2srgb"], np.float32) if meta.get("cam2srgb") else None,
                       iso=meta.get("iso"),
                       noise_profile=tuple(meta["noise_profile"]) if meta.get("noise_profile") else None)
        cams = {"GP": "Google Pixel", "IP": "iPhone 7", "S6": "Galaxy S6", "N6": "Nexus 6", "G4": "LG G4"}
        return raw, d["gt"].astype(np.float32) / 255, f"{cams.get(meta.get('camera'), 'Phone')} test scene from SIDD"
    raise ValueError("Choose a sample or upload a RAW file first.")


def degrade(raw: RawImage, noise_mult: float = 1, wb_err: float = 0, exposure_stops: float = 0,
            drop_meta: bool = False, seed: int = 0) -> RawImage:
    """Simulate harder capture conditions on a RAW mosaic (see docs/DESIGN.md §7)."""
    rng = np.random.default_rng(seed)
    b = raw.bayer.copy()
    prof = raw.noise_profile
    sim = prof if prof is not None else (1e-4, 1e-6)  # used only to synthesise noise
    if exposure_stops < 0:  # under-exposure followed by digital gain
        s = 2 ** exposure_stops
        a0, b0 = sim
        dark = b * s
        dark = dark + rng.standard_normal(b.shape).astype(np.float32) * np.sqrt(np.maximum(a0 * dark + b0, 0))
        b = np.clip(dark / s, 0, 1)
        sim = (a0 * (1 + 1 / s), b0 * (1 + 1 / s ** 2))
        prof = sim if prof is not None else None
    if noise_mult > 1:
        a0, b0 = sim
        b = np.clip(b + rng.standard_normal(b.shape).astype(np.float32)
                    * np.sqrt(np.maximum((noise_mult - 1) * (a0 * b + b0), 0)), 0, 1)
        sim = (a0 * noise_mult, b0 * noise_mult)
        prof = sim if prof is not None else None
    wb = None if raw.wb_gains is None else raw.wb_gains * np.array([1 + wb_err, 1, 1 - wb_err], np.float32)
    out = RawImage(bayer=b, cfa=raw.cfa, wb_gains=wb, cam2srgb=raw.cam2srgb, iso=raw.iso,
                   noise_profile=prof, orientation=raw.orientation, make=raw.make, model=raw.model)
    if drop_meta:
        out.wb_gains, out.cam2srgb, out.noise_profile = None, None, None
    return out


# ------------------------------------------------------------------------------------------ outputs

def encode(img: np.ndarray, quality: int = 88, max_side: int | None = None) -> str:
    im = np.clip(img, 0, 1)
    if max_side and max(im.shape[:2]) > max_side:
        s = max_side / max(im.shape[:2])
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if im.ndim == 2:
        im = np.repeat(im[..., None], 3, -1)
    ok, buf = cv2.imencode(".jpg", (im * 255 + 0.5).astype(np.uint8)[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def raw_preview(raw: RawImage) -> np.ndarray:
    """What the sensor recorded: the mosaic shown with each photosite in its filter colour."""
    return C.cfa_colorize(raw.bayer, raw.cfa)


def stage_image(v: np.ndarray, cfa: str) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return C.cfa_colorize(v, cfa) if v.ndim == 2 else v


STAGE_LABELS = {"raw_denoise": "Bayer-domain denoise", "white_balance": "White balance", "demosaic": "Demosaic",
                "noise_map": "Noise level map", "exposure_normalised": "Exposure normalised",
                "packed_input": "Packed RAW input", "detail_restore": "Detail restoration",
                "ccm": "Colour matrix", "tone_map": "Tone map", "gamma": "Gamma", "denoise": "Edge-preserving denoise",
                "sharpen": "Sharpen", "network_linear_cam": "Network output (linear camera RGB)",
                "gamma_tone": "Colour matrix + gamma", "network": "Network output"}


def centre_crop(raw: RawImage, gt: np.ndarray | None, max_mp: float):
    """Native-resolution centre crop of about `max_mp` megapixels (even offsets keep the CFA phase)."""
    H, W = raw.bayer.shape
    s = min(1.0, (max_mp * 1e6 / (H * W)) ** 0.5)
    h, w = int(H * s) // 2 * 2, int(W * s) // 2 * 2
    y, x = (H - h) // 4 * 2, (W - w) // 4 * 2
    out = RawImage(bayer=raw.bayer[y:y + h, x:x + w], cfa=raw.cfa, wb_gains=raw.wb_gains, cam2srgb=raw.cam2srgb,
                   iso=raw.iso, noise_profile=raw.noise_profile, orientation=raw.orientation,
                   make=raw.make, model=raw.model)
    return out, None if gt is None else gt[y:y + h, x:x + w]


def develop(raw: RawImage, gt: np.ndarray | None, pipelines: list[str], conditions: dict,
            stages: bool = True, view: str = "detail") -> dict:
    t0 = time.perf_counter()
    mp = raw.bayer.size / 1e6
    cropped = False
    if view == "detail" and mp > MAX_MP:
        raw, gt = centre_crop(raw, gt, MAX_MP)
        cropped = True
    f = 1
    while raw.bayer.size / 1e6 / (f * f) > MAX_MP:
        f += 1
    if min(raw.bayer.shape) // f < 32:
        raise ValueError("This image is too small to develop (minimum 32×32 pixels after resizing).")
    raw = bayer_bin(raw, f)
    if gt is not None and gt.shape[:2] != raw.bayer.shape:
        gt = cv2.resize(gt, (raw.bayer.shape[1], raw.bayer.shape[0]), interpolation=cv2.INTER_AREA)
    raw = degrade(raw, **conditions)
    orient = lambda im: C.apply_orientation(im, raw.orientation)
    res = {"input": {"width": int(raw.bayer.shape[1]), "height": int(raw.bayer.shape[0]), "binned": f,
                     "cropped": cropped, "view": view,
                     "megapixels_original": round(mp, 1), "iso": raw.iso, "cfa": raw.cfa,
                     "metadata": bool(raw.has_metadata), "noise_profile": raw.noise_profile is not None},
           "raw": encode(orient(raw_preview(raw)), 85), "gt": None if gt is None else encode(orient(gt), 90),
           "results": []}
    for pid in pipelines:
        if pid not in PIPELINES:
            continue
        name, kind, spec, _ = PIPELINES[pid]
        t = time.perf_counter()
        if kind == "classical":
            out, info = C.ClassicalISP(C.ClassicalConfig()).run(raw, keep_intermediates=stages)
            model = None
        else:
            net = get_net(spec)
            if net is None:
                continue
            out, info = net.run(raw, keep_intermediates=stages)
            model = Path(net.net.path).name
        wall = (time.perf_counter() - t) * 1000
        r = {"id": pid, "name": name, "model": model, "ms": round(wall), "image": encode(orient(out), 90),
             "timings": {STAGE_LABELS.get(k, k): round(v, 1) for k, v in info.get("timings_ms", {}).items()},
             "decisions": {k: info[k] for k in ("wb_source", "noise_source", "ccm_source", "exposure_gain_k", "detail_alpha",
                                                 "wb_gains", "noise_profile") if info.get(k) is not None}}
        if kind == "classical":
            r["decisions"]["wb_source"] = "metadata" if raw.wb_gains is not None else "grayworld"
            r["decisions"]["noise_source"] = "metadata" if raw.noise_profile is not None else "estimated"
        if gt is not None:
            m = all_metrics(out, gt, use_lpips=False)
            r["metrics"] = {k: round(float(v), 4) for k, v in m.items()}
        if stages:
            cfa = raw.cfa if kind == "classical" else "RGGB"
            r["stages"] = [{"name": STAGE_LABELS.get(k, k), "image": encode(orient(stage_image(v, cfa)), 80, 720)}
                           for k, v in info.get("intermediates", {}).items()]
        res["results"].append(r)
    res["total_ms"] = round((time.perf_counter() - t0) * 1000)
    return res


def benchmark() -> dict | None:
    """Headline tables from eval/results (merged summaries) for the benchmark section."""
    try:
        sys.path.insert(0, str(ROOT))
        from eval.make_report import merged_summary
        main = merged_summary("main")
        rob = merged_summary("robust")
    except Exception:
        return None
    if not main:
        return None
    eff = None
    p = ROOT / "eval" / "results" / "efficiency.json"
    if p.exists():
        eff = json.load(open(p))
    return {"clean": main["conditions"]["clean"], "by_iso": main["by_iso"], "n_crops": main["n_crops"],
            "robust": rob["conditions"] if rob else None, "efficiency": eff}
