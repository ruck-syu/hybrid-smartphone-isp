"""Comparative evaluation on held-out SIDD test scenes (never seen in training).

Evaluates traditional, learned and hybrid ISPs, ablations, optimisation variants and robustness
under challenging conditions. Writes eval/results/*.json and a markdown summary.

  python eval/evaluate.py --data data/prep --models models --out eval/results [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp import classical as C  # noqa: E402
from isp.hybrid import HybridISP  # noqa: E402
from isp.metrics import all_metrics  # noqa: E402
from isp.rawio import RawImage  # noqa: E402

CROP = 1024  # full-res crop size


def test_crops(root: Path, per_image: int, split: str = "test"):
    recs = [r for r in json.load(open(root / "index.json")) if r["split"] == split]
    for r in recs:
        noisy = np.load(root / f"{r['id']}_{r['k']}_noisy.npy", mmap_mode="r")
        gt = np.load(root / f"{r['id']}_gt.npy", mmap_mode="r")
        h, w = noisy.shape[:2]
        c = CROP // 2
        # deterministic grid of crops over the image
        ys = np.linspace(0, h - c, 3).astype(int)
        xs = np.linspace(0, w - c, 3).astype(int)
        cells = [(y, x) for y in ys for x in xs]
        sel = [cells[i] for i in np.linspace(0, len(cells) - 1, per_image).round().astype(int)]
        for j, (y, x) in enumerate(sel):
            p = noisy[y:y + c, x:x + c].astype(np.float32) / 65535
            t = gt[2 * y:2 * y + CROP, 2 * x:2 * x + CROP].astype(np.float32) / 255
            raw = RawImage(bayer=C.unpack_rggb(p), cfa="RGGB", wb_gains=np.array(r["wb_gains"], np.float32),
                           cam2srgb=np.array(r["cam2srgb"], np.float32), iso=r["iso"],
                           noise_profile=tuple(r["noise_profile"]) if r["noise_profile"] else None)
            yield {"rec": r, "crop": j, "raw": raw, "gt": t}


def degrade(raw: RawImage, noise_mult: float = 1.0, wb_err: float = 0.0, drop_meta: bool = False,
            rng=None) -> RawImage:
    """Simulate challenging conditions: higher sensor noise (ISO), WB metadata error, missing metadata."""
    rng = rng or np.random.default_rng(0)
    b = raw.bayer
    prof = raw.noise_profile
    if noise_mult > 1 and prof is not None:
        a, c = prof
        extra = noise_mult - 1
        b = np.clip(b + rng.standard_normal(b.shape).astype(np.float32) * np.sqrt(np.maximum(extra * (a * b + c), 0)), 0, 1)
        prof = (a * noise_mult, c * noise_mult)
    out = RawImage(bayer=b, cfa=raw.cfa, wb_gains=None if raw.wb_gains is None else raw.wb_gains.copy(),
                   cam2srgb=raw.cam2srgb, iso=raw.iso, noise_profile=prof)
    if wb_err and out.wb_gains is not None:
        out.wb_gains = out.wb_gains * np.array([1 + wb_err, 1, 1 - wb_err], np.float32)
    if drop_meta:
        out.wb_gains, out.cam2srgb, out.noise_profile = None, None, None
    return out


def build_methods(models: Path, device: str, which: set[str] | None = None):
    M = {}

    def add(name, fn):
        if which is None or name in which:
            M[name] = fn

    cl = C.ClassicalISP(C.ClassicalConfig(denoise="bilateral"))
    cl_nlm = C.ClassicalISP(C.ClassicalConfig(denoise="nlm", denoise_strength=3.0))
    cl_none = C.ClassicalISP(C.ClassicalConfig(denoise="none", sharpen=0.0))
    cl_fixed = C.ClassicalISP(C.ClassicalConfig(denoise="bilateral", adaptive=False))
    cl_nometa = C.ClassicalISP(C.ClassicalConfig(denoise="bilateral", wb="shades_of_gray", ccm="generic"))
    add("classical_nodenoise", lambda r: cl_none.run(r)[0])
    add("classical_bilateral", lambda r: cl.run(r)[0])
    add("classical_bilateral_fixed", lambda r: cl_fixed.run(r)[0])
    add("classical_nlm", lambda r: cl_nlm.run(r)[0])
    add("classical_nometa", lambda r: cl_nometa.run(r)[0])

    def maybe(path):
        return path if Path(path).exists() else None

    nets = {}
    for key, fname in [("learned", "learned.pt"), ("learned_student", "learned_student.pt"),
                       ("hybrid_teacher", "hybrid_teacher.pt"),
                       ("hybrid_student", "hybrid_student.pt"), ("hybrid_student_nokd", "hybrid_student_nokd.pt"),
                       ("hybrid_student_int8", "hybrid_student_int8.onnx"),
                       ("hybrid_student_fp32onnx", "hybrid_student_fp32.onnx"),
                       ("hybrid_student_fp16", "hybrid_student_fp16.onnx"),
                       ("learned_int8", "learned_int8.onnx")]:
        p = maybe(models / fname)
        if p and (which is None or any(w.startswith(key) for w in which)):
            nets[key] = HybridISP(p, device=device if fname.endswith(".pt") else "cpu", tile=512)

    for key, isp in nets.items():
        add(key, lambda r, isp=isp: isp.run(r)[0])
    if "hybrid_student" in nets:
        h = nets["hybrid_student"]
        # ablations (inference-time)
        add("hybrid_student__nlf_estimated", lambda r: h.run(r, nlf="estimated")[0])
        add("hybrid_student__no_exposure_norm", lambda r: h.run(r, exposure_norm=False)[0])
        add("hybrid_student__noise_map_x0.25", lambda r: h.run(r, noise_scale=0.25)[0])
        add("hybrid_student__noise_map_x4", lambda r: h.run(r, noise_scale=4.0)[0])
        add("hybrid_student__wb_grayworld", lambda r: h.run(r, wb="grayworld")[0])
        add("hybrid_student__wb_sog", lambda r: h.run(r, wb="shades_of_gray")[0])
        add("hybrid_student__generic_ccm", lambda r: h.run(r, ccm="generic")[0])
        add("hybrid_student__sharpen", lambda r: h.run(r, sharpen=0.3)[0])
        add("hybrid_student__nometa", lambda r: h.run(r, wb="shades_of_gray", ccm="generic", nlf="estimated")[0])
    return M


def iso_bucket(iso):
    return "ISO<=400" if iso <= 400 else ("ISO 800-1600" if iso <= 1600 else "ISO>=3200")


def summarise(rows):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, v in r["metrics"].items():
            agg[r["method"]][k].append(v)
        agg[r["method"]]["latency_ms"].append(r["latency_ms"])
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--per_image", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--methods", default=None, help="comma-separated subset")
    ap.add_argument("--robustness", action="store_true")
    ap.add_argument("--no_lpips", action="store_true")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--split", default="test", help="use 'val' for model-selection decisions")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    which = set(a.methods.split(",")) if a.methods else None
    methods = build_methods(Path(a.models), a.device, which)
    print("methods:", list(methods), flush=True)
    rows = []
    conditions = [("clean", {})]
    if a.robustness:
        conditions = [("clean", {}), ("noise_x2", {"noise_mult": 2}), ("noise_x4", {"noise_mult": 4}),
                      ("noise_x8", {"noise_mult": 8}), ("wb_error_15pct", {"wb_err": 0.15}),
                      ("no_metadata", {"drop_meta": True})]
    t0 = time.time()
    for n, s in enumerate(test_crops(Path(a.data), a.per_image, split=a.split)):
        for cname, ckw in conditions:
            raw = degrade(s["raw"], rng=np.random.default_rng(n), **ckw) if ckw else s["raw"]
            for mname, fn in methods.items():
                t = time.perf_counter()
                try:
                    pred = fn(raw)
                except Exception as e:  # e.g. classical needs metadata -> falls back automatically
                    print("ERR", mname, cname, e, flush=True)
                    continue
                lat = (time.perf_counter() - t) * 1000
                m = all_metrics(pred, s["gt"], use_lpips=not a.no_lpips)
                rows.append({"method": mname, "condition": cname, "id": s["rec"]["id"], "k": s["rec"]["k"],
                             "camera": s["rec"]["camera"], "iso": s["rec"]["iso"], "crop": s["crop"],
                             "metrics": m, "latency_ms": lat})
        if n % 10 == 0:
            print(f"crop {n} done ({time.time()-t0:.0f}s)", flush=True)
    json.dump(rows, open(out / f"rows_{a.tag}.json", "w"))
    summary = {}
    for cname, _ in conditions:
        summary[cname] = summarise([r for r in rows if r["condition"] == cname])
    by_iso = defaultdict(dict)
    for b in ("ISO<=400", "ISO 800-1600", "ISO>=3200"):
        by_iso[b] = summarise([r for r in rows if r["condition"] == "clean" and iso_bucket(r["iso"]) == b])
    by_cam = {cam: summarise([r for r in rows if r["condition"] == "clean" and r["camera"] == cam])
              for cam in sorted({r["camera"] for r in rows})}
    json.dump({"conditions": summary, "by_iso": by_iso, "by_camera": by_cam,
               "n_crops": len({(r['id'], r['k'], r['crop']) for r in rows})},
              open(out / f"summary_{a.tag}.json", "w"), indent=1)
    for cname, s in summary.items():
        print(f"\n## {cname}")
        print(f"{'method':42s} PSNR    SSIM    dE00   LPIPS   ms")
        for m, v in s.items():
            print(f"{m:42s} {v['psnr']:.2f}  {v['ssim']:.4f}  {v['delta_e']:.3f}  {v.get('lpips', float('nan')):.4f}  {v['latency_ms']:.0f}")


if __name__ == "__main__":
    main()
