"""Hybrid Smartphone ISP — Gradio demo.

Upload a RAW file (DNG / CR2 / NEF / ARW / ... or a SIDD sample .npz) or pick a bundled SIDD test
sample, choose ISP pipelines and inspect outputs, intermediate stages, metrics and latency.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isp import classical as C  # noqa: E402
from isp.hybrid import HybridISP  # noqa: E402
from isp.metrics import all_metrics  # noqa: E402
from isp.rawio import RawImage, load_any  # noqa: E402

MODELS = Path(os.environ.get("ISP_MODELS", ROOT / "models"))
SAMPLES = Path(os.environ.get("ISP_SAMPLES", ROOT / "samples"))
MAX_MP = float(os.environ.get("ISP_MAX_MP", "1.2"))  # megapixels processed (binning above this)
THREADS = int(os.environ.get("ISP_THREADS", "0")) or None
try:
    import cv2
    cv2.setNumThreads(THREADS or 0)
except Exception:
    pass
CPU_INFO = {"os_cpu_count": os.cpu_count(),
            "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "ort_threads": THREADS}

PIPELINES = {
    "Traditional ISP (classical)": ("classical", None),
    "Learned ISP (end-to-end U-Net, 4x MACs)": ("net", "learned_int8.onnx|learned_fp32.onnx|learned.pt"),
    "Hybrid ISP (student INT8, deployed)": ("net", "hybrid_student_int8.onnx|hybrid_student.pt"),
    "Hybrid ISP (student FP32)": ("net", "hybrid_student_fp32.onnx|hybrid_student.pt"),
    "Hybrid ISP (teacher FP32)": ("net", "hybrid_teacher_fp32.onnx|hybrid_teacher.pt"),
}
_cache: dict[str, HybridISP] = {}


def get_net(spec: str) -> HybridISP | None:
    for name in spec.split("|"):
        p = MODELS / name
        if p.exists():
            if name not in _cache:
                _cache[name] = HybridISP(p, device="cpu", tile=384, threads=THREADS)
            return _cache[name]
    return None


def available_pipelines():
    out = []
    for k, (kind, spec) in PIPELINES.items():
        if kind == "classical" or get_net(spec) is not None:
            out.append(k)
    return out


def bayer_bin(raw: RawImage, f: int) -> RawImage:
    """Bayer-preserving f x f binning of each colour plane (reduces resolution and noise)."""
    if f <= 1:
        return raw
    rggb, _ = C.to_rggb(raw.bayer, raw.cfa)
    p = C.pack_rggb(rggb[: (rggb.shape[0] // (2 * f)) * 2 * f, : (rggb.shape[1] // (2 * f)) * 2 * f])
    h, w, _ = p.shape
    p = p.reshape(h // f, f, w // f, f, 4).mean((1, 3))
    out = RawImage(bayer=C.unpack_rggb(p.astype(np.float32)), cfa="RGGB", wb_gains=raw.wb_gains,
                   cam2srgb=raw.cam2srgb, iso=raw.iso, orientation=raw.orientation,
                   noise_profile=None if raw.noise_profile is None else
                   (raw.noise_profile[0] / f ** 2, raw.noise_profile[1] / f ** 2))
    return out


def list_samples():
    if not SAMPLES.exists():
        return []
    return sorted(p.stem for p in SAMPLES.glob("*.npz"))


def load_sample(name):
    d = np.load(SAMPLES / f"{name}.npz", allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    raw = RawImage(bayer=d["bayer"].astype(np.float32), cfa=meta.get("cfa", "RGGB"),
                   wb_gains=np.array(meta["wb_gains"], np.float32) if meta.get("wb_gains") else None,
                   cam2srgb=np.array(meta["cam2srgb"], np.float32) if meta.get("cam2srgb") else None,
                   iso=meta.get("iso"), noise_profile=tuple(meta["noise_profile"]) if meta.get("noise_profile") else None)
    gt = d["gt"].astype(np.float32) / 255 if "gt" in d else None
    return raw, gt, meta


def degrade(raw: RawImage, noise_mult: float, wb_err: float, exposure_stops: float, drop_meta: bool, seed=0):
    rng = np.random.default_rng(seed)
    b = raw.bayer.copy()
    prof = raw.noise_profile
    if exposure_stops < 0:  # simulate under-exposure followed by digital gain (noise amplification)
        s = 2 ** exposure_stops
        a0, b0 = prof if prof else (1e-4, 1e-6)
        dark = b * s
        dark = dark + rng.standard_normal(b.shape).astype(np.float32) * np.sqrt(np.maximum(a0 * dark + b0, 0))
        b = np.clip(dark / s, 0, 1)
        prof = (a0 / s, b0 / s ** 2)  # approx after gain (shot ∝ 1/s, read ∝ 1/s²) + original noise
    if noise_mult > 1:
        a0, b0 = prof if prof else (1e-4, 1e-6)
        b = np.clip(b + rng.standard_normal(b.shape).astype(np.float32) * np.sqrt(np.maximum((noise_mult - 1) * (a0 * b + b0), 0)), 0, 1)
        prof = (a0 * noise_mult, b0 * noise_mult)
    out = RawImage(bayer=b, cfa=raw.cfa, wb_gains=None if raw.wb_gains is None else raw.wb_gains * np.array([1 + wb_err, 1, 1 - wb_err], np.float32),
                   cam2srgb=raw.cam2srgb, iso=raw.iso, noise_profile=prof, orientation=raw.orientation)
    if drop_meta:
        out.wb_gains, out.cam2srgb, out.noise_profile = None, None, None
    return out


def run_pipeline(name: str, raw: RawImage, keep: bool):
    kind, spec = PIPELINES[name]
    t = time.perf_counter()
    if kind == "classical":
        out, info = C.ClassicalISP(C.ClassicalConfig(wb="metadata", ccm="metadata", denoise="bilateral")).run(raw, keep_intermediates=keep)
    else:
        net = get_net(spec)
        out, info = net.run(raw, keep_intermediates=keep)
        info["model"] = Path(net.net.path).name
    info["wall_ms"] = (time.perf_counter() - t) * 1000
    return out, info


def to_u8(x):
    return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def raw_preview(raw: RawImage):
    rggb, _ = C.to_rggb(raw.bayer, raw.cfa)
    p = C.pack_rggb(rggb[: rggb.shape[0] // 2 * 2, : rggb.shape[1] // 2 * 2])
    rgb = np.stack([p[..., 0], 0.5 * (p[..., 1] + p[..., 2]), p[..., 3]], -1)
    return to_u8(C.gamma_simple(rgb / max(np.percentile(rgb, 99.5), 1e-3)))


def process(upload, sample, pipelines, noise_mult, wb_err, exposure_stops, drop_meta, show_stages, progress=gr.Progress()):
    try:
        progress(0.05, desc="Loading RAW")
        gt, meta = None, {}
        if upload is not None:
            raw = load_any(upload)
            src = Path(upload).name
        elif sample:
            raw, gt, meta = load_sample(sample)
            src = f"SIDD test sample {sample}"
        else:
            raise gr.Error("Upload a RAW file or choose a sample first.")
        if not pipelines:
            raise gr.Error("Select at least one pipeline.")
        mp = raw.bayer.size / 1e6
        f = 1
        while mp / (f * f) > MAX_MP:
            f += 1
        if f > 1:
            raw = bayer_bin(raw, f)
            if gt is not None:
                import cv2
                gt = cv2.resize(gt, (raw.bayer.shape[1], raw.bayer.shape[0]), interpolation=cv2.INTER_AREA)
        raw = degrade(raw, noise_mult, wb_err, exposure_stops, drop_meta)
        gallery, rows, stages, details = [], [], [], {}
        gallery.append((C.apply_orientation(raw_preview(raw), raw.orientation), "RAW input (packed, normalised preview)"))
        if gt is not None:
            gallery.append((to_u8(gt), "Ground truth sRGB (SIDD)"))
        for i, name in enumerate(pipelines):
            progress(0.1 + 0.8 * i / len(pipelines), desc=f"Running {name}")
            out, info = run_pipeline(name, raw, keep=show_stages)
            out = C.apply_orientation(out, raw.orientation)
            gallery.append((to_u8(out), name))
            row = [name, f"{info['wall_ms']:.0f}"]
            if gt is not None:
                m = all_metrics(out, gt[: out.shape[0], : out.shape[1]], use_lpips=False)
                row += [f"{m['psnr']:.2f}", f"{m['ssim']:.4f}", f"{m['delta_e']:.2f}"]
            else:
                row += ["–", "–", "–"]
            rows.append(row)
            if show_stages:
                for k, v in info.pop("intermediates", {}).items():
                    v = np.asarray(v)
                    if v.ndim == 2:  # CFA-domain stages
                        v = np.repeat(C.gamma_simple(np.clip(v / max(np.percentile(v, 99.5), 1e-3), 0, 1))[..., None], 3, -1)
                    stages.append((to_u8(v), f"{name.split(' (')[0]} · {k}"))
            else:
                info.pop("intermediates", None)
            details[name] = {k: v for k, v in info.items()}
        progress(1.0)
        header = (f"**Source:** {src} · **Processed size:** {raw.bayer.shape[1]}×{raw.bayer.shape[0]}"
                  f"{f' (binned {f}×)' if f > 1 else ''} · **ISO:** {raw.iso or 'n/a'} · "
                  f"**Metadata:** {'yes' if raw.has_metadata else 'missing → adaptive estimation'}")
        details["_server"] = CPU_INFO
        return header, gallery, rows, stages, json.dumps(details, indent=1, default=float)
    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Processing failed: {type(e).__name__}: {e}")


def load_results_md():
    p = ROOT / "eval" / "RESULTS.md"
    return p.read_text() if p.exists() else "Evaluation results not bundled."


def build_ui():
    pipes = available_pipelines()
    default = [p for p in pipes if p.startswith(("Traditional", "Hybrid ISP (student INT8"))]
    with gr.Blocks(title="Hybrid Smartphone ISP") as demo:
        gr.Markdown("# Hybrid Smartphone ISP\nRAW Bayer → sRGB using classical ISP stages combined with a compact, "
                    "noise-conditioned neural network (INT8). Compare it with a traditional ISP and a fully learned ISP.")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                upload = gr.File(label="Upload RAW (DNG, CR2, NEF, ARW, ORF, RW2, .npz)", type="filepath")
                dngs = sorted(str(p) for p in (SAMPLES / "dng").glob("*.dng")) if (SAMPLES / "dng").exists() else []
                if dngs:
                    gr.Examples([[d] for d in dngs], inputs=[upload],
                                label="Unseen phone sensors (CC0 DNGs from raw.pixls.us, no ground truth)")
                sample = gr.Dropdown(choices=list_samples(), value=(list_samples() or [None])[0],
                                     label="…or choose a held-out SIDD test sample (has ground truth)")
                pipelines = gr.CheckboxGroup(pipes, value=default, label="Pipelines")
                with gr.Accordion("Challenging conditions (robustness lab)", open=False):
                    noise_mult = gr.Slider(1, 8, value=1, step=0.5, label="Extra sensor noise (× variance, simulates higher ISO)")
                    exposure = gr.Slider(-4, 0, value=0, step=0.5, label="Under-exposure (stops, then digital gain)")
                    wb_err = gr.Slider(-0.3, 0.3, value=0, step=0.05, label="White-balance metadata error (R+/B−)")
                    drop_meta = gr.Checkbox(False, label="Drop metadata (WB, CCM, noise profile) → adaptive estimation")
                show_stages = gr.Checkbox(True, label="Show intermediate ISP stages")
                btn = gr.Button("Process", variant="primary")
                gr.Markdown("<small>*X-Trans sensors are not supported. Large images are Bayer-binned to "
                            f"≤{MAX_MP:g} MP for responsiveness.</small>")
            with gr.Column(scale=2):
                header = gr.Markdown()
                gallery = gr.Gallery(label="Outputs", columns=3, height=520, object_fit="contain", format="jpeg")
                table = gr.Dataframe(headers=["Pipeline", "Latency ms", "PSNR", "SSIM", "ΔE2000"], label="Metrics vs ground truth")
                with gr.Accordion("Intermediate stages", open=False):
                    stages = gr.Gallery(columns=4, height=400, object_fit="contain", format="jpeg")
                with gr.Accordion("Per-stage timings & adaptive decisions (JSON)", open=False):
                    details = gr.Code(language="json")
        with gr.Accordion("Benchmark results (held-out SIDD test scenes)", open=False):
            gr.Markdown(load_results_md())
        btn.click(process, [upload, sample, pipelines, noise_mult, wb_err, exposure, drop_meta, show_stages],
                  [header, gallery, table, stages, details], concurrency_limit=2)
        upload.change(lambda f: gr.update(value=None) if f else gr.update(), upload, sample)
    return demo


if __name__ == "__main__":
    build_ui().queue(max_size=16).launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                                         theme=gr.themes.Soft())
