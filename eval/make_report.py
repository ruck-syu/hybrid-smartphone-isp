"""Build eval/RESULTS.md and the submission file eval_P2_<TeamID>.md from evaluation JSON.

  python eval/make_report.py --team T00 --usns "USN1,USN2,USN3"
Metric order follows the problem statement: PSNR, SSIM, Delta E, LPIPS, params / size, latency / memory.
"""
import argparse
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"

MAIN = [
    ("classical_bilateral", "Traditional ISP (tuned bilateral + guided, adaptive)"),
    ("classical_nlm", "Traditional ISP (non-local means)"),
    ("learned", "Learned ISP (end-to-end U-Net, 2.6M)"),
    ("hybrid_teacher", "Hybrid ISP — teacher (2.6M, FP32)"),
    ("hybrid_student", "Hybrid ISP — student, KD (0.64M, FP32)"),
    ("hybrid_student_int8", "**Hybrid ISP — student, KD, INT8 ONNX (deployed)**"),
]
ABL = [
    ("hybrid_student", "Full hybrid (student, KD)"),
    ("hybrid_student_nokd", "− knowledge distillation (same student, ground truth only, 180k vs 131k its)"),
    ("hybrid_student__nlf_estimated", "NoiseProfile metadata → noise model estimated from the image"),
    ("hybrid_student__noise_map_x0.25", "noise map under-estimated ×0.25"),
    ("hybrid_student__noise_map_x4", "noise map over-estimated ×4"),
    ("hybrid_student__no_exposure_norm", "− exposure normalisation (k = 1)"),
    ("hybrid_student__wb_grayworld", "metadata WB → grey-world AWB"),
    ("hybrid_student__wb_sog", "metadata WB → shades-of-grey AWB"),
    ("hybrid_student__generic_ccm", "metadata CCM → generic CCM"),
    ("hybrid_student__sharpen", "+ classical unsharp mask after network"),
    ("hybrid_student__no_detail", "− noise-adaptive detail restoration"),
    ("hybrid_student__nometa", "no metadata at all (AWB + generic CCM + estimated NLF)"),
    ("classical_nodenoise", "Traditional ISP without denoising"),
    ("classical_bilateral_fixed", "Traditional ISP, fixed (non-adaptive) denoise parameters"),
    ("classical_nometa", "Traditional ISP without metadata"),
]


def fmt(v, k):
    if v is None:
        return "–"
    return {"psnr": f"{v:.2f}", "ssim": f"{v:.4f}", "delta_e": f"{v:.2f}", "lpips": f"{v:.4f}",
            "latency_ms": f"{v:.0f}"}.get(k, f"{v}")


def table(summary, rows, eff=None, cols=("psnr", "ssim", "delta_e", "lpips")):
    head = "| Method | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ |"
    sep = "|---|---|---|---|---|"
    out = [head, sep]
    for key, label in rows:
        if key not in summary:
            continue
        s = summary[key]
        out.append(f"| {label} | " + " | ".join(fmt(s.get(c), c) for c in cols) + " |")
    return "\n".join(out)


def eff_table(eff):
    out = ["| Model | Params | Size (MB) | kMAC / px | Latency 1 MP (ms) | Latency 4 MP (ms) | Peak mem 4 MP (MB) |",
           "|---|---|---|---|---|---|---|"]
    params = {}
    for name, r in eff["models"].items():
        stem = name.replace(".pt", "").replace("_fp32.onnx", "").replace("_int8.onnx", "").replace("_fp16.onnx", "")
        if "params" in r:
            params[stem] = (r["params"], r["kmacs_per_pixel"])
    for name, r in eff["models"].items():
        stem = name.replace(".pt", "").replace("_fp32.onnx", "").replace("_int8.onnx", "").replace("_fp16.onnx", "")
        p, mac = params.get(stem, (None, None))
        l1 = r.get("1024x1024", {}).get("latency_ms_median")
        l4 = r.get("2048x2048", {}).get("latency_ms_median")
        m4 = r.get("2048x2048", {}).get("peak_rss_mb")
        size = r.get("file_size_mb")
        out.append(f"| {name} | {'–' if p is None else f'{p/1e6:.3f}M'} | {'–' if size is None else f'{size:.2f}'} | "
                   f"{'–' if mac is None else f'{mac:.1f}'} | {'–' if l1 is None else f'{l1:.0f}'} | "
                   f"{'–' if l4 is None else f'{l4:.0f}'} | {'–' if m4 is None else f'{m4:.0f}'} |")
    return "\n".join(out)


def merged_summary(prefix):
    """Merge rows_<prefix>*.json files (methods evaluated in separate runs) and summarise."""
    import sys
    sys.path.insert(0, str(R.parents[1]))
    from collections import defaultdict
    from eval.evaluate import iso_bucket, summarise
    rows = []
    for f in sorted(R.glob(f"rows_{prefix}*.json")):
        rows += json.load(open(f))
    if not rows:
        return None
    conds = sorted({r["condition"] for r in rows}, key=lambda c: (c != "clean", c))
    clean = [r for r in rows if r["condition"] == "clean"]
    return {"conditions": {c: summarise([r for r in rows if r["condition"] == c]) for c in conds},
            "by_iso": {b: summarise([r for r in clean if iso_bucket(r["iso"]) == b])
                       for b in ("ISO<=400", "ISO 800-1600", "ISO>=3200")},
            "by_camera": {cam: summarise([r for r in clean if r["camera"] == cam])
                          for cam in sorted({r["camera"] for r in clean})},
            "n_crops": len({(r["id"], r["k"], r["crop"]) for r in clean})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="TXX")
    ap.add_argument("--usns", default="USN1,USN2,USN3,USN4")
    a = ap.parse_args()
    main_s = merged_summary("main")
    rob = merged_summary("robust")
    eff = json.load(open(R / "efficiency.json")) if (R / "efficiency.json").exists() else None
    clean = main_s["conditions"]["clean"]
    md = [f"# Evaluation results — P2 Hybrid Smartphone ISP\n",
          f"Held-out SIDD test scenes (scene contents 003 & 010, never used for training or tuning), "
          f"{main_s['n_crops']} full-resolution 1024×1024 crops from 36 noisy RAW captures over 5 smartphone "
          f"cameras (ISO 50–10000). Ground truth = official SIDD sRGB.\n",
          "## 1. Main comparison (traditional vs learned vs hybrid)\n", table(clean, MAIN), ""]
    if eff:
        md += ["## 2. Efficiency (CPU, %d threads, %s; full ISP incl. classical stages)\n" % (eff["threads"], eff["cpu"]),
               eff_table(eff), ""]
    md += ["## 3. Ablation study\n", table(clean, ABL), ""]
    md += ["## 4. Performance by ISO (clean condition)\n"]
    for b, s in main_s["by_iso"].items():
        md += [f"### {b}\n", table(s, MAIN), ""]
    md += ["## 5. Per camera (PSNR / ΔE)\n", "| Method | " + " | ".join(main_s["by_camera"]) + " |",
           "|---|" + "---|" * len(main_s["by_camera"])]
    for key, label in MAIN:
        if key in clean:
            md.append(f"| {label} | " + " | ".join(
                f"{main_s['by_camera'][c][key]['psnr']:.2f} / {main_s['by_camera'][c][key]['delta_e']:.2f}"
                if key in main_s["by_camera"][c] else "–" for c in main_s["by_camera"]) + " |")
    md.append("")
    if rob:
        md += ["## 6. Robustness: degradation and recovery under challenging conditions\n",
               "| Condition | " + " | ".join(l for k, l in MAIN if k in rob["conditions"]["clean"]) + " |",
               "|---|" + "---|" * sum(1 for k, _ in MAIN if k in rob["conditions"]["clean"])]
        for cond, s in rob["conditions"].items():
            md.append(f"| {cond} | " + " | ".join(
                f"{s[k]['psnr']:.2f} / {s[k]['delta_e']:.2f}" for k, _ in MAIN if k in s) + " |")
        md += ["", "Cells are PSNR (dB) / ΔE2000. `noise_xN`: N× sensor-noise variance injected with the "
               "camera's own noise model (simulates higher ISO); `wb_error_15pct`: AsShotNeutral corrupted by ±15% "
               "(R/B); `no_metadata`: WB, CCM and noise profile removed → adaptive estimation.", ""]
    md += ["## 7. Notes for reading these results\n",
           "- **Model selection used validation only.** The deployed student (distilled) was chosen because it "
           "beat the ground-truth-only student on the validation split (43.43 vs 43.09 dB). On the test split the "
           "two are equivalent (PSNR −0.05 dB, LPIPS −0.003, ΔE +0.02). We report this rather than re-picking on test.",
           "- **INT8 is mixed precision.** Per-layer sensitivity analysis (`eval/results/quant_sensitivity.json`) keeps "
           "the full-resolution U-Net level in float; naive all-layer INT8 loses about 2 dB.",
           "- **Detail restoration is noise-adaptive.** It switches itself off on every SIDD test image (all are noisy), "
           "so these tables measure the network path; its effect on clean phone photos is shown in "
           "docs/FAILURE_ANALYSIS.md §9.",
           "- **The ground truth is soft** (burst-averaged), which favours smoother outputs in PSNR/SSIM; LPIPS is the "
           "better guide to perceived detail.", ""]
    (R.parent / "RESULTS.md").write_text("\n".join(md))

    # Submission file: first line = USNs, metrics in the order of the problem statement.
    best = "hybrid_student_int8" if "hybrid_student_int8" in clean else "hybrid_student"
    base = "classical_bilateral"
    lines = [a.usns, "", f"# eval_P2_{a.team} — Intelligent and Computationally Efficient Smartphone ISP", "",
             "Test set: held-out SIDD scenes, %d 1024×1024 RAW crops, official sRGB ground truth." % main_s["n_crops"], "",
             "| Metric | Traditional ISP | Learned ISP | Hybrid ISP (deployed, INT8) |", "|---|---|---|---|"]
    trio = [base, "learned", best]
    for k, lab in [("psnr", "PSNR (dB) ↑"), ("ssim", "SSIM ↑"), ("delta_e", "Colour accuracy ΔE2000 ↓"),
                   ("lpips", "LPIPS (AlexNet) ↓")]:
        lines.append(f"| {lab} | " + " | ".join(fmt(clean.get(m, {}).get(k), k) for m in trio) + " |")
    if eff:
        def g(name, *ks):
            r = eff["models"].get(name, {})
            for k in ks:
                r = r.get(k, {}) if isinstance(r, dict) else None
            return r if not isinstance(r, dict) else None
        lp = lambda n: g(n, "params")
        lines.append(f"| Model parameters | 0 (hand-engineered) | {lp('learned.pt') or 0:,} | {lp('hybrid_student.pt') or 0:,} |")
        lines.append(f"| Model size (MB) | 0 | {g('learned.pt', 'file_size_mb') or 0:.2f} (FP32) | "
                     f"{g('hybrid_student_int8.onnx', 'file_size_mb') or 0:.2f} (INT8) |")
        for s in ("1024x1024", "2048x2048"):
            lines.append(f"| Inference latency {s} RAW, CPU (ms) | " + " | ".join(
                f"{(g(n, s, 'latency_ms_median') or 0):.0f}" for n in ("classical", "learned_fp32.onnx", "hybrid_student_int8.onnx")) + " |")
            lines.append(f"| Peak memory {s} (MB) | " + " | ".join(
                f"{(g(n, s, 'peak_rss_mb') or 0):.0f}" for n in ("classical", "learned_fp32.onnx", "hybrid_student_int8.onnx")) + " |")
    lines += ["", "Latency and memory: whole pipeline (classical stages included) through ONNX Runtime on a laptop CPU "
              "with 4 threads; the learned ISP is its FP32 ONNX export.",
              "", "Full tables (ablation, per-ISO, per-camera, robustness, efficiency of every variant): eval/RESULTS.md"]
    (R.parents[1] / f"eval_P2_{a.team}.md").write_text("\n".join(lines) + "\n")
    print("wrote RESULTS.md and", f"eval_P2_{a.team}.md")


if __name__ == "__main__":
    main()
