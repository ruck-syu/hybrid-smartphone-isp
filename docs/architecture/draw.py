"""Render the architecture diagram (docs/architecture/hybrid-isp.{png,svg}) with matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

BLUE, BLUE_E = "#dbeafe", "#1d4ed8"
ORG, ORG_E = "#ffedd5", "#c2410c"
GREY, GREY_E = "#f1f5f9", "#475569"
fig, ax = plt.subplots(figsize=(18, 8.2))
ax.set_xlim(0, 18); ax.set_ylim(0, 8.2); ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=9, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    lines = text.split("\n")
    ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=fs, color="#0f172a",
            linespacing=1.35, fontweight="normal")
    if bold_first:
        pass
    return (x, y, w, h)

def arrow(a, b, style="-|>", ls="-", color="#334155", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, lw=1.5, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))

ax.text(9, 7.9, "Hybrid Smartphone ISP: processing sequence", ha="center", fontsize=15, fontweight="bold")
y, h = 4.6, 2.2
xs = [0.1, 2.25, 4.4, 6.55, 8.7, 11.25, 13.4, 15.55]
w = [2.0, 2.0, 2.0, 2.0, 2.4, 2.0, 2.0, 2.3]
texts = [
 ("RAW Bayer + metadata\n(DNG / SIDD .MAT)\n\nAsShotNeutral,\nColorMatrix1/2,\nNoiseProfile, ISO", GREY, GREY_E),
 ("1. CFA canonicalise\n\nany 2x2 CFA -> RGGB\npack H/2 x W/2 x 4\n[classical]", BLUE, BLUE_E),
 ("2. Noise model\n(adaptive)\n\nDNG NoiseProfile or\nestimated NLF\nvar = a*x + b\n[classical]", BLUE, BLUE_E),
 ("3. White balance\n(adaptive)\n\nAsShotNeutral or\nAWB shades-of-grey\n[classical]", BLUE, BLUE_E),
 ("5. Noise-conditioned\nU-Net  [learned]\n\nin: 4ch WB raw + 4ch sigma\njoint denoise + demosaic\nout: linear camera RGB\n(PixelShuffle x2)\nstudent 0.64M, INT8 ONNX", ORG, ORG_E),
 ("6. Colour correction\n\nCCT-interpolated\nDNG ColorMatrix\n-> linear sRGB\n[classical]", BLUE, BLUE_E),
 ("7. sRGB OETF\n\ngamma + optional\ntone map / sharpen\n[classical]", BLUE, BLUE_E),
 ("sRGB output\n\n+ per-stage timings\n+ adaptive decisions\n(shown in the app)", GREY, GREY_E),
]
boxes = []
for (t, fc, ec), x, ww in zip(texts, xs, w):
    boxes.append(box(x, y, ww, h, t, fc, ec))
for i in range(len(boxes) - 1):
    a, b = boxes[i], boxes[i + 1]
    if i == 3:
        continue
    arrow((a[0] + a[2], y + h / 2), (b[0], y + h / 2))
# stage 4 (exposure normalisation) sits between WB and the network
k = box(6.6, 3.15, 1.9, 1.05, "4. Exposure norm. gain k\n(illumination adaptive)", BLUE, BLUE_E, fs=8.5)
arrow((boxes[3][0] + boxes[3][2], y + h / 2), (boxes[4][0], y + h / 2))
arrow((7.55, 4.2), (8.7, 5.1), rad=-0.2)
ax.text(10.1, 4.35, "x k  ...  / k", fontsize=8, color=BLUE_E, ha="center")

# training branch
tr = box(8.2, 0.95, 3.75, 1.75, "Training (SIDD, 136 train scenes x 2 captures)\nL1 on sRGB through differentiable\nCCM + gamma (fp32)  +  KD from 2.6M teacher\nnoise augmentation via NLF\nthen ONNX -> FP16 / INT8 (static PTQ)", ORG, ORG_E, fs=8.5)
arrow((9.9, 2.7), (9.9, 4.6), ls="--", color=ORG_E)
ax.text(10.0, 3.0, "trains", fontsize=8, color=ORG_E)

# baselines
b1 = box(0.1, 0.95, 3.9, 1.75, "Traditional ISP baseline\nWB -> Malvar demosaic -> CCM -> gamma\n-> bilateral (luma) + guided (chroma) denoise\n-> coring unsharp mask\n(noise-adaptive parameters)", BLUE, BLUE_E, fs=8.5)
b2 = box(4.2, 0.95, 3.8, 1.75, "Learned ISP baseline\npacked RAW (no metadata)\n-> U-Net (2.6M) -> sRGB\nmust learn WB, CCM, tone itself", ORG, ORG_E, fs=8.5)
arrow((0.8, 4.6), (0.8, 2.7), ls="--", color=GREY_E)
arrow((1.6, 4.6), (5.5, 2.7), ls="--", color=GREY_E, rad=0.15)
ax.text(12.25, 1.85, "All three pipelines are evaluated on held-out SIDD scenes\n(PSNR, SSIM, CIEDE2000, LPIPS, params, size, latency, memory)",
        fontsize=9, color="#334155", va="center")
ax.add_patch(FancyBboxPatch((12.2, 0.2), 0.35, 0.3, boxstyle="round,pad=0.02", fc=BLUE, ec=BLUE_E))
ax.text(12.65, 0.35, "classical", fontsize=9, va="center")
ax.add_patch(FancyBboxPatch((13.8, 0.2), 0.35, 0.3, boxstyle="round,pad=0.02", fc=ORG, ec=ORG_E))
ax.text(14.25, 0.35, "learned", fontsize=9, va="center")
out = Path(__file__).parent
fig.savefig(out / "hybrid-isp.png", dpi=130, bbox_inches="tight")
fig.savefig(out / "hybrid-isp.svg", bbox_inches="tight")
