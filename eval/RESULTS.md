# Evaluation results — P2 Hybrid Smartphone ISP

Held-out SIDD test scenes (scene contents 003 & 010, never used for training or tuning), 108 full-resolution 1024×1024 crops from 36 noisy RAW captures over 5 smartphone cameras (ISO 50–10000). Ground truth = official SIDD sRGB.

## 1. Main comparison (traditional vs learned vs hybrid)

| Method | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ |
|---|---|---|---|---|
| Traditional ISP (tuned bilateral + guided, adaptive) | 33.14 | 0.9158 | 4.89 | 0.3182 |
| Traditional ISP (non-local means) | 32.03 | 0.8931 | 5.42 | 0.2941 |
| Learned ISP (end-to-end U-Net, 2.6M) | 33.26 | 0.9243 | 4.34 | 0.2192 |
| Hybrid ISP — teacher (2.6M, FP32) | 38.86 | 0.9367 | 1.84 | 0.1667 |
| Hybrid ISP — student, KD (0.64M, FP32) | 38.51 | 0.9350 | 1.90 | 0.1713 |
| **Hybrid ISP — student, KD, INT8 ONNX (deployed)** | 37.97 | 0.9297 | 2.03 | 0.1659 |

## 2. Efficiency (CPU, 4 threads, arm64; full ISP incl. classical stages)

| Model | Params | Size (MB) | kMAC / px | Latency 1 MP (ms) | Latency 4 MP (ms) | Peak mem 4 MP (MB) |
|---|---|---|---|---|---|---|
| classical | – | – | – | 107 | 405 | 561 |
| hybrid_student.pt | 0.638M | 2.57 | 11.7 | 294 | 2496 | 1198 |
| hybrid_student_kd.pt | 0.638M | 2.57 | 11.7 | 305 | 2334 | 1206 |
| hybrid_student_nokd.pt | 0.638M | 2.57 | 11.7 | 294 | 2712 | 1203 |
| hybrid_teacher.pt | 2.543M | 10.18 | 45.2 | 547 | 5395 | 1877 |
| learned.pt | 2.541M | 10.18 | 44.9 | 499 | 4881 | 1606 |
| learned_student.pt | 0.637M | 2.56 | 11.5 | 210 | 2233 | 894 |
| hybrid_student_fp16.onnx | 0.638M | 1.29 | 11.7 | 196 | 1661 | 955 |
| hybrid_student_fp32.onnx | 0.638M | 2.56 | 11.7 | 171 | 1495 | 904 |
| hybrid_student_int8.onnx | 0.638M | 0.75 | 11.7 | 141 | 1203 | 892 |
| hybrid_teacher_fp32.onnx | 2.543M | 10.18 | 45.2 | 402 | 3548 | 1046 |
| learned_fp32.onnx | 2.541M | 10.18 | 44.9 | 338 | 3228 | 600 |
| learned_int8.onnx | 2.541M | 2.80 | 44.9 | 210 | 2041 | 693 |

## 3. Ablation study

| Method | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ |
|---|---|---|---|---|
| Full hybrid (student, KD) | 38.51 | 0.9350 | 1.90 | 0.1713 |
| − knowledge distillation (same student, ground truth only, 180k vs 131k its) | 38.56 | 0.9344 | 1.88 | 0.1743 |
| NoiseProfile metadata → noise model estimated from the image | 37.95 | 0.9310 | 2.00 | 0.1781 |
| noise map under-estimated ×0.25 | 37.82 | 0.9269 | 2.06 | 0.1677 |
| noise map over-estimated ×4 | 35.51 | 0.8973 | 3.10 | 0.2023 |
| − exposure normalisation (k = 1) | 37.49 | 0.9244 | 2.13 | 0.1832 |
| metadata WB → grey-world AWB | 34.47 | 0.9255 | 4.45 | 0.1980 |
| metadata WB → shades-of-grey AWB | 33.29 | 0.9142 | 6.12 | 0.2174 |
| metadata CCM → generic CCM | 36.50 | 0.9311 | 2.51 | 0.1727 |
| + classical unsharp mask after network | 38.51 | 0.9350 | 1.90 | 0.1711 |
| − noise-adaptive detail restoration | 38.51 | 0.9350 | 1.90 | 0.1706 |
| no metadata at all (AWB + generic CCM + estimated NLF) | 33.08 | 0.9136 | 5.93 | 0.2172 |
| Traditional ISP without denoising | 26.17 | 0.4784 | 9.64 | 0.6775 |
| Traditional ISP, fixed (non-adaptive) denoise parameters | 31.72 | 0.7993 | 5.37 | 0.3402 |
| Traditional ISP without metadata | 29.77 | 0.8959 | 7.72 | 0.3617 |

## 4. Performance by ISO (clean condition)

### ISO<=400

| Method | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ |
|---|---|---|---|---|
| Traditional ISP (tuned bilateral + guided, adaptive) | 34.26 | 0.9593 | 4.44 | 0.1693 |
| Traditional ISP (non-local means) | 33.48 | 0.9443 | 4.67 | 0.1614 |
| Learned ISP (end-to-end U-Net, 2.6M) | 34.76 | 0.9660 | 3.74 | 0.1206 |
| Hybrid ISP — teacher (2.6M, FP32) | 40.90 | 0.9756 | 1.33 | 0.0719 |
| Hybrid ISP — student, KD (0.64M, FP32) | 40.40 | 0.9737 | 1.43 | 0.0768 |
| **Hybrid ISP — student, KD, INT8 ONNX (deployed)** | 39.78 | 0.9684 | 1.57 | 0.0803 |

### ISO 800-1600

| Method | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ |
|---|---|---|---|---|
| Traditional ISP (tuned bilateral + guided, adaptive) | 33.10 | 0.9322 | 4.96 | 0.3444 |
| Traditional ISP (non-local means) | 31.83 | 0.9022 | 5.41 | 0.3301 |
| Learned ISP (end-to-end U-Net, 2.6M) | 32.35 | 0.9370 | 4.87 | 0.2312 |
| Hybrid ISP — teacher (2.6M, FP32) | 38.12 | 0.9528 | 1.92 | 0.1717 |
| Hybrid ISP — student, KD (0.64M, FP32) | 37.83 | 0.9513 | 1.97 | 0.1776 |
| **Hybrid ISP — student, KD, INT8 ONNX (deployed)** | 37.22 | 0.9450 | 2.12 | 0.1648 |

### ISO>=3200

| Method | PSNR ↑ | SSIM ↑ | ΔE2000 ↓ | LPIPS ↓ |
|---|---|---|---|---|
| Traditional ISP (tuned bilateral + guided, adaptive) | 30.25 | 0.7618 | 5.93 | 0.6543 |
| Traditional ISP (non-local means) | 28.63 | 0.7357 | 7.47 | 0.5640 |
| Learned ISP (end-to-end U-Net, 2.6M) | 31.36 | 0.7833 | 4.71 | 0.4541 |
| Hybrid ISP — teacher (2.6M, FP32) | 35.14 | 0.7952 | 3.00 | 0.4083 |
| Hybrid ISP — student, KD (0.64M, FP32) | 35.07 | 0.7937 | 3.02 | 0.4086 |
| **Hybrid ISP — student, KD, INT8 ONNX (deployed)** | 34.87 | 0.7907 | 3.07 | 0.3967 |

## 5. Per camera (PSNR / ΔE)

| Method | G4 | GP | IP | N6 | S6 |
|---|---|---|---|---|---|
| Traditional ISP (tuned bilateral + guided, adaptive) | 35.97 / 3.36 | 31.68 / 5.44 | 36.56 / 3.39 | 29.35 / 7.45 | 30.67 / 5.67 |
| Traditional ISP (non-local means) | 34.94 / 3.97 | 31.06 / 5.77 | 34.75 / 3.89 | 28.53 / 7.72 | 29.17 / 7.19 |
| Learned ISP (end-to-end U-Net, 2.6M) | 37.89 / 2.43 | 31.53 / 5.10 | 34.32 / 4.11 | 32.93 / 4.10 | 32.35 / 4.40 |
| Hybrid ISP — teacher (2.6M, FP32) | 40.87 / 1.64 | 36.91 / 2.05 | 41.53 / 1.34 | 38.65 / 1.65 | 37.24 / 2.72 |
| Hybrid ISP — student, KD (0.64M, FP32) | 40.41 / 1.74 | 36.44 / 2.15 | 41.55 / 1.31 | 37.93 / 1.78 | 36.85 / 2.80 |
| **Hybrid ISP — student, KD, INT8 ONNX (deployed)** | 39.86 / 1.86 | 35.89 / 2.30 | 41.25 / 1.38 | 36.77 / 2.06 | 36.33 / 2.90 |

## 6. Robustness: degradation and recovery under challenging conditions

| Condition | Traditional ISP (tuned bilateral + guided, adaptive) | Learned ISP (end-to-end U-Net, 2.6M) | Hybrid ISP — teacher (2.6M, FP32) | Hybrid ISP — student, KD (0.64M, FP32) | **Hybrid ISP — student, KD, INT8 ONNX (deployed)** |
|---|---|---|---|---|---|
| clean | 33.64 / 3.92 | 32.98 / 4.08 | 38.06 / 1.95 | 37.65 / 2.00 | 37.12 / 2.13 |
| no_metadata | 27.55 / 8.75 | 32.98 / 4.08 | 32.74 / 5.56 | 32.17 / 5.60 | 31.78 / 5.69 |
| noise_x2 | 33.03 / 4.05 | 32.29 / 4.33 | 37.41 / 2.06 | 36.64 / 2.18 | 36.19 / 2.30 |
| noise_x4 | 31.84 / 4.47 | 31.21 / 4.75 | 37.20 / 2.14 | 36.49 / 2.26 | 36.10 / 2.36 |
| noise_x8 | 30.38 / 5.16 | 29.74 / 5.40 | 36.05 / 2.47 | 34.69 / 2.87 | 34.38 / 2.94 |
| wb_error_15pct | 34.14 / 3.92 | 32.98 / 4.08 | 34.22 / 3.34 | 34.15 / 3.39 | 33.92 / 3.48 |

Cells are PSNR (dB) / ΔE2000. `noise_xN`: N× sensor-noise variance injected with the camera's own noise model (simulates higher ISO); `wb_error_15pct`: AsShotNeutral corrupted by ±15% (R/B); `no_metadata`: WB, CCM and noise profile removed → adaptive estimation.

## 7. Notes for reading these results

- **Model selection used validation only.** The deployed student (distilled) was chosen because it beat the ground-truth-only student on the validation split (43.43 vs 43.09 dB). On the test split the two are equivalent (PSNR −0.05 dB, LPIPS −0.003, ΔE +0.02). We report this rather than re-picking on test.
- **INT8 is mixed precision.** Per-layer sensitivity analysis (`eval/results/quant_sensitivity.json`) keeps the full-resolution U-Net level in float; naive all-layer INT8 loses about 2 dB.
- **Detail restoration is noise-adaptive.** It switches itself off on every SIDD test image (all are noisy), so these tables measure the network path; its effect on clean phone photos is shown in docs/FAILURE_ANALYSIS.md §9.
- **The ground truth is soft** (burst-averaged), which favours smoother outputs in PSNR/SSIM; LPIPS is the better guide to perceived detail.
