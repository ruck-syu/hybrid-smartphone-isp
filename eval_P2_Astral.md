1RUA25CSE7013, 1RUA25CSE7015, 1RUA24CSE0535


# eval_P2_Astral — Intelligent and Computationally Efficient Smartphone ISP

Test set: held-out SIDD scenes, 108 1024×1024 RAW crops, official sRGB ground truth.

| Metric | Traditional ISP | Learned ISP | Hybrid ISP (deployed, INT8) |
|---|---|---|---|
| PSNR (dB) ↑ | 33.14 | 33.26 | 37.97 |
| SSIM ↑ | 0.9158 | 0.9243 | 0.9297 |
| Colour accuracy ΔE2000 ↓ | 4.89 | 4.34 | 2.03 |
| LPIPS (AlexNet) ↓ | 0.3182 | 0.2192 | 0.1659 |
| Model parameters | 0 (hand-engineered) | 2,541,356 | 637,660 |
| Model size (MB) | 0 | 10.18 (FP32) | 0.75 (INT8) |
| Inference latency 1024x1024 RAW, CPU (ms) | 107 | 338 | 141 |
| Peak memory 1024x1024 (MB) | 169 | 400 | 405 |
| Inference latency 2048x2048 RAW, CPU (ms) | 405 | 3228 | 1203 |
| Peak memory 2048x2048 (MB) | 561 | 600 | 892 |

Latency and memory: whole pipeline (classical stages included) through ONNX Runtime on a laptop CPU with 4 threads; the learned ISP is its FP32 ONNX export.

Full tables (ablation, per-ISO, per-camera, robustness, efficiency of every variant): eval/RESULTS.md
