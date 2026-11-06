"""Tune the hybrid's detail-restoration strength (alpha) and coring (c) on the SIDD VALIDATION split.

Objective: minimise LPIPS (perceptual) subject to PSNR dropping by at most 0.15 dB vs alpha = 0.
  python scripts/tune_detail.py --model models/hybrid_student.pt --data data/prep
"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.evaluate import test_crops  # noqa: E402
from isp.hybrid import HybridISP  # noqa: E402
from isp.metrics import lpips_dist, psnr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/hybrid_student.pt")
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--out", default="eval/results/detail_tuning.json")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    isp = HybridISP(a.model, device=a.device)
    crops = list(test_crops(Path(a.data), 3, split="val"))
    grid = [(0.0, 2.0)] + list(itertools.product([0.2, 0.4, 0.6, 0.8, 1.0], [1.0, 1.5, 2.0, 3.0]))
    res = []
    for al, c in grid:
        ps, lp = [], []
        for s in crops:
            o = isp.run(s["raw"], detail=al, coring=c)[0]
            ps.append(psnr(o, s["gt"])); lp.append(lpips_dist(o, s["gt"]))
        res.append({"alpha": al, "coring": c, "psnr": float(np.mean(ps)), "lpips": float(np.mean(lp))})
        print(res[-1], flush=True)
    base = res[0]["psnr"]
    ok = [r for r in res if r["psnr"] >= base - 0.15]
    best = min(ok, key=lambda r: r["lpips"])
    print("BEST", best, "baseline", res[0])
    json.dump({"grid": res, "best": best, "n_crops": len(crops), "split": "val"}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
