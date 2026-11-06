"""Grid-search the classical ISP's denoise / sharpening strengths on TRAINING scenes only.

  python scripts/tune_classical.py --data data/prep --out eval/results/classical_tuning.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.evaluate import test_crops  # noqa: E402
from isp import classical as C  # noqa: E402
from isp.metrics import psnr, ssim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="eval/results/classical_tuning.json")
    ap.add_argument("--max_images", type=int, default=24)
    a = ap.parse_args()
    crops = []
    for s in test_crops(Path(a.data), per_image=1, split="train"):
        if s["rec"]["k"] == "010":
            crops.append(s)
        if len(crops) >= a.max_images:
            break
    grid = {"bilateral": list(itertools.product([0.5, 1.0, 1.5, 2.0, 3.0], [0.0, 0.3, 0.6])),
            "nlm": list(itertools.product([0.6, 1.0, 1.5, 2.0, 3.0], [0.0, 0.2]))}
    res = []
    for den, combos in grid.items():
        for strength, sh in combos:
            isp = C.ClassicalISP(C.ClassicalConfig(denoise=den, denoise_strength=strength, sharpen=sh))
            ps, ss = [], []
            for s in crops:
                o = isp.run(s["raw"])[0]
                ps.append(psnr(o, s["gt"]))
                ss.append(ssim(o, s["gt"]))
            res.append({"denoise": den, "strength": strength, "sharpen": sh,
                        "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))})
            print(res[-1], flush=True)
    best = {d: max((r for r in res if r["denoise"] == d), key=lambda r: r["psnr"]) for d in grid}
    print("BEST", best)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"grid": res, "best": best, "n_images": len(crops)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
