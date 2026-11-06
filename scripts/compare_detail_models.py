"""Compare candidate hybrid students on VALIDATION data only (native + 2x-binned crops) and on the
detail they keep on unseen phone DNGs (network alone: detail restoration disabled).

  python scripts/compare_detail_models.py models/hybrid_student.pt runs/ft_g05/best_combined.pt ...
"""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "app"))
from isp.data import SIDDPatches  # noqa: E402
from isp.hybrid import HybridISP  # noqa: E402
from isp.metrics import lpips_dist, psnr  # noqa: E402
from isp.rawio import RawImage  # noqa: E402
from isp import classical as C  # noqa: E402
import core  # noqa: E402


def sharp(x):
    y = (x @ np.array([0.2126, 0.7152, 0.0722], np.float32)).astype(np.float32)
    return float(cv2.Laplacian(y, cv2.CV_32F).var() * 1e4)


def main():
    cands = sys.argv[1:]
    vds = SIDDPatches(str(ROOT / "data/prep"), "val", 256, "hybrid", augment=False)
    # rebuild val crops as RawImages so the full HybridISP (incl. classical stages) is exercised
    sets = {}
    for scaled in (False, True):
        rng = np.random.default_rng(7)
        items = []
        for r in vds.recs:
            if r["k"] != "010":
                continue
            noisy, gt = vds._arrays(r)
            s = 2 if scaled else 1
            ph = 256
            for _ in range(3):
                y = int(rng.integers(0, noisy.shape[0] - ph * s)); x = int(rng.integers(0, noisy.shape[1] - ph * s))
                p = noisy[y:y + ph * s, x:x + ph * s].astype(np.float32) / 65535
                t = gt[2 * y:2 * y + 2 * ph * s, 2 * x:2 * x + 2 * ph * s].astype(np.float32) / 255
                prof = tuple(r["noise_profile"])
                if s > 1:
                    p = p.reshape(ph, s, ph, s, 4).mean((1, 3)); prof = (prof[0] / 4, prof[1] / 4)
                    lin = C.srgb_eotf(t).reshape(2 * ph, s, 2 * ph, s, 3).mean((1, 3)); t = C.srgb_oetf(lin)
                raw = RawImage(bayer=C.unpack_rggb(p), cfa="RGGB", wb_gains=np.array(r["wb_gains"], np.float32),
                               cam2srgb=np.array(r["cam2srgb"], np.float32), noise_profile=prof)
                items.append((raw, t))
        sets["x2" if scaled else "native"] = items
    dngs = []
    for sid in ["dng:pixel4a_cc0.dng", "dng:pixel7pro_cc0.dng", "dng:iphonexs_cc0.dng"]:
        raw, _, _ = core.load_input(None, sid)
        dngs.append(core.centre_crop(raw, None, 1.0)[0])
    res = {}
    for c in cands:
        isp = HybridISP(c, device="mps")
        r = {}
        for name, items in sets.items():
            outs = [(isp.run(raw, detail=0.0)[0], t) for raw, t in items]
            r[f"psnr_{name}"] = float(np.mean([psnr(o, t) for o, t in outs]))
            r[f"lpips_{name}"] = float(np.mean([lpips_dist(o, t) for o, t in outs]))
        r["dng_sharpness_net_only"] = float(np.mean([sharp(isp.run(d, detail=0.0)[0]) for d in dngs]))
        r["dng_sharpness_trad"] = float(np.mean([sharp(C.ClassicalISP().run(d)[0]) for d in dngs]))
        res[c] = r
        print(c, json.dumps({k: round(v, 4) for k, v in r.items()}), flush=True)
    json.dump(res, open(ROOT / "eval/results/detail_finetune_comparison.json", "w"), indent=1)


if __name__ == "__main__":
    main()
