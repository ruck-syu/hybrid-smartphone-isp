"""Generate failure-case figures for docs/FAILURE_ANALYSIS.md (docs/figures/*.jpg) + numbers (JSON).

  python eval/failure_figures.py --data data/prep --models models
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.evaluate import test_crops  # noqa: E402
from isp import classical as C  # noqa: E402
from isp.hybrid import HybridISP, estimate_nlf  # noqa: E402
from isp.metrics import delta_e2000, psnr  # noqa: E402
from isp.rawio import RawImage, load_dng  # noqa: E402

OUT = ROOT / "docs" / "figures"


def label(img, text):
    im = (np.clip(img, 0, 1) * 255).astype(np.uint8).copy()
    cv2.rectangle(im, (0, 0), (im.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(im, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def save_row(name, panels):
    h = min(p.shape[0] for _, p in panels)
    row = np.concatenate([label(p[:h], t) for t, p in panels], 1)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{name}.jpg"), row[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    M = Path(a.models)
    cl = C.ClassicalISP(C.ClassicalConfig())
    learned = HybridISP(M / "learned.pt")
    hyb = HybridISP(M / "hybrid_student_int8.onnx")
    stats = {}

    # 1) worst classical crop (high ISO, low light): clipped-noise bias -> colour/black-level shift
    crops = list(test_crops(Path(a.data), 3))
    worst, worst_de = None, -1
    for s in crops:
        if s["rec"]["iso"] >= 1600 and s["rec"]["k"] == "010":
            o = cl.run(s["raw"])[0]
            de = delta_e2000(o, s["gt"])
            if de > worst_de:
                worst, worst_de, wo = s, de, o
    c = slice(256, 768)
    ol, oh = learned.run(worst["raw"])[0], hyb.run(worst["raw"])[0]
    stats["high_iso_dark"] = {"scene": worst["rec"]["name"], "iso": worst["rec"]["iso"],
                              **{k: {"psnr": psnr(o, worst["gt"]), "delta_e": delta_e2000(o, worst["gt"])}
                                 for k, o in [("classical", wo), ("learned", ol), ("hybrid_int8", oh)]}}
    save_row("fail_high_iso", [("ground truth", worst["gt"][c, c]), (f"traditional dE={worst_de:.1f}", wo[c, c]),
                               (f"learned dE={stats['high_iso_dark']['learned']['delta_e']:.1f}", ol[c, c]),
                               (f"hybrid INT8 dE={stats['high_iso_dark']['hybrid_int8']['delta_e']:.1f}", oh[c, c])])

    # 2) learned ISP colour failure: crop where learned has the largest dE vs hybrid
    best_gap, sel = -1, None
    for s in crops[::2]:
        ol = learned.run(s["raw"])[0]
        gap = delta_e2000(ol, s["gt"])
        if gap > best_gap:
            best_gap, sel, sel_o = gap, s, ol
    oh = hyb.run(sel["raw"])[0]
    stats["learned_colour"] = {"scene": sel["rec"]["name"], "camera": sel["rec"]["camera"],
                               "learned_dE": best_gap, "hybrid_dE": delta_e2000(oh, sel["gt"])}
    save_row("fail_learned_colour", [("ground truth", sel["gt"][c, c]), (f"learned ISP dE={best_gap:.1f}", sel_o[c, c]),
                                     (f"hybrid dE={stats['learned_colour']['hybrid_dE']:.1f}", oh[c, c])])

    # 3) unseen sensor, textured scene: image-based noise estimate is fooled by foliage
    dng = ROOT / "samples" / "dng" / "pixel4a_cc0.dng"
    if dng.exists():
        r = load_dng(dng)
        sys.path.insert(0, str(ROOT / 'app'))
        from core import bayer_bin
        r = bayer_bin(r, 2)
        rg, _ = C.to_rggb(r.bayer, r.cfa)
        est = estimate_nlf(C.pack_rggb(rg[: rg.shape[0] // 2 * 2, : rg.shape[1] // 2 * 2]))
        om, im_ = hyb.run(r)
        oe, ie = hyb.run(r, nlf="estimated")
        stats["nlf_texture"] = {"metadata_nlf": list(r.noise_profile), "estimated_nlf": list(est)}
        y0, x0 = 700, 600
        save_row("fail_nlf_texture", [("hybrid, DNG NoiseProfile", om[y0:y0 + 400, x0:x0 + 500]),
                                      ("hybrid, NLF estimated (over-smoothed)", oe[y0:y0 + 400, x0:x0 + 500])])
        # 4) highlight clipping: WB applied without clipping saturated photosites -> magenta sky
        x = C.apply_wb_bayer(r.bayer, r.cfa, r.wb_gains)
        rgb = np.clip(C.apply_ccm(C.demosaic(x, r.cfa, "malvar"), r.cam2srgb), 0, 1)
        bad = C.srgb_oetf(rgb)
        good = cl.run(r)[0]
        save_row("fail_highlights", [("no clip after WB (magenta)", bad[0:350, 300:900]),
                                     ("clip to white after WB (fixed)", good[0:350, 300:900])])

    # 5) naive INT8 vs sensitivity-guided mixed precision
    naive = ROOT / "runs" / "q" / "hybrid_student_int8_naive.onnx"
    fp = M / "hybrid_student_fp32.onnx"
    if naive.exists() and fp.exists():
        s = worst
        o_fp, o_q, o_mp = HybridISP(fp).run(s["raw"])[0], HybridISP(naive).run(s["raw"])[0], hyb.run(s["raw"])[0]
        stats["int8"] = {k: psnr(o, s["gt"]) for k, o in [("fp32", o_fp), ("int8_naive", o_q), ("int8_mixed", o_mp)]}
        z = slice(300, 556)
        up = lambda im: cv2.resize(im[z, z], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        save_row("fail_int8", [("FP32", up(o_fp)), (f"naive INT8 ({stats['int8']['int8_naive']:.1f} dB)", up(o_q)),
                               (f"mixed INT8 ({stats['int8']['int8_mixed']:.1f} dB)", up(o_mp))])
    json.dump(stats, open(ROOT / "eval" / "results" / "failure_cases.json", "w"), indent=1, default=float)
    print(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()
