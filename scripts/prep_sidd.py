"""Prepare SIDD (Medium RAW + Small sRGB GT) into memory-mappable arrays for training.

For every scene instance we store
  <out>/<id>_<k>_noisy.npy : (H/2, W/2, 4) uint16 packed RGGB noisy raw (no WB), k in {010, 011}
  <out>/<id>_gt.npy        : (H, W, 3) uint8 ground-truth sRGB (official SIDD GT_SRGB_010)
  <out>/index.json         : metadata (wb gains, cam->sRGB CCM, noise profile, ISO, camera, split)

Bayer patterns are canonicalised to RGGB by cropping (dy, dx) rows/cols from the top/left of both
raw and sRGB, so every packed array is [R, Gr, Gb, B].

Split: by *scene content id* (2nd field of the folder name) so test scenes are never seen in
training, regardless of camera/ISO/illumination.

usage: python prep_sidd.py --raw /content/dl/raw/SIDD_Medium_Raw/Data --srgb /content/dl/srgb/SIDD_Small_sRGB_Only/Data --out /content/prep
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp.rawio import load_sidd_raw  # noqa: E402

SHIFT = {"RGGB": (0, 0), "GRBG": (0, 1), "GBRG": (1, 0), "BGGR": (1, 1)}
TEST_SCENES = {"003", "010"}   # held-out scene contents
VAL_SCENES = {"009"}


def process(args):
    scene_dir, srgb_dir, out = args
    name = scene_dir.name
    sid = name.split("_")[0]
    scene = name.split("_")[1]
    gt_path = srgb_dir / name / "GT_SRGB_010.PNG"
    if not gt_path.exists():
        return None
    recs = []
    gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)[..., ::-1]
    for k in ("010", "011"):
        rp = scene_dir / f"{sid}_NOISY_RAW_{k}.MAT"
        mp = scene_dir / f"{sid}_METADATA_RAW_{k}.MAT"
        if not rp.exists() or not mp.exists():
            continue
        img = load_sidd_raw(rp, mp)
        dy, dx = SHIFT[img.cfa]
        b = img.bayer[dy:, dx:]
        H, W = (b.shape[0] // 2) * 2, (b.shape[1] // 2) * 2
        b = b[:H, :W]
        if gt.shape[:2] != img.bayer.shape:
            print("shape mismatch", name, gt.shape, img.bayer.shape, flush=True)
            return None
        packed = np.stack([b[0::2, 0::2], b[0::2, 1::2], b[1::2, 0::2], b[1::2, 1::2]], -1)
        np.save(out / f"{sid}_{k}_noisy.npy", np.round(packed * 65535).astype(np.uint16))
        if k == "010":
            np.save(out / f"{sid}_gt.npy", np.ascontiguousarray(gt[dy:dy + H, dx:dx + W]))
        recs.append({"id": sid, "k": k, "name": name, "scene": scene, "camera": name.split("_")[2],
                     "iso": img.iso, "exposure": img.exposure, "cfa": img.cfa, "shift": [dy, dx],
                     "H": H, "W": W, "wb_gains": img.wb_gains.tolist(), "cam2srgb": img.cam2srgb.tolist(),
                     "noise_profile": img.noise_profile, "orientation": img.orientation,
                     "illum": name.split("_")[5], "brightness": name.split("_")[6],
                     "split": "test" if scene in TEST_SCENES else "val" if scene in VAL_SCENES else "train"})
    print("done", name, flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--srgb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted(p for p in Path(a.raw).iterdir() if p.is_dir())
    recs = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(process, [(d, Path(a.srgb), out) for d in dirs]):
            if r:
                recs.extend(r)
    json.dump(recs, open(out / "index.json", "w"), indent=1)
    from collections import Counter
    print(Counter((r["split"], r["k"]) for r in recs))


if __name__ == "__main__":
    main()
