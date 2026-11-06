"""Export a few held-out TEST-scene crops (RAW + metadata + GT sRGB) as .npz samples for the app."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp.classical import unpack_rggb

root, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
recs = [r for r in json.load(open(root / "index.json")) if r["split"] == "test" and r["k"] == "010"]
recs.sort(key=lambda r: r["iso"])
picked, seen = [], set()
for r in recs:  # one per (camera, ISO bucket), spread over the ISO range
    key = (r["camera"], r["iso"] >= 1600)
    if key not in seen:
        seen.add(key); picked.append(r)
for r in picked[:8]:
    n = np.load(root / f"{r['id']}_010_noisy.npy", mmap_mode="r")
    g = np.load(root / f"{r['id']}_gt.npy", mmap_mode="r")
    h, w = n.shape[:2]; c = 512  # packed -> 1024x1024 full-res crop (1 MP keeps the hosted demo fast)
    y, x = (h - c) // 2, (w - c) // 2
    bayer = unpack_rggb(n[y:y + c, x:x + c].astype(np.float32) / 65535)
    meta = {k: r[k] for k in ("cfa", "wb_gains", "cam2srgb", "iso", "noise_profile", "camera", "name")}
    meta["cfa"] = "RGGB"
    name = f"{r['camera']}_ISO{int(r['iso']) if r['iso'] else 'na'}_{r['id']}"
    np.savez_compressed(out / f"{name}.npz", bayer=bayer.astype(np.float16), gt=np.asarray(g[2*y:2*y+2*c, 2*x:2*x+2*c]), meta=json.dumps(meta))
    print(name)
