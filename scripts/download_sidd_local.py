"""Download only the files needed (NOISY_RAW_010/011 + METADATA + GT_SRGB_010) from Kaggle mirrors of SIDD."""
import os, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
api = KaggleApi(); api.authenticate()
names = []
tok = None
while True:
    r = api.dataset_list_files("rajat95gupta/smartphone-image-denoising-dataset", page_token=tok, page_size=200)
    names += [f.name for f in r.files if f.name.endswith("GT_SRGB_010.PNG")]
    tok = r.next_page_token
    if not tok: break
scenes = sorted({n.split("/")[-2] for n in names})
print(len(scenes), "scenes", flush=True)
jobs = []
for s in scenes:
    sid = s.split("_")[0]
    jobs.append(("rajat95gupta/smartphone-image-denoising-dataset", f"SIDD_Small_sRGB_Only/Data/{s}/GT_SRGB_010.PNG", out / "srgb" / s))
    for k in ("010", "011"):
        for kind in ("NOISY_RAW", "METADATA_RAW"):
            jobs.append(("skibidiohiorizz/sidd-medium-raw", f"SIDD_Medium_Raw/Data/{s}/{sid}_{kind}_{k}.MAT", out / "raw" / s))

def get(j):
    ds, f, d = j
    d.mkdir(parents=True, exist_ok=True)
    target = d / Path(f).name
    if target.exists(): return
    for attempt in range(3):
        try:
            api.dataset_download_file(ds, f, path=str(d), quiet=True)
            z = d / (Path(f).name + ".zip")
            if z.exists():
                with zipfile.ZipFile(z) as zz: zz.extractall(d)
                z.unlink()
            return
        except Exception as e:
            print("retry", f, e, flush=True)
with ThreadPoolExecutor(6) as ex:
    for i, _ in enumerate(ex.map(get, jobs)):
        if i % 50 == 0: print(i, "/", len(jobs), flush=True)
print("DONE", flush=True)
