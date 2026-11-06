"""Develop every RAW file in a folder with one or more ISP pipelines and write sRGB images.

  python scripts/develop.py --input samples/dng --output results --pipelines traditional,hybrid
  python scripts/develop.py --input my_photo.dng --output results --pipelines traditional,learned,hybrid

Inputs: DNG / CR2 / NEF / ARW / ORF / RW2 (LibRaw) and the bundled SIDD .npz samples.
Pipelines: traditional, learned, hybrid (deployed INT8), hybrid_teacher. Full resolution (tiled).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))
import core  # noqa: E402
from isp import classical as C  # noqa: E402

RAW_EXT = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".raf", ".pef", ".srw"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="a RAW file or a folder of RAW files")
    ap.add_argument("--output", required=True, help="folder for the developed JPEGs")
    ap.add_argument("--pipelines", default="hybrid", help="comma-separated: traditional,learned,hybrid,hybrid_teacher")
    a = ap.parse_args()
    src = Path(a.input)
    files = [src] if src.is_file() else sorted(p for p in src.iterdir()
                                                if p.suffix.lower() in RAW_EXT or p.suffix == ".npz")
    if not files:
        sys.exit(f"No RAW files found in {src}")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    pids = [p.strip() for p in a.pipelines.split(",") if p.strip()]
    for f in files:
        try:
            if f.suffix == ".npz":
                raw, gt, _ = core.load_input(None, f.stem) if (core.SAMPLES / f.name).exists() else (None, None, None)
                if raw is None:
                    raise ValueError("only the bundled SIDD .npz samples are supported")
            else:
                raw, gt, _ = core.load_input(str(f))
        except Exception as e:  # corrupt or unsupported file: report and continue
            print(f"skip {f.name}: {e}")
            continue
        for pid in pids:
            name, kind, spec, _ = core.PIPELINES[pid]
            t = time.perf_counter()
            if kind == "classical":
                img = C.ClassicalISP(C.ClassicalConfig()).run(raw)[0]
            else:
                img = core.get_net(spec).run(raw)[0]
            img = C.apply_orientation(img, raw.orientation)
            dst = out / f"{f.stem}_{pid}.jpg"
            cv2.imwrite(str(dst), (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"{f.name} -> {dst.name}  ({name}, {img.shape[1]}x{img.shape[0]}, "
                  f"{(time.perf_counter() - t) * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
