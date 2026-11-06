"""Efficiency benchmark: parameters, model size, FLOPs (MACs), inference latency and peak memory.

  python eval/benchmark.py --models models --out eval/results/efficiency.json [--sizes 512,1024,2048]

Latency is measured on the network alone (packed input -> output) and end-to-end through the full
ISP (incl. classical stages) for a W x H full-resolution RAW, on CPU with a fixed thread count
(proxy for a mobile big-core cluster). Peak memory = peak RSS increase of a fresh subprocess.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def count_macs(ckpt):
    import torch
    from isp.models import load_checkpoint
    m = load_checkpoint(ckpt)
    macs = 0

    def hook(mod, inp, out):
        nonlocal macs
        if isinstance(mod, torch.nn.Conv2d):
            macs += out.numel() * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]
        elif isinstance(mod, torch.nn.ConvTranspose2d):
            macs += inp[0].numel() * mod.out_channels * mod.kernel_size[0] * mod.kernel_size[1]
    for mod in m.modules():
        mod.register_forward_hook(hook)
    with torch.no_grad():
        m(torch.zeros(1, m.in_ch, 256, 256))  # packed 256x256 == 512x512 full-res RAW = 0.262 MP
    params = sum(p.numel() for p in m.parameters())
    return params, macs / (512 * 512) # MACs per output pixel


def _worker(path, size, threads, reps, mode):
    """Run in a subprocess: measure latency + peak RSS."""
    import psutil
    proc = psutil.Process()
    base = proc.memory_info().rss
    from isp.classical import ClassicalConfig, ClassicalISP
    from isp.hybrid import HybridISP
    from isp.rawio import RawImage
    rng = np.random.default_rng(0)
    bayer = np.clip(rng.random((size, size), dtype=np.float32) * 0.3, 0, 1)
    raw = RawImage(bayer=bayer, cfa="RGGB", wb_gains=np.array([2, 1, 1.6], np.float32),
                   cam2srgb=np.eye(3, dtype=np.float32), noise_profile=(1e-3, 1e-6))
    if path == "classical":
        isp = ClassicalISP(ClassicalConfig())
        fn = lambda: isp.run(raw)
    else:
        isp = HybridISP(path, device="cpu", tile=512, threads=threads)
        fn = lambda: isp.run(raw)
    peak = 0
    import threading
    stop = False

    def poll():
        nonlocal peak
        while not stop:
            peak = max(peak, proc.memory_info().rss)
            time.sleep(0.005)
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    stop = True
    th.join()
    print(json.dumps({"latency_ms_median": float(np.median(ts)), "latency_ms_min": float(np.min(ts)),
                      "peak_rss_mb": (peak - base) / 1e6}))


def measure(path, size, threads, reps):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import torch; torch.set_num_threads({threads}); "
            f"from eval.benchmark import _worker; _worker({str(path)!r}, {size}, {threads}, {reps}, 'e2e')")
    if not str(path).endswith(".pt"):
        code = code.replace("import torch; torch.set_num_threads(%d); " % threads, "")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not line:
        print(r.stderr[-2000:])
        return None
    return json.loads(line[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="eval/results/efficiency.json")
    ap.add_argument("--sizes", default="1024,2048")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    md = Path(a.models)
    res = {"threads": a.threads, "cpu": os.uname().machine, "models": {}}
    entries = [("classical", "classical")]
    for f in sorted(md.glob("*.pt")) + sorted(md.glob("*.onnx")):
        entries.append((f.name, f))
    for name, path in entries:
        r = {"file_size_mb": None if path == "classical" else Path(path).stat().st_size / 1e6}
        if str(path).endswith(".pt"):
            p, macs = count_macs(path)
            r.update(params=p, kmacs_per_pixel=macs / 1e3, gmacs_12mp=macs * 12e6 / 1e9)
        for s in map(int, a.sizes.split(",")):
            m = measure(path, s, a.threads, a.reps)
            if m:
                r[f"{s}x{s}"] = m
        res["models"][name] = r
        print(name, json.dumps(r), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
