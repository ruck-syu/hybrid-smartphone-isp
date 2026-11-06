"""Per-layer INT8 sensitivity: quantise one Conv at a time, measure PSNR drop on VAL crops.

  python scripts/quant_sensitivity.py --ckpt models/hybrid_student.pt --data data/prep --out eval/results/quant_sensitivity.json
"""
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_optimize import calib_batches, export_onnx  # noqa
from eval.evaluate import test_crops  # noqa
from isp.hybrid import HybridISP  # noqa
from isp.metrics import psnr  # noqa
from isp.models import load_checkpoint  # noqa


def quant(pre, out, data, mode, only=None, exclude=()):
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    class R(CalibrationDataReader):
        def __init__(self): self.it = iter({"input": b} for b in calib_batches(data, mode, n=24))
        def get_next(self): return next(self.it, None)
    g = onnx.load(pre).graph
    convs = [n.name for n in g.node if n.op_type in ("Conv", "ConvTranspose")]
    excl = [c for c in convs if c != only] if only else list(exclude)
    quantize_static(pre, out, R(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True, calibrate_method=CalibrationMethod.Percentile,
                    op_types_to_quantize=["Conv", "ConvTranspose"], nodes_to_exclude=excl,
                    extra_options={"CalibPercentile": 99.999})
    m = onnx.load(out); p = m.metadata_props.add(); p.key, p.value = "mode", mode; onnx.save(m, out)
    return convs


def score(path, crops):
    h = HybridISP(path)
    return float(np.mean([psnr(h.run(s["raw"])[0], s["gt"]) for s in crops]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="eval/results/quant_sensitivity.json")
    a = ap.parse_args()
    from onnxruntime.quantization.shape_inference import quant_pre_process
    tmp = Path(tempfile.mkdtemp())
    model = load_checkpoint(a.ckpt)
    fp32 = str(tmp / "fp32.onnx"); export_onnx(model, fp32, model.in_ch)
    pre = str(tmp / "pre.onnx"); quant_pre_process(fp32, pre, skip_symbolic_shape=True)
    crops = list(test_crops(Path(a.data), 1, split="val"))
    base = score(fp32, crops)
    import onnx
    convs = [n.name for n in onnx.load(pre).graph.node if n.op_type in ("Conv", "ConvTranspose")]
    res = {}
    for c in convs:
        q = str(tmp / "q.onnx"); quant(pre, q, a.data, model.mode, only=c)
        res[c] = base - score(q, crops)
        print(f"{c:40s} drop {res[c]:.3f} dB", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"fp32_val_psnr": base, "drop_db": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
