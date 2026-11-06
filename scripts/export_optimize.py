"""Export trained checkpoints to ONNX (FP32), FP16 and INT8 (static post-training quantisation, QDQ).

  python scripts/export_optimize.py --ckpt models/hybrid_student.pt --data data/prep --out models/

Calibration for INT8 uses real (training-split) SIDD patches passed through the same classical
front-end as at inference time, so activation ranges match deployment.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isp.models import load_checkpoint  # noqa: E402


def export_onnx(model, path, in_ch):
    dummy = torch.rand(1, in_ch, 128, 128)
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "n", 2: "h", 3: "w"}, "output": {0: "n", 2: "H", 3: "W"}},
                      opset_version=17, dynamo=False)
    import onnx
    m = onnx.load(str(path))
    meta = m.metadata_props.add()
    meta.key, meta.value = "mode", model.mode
    onnx.save(m, str(path))


def calib_batches(data_root, mode, n=48, patch=256):
    from isp.data import SIDDPatches
    ds = SIDDPatches(data_root, "train", patch, mode, augment=False, length=n)
    rng = np.random.default_rng(0)
    for i in range(n):
        r = ds.recs[int(rng.integers(len(ds.recs)))]
        yield ds.sample(r, rng=rng)["inp"].numpy()[None]


def quantize_int8(fp32_path, int8_path, data_root, mode, exclude_ends=True, method="percentile"):
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter({"input": b} for b in calib_batches(data_root, mode))

        def get_next(self):
            return next(self.it, None)

    pre = str(fp32_path).replace(".onnx", "_pre.onnx")
    quant_pre_process(str(fp32_path), pre, skip_symbolic_shape=True)
    import onnx
    g = onnx.load(pre).graph
    # Mixed precision: quantise only the convolution body; keep the first/last conv and the
    # full-resolution residual path (Resize + Add) in float — they carry the image signal itself.
    # Sensitivity analysis (scripts/quant_sensitivity.py) shows only the full-resolution U-Net level
    # is fragile under INT8, so it stays float; the deeper levels (~75% of MACs) run in INT8.
    lvl0 = ("/head/", "/tail/", "/enc.0/", "/down.0/", "/dec.2/")
    exclude = [n.name for n in g.node if n.op_type in ("Conv", "ConvTranspose")
               and n.name.startswith(lvl0)] if exclude_ends else []
    cm = {"percentile": CalibrationMethod.Percentile, "minmax": CalibrationMethod.MinMax,
          "entropy": CalibrationMethod.Entropy}[method]
    quantize_static(pre, str(int8_path), Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=cm, op_types_to_quantize=["Conv", "ConvTranspose"],
                    nodes_to_exclude=exclude,
                    extra_options={"CalibPercentile": 99.999, "ActivationSymmetric": False})
    Path(pre).unlink(missing_ok=True)
    import onnx
    m = onnx.load(str(int8_path))
    meta = m.metadata_props.add()
    meta.key, meta.value = "mode", mode
    onnx.save(m, str(int8_path))


def convert_fp16(fp32_path, fp16_path, mode):
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(str(fp32_path))
    m16 = float16.convert_float_to_float16(m, keep_io_types=False)
    onnx.save(m16, str(fp16_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=None, help="prep dir for INT8 calibration")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--no_exclude_ends", action="store_true")
    ap.add_argument("--calib", default="percentile", choices=["percentile", "minmax", "entropy"])
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_checkpoint(a.ckpt).eval()
    stem = Path(a.ckpt).stem
    fp32 = out / f"{stem}_fp32.onnx"
    export_onnx(model, fp32, model.in_ch)
    print("exported", fp32, fp32.stat().st_size / 1e6, "MB")
    if a.data:
        int8 = out / f"{stem}_int8{a.suffix}.onnx"
        quantize_int8(fp32, int8, a.data, model.mode, not a.no_exclude_ends, a.calib)
        print("int8", int8, int8.stat().st_size / 1e6, "MB")
    if a.fp16:
        try:
            fp16 = out / f"{stem}_fp16.onnx"
            convert_fp16(fp32, fp16, model.mode)
            print("fp16", fp16, fp16.stat().st_size / 1e6, "MB")
        except ImportError:
            print("onnxconverter_common not installed; skipping fp16")


if __name__ == "__main__":
    main()
