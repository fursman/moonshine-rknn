#!/usr/bin/env python3
"""Compile an exported ONNX graph for the RK3588 NPU (fp16, no quantization).

    compile_rknn.py encoder_t256.onnx  ->  encoder_t256.rknn
"""
import sys
from pathlib import Path

from rknn.api import RKNN

src = Path(sys.argv[1])
dst = src.with_suffix(".rknn")
r = RKNN(verbose=False)
r.config(target_platform="rk3588", optimization_level=3)
if r.load_onnx(model=str(src)) != 0:
    raise SystemExit("load_onnx failed")
if r.build(do_quantization=False) != 0:
    raise SystemExit("build failed")
if r.export_rknn(str(dst)) != 0:
    raise SystemExit("export failed")
print(f"wrote {dst} ({dst.stat().st_size / 1e6:.0f} MB)")
