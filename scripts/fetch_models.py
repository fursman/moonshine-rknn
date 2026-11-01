#!/usr/bin/env python3
"""Download everything moonshine_rknn needs into one directory, checksummed.

    fetch_models.py models/            with the prebuilt NPU encoder from this repo's release
    fetch_models.py models/ --cpu      with the CPU encoder instead (int8 ONNX; no NPU needed)
    fetch_models.py models/ --no-rknn  neither (build the encoder with export/)

Sources, all MIT licensed:
    frontend.onnx, streaming_config.json, tokenizer.json
        Moonshine's published ONNX export (huggingface.co/moonshine-ai/moonshine-streaming)
    adapter.ort, cross_kv.ort, decoder_kv.ort
        the build moonshine-voice itself downloads (quantized_26_08_21). Its decoder,
        not the one published as ONNX: on our tests the published one invented
        words at the end of utterances that this one does not.
    silero_vad.onnx
        Silero VAD v5 (github.com/snakers4/silero-vad), to cut lines at pauses
    stream_encoder_c16.rknn, stream_encoder_c16_int8.onnx
        the streaming encoder, compiled for the RK3588 / for the CPU (export/ makes both)
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/moonshine-ai/moonshine-streaming/resolve/main/onnx/medium/"
LIB = "https://download.moonshine.ai/model/medium-streaming-en/quantized_26_08_21/"
RELEASE = "https://github.com/fursman/moonshine-rknn/releases/download/v0.1.0/"

FILES = {
    "frontend.onnx": (HF, "6b20310cb8d43c1baa0c46906312adc353a0d94dc43f3dad7af9ff3b5fec502f"),
    "streaming_config.json": (HF, "28e83b7a28e91472692a035e0dae3116422ae43aeb2bef5ed822c44ce89b88af"),
    "tokenizer.json": (HF, "b68b995a58b3373db6c4f46864a9db50cd1333d086150ffa2c1096581ede9c10"),
    "adapter.ort": (LIB, "3f2a287def57cc094367a0eec3c4f5fc36a32ec420e86764b696920991b20281"),
    "cross_kv.ort": (LIB, "642f6e21cd305be79342207c6f9e6b681d469d55bc48c72b27b84846fb71fd1e"),
    "decoder_kv.ort": (LIB, "193bb366492b74fc4ad338c6778e8d8eb916aaa11b5aa264f9057f4db7759486"),
    "silero_vad.onnx": ("https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/",
                        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"),
}
RKNN = {"stream_encoder_c16.rknn": (RELEASE, "a9ad4c135318b6c297a17a7a0155073c04e03157de8965f663e57859e153ddde")}
CPU = {"stream_encoder_c16_int8.onnx": (RELEASE, "bfba2e082a4d5200f86027f9f32d71f68e77595d18bc4efe94e003a4559c1e34")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--no-rknn", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    files = dict(FILES, **(CPU if a.cpu else {} if a.no_rknn else RKNN))
    bad = 0
    for name, (base, digest) in files.items():
        dst = out / name
        if dst.exists() and sha256(dst) == digest:
            print(f"have  {name}")
            continue
        print(f"fetch {name} ...", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        # download.moonshine.ai refuses Python's default user agent.
        req = urllib.request.Request(base + name, headers={"User-Agent": "moonshine-rknn/0.1"})
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
        if sha256(tmp) != digest:
            print(f"      checksum mismatch for {name}; left as {tmp.name}", file=sys.stderr)
            bad += 1
            continue
        tmp.rename(dst)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
