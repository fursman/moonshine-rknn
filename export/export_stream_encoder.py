#!/usr/bin/env python3
"""Moonshine streaming medium's encoder as a true streaming graph for the NPU.

The library re-encodes a ~256-frame window on every update (224 frames of
left context it has encoded before), 199 ms of int8 CPU per update on an RK1.
But each layer only ever looks 15 frames back and, in four layers, 3 frames
ahead (sliding windows (16, 4) and (16, 0), where the right bound is
exclusive). So each layer can keep its last 18 input frames and process only
what is new:

    x_new  [1, C, 768]          C new frontend frames
    cache  [14, 1, 18, 768]     each layer's previous 18 input frames
    kvalid [14, 1, 1, 1, W]     0 / -1e4 per key position (W = 18 + C):
                                frames before the utterance or past its end
 -> y      [1, C, 768]          C finished encoder frames (lagging the input
                                by the total lookahead, 12 frames)
    cache' [14, 1, 18, 768]

Per layer the window is cache + new frames (W positions). A layer with
lookahead R emits the C queries at positions [W-C-R, W-R): each needs keys
[q-15, q+R], all inside the window. Every output is exactly what the full
utterance encoder computes for that frame; `check` proves it against the
transformers model run over the whole utterance.

Usage:
    export_stream_encoder.py check   [--c 16]
    export_stream_encoder.py onnx    [--c 16]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from export_encoder import LayerNorm, load, hf_reference
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moonshine_rknn.stream import CACHE, LEFT, NEG, Stream  # noqa: E402

HERE = Path(__file__).resolve().parent


class StreamLayer(nn.Module):
    def __init__(self, hf_layer, heads, head_dim, eps, C, R):
        super().__init__()
        a, m = hf_layer.self_attn, hf_layer.mlp
        self.heads, self.head_dim, self.C, self.R = heads, head_dim, C, R
        self.n1 = LayerNorm(hf_layer.input_layernorm.gamma, eps)
        self.n2 = LayerNorm(hf_layer.post_attention_layernorm.gamma, eps)
        self.q = a.q_proj
        w = torch.cat([a.k_proj.weight, a.v_proj.weight], 0)
        self.kv = nn.Linear(w.shape[1], w.shape[0], bias=False)
        self.kv.weight.data = w.detach().clone()
        self.o, self.fc1, self.fc2 = a.o_proj, m.fc1, m.fc2
        self.scale = head_dim ** -0.5
        W = CACHE + C
        self.q0 = W - C - R
        qpos = torch.arange(self.q0, self.q0 + C)[:, None]
        kpos = torch.arange(W)[None, :]
        d = qpos - kpos
        ok = ((d >= 0) & (d <= LEFT)) | ((d < 0) & (-d <= R))
        self.register_buffer("band", torch.where(ok, 0.0, NEG)[None, None])

    def forward(self, win, kvalid):
        """win [1, W, 768] -> C outputs for positions [q0, q0 + C)."""
        W, C = win.shape[1], self.C
        h = self.n1(win)
        kv = self.kv(h).reshape(1, W, 2, self.heads, self.head_dim)
        k = kv[:, :, 0].permute(0, 2, 3, 1)
        v = kv[:, :, 1].permute(0, 2, 1, 3)
        xq = win[:, self.q0:self.q0 + C]
        q = self.q(h[:, self.q0:self.q0 + C]).reshape(1, C, self.heads, self.head_dim)
        q = q.permute(0, 2, 1, 3) * self.scale
        att = torch.softmax(torch.matmul(q, k) + self.band + kvalid, dim=-1)
        y = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(1, C, self.heads * self.head_dim)
        x = xq + self.o(y)
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.n2(x))))


class StreamEncoder(nn.Module):
    def __init__(self, hf_encoder, config, C):
        super().__init__()
        eps = hf_encoder.final_norm.ln.eps
        self.R = [max(0, r - 1) for _, r in config.sliding_windows]
        self.layers = nn.ModuleList(
            StreamLayer(l, config.num_attention_heads, config.head_dim, eps, C, r)
            for l, r in zip(hf_encoder.layers, self.R))
        self.final = LayerNorm(hf_encoder.final_norm.gamma, eps)

    def forward(self, x_new, cache, kvalid):
        new_cache = []
        x = x_new
        for i, layer in enumerate(self.layers):
            win = torch.cat([cache[i], x], 1)
            new_cache.append(win[:, -CACHE:])
            x = layer(win, kvalid[i])
        return self.final(x), torch.stack(new_cache, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "onnx"])
    ap.add_argument("--c", type=int, default=16)
    args = ap.parse_args()
    _, hf, cfg = load()
    hf.config._attn_implementation = "eager"
    enc = StreamEncoder(hf, cfg, args.c).eval()
    C = args.c

    if args.cmd == "check":
        # 250 frames (5 s) of frontend-like features: the claim is exact
        # equality with the whole-utterance encoder, whatever the input.
        feats = (np.random.default_rng(0).standard_normal((250, 768)) * 0.45).astype(np.float32)
        with torch.no_grad():
            ref = hf_reference(hf, torch.from_numpy(feats[None]))[0].numpy()

            def run(x, cache, kv):
                y, c = enc(torch.from_numpy(x), torch.from_numpy(cache), torch.from_numpy(kv))
                return y.numpy(), c.numpy()
            s = Stream(enc.R, C, run)
            n = (feats.shape[0] // C) * C
            for i in range(0, n, C):
                s.push(feats[i:i + C])
            partial = len(s.out)
            s.finish(feats[n:])
        out = np.stack(s.out)
        err = np.abs(out - ref)
        print(f"{feats.shape[0]} frames: {partial} finished while streaming, {len(out)} after finish; "
              f"vs full-utterance encoder max abs {err.max():.2e}, mean {err.mean():.2e}")
        return

    W = CACHE + C
    x = torch.zeros(1, C, 768)
    cache = torch.zeros(len(enc.R), 1, CACHE, 768)
    kv = torch.zeros(len(enc.R), 1, 1, 1, W)
    path = HERE / f"stream_encoder_c{C}.onnx"
    torch.onnx.export(enc, (x, cache, kv), str(path),
                      input_names=["x_new", "cache", "kvalid"], output_names=["y", "cache_out"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
