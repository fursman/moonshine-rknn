#!/usr/bin/env python3
"""Moonshine streaming medium's encoder, as a fixed-shape graph for the RK3588 NPU.

The streaming library runs the encoder on a sliding window of frontend
features: 16 frames of left context per layer (224 for 14 layers), the frames
it is about to emit, and 16 frames of lookahead. The encoder has no positional
encoding (positions are added later, in the adapter), so a window padded at
the end to a fixed length and masked gives exactly the outputs the library's
variable-length window does. That fixed length is what the NPU needs.

    features [1, T, 768] + two additive masks [1, 1, T, T] -> encoded [1, T, 768]

The masks carry both the per-layer sliding windows and the padding: one for
the layers that look 4 frames ahead, one for the layers that do not. Building
them on the CPU keeps the graph free of index arithmetic the NPU compiler
handles badly. Attention, LayerNorm and the MLP are written out in plain ops
(MatMul, Softmax, ReduceMean, Erf GELU) for the same reason.

Usage:
    export_encoder.py check            fp32 graph vs transformers on random input
    export_encoder.py onnx --t 256     write encoder_t256.onnx
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO = "moonshine-ai/moonshine-streaming-medium"
NEG = -1.0e4            # fp16-safe "minus infinity" for additive masks


def window_masks(T: int, valid: int, windows) -> np.ndarray:
    """Additive masks [n_distinct, 1, 1, T, T] and, per layer, which one it uses."""
    distinct = sorted(set(map(tuple, windows)))
    q = np.arange(T)[:, None]
    k = np.arange(T)[None, :]
    dist = q - k
    out = []
    for left, right in distinct:
        ok = ((dist >= 0) & (dist < left)) | ((dist < 0) & (-dist < right))
        ok &= (k < valid)
        # A padded query row with nothing visible would softmax to NaN; let it
        # see itself. Its output is never used.
        ok |= (q == k) & (q >= valid)
        out.append(np.where(ok, 0.0, NEG).astype(np.float32)[None, None])
    index = [distinct.index(tuple(w)) for w in windows]
    return np.stack(out), index


class LayerNorm(nn.Module):
    """LayerNorm without affine, times (gamma + 1): Moonshine's unit-offset norm."""

    def __init__(self, gamma: torch.Tensor, eps: float):
        super().__init__()
        self.register_buffer("scale", (gamma + 1.0).detach().clone())
        self.eps = eps

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(-1, keepdim=True)
        return xc * torch.rsqrt(var + self.eps) * self.scale


class Layer(nn.Module):
    def __init__(self, hf_layer, heads: int, head_dim: int, eps: float):
        super().__init__()
        a, m = hf_layer.self_attn, hf_layer.mlp
        self.heads, self.head_dim = heads, head_dim
        self.n1 = LayerNorm(hf_layer.input_layernorm.gamma, eps)
        self.n2 = LayerNorm(hf_layer.post_attention_layernorm.gamma, eps)
        # q, k and v as one projection: one big MatMul instead of three.
        w = torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight], 0)
        self.qkv = nn.Linear(w.shape[1], w.shape[0], bias=False)
        self.qkv.weight.data = w.detach().clone()
        self.o = a.o_proj
        self.fc1, self.fc2 = m.fc1, m.fc2
        self.scale = head_dim ** -0.5

    def forward(self, x, mask):
        T = x.shape[1]
        h = self.n1(x)
        qkv = self.qkv(h).reshape(1, T, 3, self.heads, self.head_dim)
        q = qkv[:, :, 0].permute(0, 2, 1, 3) * self.scale
        k = qkv[:, :, 1].permute(0, 2, 3, 1)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)
        att = torch.softmax(torch.matmul(q, k) + mask, dim=-1)
        y = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(1, T, self.heads * self.head_dim)
        x = x + self.o(y)
        # Exact (erf) GELU, as trained: the tanh approximation alone moved
        # outputs by up to 0.05 (std 0.39), where padding moved them by 1e-4.
        h = torch.nn.functional.gelu(self.fc1(self.n2(x)))
        return x + self.fc2(h)


class Encoder(nn.Module):
    def __init__(self, hf_encoder, config):
        super().__init__()
        eps = hf_encoder.final_norm.ln.eps
        self.layers = nn.ModuleList(
            Layer(l, config.num_attention_heads, config.head_dim, eps) for l in hf_encoder.layers)
        self.final = LayerNorm(hf_encoder.final_norm.gamma, eps)
        _, self.mask_index = window_masks(4, 4, config.sliding_windows)

    def forward(self, features, mask_ahead, mask_back):
        masks = (mask_back, mask_ahead)          # window_masks sorts (16, 0) before (16, 4)
        x = features
        for layer, mi in zip(self.layers, self.mask_index):
            x = layer(x, masks[mi])
        return self.final(x)


def load():
    from transformers import MoonshineStreamingForConditionalGeneration
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(REPO, torch_dtype=torch.float32)
    model.eval()
    enc = model.model.encoder
    return model, enc, enc.config


def hf_reference(hf_enc, features: torch.Tensor) -> torch.Tensor:
    """The transformers encoder layers on `features` (no padding), eager attention."""
    from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
        sliding_window_mask_function)
    T = features.shape[1]
    cfg = hf_enc.config
    x = features
    for i, layer in enumerate(hf_enc.layers):
        fn = sliding_window_mask_function(cfg.sliding_windows[i])
        m = torch.tensor([[fn(0, 0, q, k) for k in range(T)] for q in range(T)])
        add = torch.where(m, 0.0, float("-inf"))[None, None]
        x = layer(x, attention_mask=add)
    return hf_enc.final_norm(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "onnx"])
    ap.add_argument("--t", type=int, default=256)
    ap.add_argument("--valid", type=int, default=200)
    args = ap.parse_args()
    torch.manual_seed(0)
    _, hf_enc, cfg = load()
    hf_enc.config._attn_implementation = "eager"
    enc = Encoder(hf_enc, cfg).eval()
    T, L = args.t, args.valid
    masks, _ = window_masks(T, L, cfg.sliding_windows)

    if args.cmd == "check":
        feats = torch.randn(1, L, 768) * 0.5
        with torch.no_grad():
            ref = hf_reference(hf_enc, feats)
            padded = torch.cat([feats, torch.zeros(1, T - L, 768)], 1)
            out = enc(padded, torch.from_numpy(masks[1]), torch.from_numpy(masks[0]))[:, :L]
        err = (out - ref).abs()
        print(f"padded fixed-T graph vs transformers: max abs {err.max():.2e}, "
              f"mean {err.mean():.2e}, ref std {ref.std():.3f}")
        return

    path = HERE / f"encoder_t{T}.onnx"
    feats = torch.zeros(1, T, 768)
    torch.onnx.export(
        enc, (feats, torch.from_numpy(masks[1]), torch.from_numpy(masks[0])), str(path),
        input_names=["features", "mask_ahead", "mask_back"], output_names=["encoded"],
        opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
