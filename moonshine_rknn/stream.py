"""Host side of the streaming Moonshine encoder (numpy only, runs on the board).

The graph (export_stream_encoder.py) processes C new frames per call against
an 18-frame cache per layer. This tracks which global frame each window
position holds, so keys before the utterance and past its end are masked, and
collects the finished frames in order.
"""
from __future__ import annotations

import numpy as np

NEG = -1.0e4
LEFT = 15               # keys at distance 0..15 behind the query
CACHE = 18              # LEFT + the largest lookahead
DIM = 768
# Moonshine streaming medium: (16, 4) windows in layers 0, 1, 12, 13.
WINDOWS = [(16, 4)] * 2 + [(16, 0)] * 10 + [(16, 4)] * 2


def lookaheads(windows=WINDOWS):
    # (16, r): right_mask is -dist < r, so the query sees r - 1 frames ahead.
    return [max(0, r - 1) for _, r in windows]


class Stream:
    """Host-side bookkeeping: which global frame each window position holds,
    so start-of-utterance and end-of-utterance keys can be masked. `run` is
    the graph (torch module or NPU), called with numpy arrays."""

    def __init__(self, R, C, run, dim=DIM):
        self.R, self.C, self.run, self.dim = R, C, run, dim
        self.depth = len(R)
        self.cache = np.zeros((self.depth, 1, CACHE, dim), np.float32)
        # Global index of window position 0, per layer, for the next call.
        base, self.base = -CACHE, []
        for r in R:
            self.base.append(base)
            base = base - r          # this layer's outputs lag its input by r
        self.fed = 0                 # frames given to layer 0
        self.total = None            # frame count once the utterance has ended
        self.out = []                # finished encoder frames, in order
        self.emitted_upto = -sum(R)  # global index of the next output frame

    def _kvalid(self):
        W = CACHE + self.C
        kv = np.zeros((self.depth, 1, 1, 1, W), np.float32)
        end = self.total if self.total is not None else self.fed
        for i in range(self.depth):
            idx = self.base[i] + np.arange(W)
            # Layer i's input frames exist for indices [0, end): its inputs
            # are layer i-1's outputs, which are only ever real frames.
            kv[i, 0, 0, 0] = np.where((idx >= 0) & (idx < end), 0.0, NEG)
        return kv

    def push(self, frames: np.ndarray):
        """frames [C, dim], real features."""
        assert frames.shape[0] == self.C
        self.fed += self.C
        self._call(frames)

    def _call(self, frames):
        y, self.cache = self.run(frames[None].astype(np.float32), self.cache, self._kvalid())
        for i in range(self.depth):
            self.base[i] += self.C
        lo = self.emitted_upto
        self.emitted_upto += self.C
        end = self.total if self.total is not None else self.fed
        for j in range(self.C):
            if 0 <= lo + j < end:
                self.out.append(y[0, j])

    def finish(self, tail: np.ndarray):
        """The last frames (fewer than C), then zero-padded calls until every
        real frame has come out of the last layer."""
        self.total = self.fed + tail.shape[0]
        pad = np.zeros((self.C, self.dim), np.float32)
        if tail.shape[0]:
            pad[:tail.shape[0]] = tail
            self.fed = self.total
            self._call(pad)
        while self.emitted_upto < self.total:
            self._call(np.zeros((self.C, self.dim), np.float32))
