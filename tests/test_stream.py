"""The streaming bookkeeping (moonshine_rknn.stream.Stream) gives exactly what
the whole sequence gives, for any length and chunk size.

The real graph is a 14-layer transformer on the NPU; here each layer is a
tiny numpy stand-in with the same sliding windows (15 frames back, 0 or 3
ahead), written twice: once over the whole sequence, once the way the
exported graph works (an 18-frame cache, C new frames, queries at
[W-C-R, W-R), keys masked by `kvalid`). If Stream's frame indices, masks or
emission were off by one anywhere, the two would differ.
"""
import numpy as np
import pytest

from moonshine_rknn.stream import CACHE, LEFT, NEG, Stream, lookaheads

D = 8                     # feature width of the stand-in
R = lookaheads()          # per-layer lookahead, as in Moonshine medium


def weights(seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((D, D)) / D ** 0.5, rng.standard_normal((D, D)) / D ** 0.5)
            for _ in R]


def layer_full(x, wq, wk, r):
    """Each frame mixes itself with the mean of frames [t-15, t+r]."""
    n = x.shape[0]
    out = np.empty_like(x)
    for t in range(n):
        keys = x[max(0, t - LEFT):min(n, t + r + 1)]
        out[t] = np.tanh(x[t] @ wq + (keys @ wk).mean(0))
    return out


def full(x, W):
    for (wq, wk), r in zip(W, R):
        x = layer_full(x, wq, wk, r)
    return x


def graph(W, C):
    """The exported graph's contract, in numpy."""
    Wn = CACHE + C

    def run(x_new, cache, kvalid):
        x = x_new[0]
        new_cache = []
        for i, ((wq, wk), r) in enumerate(zip(W, R)):
            win = np.concatenate([cache[i, 0], x], 0)
            new_cache.append(win[-CACHE:])
            q0 = Wn - C - r
            valid = kvalid[i, 0, 0, 0] > NEG / 2
            out = np.empty((C, D))
            for j in range(C):
                q = q0 + j
                band = np.zeros(Wn, bool)
                band[max(0, q - LEFT):min(Wn, q + r + 1)] = True
                keys = win[band & valid]
                mix = (keys @ wk).mean(0) if len(keys) else np.zeros(D)
                out[j] = np.tanh(win[q] @ wq + mix)
            x = out
        return x[None].astype(np.float32), np.stack(new_cache)[:, None].astype(np.float32)
    return run


@pytest.mark.parametrize("C", [8, 16, 32])
@pytest.mark.parametrize("n", [1, 5, 15, 16, 17, 40, 64, 101])
def test_stream_matches_whole_sequence(C, n):
    W = weights(n + C)
    x = np.random.default_rng(n).standard_normal((n, D)).astype(np.float32)
    s = Stream(R, C, graph(W, C), dim=D)
    whole = (n // C) * C
    for i in range(0, whole, C):
        s.push(x[i:i + C])
    s.finish(x[whole:])
    got = np.stack(s.out)
    assert got.shape == (n, D)
    np.testing.assert_allclose(got, full(x.astype(np.float64), W), atol=1e-5)


def test_frames_come_out_while_streaming_with_the_lookahead_lag():
    C = 16
    W = weights()
    s = Stream(R, C, graph(W, C), dim=D)
    x = np.random.default_rng(1).standard_normal((64, D)).astype(np.float32)
    for i in range(0, 64, C):
        s.push(x[i:i + C])
    # Everything but the total lookahead (sum of R) is final already.
    assert len(s.out) == 64 - sum(R)
