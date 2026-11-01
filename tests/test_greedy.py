"""Decoding with a realigned draft gives exactly plain greedy decoding, and
never takes more decoder calls than the library's way of using a draft.

The decoder here is a stand-in with Moonshine's interface (tokens in, logits
and a self-attention cache out) whose next token is a deterministic function
of the whole prefix, so "what greedy decoding says" is known exactly. The
cache carries the token ids themselves, which is enough to check that the
engine trims and reuses it correctly.
"""
import random
import threading

import numpy as np
import pytest

from moonshine_rknn.engine import _Segment

BOS, EOS, VOCAB, DEPTH, HEADS, HD = 1, 2, 64, 2, 1, 1


class FakeDecoder:
    def __init__(self, script):
        self.script = script            # what greedy decoding must produce
        self.calls = 0

    def next(self, prefix):
        """Greedy's choice after `prefix` ([BOS] + tokens): the script while
        the prefix follows it, then EOS; off-script, a fixed junk token."""
        body = prefix[1:]
        if body == self.script[:len(body)]:
            return self.script[len(body)] if len(body) < len(self.script) else EOS
        return 3 + sum(body) % (VOCAB - 3)

    def run(self, _names, feed):
        self.calls += 1
        past = [int(t) for t in feed["k_self"][0, 0, 0, :, 0]]
        new = [int(t) for t in feed["token"][0]]
        logits = np.zeros((1, len(new), VOCAB), np.float32)
        for i in range(len(new)):
            logits[0, i, self.next(past + new[:i + 1])] = 1.0
        cache = np.array(past + new, np.float32).reshape(1, 1, 1, -1, 1)
        cache = np.broadcast_to(cache, (DEPTH, 1, HEADS, len(past) + len(new), HD)).copy()
        return [logits, cache, cache.copy()]


def segment(decoder):
    s = _Segment.__new__(_Segment)
    s.m = type("M", (), {"decoder": decoder, "config": {
        "bos_id": BOS, "eos_id": EOS, "depth": DEPTH, "nheads": HEADS, "head_dim": HD}})()
    s.calls, s.passes = 0, 0
    s.timings = {"decoder": 0.0}
    s._closing, s._spec_job = False, None
    s._cond = threading.Condition()
    return s


def plain_greedy_calls(script):
    return len(script) + 1


def library_calls(script, draft):
    """The library: one call for [BOS] + draft, then one per token after the
    first disagreement."""
    agree = 0
    while agree < min(len(draft), len(script)) and draft[agree] == script[agree]:
        agree += 1
    return 1 + (len(script) - agree)


def decode(script, draft):
    dec = FakeDecoder(script)
    s = segment(dec)
    cross = np.zeros((DEPTH, 1, HEADS, 1000, HD), np.float32)
    # n frames large enough that max_tokens never binds
    return s._greedy(cross, cross, 1000, list(draft)), dec.calls


def variants(script, rng):
    yield list(script)                          # unchanged
    yield []                                    # no draft
    yield list(script[:len(script) // 2])       # stale: shorter
    yield list(script) + [7, 8, 9]              # longer than the truth
    for _ in range(20):
        d = list(script)
        for _ in range(rng.randint(1, 3)):
            op, i = rng.choice("sid"), rng.randrange(len(d) + 1)
            if op == "s" and i < len(d):
                d[i] = rng.randrange(3, VOCAB)
            elif op == "i":
                d.insert(i, rng.randrange(3, VOCAB))
            elif op == "d" and i < len(d):
                del d[i]
        yield d


@pytest.mark.parametrize("seed", range(8))
def test_realigned_draft_is_exactly_greedy(seed):
    rng = random.Random(seed)
    script = [rng.randrange(3, VOCAB) for _ in range(rng.randint(1, 40))]
    for draft in variants(script, rng):
        got, calls = decode(script, draft)
        assert got == script, (draft, got)
        assert calls <= library_calls(script, draft)


def test_one_revision_costs_one_call_not_the_rest_of_the_sentence():
    script = list(range(10, 40))
    draft = list(script)
    draft[3] = 63                                # the decoder changes its mind about one token
    _, calls = decode(script, draft)
    assert calls == 2
    assert library_calls(script, draft) == 1 + 27


def test_empty_draft_is_plain_greedy():
    script = list(range(10, 22))
    _, calls = decode(script, [])
    assert calls == plain_greedy_calls(script)
