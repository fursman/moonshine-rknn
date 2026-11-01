#!/usr/bin/env python3
"""Moonshine streaming medium with its encoder on the RK3588 NPU.

The pipeline is the moonshine-voice library's (core/moonshine-streaming-model.cpp)
with the work it repeats taken out:

    frontend   CPU  official frontend.onnx, stateful, fed as audio arrives
    encoder    NPU  stream_encoder_c16.rknn: 16 new frames per call against an
                    18-frame cache per layer, exactly the full-utterance encoder
                    (the library re-encodes a 256-frame window every update)
    adapter    CPU  official adapter.onnx, on the new frames only
    cross K/V  CPU  official cross_kv.onnx, on the new frames only, appended
                    (it is per-frame linear; the library recomputes all of it)
    decoder    CPU  official decoder_kv.onnx: each update re-checks the last
                    transcript in one parallel pass, keeps the agreed prefix's
                    cache, and continues greedily, as the library does

Off the board (no rknnlite), the encoder runs stream_encoder_c16.onnx on the
CPU, which is how this file is tested on a desktop.

    s = MoonshineNpu(models_dir).stream()   # python -m moonshine_rknn.engine clip.wav --models models/
    s.feed(samples_16k)            # any number of times; partial() as it goes
    text = s.finish()
"""
from __future__ import annotations

import collections
import copy
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .stream import CACHE, Stream, lookaheads

SAMPLE_RATE = 16000
FRAME_S = 0.020          # one encoder frame
TOKENS_PER_S = 6.5       # the library's cap on transcript length per second
FRONTEND_CHUNK = 1280    # samples per frontend call
# Benchmarks: decode on the feeding thread after every chunk, as a real-time
# stream gets to, instead of skipping states the decoder thread is too slow
# for (which a faster-than-real-time feed would make look cheap).
SYNC = os.getenv("MOONSHINE_SYNC") == "1"
# Seconds between background decodes for partial(): a pass costs 30-200 ms of
# the cores the final transcript needs, and partials are read a few times a
# second at most. The early final is not held back by this.
PARTIAL_INTERVAL = float(os.getenv("MOONSHINE_PARTIAL_INTERVAL", "0.3"))


class _Encoder:
    """The streaming encoder graph, on the NPU when there is one."""

    def __init__(self, models: Path, chunk: int, threads: int, core: str):
        self.chunk = chunk
        rknn = models / f"stream_encoder_c{chunk}.rknn"
        self.npu = None
        try:
            if os.getenv("MOONSHINE_ENCODER", "npu") == "cpu":
                raise ImportError("MOONSHINE_ENCODER=cpu")
            from rknnlite.api import RKNNLite
            r = RKNNLite(verbose=False)
            if r.load_rknn(str(rknn)) != 0:
                raise RuntimeError(f"load_rknn {rknn}")
            mask = {"0": RKNNLite.NPU_CORE_0, "1": RKNNLite.NPU_CORE_1,
                    "2": RKNNLite.NPU_CORE_2}.get(core, RKNNLite.NPU_CORE_AUTO)
            if r.init_runtime(core_mask=mask) != 0:
                raise RuntimeError("init_runtime")
            self.npu = r
            self.where = "NPU"
        except (ImportError, RuntimeError, OSError) as e:
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.add_session_config_entry("session.intra_op.allow_spinning",
                                        os.getenv("MOONSHINE_SPIN", "0"))
            # The int8 export when there is one (as fast, half the size),
            # else fp32; MOONSHINE_ENCODER_ONNX picks a file explicitly.
            name = os.getenv("MOONSHINE_ENCODER_ONNX") or next(
                (n for n in (f"stream_encoder_c{chunk}_int8.onnx", f"stream_encoder_c{chunk}.onnx")
                 if (models / n).exists()), f"stream_encoder_c{chunk}.onnx")
            self.cpu = ort.InferenceSession(str(models / name), so,
                                            providers=["CPUExecutionProvider"])
            self.where = f"CPU ({e.__class__.__name__}: {e})" if not isinstance(e, ImportError) else "CPU"
        self.lock = threading.Lock()     # one NPU context, one call at a time

    def __call__(self, x, cache, kvalid):
        with self.lock:
            if self.npu is not None:
                y, c = self.npu.inference(inputs=[x, cache, kvalid], data_format=["nchw"] * 3)
                return np.asarray(y, np.float32), np.asarray(c, np.float32).reshape(cache.shape)
            y, c = self.cpu.run(None, {"x_new": x, "cache": cache, "kvalid": kvalid})
            return y, c


class MoonshineNpu:
    def __init__(self, models: str | Path, chunk: int = 16, threads: int = 4, core: str = "auto"):
        models = Path(models)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        # onnxruntime's threads otherwise spin between calls, and this
        # pipeline makes many small ones: spinning was half the CPU it used.
        so.add_session_config_entry("session.intra_op.allow_spinning",
                                    os.getenv("MOONSHINE_SPIN", "0"))
        cpu = ["CPUExecutionProvider"]
        def load(name):
            # .ort (the library's own builds) or .onnx (the published ones).
            path = next((p for p in (models / f"{name}.ort", models / f"{name}.onnx") if p.exists()),
                        models / f"{name}.onnx")
            return ort.InferenceSession(str(path), so, providers=cpu)
        self.frontend = load("frontend")
        self.adapter = load("adapter")
        self.cross_kv = load("cross_kv")
        self.decoder = load("decoder_kv")
        self.encoder = _Encoder(models, chunk, threads, core)
        self.config = json.loads((models / "streaming_config.json").read_text())
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(models / "tokenizer.json"))
        self.vad = load("silero_vad")
        self._int_inputs = {i.name for i in self.frontend.get_inputs() if "int" in i.type}
        self._frontend_outputs = [o.name for o in self.frontend.get_outputs()]

    def stream(self):
        return _Utterance(self)

    def transcribe(self, audio: np.ndarray) -> str:
        s = self.stream()
        s.feed(audio)
        return s.finish()

    def text(self, tokens) -> str:
        return self.tokenizer.decode(list(tokens), skip_special_tokens=True).strip()


class _Segment:
    """One VAD segment (a line). feed() runs the frontend and, for every 16 frames that
    are complete, the encoder, adapter and cross K/V; a decoding thread keeps
    partial() up to date from the newest memory, skipping states it is too
    slow to see (a pass grows with the utterance, and an RK1 core is not fast
    enough to decode after every chunk of a long one). finish() flushes the
    encoder and decodes once more, starting from the last partial, which it
    mostly only has to confirm.

    The decoder never attends to trailing silence: the recording always ends
    with the pause that ended the turn, and Moonshine asked to transcribe
    silence invents words ("... if anything changes. 3."). The library avoids
    this by cutting lines with its VAD; here the memory the decoder sees ends
    TRAIL_KEEP (160 ms) after the last frame within SILENCE_DB of the loudest:
    300 ms still let it add "or 11" to a sentence; 100-200 ms kept every
    ending in the test clips.
    """

    SILENCE_DB = 35.0
    TRAIL_KEEP = int(os.getenv("MOONSHINE_TRAIL_KEEP", "8"))  # frames (160 ms) kept after the last voiced one; 15 let it invent "or 11"

    def __init__(self, m: MoonshineNpu):
        self.m = m
        cfg = m.config
        st = cfg["frontend_state_shapes"]
        self.fstate = {k: np.zeros(v, np.int64 if k in m._int_inputs else np.float32)
                       for k, v in st.items()}
        self.pending = np.zeros((0, 768), np.float32)   # frontend frames not yet encoded
        self.enc = Stream(lookaheads(), m.encoder.chunk, m.encoder)
        self.encoded_seen = 0
        self.mem_len = 0
        self.k_cross = None
        self.v_cross = None
        self.tokens = []
        self.levels = []                                  # dB per 20 ms frame of audio
        self._spare = np.zeros(0, np.float32)
        self._unaligned = np.zeros(0, np.float32)
        self._cond = threading.Condition()
        self._dirty = False
        self._closing = False
        self._decoder = None
        self.timings = {"frontend": 0.0, "encoder": 0.0, "adapter+cross": 0.0, "decoder": 0.0}
        self.passes = 0
        self.calls = 0
        self.tokens_n = None       # memory frames the tokens were decoded from
        self._spec_for = None      # `last` the early final was taken for
        self._spec_job = None
        self._spec = None          # (last, tokens): the final transcript, early
        self._last_pass = 0.0      # when the last background decode began

    # -- audio in --
    def feed(self, samples: np.ndarray):
        # 1280-sample chunks, as the library feeds it: the published frontend
        # graph drops a chunk's remainder past a multiple of 80 samples (traced
        # away at export; see transcribe_segment_with_streaming_model), and its
        # conv buffers come back the wrong length for chunks under ~4 frames.
        samples = np.concatenate([self._unaligned, np.asarray(samples, np.float32).reshape(-1)])
        cut = samples.size - samples.size % FRONTEND_CHUNK
        samples, self._unaligned = samples[:cut], samples[cut:]
        if samples.size == 0:
            return
        self._measure(samples)
        t = time.perf_counter()
        feeds = {"audio_chunk": samples[None], **self.fstate}
        outs = self.m.frontend.run(None, feeds)
        res = dict(zip(self.m._frontend_outputs, outs))
        for k in self.fstate:
            self.fstate[k] = res[k + "_out"].astype(self.fstate[k].dtype).reshape(self.fstate[k].shape)
        self.timings["frontend"] += time.perf_counter() - t
        if res["features"].shape[1]:
            self.pending = np.concatenate([self.pending, res["features"][0]], 0)
        C = self.m.encoder.chunk
        grew = False
        while self.pending.shape[0] >= C:
            t = time.perf_counter()
            self.enc.push(self.pending[:C])
            self.timings["encoder"] += time.perf_counter() - t
            self.pending = self.pending[C:]
            grew = True
        if grew:
            self._extend_memory()
            self._kick()
        self._maybe_speculate()

    def _measure(self, samples):
        hop = int(SAMPLE_RATE * FRAME_S)
        buf = np.concatenate([self._spare, samples])
        n = buf.size // hop
        if n:
            frames = buf[:n * hop].reshape(n, hop)
            rms = np.sqrt((frames * frames).mean(1) + 1e-12)
            self.levels.extend((20 * np.log10(rms)).tolist())
        self._spare = buf[n * hop:]

    def _last_needed(self):
        """End (exclusive) of the frames the transcript needs: TRAIL_KEEP past
        the last frame within SILENCE_DB of the loudest. None before speech."""
        if not self.levels:
            return None
        lv = np.asarray(self.levels)
        loud = np.nonzero(lv > lv.max() - self.SILENCE_DB)[0]
        return int(loud[-1]) + 1 + self.TRAIL_KEEP if loud.size else None

    def _voiced_frames(self) -> int:
        """Memory frames worth decoding now."""
        last = self._last_needed()
        return 0 if last is None else min(self.mem_len, last)

    def partial(self) -> str:
        return self.m.text(self.tokens)

    # -- the final transcript, early --
    def _maybe_speculate(self):
        """Once the audio runs TRAIL_KEEP past the last voiced frame, the final
        transcript depends on nothing still to come: it is the decode of
        memory frames [0, last), which end the stream at `last` with the
        lookahead past it masked, exactly as finish() would. So do it now, on
        a copy of the encoder, while the speaker is still pausing; finish()
        then hands it over instead of flushing and decoding after the turn has
        ended. New speech moves `last` and makes the guess stale; it is
        dropped then.

        When the main stream has already gone past `last` (fed frames beyond
        it), the frames before `last` come out of it final within one more
        chunk, and the background decode at n = last is the final one."""
        last = self._last_needed()
        if last is None or last == self._spec_for:
            return
        if self.enc.fed + self.pending.shape[0] < last:
            return
        self._spec_for = last
        if last <= self.enc.fed:
            return
        enc = copy.copy(self.enc)
        enc.cache = self.enc.cache.copy()
        enc.base = list(self.enc.base)
        enc.out = list(self.enc.out)
        job = (last, enc, self.pending[:last - self.enc.fed].copy(), self.mem_len,
               self.k_cross, self.v_cross)
        if SYNC:
            self._spec = None
            self._run_spec(job)
            return
        with self._cond:
            self._spec = None
            self._spec_job = job
            self._start_decoder()
            self._cond.notify()

    def _run_spec(self, job):
        """On the decoder thread. Running the flush on a thread of its own, to
        overlap the decoder call in flight, was tried: on four cores the two
        fought over CPU and memory bandwidth, and the early final came out
        later and far less predictably (112-513 ms vs 157-198 ms)."""
        last, enc, tail, mem_len, k_cross, v_cross = job
        t = time.perf_counter()
        enc.finish(tail)
        new = enc.out[mem_len:]
        if new:
            k, v = self._memory(np.stack(new)[None].astype(np.float32), mem_len)
            k_cross = k if k_cross is None else np.concatenate([k_cross, k], 3)
            v_cross = v if v_cross is None else np.concatenate([v_cross, v], 3)
        self.timings["encoder"] += time.perf_counter() - t
        if k_cross is None:
            return
        tokens = self._greedy(k_cross, v_cross, last, list(self.tokens))
        with self._cond:
            if self._spec_for == last:
                self._spec = (last, tokens)
                self.tokens, self.tokens_n = tokens, last

    def finish(self) -> str:
        t0 = time.perf_counter()
        last = self._last_needed()
        self._stop_decoder()          # runs a queued early final to the end first
        t1 = time.perf_counter()
        how = "decoded after the end"
        if last is None:
            self.tokens = []
            how = "no speech"
        elif self._spec is not None and self._spec[0] == last:
            self.tokens = self._spec[1]
            how = "ready before the end"
        elif self.tokens_n == last and last <= self.mem_len:
            how = "background decode was final"
        else:
            self.enc.finish(self.pending)
            self.pending = self.pending[:0]
            self._extend_memory()
            self.tokens = self._greedy(self.k_cross, self.v_cross, min(self.mem_len, last),
                                       list(self.tokens)) if self.k_cross is not None else []
        t2 = time.perf_counter()
        self.finish_breakdown = {"waited for background": t1 - t0, "finish work": t2 - t1, "path": how}
        return self.partial()

    def close(self):
        self._stop_decoder()

    # -- encoder frames -> decoder memory --
    def _memory(self, enc, pos):
        mem = self.m.adapter.run(None, {"encoded": enc, "pos_offset": np.array([pos], np.int64)})[0]
        return self.m.cross_kv.run(None, {"memory": mem})

    def _extend_memory(self):
        new = self.enc.out[self.encoded_seen:]
        if not new:
            return
        t = time.perf_counter()
        enc = np.stack(new)[None].astype(np.float32)
        self.encoded_seen += len(new)
        k, v = self._memory(enc, self.mem_len)
        k = k if self.k_cross is None else np.concatenate([self.k_cross, k], 3)
        v = v if self.v_cross is None else np.concatenate([self.v_cross, v], 3)
        with self._cond:            # the decoder thread reads these three together
            self.k_cross, self.v_cross = k, v
            self.mem_len += enc.shape[1]
        self.timings["adapter+cross"] += time.perf_counter() - t

    # -- decoding --
    def _start_decoder(self):
        if self._decoder is None:
            self._decoder = threading.Thread(target=self._decode_loop, name="moonshine-dec",
                                             daemon=True)
            self._decoder.start()

    def _kick(self):
        if SYNC:
            self._decode()
            return
        with self._cond:
            self._dirty = True
            self._start_decoder()
            self._cond.notify()

    def _decode_loop(self):
        while True:
            with self._cond:
                while not self._dirty and not self._closing and self._spec_job is None:
                    self._cond.wait()
                job, self._spec_job = self._spec_job, None
                if job is None:
                    if self._closing:
                        return
                    self._dirty = False
            if job is not None:
                self._run_spec(job)
                continue
            wait = self._last_pass + PARTIAL_INTERVAL - time.perf_counter()
            if wait > 0:
                with self._cond:
                    self._cond.wait_for(lambda: self._spec_job is not None or self._closing, wait)
                    if self._spec_job is not None or self._closing:
                        self._dirty = True
                        continue
            self._last_pass = time.perf_counter()
            self._decode()

    def _stop_decoder(self):
        with self._cond:
            self._closing = True
            self._cond.notify()
            th = self._decoder
        if th is not None:
            th.join()
        self._decoder = None

    def _run_decoder(self, tokens, k_self, v_self, k_cross, v_cross):
        return self.m.decoder.run(None, {
            "token": np.array([tokens], np.int64), "k_self": k_self, "v_self": v_self,
            "out_k_cross": k_cross, "out_v_cross": v_cross})

    def _decode(self):
        """A background pass: the transcript of the memory so far, for
        partial(). Gives up between calls once finish() has been called or
        the early final has been queued."""
        with self._cond:
            n = self._voiced_frames()
            k_cross, v_cross = self.k_cross, self.v_cross
        if n == 0 or k_cross is None:
            return
        tokens = self._greedy(k_cross, v_cross, n, list(self.tokens), abortable=True)
        if tokens is not None:
            self.tokens, self.tokens_n = tokens, n

    def _greedy(self, k_cross, v_cross, n, draft, abortable=False):
        """Greedy decoding of memory frames [0, n), using `draft` (the last
        transcript) to skip ahead (the library's decode_full, with one change).

        Each decoder call feeds the last accepted token plus the whole draft,
        and accepts the draft's longest prefix that greedy decoding agrees
        with; the result is exactly what plain greedy decoding gives. The
        library drops the rest of the draft at the first disagreement and
        goes on one token per call. But as the memory grows the decoder
        mostly revises single tokens (a comma, "3" for "3:00"), and the old
        text after them is still right, so here the draft is realigned past
        the revision and checked again in one call: a revision costs one
        call, not one call per token after it. On an RK1 a decoder call costs
        about the same for 1 token as for 20 (it is bound by reading the
        weights), so calls are what to save."""
        t = time.perf_counter()
        k_cross, v_cross = k_cross[:, :, :, :n], v_cross[:, :, :, :n]
        cfg = self.m.config
        bos, eos = cfg["bos_id"], cfg["eos_id"]
        depth, heads, hd = cfg["depth"], cfg["nheads"], cfg["head_dim"]
        max_tokens = min(math.ceil(n * FRAME_S * TOKENS_PER_S), 448)
        k_self = v_self = np.zeros((depth, 1, heads, 0, hd), np.float32)
        out, pending = [], bos
        draft = draft[:max_tokens]
        self.calls = 0
        while True:
            # A background pass is obsolete once the early final is queued
            # (it decodes the same memory and more) or finish() has begun.
            if abortable and (self._closing or self._spec_job is not None):
                return None
            draft = draft[:max(0, max_tokens - len(out))]
            cached = k_self.shape[3]
            self.calls += 1
            logits, k_self, v_self = self._run_decoder(
                [pending] + draft, k_self, v_self, k_cross, v_cross)[:3]
            pred = logits[0].argmax(-1)
            j = 0
            while j < len(draft) and pred[j] == draft[j]:
                j += 1
            out.extend(draft[:j])
            # Causal attention: the cache up to the agreed prefix is exactly
            # what a run over just that prefix gives.
            keep = cached + 1 + j
            k_self, v_self = k_self[:, :, :, :keep], v_self[:, :, :, :keep]
            y = int(pred[j])
            if y == eos or len(out) >= max_tokens:
                break
            out.append(y)
            pending = y
            rest = draft[j:]
            # Realign: y replaced rest[0], or was inserted before a token a
            # little further on. A wrong guess costs one call that accepts
            # nothing, which is what the library pays for every token.
            near = rest[:4]
            draft = rest[near.index(y) + 1:] if y in near else rest[1:]
        self.passes += 1
        self.timings["decoder"] += time.perf_counter() - t
        return out


class _Vad:
    """Silero VAD, the way the library's VoiceActivityDetector uses it:
    512-sample hops, probabilities averaged over a 0.5 s window, threshold 0.5."""

    HOP = 512
    WINDOW = 16              # hops in 0.5 s

    def __init__(self, session):
        self.s = session
        self.state = np.zeros((2, 1, 128), np.float32)
        self.context = np.zeros(64, np.float32)   # silero v5 wants the last 64 samples prepended
        self.probs = collections.deque(maxlen=self.WINDOW)

    def __call__(self, hop: np.ndarray) -> float:
        x = np.concatenate([self.context, hop])[None].astype(np.float32)
        out, self.state = self.s.run(None, {"input": x, "state": self.state,
                                            "sr": np.array(SAMPLE_RATE, np.int64)})
        self.context = hop[-64:]
        self.probs.append(float(out[0, 0]))
        return sum(self.probs) / len(self.probs)


class _Utterance:
    """What the assistant sends as one turn, cut into lines where the speaker
    pauses, each transcribed by a fresh _Segment: the library resets its
    state per line too. Decoding a line from its own start keeps every pass
    short, and Moonshine never has to decode a long pause, which is where it
    invents words ("... if anything changes. Thank you.")."""

    THRESHOLD = 0.5
    LOOK_BEHIND = 8192       # samples of audio before speech onset kept
    MAX_SEGMENT = 15.0       # seconds; a longer line is cut

    def __init__(self, m: MoonshineNpu):
        self.m = m
        self.vad = _Vad(m.vad)
        self.hop_buf = np.zeros(0, np.float32)
        self.behind = collections.deque(maxlen=self.LOOK_BEHIND // _Vad.HOP)
        self.seg = None
        self.seg_samples = 0
        self.lines = []
        self.timings = collections.Counter()
        self.passes = 0
        self.calls = 0

    def feed(self, samples: np.ndarray):
        buf = np.concatenate([self.hop_buf, np.asarray(samples, np.float32).reshape(-1)])
        n = buf.size // _Vad.HOP
        for i in range(n):
            self._hop(buf[i * _Vad.HOP:(i + 1) * _Vad.HOP])
        self.hop_buf = buf[n * _Vad.HOP:]

    def _hop(self, hop):
        speech = self.vad(hop) >= self.THRESHOLD
        if self.seg is None:
            self.behind.append(hop)
            if speech:
                self.seg = _Segment(self.m)
                self.seg_samples = 0
                for h in self.behind:
                    self._feed_seg(h)
                self.behind.clear()
            return
        self._feed_seg(hop)
        if not speech or self.seg_samples >= self.MAX_SEGMENT * SAMPLE_RATE:
            self._close_segment()

    def _feed_seg(self, audio):
        self.seg.feed(audio)
        self.seg_samples += audio.size

    def _close_segment(self):
        seg, self.seg = self.seg, None
        text = seg.finish()
        if os.getenv("MOONSHINE_DEBUG"):
            print("segment closed: " + ", ".join(
                f"{k} {v * 1000:.0f} ms" if isinstance(v, float) else f"{k} {v}"
                for k, v in seg.finish_breakdown.items()), flush=True)
        if text:
            self.lines.append(text)
        self.timings.update(seg.timings)
        self.passes += seg.passes
        self.calls = getattr(seg, "calls", 0)

    def partial(self) -> str:
        cur = self.seg.partial() if self.seg is not None else ""
        return " ".join(t for t in self.lines + [cur] if t)

    def finish(self) -> str:
        if self.seg is not None:
            if self.hop_buf.size:
                self._feed_seg(self.hop_buf)
            self._close_segment()
        self.hop_buf = self.hop_buf[:0]
        return " ".join(self.lines)

    def close(self):
        if self.seg is not None:
            self.seg.close()
            self.seg = None


if __name__ == "__main__":
    import argparse
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--models", required=True, help="directory made by scripts/fetch_models.py")
    ap.add_argument("--realtime", action="store_true", help="feed at the pace of speech")
    args = ap.parse_args()
    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly
        audio = resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)
    m = MoonshineNpu(args.models)
    print(f"encoder on {m.encoder.where}")
    s = m.stream()
    step = 1600
    t0 = time.perf_counter()
    for i in range(0, len(audio), step):
        s.feed(audio[i:i + step])
        if args.realtime:
            time.sleep(max(0.0, t0 + (i + step) / SAMPLE_RATE - time.perf_counter()))
    t = time.perf_counter()
    text = s.finish()
    done = time.perf_counter()
    late = f", text ready {1000 * (done - (t0 + len(audio) / SAMPLE_RATE)):.0f} ms after the audio ended" \
        if args.realtime else ""
    print(f"finish {1000 * (done - t):.0f} ms{late}; {len(audio) / SAMPLE_RATE:.1f}s audio; "
          + ", ".join(f"{k} {v * 1000:.0f} ms" for k, v in s.timings.items()) + f", {s.passes} decodes, final pass {s.calls} calls")
    print(text)
