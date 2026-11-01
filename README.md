# moonshine-rknn

**Streaming speech-to-text on the Rockchip RK3588 NPU.** [Moonshine](https://github.com/moonshine-ai/moonshine)
medium streaming, rebuilt so its encoder streams, runs on the NPU, and the rest
of the pipeline stops redoing work it has already done. Same model, same
accuracy; the transcript is ready the moment you stop talking, for a fifth of
the CPU.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.10–3.12](https://img.shields.io/badge/python-3.10–3.12-blue.svg)
![RK3588](https://img.shields.io/badge/NPU-RK3588-orange.svg)

Measured on a Turing RK1 (RK3588, 32 GB), streaming in real time:

| | moonshine-voice 0.1.5 | **moonshine-rknn** (NPU) | moonshine-rknn (CPU only) |
|---|---|---|---|
| Final text after the audio ends — median / p90 / max | 30 / 951 / 2,694 ms | **0 / 0 / 0 ms** | 0 / 0 / 0 ms |
| CPU while listening | 6.3 cores | **1.3 cores** | 1.7 cores |
| Word error rate, LibriSpeech test-clean (300 utt.) | 2.70% | **2.66%** | 2.78% |

Utterances end in 0.5 s of silence, as a voice assistant's turns do. Most of
the gain comes from how the work is done and holds without an NPU (last
column). Method, raw outputs and caveats: [bench/RESULTS.md](bench/RESULTS.md).

---

- [Quick start](#quick-start)
- [Using it](#using-it): Python, the server and its protocol, configuration
- [How it works](#how-it-works)
- [Running it alongside other models](#running-it-alongside-other-models)
- [Building the encoder yourself](#building-the-encoder-yourself)
- [Troubleshooting](#troubleshooting) · [Limitations](#limitations) · [Credits](#credits)

## Quick start

On an RK3588 board (Ubuntu 22.04 tested) with the RKNPU driver 0.9.2 or later
(`sudo cat /sys/kernel/debug/rknpu/version`):

```sh
sudo apt install -y git python3-venv

# Rockchip's NPU runtime
sudo curl -L -o /usr/lib/librknnrt.so \
  https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

git clone https://github.com/fursman/moonshine-rknn && cd moonshine-rknn
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_models.py models    # ~460 MB, every file checksummed

.venv/bin/python -m moonshine_rknn.engine speech.wav --models models
```

```
encoder on NPU
finish 0 ms; 5.0s audio; frontend 592 ms, encoder 1000 ms, ...
Good afternoon. The weather looks clear for the rest of the day with a light breeze on
```

No NPU? `fetch_models.py models --cpu` fetches an int8 CPU encoder instead,
and everything else works the same on any aarch64 or x86-64 machine.

## Using it

### From Python

```python
from moonshine_rknn.engine import MoonshineNpu

m = MoonshineNpu("models")          # loads in ~1 s; share it between streams
s = m.stream()                      # one per utterance or conversation turn
for chunk in microphone():          # float32 mono at 16 kHz, any chunk size
    s.feed(chunk)
    print(s.partial(), end="\r")    # the transcript so far, updated ~3x a second
print(s.finish())                   # the final transcript
```

`m.transcribe(audio)` does a whole clip at once. A stream cuts lines at
pauses and joins them, so one stream can be a whole turn.

### As a server

```sh
.venv/bin/python -m moonshine_rknn.server --models models     # :8772
python examples/stream_file.py speech.wav --server board:8772   # from anywhere
```

```
  ... Good afternoon. The weather looks clear for the rest of the day
  ... Good afternoon. The weather looks clear for the rest of the day with a light breeze on
final (163 ms after the audio ended): Good afternoon. The weather looks clear ...
```

`systemd/moonshine-rknn.service` runs it at boot. The protocol is plain TCP,
so a client needs nothing beyond its language's standard library: every
message is a type byte, a 4-byte big-endian length and a payload.

| Client sends | Meaning | Server answers |
|---|---|---|
| `H` `{}` | health | `O` `{"ok": true, "encoder": "NPU", ...}` |
| `B` `{"rate": 16000}` | begin an utterance | |
| `A` float32 LE samples | audio, as it arrives | `P` text: the transcript so far, whenever it changes |
| `E` (empty) | end of utterance | `F` text: the final transcript |
| `T` float32 LE, 16 kHz | a whole clip | `F` text |
| | | `X` text: an error; the utterance is over |

One connection carries any number of utterances, one at a time. Audio at
other sample rates is resampled.

### Configuration

| Server flag | Default | |
|---|---|---|
| `--models DIR` | (required) | made by `scripts/fetch_models.py` |
| `--host`, `--port` | `0.0.0.0`, `8772` | |
| `--npu-core` | `auto` | `0`, `1` or `2`: pin it when another model shares the NPU |
| `--threads` | `4` | onnxruntime threads (the decoder is the one that matters) |

| Environment | Default | |
|---|---|---|
| `MOONSHINE_ENCODER` | `npu` | `cpu` runs the encoder on the CPU even with an NPU |
| `MOONSHINE_ENCODER_ONNX` | int8, else fp32 | which ONNX encoder the CPU path loads |
| `MOONSHINE_PARTIAL_INTERVAL` | `0.3` | seconds between partial-transcript decodes; the final is never held back |
| `MOONSHINE_TRAIL_KEEP` | `8` | 20 ms frames of silence after the last word that the decoder sees |
| `MOONSHINE_SPIN` | `0` | `1` lets onnxruntime threads spin between calls (costs CPU, gains nothing here) |
| `MOONSHINE_DEBUG` | | `1` prints how each line's final transcript was reached |

Pin the process to the four Cortex-A76 cores (`taskset -c 4-7`, or
`CPUAffinity=4-7` in the unit): on the A55s the decoder is several times slower.

## How it works

```
 mic ─▶ Silero VAD ─▶ frontend ─▶ encoder ─▶ adapter ─▶ cross K/V ─▶ decoder ─▶ text
        (lines)       (CPU)       (NPU)      (CPU)      (CPU)        (CPU, own thread)
                                  16 frames  new        appended,    greedy; the last transcript
                                  per call,  frames     not          is the draft; the final is
                                  18-frame   only       recomputed   decoded during the pause
                                  cache/layer
```

Moonshine's streaming model is an encoder of 14 sliding-window attention layers
and an autoregressive decoder ([Moonshine v2](https://arxiv.org/abs/2602.12241)).
moonshine-voice runs it well; here it is rearranged for a board where memory
bandwidth, not arithmetic, is the limit.

**1. The encoder streams, exactly.** Each layer sees 15 frames back and, in four
of them, 3 ahead, with no positional encoding. moonshine-voice re-encodes the
last ~250 frames on every update, four times a second, to get ~12 new ones.
Here each layer keeps its last 18 input frames and processes only the new
ones: about 13× less work per update, and the same frames (to float rounding,
checked against the transformers model over whole utterances).

**2. The encoder runs on the NPU.** It is exported with fixed shapes (16 new
frames, the cache, and a mask for the start and end of an utterance) and
compiled to fp16. It tracks the fp32 encoder at correlation 0.99998, closer
than the int8 CPU encoder moonshine-voice ships (0.989). A call takes ~68 ms
for 0.32 s of audio, so it keeps up using about a fifth of one of the three
NPU cores.

**3. Cross-attention keys and values are appended.** They are a per-frame
linear map of the encoder output; moonshine-voice recomputes them for the
whole utterance after every update.

**4. A revised transcript costs one decoder call.** Each update re-checks the
previous transcript in one parallel pass, as moonshine-voice does. When the
decoder changes its mind about one token (a comma, "3" for "3:00"), the old
text after it is usually still right, so it is realigned and checked again in
one call instead of decoded one token per call. The result is exactly greedy
decoding either way ([tests/test_greedy.py](tests/test_greedy.py)); on these
cores a call costs about the same for 1 token as for 20, so calls are what count.

**5. The final transcript is ready before it is asked for.** 160 ms after the
last voiced frame, the final transcript depends on nothing still to come. It
is decoded then, on a copy of the encoder state, while the speaker pauses;
speech that resumes discards it.

Two details matter for quality. Lines are cut at pauses with Silero VAD and
each starts fresh, as in moonshine-voice. And the decoder never sees more than
160 ms of trailing silence: given more, Moonshine sometimes invents words
("…14 degrees, or 11").

## Running it alongside other models

One board can run this and a text-to-speech model on the same NPU. We measured
it next to Kokoro-82M (decoder on the NPU) on one RK1: recognition was
unaffected even with speech being synthesised the whole time, and synthesis
started ~0.25 s later while both ran but stayed faster than real time.
Splitting the NPU cores between them changed nothing (`--npu-core` is there if
you want it): what they share is the A76 cores and memory bandwidth. Moving
either to the A55s made it much slower.

Several boards need nothing server-side: a client that sends each request to
an idle board, taking turns, spreads the load and survives a board going away.
Ours does exactly that across four RK1s for a voice assistant.

## Building the encoder yourself

The release has both encoders prebuilt. To make them:

```sh
pip install -r requirements-export.txt     # torch, transformers 5, rknn-toolkit2 2.5.0
cd export
python export_encoder.py check                 # fixed-shape encoder == transformers
python export_stream_encoder.py check          # streaming == whole-utterance encoder
python export_stream_encoder.py onnx --c 16    # stream_encoder_c16.onnx
python compile_rknn.py stream_encoder_c16.onnx # stream_encoder_c16.rknn
python quantize_cpu.py stream_encoder_c16.onnx # stream_encoder_c16_int8.onnx
```

16 frames per call is the default because a call costs about the same for 8,
16 or 32 frames (it is bound by reading 183 MB of weights): 16 keeps partial
transcripts within ~0.3 s of the audio at a fifth of an NPU core.

`pytest tests` needs neither an NPU nor the models: the streaming bookkeeping
is checked against a whole-sequence reference with the same windows, and the
decoder's draft handling against plain greedy decoding.

## Troubleshooting

| Symptom | Cause |
|---|---|
| The engine prints `encoder on CPU (...)` on an RK3588 | `librknnrt.so` is not in `/usr/lib`, or the RKNPU driver is older than 0.9.2 |
| `W The input[1] need NHWC data format...` on every call | harmless: the runtime converts the cache's layout (we measured no cost) |
| NaNs from an encoder you compiled yourself | compile with rknn-toolkit2 **2.5.0**; 2.3.2 breaks this graph |
| Transcripts arrive late, or partials stall | the process is on the A55 cores; pin it to 4-7 |
| `python3 -m venv` fails | `sudo apt install python3-venv` |

## Limitations

- English, medium streaming only. The other Moonshine models would need their
  own export; the method carries over.
- Tested on Turing RK1s. Other RK3588 boards (Rock 5, Orange Pi 5, …) use the
  same NPU and runtime and should work, but we have not run them.
- Accuracy was measured on read English (LibriSpeech), not on noisy or
  far-field audio.
- The adapter, cross-attention and decoder run moonshine-voice's own int8
  builds on the CPU; only the encoder is on the NPU. The decoder is faster on
  the CPU: one token per call leaves the NPU waiting on its weights.

## Credits

- [Moonshine](https://github.com/moonshine-ai/moonshine) and its weights: Moonshine AI, MIT.
  Changes 1, 3 and 4 are proposed upstream in
  [moonshine-ai/moonshine#236](https://github.com/moonshine-ai/moonshine/issues/236).
- [Silero VAD](https://github.com/snakers4/silero-vad): MIT.
- [RKNN toolkit and runtime](https://github.com/airockchip/rknn-toolkit2): Rockchip.

This repository is MIT licensed.
