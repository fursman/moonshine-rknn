# moonshine-rknn

Streaming speech-to-text on the Rockchip RK3588 NPU: [Moonshine](https://github.com/moonshine-ai/moonshine)
medium streaming, with its encoder running on the NPU and the work the stock
streaming pipeline repeats taken out.

On a Turing RK1 (RK3588, 32 GB), streaming in real time:

| | moonshine-voice 0.1.5 (CPU) | moonshine-rknn (NPU) | moonshine-rknn (CPU only) |
|---|---|---|---|
| final text after the audio ends (median / p90 / max) | 30 / 951 / 2,694 ms | 0 / 0 / 0 ms | 0 / 0 / 0 ms |
| CPU while listening | 6.3 cores | 1.3 cores | 1.7 cores |
| word error rate, LibriSpeech test-clean, 300 utterances | 2.70% | 2.66% | 2.78% |

Same model and weights. Utterances end in 0.5 s of silence, as a voice
assistant's turns do; most of the gain is in how the work is done (1, 3-5
below) and holds without the NPU. See [bench/RESULTS.md](bench/RESULTS.md) for
how these were measured, and what they leave out.

## What is different

**1. The encoder streams.** Moonshine's streaming encoder is a stack of 14
sliding-window attention layers: each frame sees 15 frames back and, in four
layers, 3 ahead. moonshine-voice runs it on a window of the last ~4.5 s of
audio on every update (four times a second), re-encoding 224 frames of context
it has already encoded to get ~12 new ones. Here each layer keeps its last 18
input frames and processes only the new ones: about 13x less work per update,
with the same output (differences under 1e-4 in fp32, float rounding, checked against the
transformers model over whole utterances; `export/export_stream_encoder.py check`).

**2. The encoder is on the NPU.** That streaming encoder, exported with fixed
shapes (16 new frames per call, the cache, a validity mask for the start and
end of an utterance) and compiled for the RK3588 in fp16. It matches the fp32
encoder at correlation 0.99998, closer than the int8 CPU encoder that ships
with moonshine-voice (0.989). The CPU is left to the decoder.

**3. Cross-attention keys and values are appended, not recomputed.** They are
a per-frame linear map of the encoder output; moonshine-voice recomputes them
for the whole utterance after every update.

**4. A revised transcript costs one decoder call.** On every update the
decoder re-checks the previous transcript in one parallel pass, as in
moonshine-voice. When the decoder changes its mind about a token (a comma,
"3" for "3:00"), moonshine-voice drops the rest of the old transcript and goes
on one token per call; here the old transcript is realigned past the revision
and checked again in one call. The result is exactly greedy decoding either
way (`tests/test_greedy.py`); on an RK1 a decoder call costs about the same
for 1 token as for 20, so calls are what is worth saving.

**5. The final transcript is ready before it is asked for.** Once the audio
has run 160 ms past the last voiced frame, the final transcript depends on
nothing still to come; it is computed then, on a copy of the encoder state,
while the speaker pauses. Speech that resumes discards it.

Lines are cut at pauses with Silero VAD, as moonshine-voice does, and the
decoder never sees more than 160 ms of trailing silence: given more,
Moonshine sometimes invents words ("... 14 degrees, or 11").

## Install (on the board)

Needs an RK3588 with the RKNPU driver (0.9.2 or later) and Rockchip's runtime:

```sh
# Rockchip's runtime library, from github.com/airockchip/rknn-toolkit2
sudo curl -L -o /usr/lib/librknnrt.so \
  https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so

git clone https://github.com/fursman/moonshine-rknn && cd moonshine-rknn
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/fetch_models.py models      # ~460 MB, checksummed
```

Then

```sh
.venv/bin/python -m moonshine_rknn.server --models models     # port 8772
python examples/stream_file.py speech.wav --server board:8772  # anywhere
```

or from Python:

```python
from moonshine_rknn.engine import MoonshineNpu
m = MoonshineNpu("models")
s = m.stream()
for chunk in microphone():          # float32, 16 kHz, any chunk size
    s.feed(chunk)
    print(s.partial())
print(s.finish())
```

`systemd/moonshine-rknn.service` runs the server at boot. The server's wire
protocol is described at the top of `moonshine_rknn/server.py`.

Without an NPU, fetch with `--cpu` and the same streaming encoder runs on the
CPU (int8, `stream_encoder_c16_int8.onnx`): changes 1 and 3-5 alone, at the
same accuracy and 1.7 cores while listening on an RK1. `MOONSHINE_ENCODER=cpu`
forces it where there is an NPU.

## Building the NPU encoder yourself

```sh
pip install -r requirements-export.txt
cd export
python export_encoder.py check                 # fixed-shape encoder == transformers
python export_stream_encoder.py check          # streaming == whole-utterance encoder
python export_stream_encoder.py onnx --c 16    # stream_encoder_c16.onnx
python compile_rknn.py stream_encoder_c16.onnx # stream_encoder_c16.rknn
python quantize_cpu.py stream_encoder_c16.onnx # stream_encoder_c16_int8.onnx, for CPUs
```

Use rknn-toolkit2 2.5.0: 2.3.2 compiles this graph to NaNs. 16 frames per call
is the default because an NPU call costs about the same for 8, 16 or 32 frames
(~70 ms: it is bound by reading 183 MB of fp16 weights), so 16 keeps partial
transcripts within ~0.3 s while using ~21% of the NPU while someone talks.

## Tests

`pytest tests` needs neither an NPU nor the models: the streaming bookkeeping
against a whole-sequence reference with the same windows, and the decoder's
draft handling against plain greedy decoding.

## Credits

Moonshine and its weights: [Moonshine AI](https://github.com/moonshine-ai/moonshine), MIT.
Silero VAD: [snakers4/silero-vad](https://github.com/snakers4/silero-vad), MIT.
RKNN runtime and toolkit: [Rockchip](https://github.com/airockchip/rknn-toolkit2).
This repository: MIT.
