# Results

Turing RK1 (RK3588, 32 GB), Ubuntu 22.04, RKNPU driver 0.9.2, librknnrt 2.5.0,
processes pinned to the four Cortex-A76 cores (4-7). Same model everywhere:
Moonshine medium streaming, English.

- **stock**: moonshine-voice 0.1.5's streaming transcriber, `update_interval=0.25`,
  `MOONSHINE_ORT_SINGLE_THREAD=0` (with its default single thread, medium
  streaming falls behind real time on an RK1).
- **ours, NPU**: this repository, encoder on the NPU.
- **ours, CPU**: this repository with `MOONSHINE_ENCODER=cpu`, the streaming
  encoder run by onnxruntime, int8 (dynamic quantization of the same export).

## Accuracy

300 utterances of LibriSpeech test-clean (41 min, all 40 speakers, 8 or fewer
each, seed 0: `make_subset.py`), each with 0.5 s of silence appended. Text
normalized with Whisper's English normalizer for every system (`score.py`).

| | WER | substitutions | deletions | insertions |
|---|---|---|---|---|
| stock | 2.70% | 114 | 21 | 40 |
| ours, NPU | 2.66% | 126 | 22 | 24 |
| ours, CPU (int8) | 2.78% | 132 | 21 | 27 |

The same accuracy, within what 6,400 words can tell apart. Ours inserts fewer
words (the decoder is not shown the trailing silence) and substitutes a few
more; the extra substitutions we looked at are rare names split in two
("Margot Lot" for Margolotte), which come and go between runs.

## Streaming, in real time

12 of those utterances (seed 1), fed at the pace of speech in 100 ms chunks,
as a microphone would (`run_realtime.py`).

| | final text after the audio ends: median / p90 / max | CPU while listening |
|---|---|---|
| stock | 30 / 951 / 2,694 ms | 6.30 cores |
| ours, NPU | 0 / 0 / 0 ms | 1.34 cores |
| ours, CPU (int8) | 0 / 0 / 0 ms | 1.65 cores |

"CPU" is process CPU time over audio time (all threads), so 6.3 means the
stock pipeline kept more than the four A76 cores busy: onnxruntime's threads
spin between calls, and the streaming encoder re-encodes ~4.5 s of audio four
times a second. Ours runs with spinning off and decodes partial transcripts at
most every 0.3 s.

With 0.5 s of silence after the last word, ours has the final transcript
before the audio has finished (the early final). With 0.45 s, the shortest
pause our voice assistant ends a turn on, it is ready 90-200 ms after the end
of the audio in our tests; the stock pipeline took 0.8-1.6 s there.

## Where the time goes (ours, NPU)

- Encoder: ~68 ms of one NPU core per 16 frames (0.32 s of audio), the same
  for 8, 16 or 32 frames and for 1, 2 or 3 cores: it is bound by reading
  183 MB of fp16 weights.
- Decoder (CPU, int8): ~35 ms per call for 1 token, ~70 ms for 20.
- Ending a line takes 1-2 encoder calls (the last frames' lookahead) and a
  decode of the words those frames hold; the early final does both during the
  pause.

## What this does not show

- Accuracy on noisy or far-field audio, accents, or anything but read English.
- The real-time numbers are 12 utterances; the latency tail of the stock
  pipeline in particular would need more.
- How much of the stock pipeline's CPU is spinning rather than work. Turning
  spinning off in moonshine-voice would bring its number down too; the 13x
  smaller encoder update and the cheaper decoding would remain.
