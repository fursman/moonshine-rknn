#!/usr/bin/env python3
"""Stream utterances at the pace of speech, as a microphone would, and
measure what a deployment sees: CPU load while listening, and how soon after
the audio ends the final text is ready.

    run_realtime.py stock|ours N OUT.tsv     (MODELS=dir, MOONSHINE_ENCODER=cpu, ... for ours)
"""
import os, random, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
mode, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sub = Path(__file__).resolve().parent / "subset"
uids = sorted(l.split("\t")[0] for l in (sub / "refs.tsv").read_text().splitlines())
uids = random.Random(1).sample(uids, n)
if mode == "stock":
    os.environ.setdefault("MOONSHINE_ORT_SINGLE_THREAD", "0")
    import moonshine_voice as mv
    from moonshine_voice.transcriber import Transcriber
    p, _ = mv.get_model_for_language("en", mv.ModelArch.MEDIUM_STREAMING)
    tr = Transcriber(p, mv.ModelArch.MEDIUM_STREAMING)
    class S:
        def __init__(self):
            self.st = tr.create_stream(update_interval=0.25); self.st.start()
        def feed(self, a): self.st.add_audio(a.tolist(), 16000)
        def finish(self):
            r = self.st.stop(); self.st.close(); return " ".join(l.text for l in r.lines)
    make = S
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from moonshine_rknn.engine import MoonshineNpu
    m = MoonshineNpu(os.environ.get("MODELS", "models"))
    make = m.stream
with open(out, "w") as f:
    for uid in uids:
        a, _ = sf.read(sub / f"{uid}.wav", dtype="float32")
        s = make(); c0 = time.process_time(); t0 = time.perf_counter()
        for i in range(0, len(a), 1600):          # 100 ms per chunk, in real time
            s.feed(a[i:i + 1600])
            time.sleep(max(0.0, t0 + (i + 1600) / 16000 - time.perf_counter()))
        end = time.perf_counter()
        text = s.finish()
        ready = time.perf_counter() - end
        cpu = time.process_time() - c0
        f.write(f"{uid}\t{text}\t{len(a) / 16000:.2f}\t{ready:.3f}\t{cpu:.3f}\n"); f.flush()
        print(f"{uid} {len(a)/16000:5.1f}s  CPU {cpu / (len(a)/16000):4.2f} cores  ready +{ready*1000:4.0f} ms  {text[:50]}", flush=True)
