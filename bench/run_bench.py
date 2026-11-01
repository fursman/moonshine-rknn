#!/usr/bin/env python3
"""Transcribe the benchmark subset, streaming, as fast as the engine goes.

    run_bench.py stock  OUT.tsv      moonshine-voice's own streaming transcriber
    run_bench.py ours   OUT.tsv      moonshine_rknn (MODELS=dir; MOONSHINE_ENCODER=cpu for CPU only;
                                     MOONSHINE_SYNC=1 to decode as often as a real-time stream does)

Writes uid, hypothesis, seconds of audio, wall seconds, CPU seconds per line.
"""
import os, sys, time
from pathlib import Path
import numpy as np, soundfile as sf
mode, out = sys.argv[1], sys.argv[2]
sub = Path(__file__).resolve().parent / "subset"
uids = [l.split("\t")[0] for l in (sub / "refs.tsv").read_text().splitlines()]
if mode == "stock":
    os.environ.setdefault("MOONSHINE_ORT_SINGLE_THREAD", "0")
    import moonshine_voice as mv
    from moonshine_voice.transcriber import Transcriber
    p, _ = mv.get_model_for_language("en", mv.ModelArch.MEDIUM_STREAMING)
    tr = Transcriber(p, mv.ModelArch.MEDIUM_STREAMING)
    def run(a):
        st = tr.create_stream(update_interval=0.25); st.start()
        for i in range(0, len(a), 1600): st.add_audio(a[i:i + 1600].tolist(), 16000)
        r = st.stop(); st.close()
        return " ".join(l.text for l in r.lines)
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from moonshine_rknn.engine import MoonshineNpu
    m = MoonshineNpu(os.environ.get("MODELS", "models"))
    print("encoder on", m.encoder.where, flush=True)
    def run(a):
        s = m.stream()
        for i in range(0, len(a), 1600): s.feed(a[i:i + 1600])
        return s.finish()
done = set()
if Path(out).exists():
    done = {l.split("\t")[0] for l in Path(out).read_text().splitlines()}
with open(out, "a") as f:
    for k, uid in enumerate(uids):
        if uid in done: continue
        a, _ = sf.read(sub / f"{uid}.wav", dtype="float32")
        w, c = time.perf_counter(), time.process_time()
        text = run(a)
        f.write(f"{uid}\t{text}\t{len(a) / 16000:.2f}\t{time.perf_counter() - w:.3f}\t{time.process_time() - c:.3f}\n"); f.flush()
        if k % 25 == 0: print(k, uid, text[:60], flush=True)
