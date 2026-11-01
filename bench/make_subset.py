"""300 LibriSpeech test-clean utterances, balanced over its 40 speakers, as
16 kHz wavs with 0.5 s of silence appended (an assistant turn ends in a pause)."""
import random, sys
from pathlib import Path
import numpy as np, soundfile as sf
root = Path("LibriSpeech/test-clean"); out = Path("subset"); out.mkdir(exist_ok=True)
by_spk = {}
for tr in sorted(root.glob("*/*/*.trans.txt")):
    for line in tr.read_text().splitlines():
        uid, text = line.split(" ", 1)
        by_spk.setdefault(uid.split("-")[0], []).append((uid, text, tr.parent / f"{uid}.flac"))
rng = random.Random(0)
pick = []
for spk in sorted(by_spk):
    pick += rng.sample(by_spk[spk], min(8, len(by_spk[spk])))
pick = sorted(rng.sample(pick, 300))
total = 0.0
with open(out / "refs.tsv", "w") as f:
    for uid, text, flac in pick:
        a, sr = sf.read(flac, dtype="float32"); assert sr == 16000
        sf.write(out / f"{uid}.wav", np.concatenate([a, np.zeros(8000, np.float32)]), 16000)
        f.write(f"{uid}\t{text}\n"); total += len(a) / sr
print(f"{len(pick)} utterances, {total / 60:.1f} min of speech, {len({p[0].split('-')[0] for p in pick})} speakers")
