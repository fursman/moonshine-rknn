#!/usr/bin/env python3
"""Word error rate and cost of each run, against LibriSpeech's references.

Everything goes through Whisper's English normalizer (case, punctuation,
numbers spelled out or not, British/American spellings), the same for every
system, so what is compared is recognition, not formatting.

    score.py stock.tsv ours_npu.tsv ...
"""
import sys
from pathlib import Path
import jiwer
from whisper_normalizer.english import EnglishTextNormalizer
norm = EnglishTextNormalizer()
refs = dict(l.split("\t") for l in (Path(__file__).parent / "subset/refs.tsv").read_text().splitlines())
runs = {p: [l.split("\t") for l in Path(p).read_text().splitlines()] for p in sys.argv[1:]}
common = set(refs)
for rows in runs.values():
    common &= {r[0] for r in rows}
common = sorted(common)
print(f"{len(common)} utterances scored (common to all runs)\n")
print(f"{'run':24s} {'WER':>6s} {'subs':>5s} {'del':>5s} {'ins':>5s}  {'CPU s per s of speech':>21s}  {'wall s per s':>12s}")
for p, rows in runs.items():
    by = {r[0]: r for r in rows}
    R = [norm(refs[u]) for u in common]
    H = [norm(by[u][1]) for u in common]
    o = jiwer.process_words(R, H)
    audio = sum(float(by[u][2]) - 0.5 for u in common)
    wall = sum(float(by[u][3]) for u in common)
    cpu = sum(float(by[u][4]) for u in common)
    print(f"{Path(p).stem:24s} {o.wer * 100:5.2f}% {o.substitutions:5d} {o.deletions:5d} {o.insertions:5d}  {cpu / audio:21.2f}  {wall / audio:12.2f}")
