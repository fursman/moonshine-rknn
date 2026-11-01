#!/usr/bin/env python3
"""A streaming speech-to-text server around moonshine_rknn.engine.

Clients stream microphone audio while the user talks and get the transcript
back as it forms. Plain TCP, so a client needs nothing beyond its language's
standard library. Every message is a frame: one type byte, a 4-byte
big-endian length, the payload.

    client -> server
      H  {}                      health; answered with O
      B  {"rate": 16000}         begin an utterance
      A  float32 LE samples      audio for the current utterance
      E  (empty)                 end of utterance; answered with F
      T  float32 LE at 16 kHz    a whole clip at once; answered with F
    server -> client
      O  {"ok": true, ...}
      P  utf-8 text              the transcript so far, whenever it changes
      F  utf-8 text              the final transcript
      X  utf-8 text              an error; the utterance is over

One connection carries any number of utterances, one at a time.

    python -m moonshine_rknn.server --models models/ [--port 8772]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import struct
import threading
import time

import numpy as np

from .engine import SAMPLE_RATE, MoonshineNpu

log = logging.getLogger("moonshine-rknn")
HDR = struct.Struct(">cI")
MAX_FRAME = 64 * 1024 * 1024


class Utterance:
    """One streamed utterance. Audio is fed from a thread of its own, so the
    socket is never held up by the frontend or the encoder; the transcript so
    far goes out whenever the engine's decoder has changed it."""

    def __init__(self, m: MoonshineNpu, on_partial):
        self.s = m.stream()
        self.on_partial = on_partial
        self.samples = 0
        self.last = ""
        self.failed = None
        self.q = queue.Queue()
        self.worker = threading.Thread(target=self._pump, name="feed", daemon=True)
        self.worker.start()

    def _pump(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if self.failed:
                continue
            chunk, rate = item
            try:
                if rate != SAMPLE_RATE:
                    n = int(round(chunk.size * SAMPLE_RATE / rate))
                    chunk = np.interp(np.linspace(0, chunk.size - 1, n), np.arange(chunk.size),
                                      chunk).astype(np.float32)
                self.s.feed(chunk)
                text = self.s.partial()
                if text != self.last:
                    self.last = text
                    self.on_partial(text)
            except Exception as e:
                self.failed = str(e)

    def feed(self, chunk: np.ndarray, rate: int):
        self.samples += chunk.size
        self.q.put((chunk, rate))

    def finish(self) -> str:
        self.q.put(None)
        self.worker.join()
        if self.failed:
            self.s.close()
            raise RuntimeError(self.failed)
        return self.s.finish()

    def close(self):
        if self.worker.is_alive():
            self.q.put(None)
            self.worker.join(timeout=10)
        self.s.close()


async def handle(m: MoonshineNpu, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    loop = asyncio.get_running_loop()
    peer = writer.get_extra_info("peername")
    utt = None
    rate = SAMPLE_RATE
    began = 0.0

    def send(kind: bytes, payload: bytes = b""):
        writer.write(HDR.pack(kind, len(payload)) + payload)

    def partial(text: str):
        loop.call_soon_threadsafe(send, b"P", text.encode())

    try:
        while True:
            try:
                kind, size = HDR.unpack(await reader.readexactly(HDR.size))
            except asyncio.IncompleteReadError:
                return
            if size > MAX_FRAME:
                send(b"X", b"frame too large")
                return
            payload = await reader.readexactly(size) if size else b""
            if kind == b"H":
                send(b"O", json.dumps({"ok": True, "engine": "moonshine-rknn",
                                       "model": "medium_streaming", "encoder": m.encoder.where,
                                       "streaming": True}).encode())
            elif kind == b"B":
                if utt is not None:
                    await loop.run_in_executor(None, utt.close)
                rate = int(json.loads(payload or b"{}").get("rate", SAMPLE_RATE))
                utt = await loop.run_in_executor(None, Utterance, m, partial)
                began = time.perf_counter()
            elif kind == b"A":
                if utt is not None:
                    utt.feed(np.frombuffer(payload, "<f4").astype(np.float32), rate)
            elif kind == b"E":
                if utt is None:
                    send(b"F", b"")
                    continue
                done, utt = utt, None
                t0 = time.perf_counter()
                try:
                    text = await loop.run_in_executor(None, done.finish)
                    send(b"F", text.encode())
                    log.info(f"{peer[0]}: {done.samples / rate:4.1f}s streamed over {t0 - began:4.1f}s, "
                             f"final {(time.perf_counter() - t0) * 1000:.0f} ms after the end: {text[:70]!r}")
                except Exception as e:
                    send(b"X", str(e).encode())
            elif kind == b"T":
                audio = np.frombuffer(payload, "<f4").astype(np.float32)
                t0 = time.perf_counter()
                try:
                    text = await loop.run_in_executor(None, m.transcribe, audio)
                    send(b"F", text.encode())
                    log.info(f"{peer[0]}: whole clip {audio.size / SAMPLE_RATE:4.1f}s in "
                             f"{time.perf_counter() - t0:4.2f}s: {text[:70]!r}")
                except Exception as e:
                    send(b"X", str(e).encode())
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        if utt is not None:
            await loop.run_in_executor(None, utt.close)
        writer.close()


async def serve(m: MoonshineNpu, host: str, port: int):
    server = await asyncio.start_server(lambda r, w: handle(m, r, w), host, port)
    log.info(f"listening on {host}:{port}")
    async with server:
        await server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", required=True, help="directory made by scripts/fetch_models.py")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8772)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    m = MoonshineNpu(args.models)
    m.transcribe(np.zeros(SAMPLE_RATE, np.float32))
    log.info(f"Moonshine medium streaming, encoder on the {m.encoder.where}, ready in {time.time() - t0:.1f}s")
    asyncio.run(serve(m, args.host, args.port))


if __name__ == "__main__":
    main()
