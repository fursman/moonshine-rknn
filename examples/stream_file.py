#!/usr/bin/env python3
"""Stream a WAV file to a moonshine_rknn server at the pace of speech, as a
microphone would, and print the transcript as it forms. Standard library
plus numpy and soundfile; the protocol is in moonshine_rknn/server.py.

    stream_file.py speech.wav [--server rk1.local:8772]
"""
import argparse
import json
import socket
import struct
import threading
import time

import numpy as np
import soundfile as sf

HDR = struct.Struct(">cI")


def send(sock, kind, payload=b""):
    sock.sendall(HDR.pack(kind, len(payload)) + payload)


def recv(sock):
    head = b""
    while len(head) < HDR.size:
        head += sock.recv(HDR.size - len(head))
    kind, size = HDR.unpack(head)
    body = b""
    while len(body) < size:
        body += sock.recv(size - len(body))
    return kind, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--server", default="localhost:8772")
    a = ap.parse_args()
    host, port = a.server.rsplit(":", 1)
    audio, rate = sf.read(a.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)
    sock = socket.create_connection((host, int(port)))
    send(sock, b"H")
    print("server:", json.loads(recv(sock)[1]))
    send(sock, b"B", json.dumps({"rate": rate}).encode())
    final = {}

    def reader():
        while True:
            kind, body = recv(sock)
            if kind == b"P":
                print("  ...", body.decode())
            else:
                final["at"], final["kind"], final["text"] = time.perf_counter(), kind, body.decode()
                return

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    step = rate // 10
    t0 = time.perf_counter()
    for i in range(0, len(audio), step):
        send(sock, b"A", audio[i:i + step].astype("<f4").tobytes())
        time.sleep(max(0.0, t0 + (i + step) / rate - time.perf_counter()))
    end = time.perf_counter()
    send(sock, b"E")
    th.join()
    print(f"final ({1000 * (final['at'] - end):.0f} ms after the audio ended):", final["text"])


if __name__ == "__main__":
    main()
