"""A one-minute demo of the night's parrot (cadence#334): the room, the question asked aloud, the
parrot's answer in its own voice.

    python demo.py <exchanges.json> <voices_dir> <out.mp4>

``exchanges.json`` is a list of exchanges as the exam records them: the room's things (kind,
colour, material, direction in degrees from where the parrot faces, distance), the question's
text and its audio file (a trainer's voice), and the words the parrot said. The parrot's voice
says each word as the take of the trainer's word its memory keeps (``take_<word>.wav``, 8 kHz).
Each exchange is a still of the room seen from above with the question and the answer as
captions, its audio the question, a pause and the parrot's words.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

RATE = 22050
COLOURS = {"red": "#d62728", "yellow": "#e6c229", "green": "#2ca02c", "blue": "#1f77b4",
           "white": "#f2f2f2", "black": "#222222"}


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as f:
        rate, n = f.getframerate(), f.getnframes()
        audio = np.frombuffer(f.readframes(n), dtype=np.int16).astype(np.float64) / 32768
    return audio, rate


def to_rate(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE:
        return audio
    n = int(round(len(audio) * RATE / rate))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio)


def write_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def scene(ex: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor("#10141a")
    ax.set_facecolor("#10141a")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.0, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_position((0.05, 0.13, 0.9, 0.74))  # the captions above and below the room
    ax.add_patch(plt.Circle((0, 0), 0.16, color="#9aa3ad"))
    ax.annotate("", xy=(0, 0.45), xytext=(0, 0.18),
                arrowprops={"arrowstyle": "->", "color": "#9aa3ad", "lw": 2})
    ax.text(0, -0.32, "parrot", color="#9aa3ad", ha="center", fontsize=12)
    for t in ex["things"]:
        a = math.radians(90 + t["direction"])  # the parrot faces up the page
        x, y = t["distance"] * math.cos(a), t["distance"] * math.sin(a)
        edge = "#ffd166" if t.get("focus") else "#55606b"
        ax.add_patch(plt.Circle((x, y), 0.17, color=COLOURS.get(t["colour"], "#888"),
                                ec=edge, lw=3 if t.get("focus") else 1))
        ax.text(x, y - 0.3, f"{t['kind']} ({t['material']})", color="#d8dee5", ha="center",
                fontsize=11)
    fig.text(0.5, 0.92, f"Trainer: {ex['question_text']}", color="#ffffff", ha="center",
             fontsize=20)
    said = ", ".join(ex["said"]) if ex["said"] else "(silence)"
    mark = "" if ex.get("right", True) else "   (wrong)"
    fig.text(0.5, 0.04, f"Parrot: {said}{mark}", color="#ffd166", ha="center", fontsize=22)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv: list[str]) -> None:
    exchanges = json.loads(Path(argv[1]).read_text())
    voices, out = Path(argv[2]), Path(argv[3])
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, ex in enumerate(exchanges):
            q, qr = read_wav(voices / ex["question_audio"])
            audio = [to_rate(q, qr), np.zeros(int(0.5 * RATE))]
            for w in ex["said"]:
                take, tr = read_wav(voices / f"take_{w}.wav")
                audio += [to_rate(take, tr), np.zeros(int(0.15 * RATE))]
            audio.append(np.zeros(int(1.0 * RATE)))
            wav, png, mp4 = (Path(tmp) / f"{i}.wav", Path(tmp) / f"{i}.png",
                             Path(tmp) / f"{i}.mp4")
            write_wav(wav, np.concatenate(audio))
            scene(ex, png)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png),
                            "-i", str(wav), "-c:v", "libx264", "-tune", "stillimage",
                            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(mp4)],
                           check=True)
            parts.append(mp4)
        listing = Path(tmp) / "list.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-c", "copy", str(out)], check=True)


if __name__ == "__main__":
    main(sys.argv)
