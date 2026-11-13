"""Export the parrot's whole brain for the 3D views: four moments of one composed brain.

``python -m benchmarks.parrot.export_brain --out runs/brain3d [--brain brain.npz]
[--word apple.wav]``. The moments: it sees an apple; it hears "apple" (a 16 kHz mono WAV of
the word; without one, the moment hears silence and says so in its name); it focuses and says
it; it gets a pistachio. The regions the later steps teach are present at their founders and
marked untrained.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from cadence.parrot import Scene, Thing, render
from cadence.parrot.brain import Parrot
from cadence.parrot.whole import WholeParrot


def samples_of(path: Path) -> np.ndarray:
    """A 16 kHz mono 16-bit WAV as samples in [-1, 1]."""
    with wave.open(str(path), "rb") as clip:
        if clip.getframerate() != 16000 or clip.getnchannels() != 1 or clip.getsampwidth() != 2:
            raise ValueError("the word is a 16 kHz mono 16-bit WAV")
        raw = clip.readframes(clip.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--brain", type=Path, default=None)
    parser.add_argument("--word", type=Path, default=None)
    args = parser.parse_args()
    first = Parrot.load(args.brain) if args.brain else Parrot(seed=0)
    whole = WholeParrot(first, seed=0)
    scene = Scene([Thing("apple-1", "apple", 0.03, 0.33, stand=0.12),
                   Thing("cup-1", "cup", -0.35, 0.2, stand=0.05)])

    def eyes() -> np.ndarray:
        return render(scene, whole.parrot.body.eyes()).images

    for _ in range(6):
        whole.moment(eyes())
    whole.moment(eyes(), record="sees an apple")
    word = samples_of(args.word) if args.word else np.zeros(4800)
    pieces = [word[k: k + 1600] for k in range(0, len(word), 1600)]
    loud = int(np.argmax([float(np.abs(p).sum()) for p in pieces]))
    for k, piece in enumerate(pieces):
        piece = np.pad(piece, (0, 1600 - len(piece)))
        name = ("hears apple" if args.word else "hears silence") if k == loud else None
        whole.moment(eyes(), samples=piece, record=name)
    whole.moment(eyes(), record="focuses and says it")
    whole.moment(eyes(), reward=1.0, record="gets a pistachio")
    whole.export(args.out)
    print("exported", args.out)


if __name__ == "__main__":
    main()
