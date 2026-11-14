"""Talk to a trained text parrot about a scene it sees.

    python chat.py <brain.npz> <founder> [--arm d4a25]

The parrot sees a scene (the owner's room at the start) and answers what you type, one word a
moment, as it was taught. Beside each reply the chat shows what its brain did: the association
units most active while it answered, and for each thing word in your question the properties
that word alone brings up in its concept code (round, red, food: the seeing probes read on the
word's code). Commands:

    scene [n]                 a new scene of n things (3 by default)
    owner                     the owner's room
    show                      what is where
    put <kind> <colour> <direction> [far]   put a thing there (replaces what stands there)
    word <kind>               what that word alone brings up
    point on|off              the trainer points at the thing a question names (the focus
                              given, as the secondary reading of cadence#338); off: the parrot's
                              own gaze finds it
    quit
    anything else             a question, e.g. "what objects are round ?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import textchamber as tc  # noqa: E402
import textintrospect as ti  # noqa: E402
import textparrot as tp  # noqa: E402
import textworld as tw  # noqa: E402


def load(path: str, founder: int) -> Any:
    from cadence import Brain

    layout = tc.Layout.founder(founder)
    return tp.BrainPolicy.from_brain(Brain.load(path, backend="cpu"), layout)


def show(scene: tw.Scene) -> str:
    return "\n".join(f"  the {t.colour} {t.kind} ({t.shape}, {t.material}) is "
                     f"{t.direction}{' far' if t.far else ''}" for t in scene.scanned())


def words_of(line: str) -> list[str]:
    line = line.lower().replace("?", " ?").replace("color", "colour")
    words = line.split()
    if words and words[-1] != "?":
        words.append("?")
    return words


def evoked(policy: Any, pr: dict[str, Any], word: str) -> dict[str, float]:
    x = ti.hearing_codes(policy, [word] + list(tw.TRAIN_KINDS))
    centre = x.mean(0)
    return {prop: round(float(ti.probe_read(pr["probe"][prop], x[:1], centre)[0]), 2)
            for prop in ("round", "red", "food")}


def named(scene: tw.Scene, words: list[str]) -> tw.Thing | None:
    """The thing a question names (the last kind word, with a colour word before it), where the
    trainer points."""
    for i in range(len(words) - 1, -1, -1):
        if words[i] in tw.KINDS:
            colour = words[i - 1] if i > 0 and words[i - 1] in tw.COLOURS else None
            found = scene.find(words[i], colour)
            return found[0] if found else None
    return None


def answer(policy: Any, scene: tw.Scene, words: list[str], point: bool = False
           ) -> tuple[list[str], np.ndarray]:
    rec = ti.Recorder(policy)
    ref = named(scene, words) if point else None
    q = tw.Question("ask", list(words), ref, {}, [])
    eps = tc.run(rec, policy.layout, [(scene, q)], "dagger" if ref is not None else "free",
                 learn=False)
    codes = np.stack([c[0] for c in rec.codes[len(words):]]) if len(rec.codes) > len(words) \
        else np.zeros((1, len(policy.association)))
    return eps[0].reply(), codes.mean(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("brain")
    ap.add_argument("founder", type=int)
    a = ap.parse_args()
    policy = load(a.brain, a.founder)
    print("fitting the probes on what the parrot sees ...", flush=True)
    pr = ti.probes(policy, np.random.default_rng(0), n=256)
    print("probes on fresh things:", {k: round(v, 2) for k, v in pr["accuracy"].items()})
    rng = np.random.default_rng()
    scene = ti.owner_room()
    print("the parrot sees:\n" + show(scene))
    known = set(tw.VOCABULARY)
    point = False
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            break
        if not line:
            continue
        cmd = line.split()
        if cmd[0] == "quit":
            break
        if cmd[0] == "scene":
            n = int(cmd[1]) if len(cmd) > 1 else 3
            scene = tw.scene(rng, n)
            print(show(scene))
            continue
        if cmd[0] == "owner":
            scene = ti.owner_room()
            print(show(scene))
            continue
        if cmd[0] == "show":
            print(show(scene))
            continue
        if cmd[0] == "put" and len(cmd) >= 4:
            kind, colour, direction = cmd[1], cmd[2], cmd[3]
            place = (direction, "far" if len(cmd) > 4 and cmd[4] == "far" else "near")
            if kind not in tw.KINDS or colour not in tw.COLOURS or place not in tw.PLACES:
                print("  (unknown kind, colour or place)")
                continue
            things = [t for t in scene.things if t.place != place]
            things.append(tw.Thing(kind, colour, place, tw.look(kind, colour, place, rng)))
            scene = tw.Scene(things)
            print(show(scene))
            continue
        if cmd[0] == "point" and len(cmd) > 1:
            point = cmd[1] == "on"
            print("  the trainer points at the thing named" if point else
                  "  the parrot's own gaze finds the thing named")
            continue
        if cmd[0] == "word" and len(cmd) > 1:
            print("  evokes:", evoked(policy, pr, cmd[1]))
            continue
        words = words_of(line)
        unknown = [w for w in words if w not in known]
        if unknown:
            print(f"  (the parrot never read: {' '.join(unknown)})")
            continue
        reply, code = answer(policy, scene, words, point)
        print("parrot>", " ".join(w for w in reply if w != ".") or "(nothing)")
        top = np.argsort(-np.abs(code))[:8]
        print("  association units most active:",
              " ".join(f"{int(u)}:{code[u]:+.2f}" for u in top))
        for w in words:
            if w in tw.KINDS:
                print(f"  '{w}' alone evokes:", evoked(policy, pr, w))


if __name__ == "__main__":
    main()
