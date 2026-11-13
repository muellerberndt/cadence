"""The relations chamber's matched controls (cadence#327): an LSTM and a causal transformer
trained by gradient on the same script as the brain (``relations.script``), with the words
chamber's training (``controls.py``: the same networks, rates, epochs and greedy answers).

**What a control is given.** The brain holds the room on its map and knows where it faces; a
control is given the room as tokens at every sequence: for each thing, ``THING``, its kind and
its place in the parrot's frame as it faces at that moment (``x`` ahead, ``y`` to its left),
each axis in ``BINS`` bins of 4 m / ``BINS`` over the room. Then:

- a naming, as two sequences: ``NAME kind SAY -> word END`` and ``NAME word SAY -> kind END``;
  the parrot named itself: ``NAME SELF SAY -> word END`` and ``NAME word SAY -> SELF END``;
- a question of practice or of the test: ``ASK`` and its words, ``SAY``, the answer's words and
  ``END``.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import controls  # noqa: E402
import relations  # noqa: E402
import world  # noqa: E402

BINS = 16
MARKERS = ("PAD", "THING", "NAME", "ASK", "SAY", "END", "SELF")


class Tokens:
    """A founder's vocabulary: the markers, the kinds, the place bins and the founder's words."""

    def __init__(self, w: relations.Nonce):
        self.names = list(MARKERS) + [f"kind{k}" for k in range(relations.KINDS)]
        self.names += [f"x{i}" for i in range(BINS)] + [f"y{i}" for i in range(BINS)]
        self.words = list(dict.fromkeys(w.base.kind[: relations.KINDS] + list(
            w.base.question.values()) + [w.base.none] + w.relation + list(w.frame.values())))
        self.names += self.words
        self.index = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def __call__(self, name: str) -> int:
        return self.index[name]


def scene(tok: Tokens, r: world.Room) -> list[int]:
    """The room as tokens in the parrot's frame as it faces."""
    out: list[int] = []
    c, s = math.cos(-r.heading), math.sin(-r.heading)
    for t in r.things:
        x, y = c * t.place[0] - s * t.place[1], s * t.place[0] + c * t.place[1]
        bx = min(max(int((x + 2.0) / 4.0 * BINS), 0), BINS - 1)
        by = min(max(int((y + 2.0) / 4.0 * BINS), 0), BINS - 1)
        out += [tok("THING"), tok(f"kind{t.kind}"), tok(f"x{bx}"), tok(f"y{by}")]
    return out


def sequences(seed: int, **given: Any) -> tuple[Tokens, list[tuple[str, Any, Any]]]:
    """The script as the controls' items, in the brain's order: ``("train", pairs, None)`` and
    ``("ask", gate, (prompt, asked))``; the held-out word's lessons are ``("train", ...)`` in
    their place in the script."""
    w = relations.words_of(seed)
    tok = Tokens(w)
    out: list[tuple[str, Any, Any]] = []
    r: world.Room | None = None
    for event in relations.script(seed, **given):
        kind = event[0]
        if kind == "room":
            r = event[1]
        elif kind == "turn":
            assert r is not None
            r.heading = event[1]
        elif kind == "name":
            assert r is not None
            place, word = event[1], event[2]
            t = min(r.things, key=lambda th: float(np.linalg.norm(th.place - place)))
            room = scene(tok, r)
            out.append(("train", [
                (room + [tok("NAME"), tok(f"kind{t.kind}"), tok("SAY")], [tok(word), tok("END")]),
                (room + [tok("NAME"), tok(word), tok("SAY")], [tok(f"kind{t.kind}"), tok("END")])],
                None))
        elif kind == "self":
            out.append(("train", [([tok("NAME"), tok("SELF"), tok("SAY")],
                                   [tok(event[1]), tok("END")]),
                                  ([tok("NAME"), tok(event[1]), tok("SAY")],
                                   [tok("SELF"), tok("END")])], None))
        else:
            assert r is not None
            a = event[1] if kind == "practice" else event[2]
            prompt = scene(tok, r) + [tok("ASK")] + [tok(x) for x in a.text.split()] + [
                tok("SAY")]
            if kind == "practice":
                out.append(("train", [(prompt, [tok(x) for x in a.answer] + [tok("END")])], None))
            else:
                out.append(("ask", event[1], (prompt, a)))
    return tok, out


def run(seed: int, model: str, mode: str, parameters: int, **given: Any) -> dict[str, Any]:
    """One control's life on the founder's script: trained on every item before the test (in
    order online; in shuffled epochs offline), then the test asked in order, the held-out
    word's lessons taken as one update each where they come."""
    import torch

    torch.set_num_threads(1)
    tok, items = sequences(seed, **given)
    held = relations.held_out_pairs(seed, {**relations.DEFAULTS, **given}["held_pairs"])
    width = controls.matched_width(model, parameters, len(tok))
    control = controls.Control(model, width, len(tok), seed, controls.RATES[mode])
    began = time.monotonic()
    first_ask = next(i for i, item in enumerate(items) if item[0] == "ask")
    train = [pairs for kind, pairs, _ in items[:first_ask] if kind == "train"]
    if mode == "online":
        for pairs in train:
            control.update(pairs)
    else:
        flat = [p for pairs in train for p in pairs]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), relations.SALT, 9]))
        order = rng.permutation(len(flat))
        aside = [flat[i] for i in order[: len(flat) // 10]]
        fit = [flat[i] for i in order[len(flat) // 10:]]
        best, best_state, since, since_halved, rate = math.inf, None, 0, 0, controls.RATES[mode]
        for _ in range(controls.EPOCHS):
            shuffled = rng.permutation(len(fit))
            for start in range(0, len(fit), controls.BATCH):
                control.update([fit[int(i)] for i in shuffled[start: start + controls.BATCH]])
            with torch.no_grad():
                loss = float(np.mean([control.loss(aside[i: i + controls.BATCH]).item()
                                      for i in range(0, len(aside), controls.BATCH)]))
            if loss < best - 1e-6:
                best, since, since_halved = loss, 0, 0
                best_state = {k: v.clone() for k, v in control.net.state_dict().items()}
            else:
                since += 1
                since_halved += 1
                if since_halved >= controls.HALVE and rate > controls.FLOOR:
                    rate = max(rate / 2.0, controls.FLOOR)
                    since_halved = 0
                    for group in control.opt.param_groups:
                        group["lr"] = rate
            if since >= controls.PATIENCE:
                break
        if best_state is not None:
            control.net.load_state_dict(best_state)
    trained_at = time.monotonic()
    readings: dict[str, list[bool]] = {}
    for kind, first, second in items[first_ask:]:
        if kind == "train":
            control.update(first)  # the held-out word's lessons: one update each
            continue
        gate, (prompt, a) = first, second
        said = [tok.names[t] for t in control.say(prompt, tok("END"))]
        ok = relations.right(a, said)
        readings.setdefault(gate, []).append(ok)
        readings.setdefault(f"{gate}_{a.form}", []).append(ok)
        if gate != "R5_new" and relations.held_pair(a, held):
            readings.setdefault("R4_held", []).append(ok)
    out = relations.readings_of(seed, {}, readings, None, {
        "train": trained_at - began, "test": time.monotonic() - trained_at})
    out.update({"model": model, "mode": mode, "width": width,
                "parameters": controls.count(control.net), "updates": control.work["updates"],
                "macs": control.work["macs"]})
    return out


if __name__ == "__main__":
    import json

    seed, model, mode = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    parameters = int(sys.argv[4]) if len(sys.argv) > 4 else 2368
    told = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    print(json.dumps(run(seed, model, mode, parameters, **told), indent=1, default=float))
