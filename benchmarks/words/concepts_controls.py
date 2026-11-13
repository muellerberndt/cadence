"""The learned-concepts chamber's matched controls: an LSTM and a causal transformer trained by
gradient on the same script as the brain (cadence#326, learned concepts).

**What a control is given.** A control reads every thing of the room as its raw readings, as
the brain's eyes give them and more: each thing is one position, the token ``THING`` with seven
readings beside it (the hue as its cosine and sine, the three shape features, the direction from
where the parrot's body faces as its cosine and sine), every thing of the room in view or out of
it. Words are tokens, one a nonce form of the founder; a control reads no letter and is given no
kind, colour or place as a token. Then:

- a naming: the room, ``NAME``, the attended thing (one ``THING`` position with its readings),
  ``SAY``, and the heard word and ``END``;
- a question of practice or of the test: the room, ``ASK`` and its words, ``SAY``, and the
  answer's words and ``END``.

The loss is the cross-entropy of the tokens after ``SAY``. The readings enter through a linear
map into the width, added to the token's embedding; a position with no thing reads zeros.

**Modes** are protocol 1's (``controls.py``): ``online``, one pass over the script in the
brain's order, one Adam step a naming or practice question; ``offline``, shuffled epochs with
early stopping on a tenth held aside. In both modes the new words' three namings are one step
each and the test is answered greedily. The width is the one whose parameter count is nearest
the brain's.

    python concepts_controls.py <seed> <lstm|transformer> <online|offline> [parameters]
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

import concepts  # noqa: E402
import world  # noqa: E402


def _controls() -> Any:
    """Protocol 1's controls under the name they take in this chamber's tests: several chambers
    hold a ``controls.py`` (#306)."""
    import importlib.util

    if "words_controls" not in sys.modules:
        spec = importlib.util.spec_from_file_location("words_controls", HERE / "controls.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["words_controls"] = module
        spec.loader.exec_module(module)
    return sys.modules["words_controls"]


controls = _controls()

READINGS = 7
MARKERS = ("PAD", "THING", "NAME", "ASK", "SAY", "END")


class Tokens:
    """A founder's vocabulary: the markers and the founder's nonce words."""

    def __init__(self, lex: world.Lexicon):
        self.names = list(MARKERS) + (lex.kind + lex.colour + lex.place
                                      + list(lex.question.values()) + [lex.none] + lex.foils)
        self.index = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def __call__(self, name: str) -> int:
        return self.index[name]


def readings(t: world.Thing, heading: float) -> np.ndarray:
    d = concepts.direction(heading, t.place)
    hue = float(t.hue)  # type: ignore[attr-defined]
    return np.concatenate([[math.cos(hue), math.sin(hue)], t.shape,  # type: ignore[attr-defined]
                           [math.cos(d), math.sin(d)]])


Row = tuple[list[int], list[np.ndarray]]


def scene(tok: Tokens, r: world.Room, heading: float) -> Row:
    return ([tok("THING")] * len(r.things), [readings(t, heading) for t in r.things])


def sequences(seed: int, **given: Any) -> list[tuple[str, Any, Any]]:
    """The script as the controls' items, in the brain's order: ``("train", [(prompt,
    target)])`` for a naming or a practice question, ``("ask", gate, (prompt, question))`` for a
    test question and ``("new", [(prompt, target)])`` for a new word's naming. A prompt is
    ``(tokens, readings)``."""
    lex = world.lexicon(seed)
    tok = Tokens(lex)
    zero = np.zeros(READINGS)
    out: list[tuple[str, Any, Any]] = []
    r: world.Room | None = None
    heading = 0.0
    for event in concepts.script(seed, **given):
        kind = event[0]
        if kind == "room":
            r, heading = event[1], event[1].heading
        elif kind == "turn":
            heading = event[1]
        elif kind == "name":
            assert r is not None
            t = min(r.things, key=lambda th: float(np.linalg.norm(th.place - event[1])))
            tokens, values = scene(tok, r, heading)
            prompt = (tokens + [tok("NAME"), tok("THING"), tok("SAY")],
                      values + [zero, readings(t, heading), zero])
            out.append(("train" if event[3] == "naming" else "new",
                        [(prompt, [tok(event[2]), tok("END")])], None))
        elif kind in ("practice", "ask"):
            assert r is not None
            q = event[1] if kind == "practice" else event[2]
            words = [tok(w) for w in q.text.split()]
            tokens, values = scene(tok, r, heading)
            prompt = (tokens + [tok("ASK")] + words + [tok("SAY")],
                      values + [zero] * (len(words) + 2))
            if kind == "practice":
                out.append(("train", [(prompt, [tok(w) for w in q.answer] + [tok("END")])],
                            None))
            else:
                out.append(("ask", event[1], (prompt, q)))
    return out


def build(model: str, width: int, vocabulary: int, seed: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    inner = controls.build(model, width, vocabulary, seed)

    class Sensed(nn.Module):  # type: ignore[misc]
        """The protocol 1 control with the readings mapped into its width beside each
        token's embedding."""

        def __init__(self) -> None:
            super().__init__()
            self.inner = inner
            self.sense = nn.Linear(READINGS, width)

        def forward(self, x: Any, values: Any) -> Any:
            h = self.inner.embed(x) + self.sense(values)
            if model == "lstm":
                out, _ = self.inner.cell(h)
                return self.inner.out(out)
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            h = h + self.inner.position[:n]
            return self.inner.out(self.inner.body(h, mask=mask, is_causal=True))

        def macs(self, length: int) -> int:
            return int(self.inner.macs(length) + length * READINGS * width)

    return Sensed()


def count(net: Any) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def matched_width(model: str, target: int, vocabulary: int) -> int:
    step = 2 if model == "transformer" else 1
    return int(min(range(4, 513, step),
                   key=lambda w: (abs(count(build(model, w, vocabulary, 0)) - target), w)))


class Control:
    def __init__(self, model: str, width: int, vocabulary: int, seed: int, rate: float):
        import torch

        self.torch = torch
        self.net = build(model, width, vocabulary, seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.work = {"updates": 0, "macs": 0}

    def batch(self, pairs: list[tuple[Row, list[int]]]) -> tuple[Any, Any, Any]:
        torch = self.torch
        n = max(len(p[0]) + len(t) for p, t in pairs)
        x = np.zeros((len(pairs), n), dtype=np.int64)
        v = np.zeros((len(pairs), n, READINGS), dtype=np.float32)
        y = np.full((len(pairs), n), -100, dtype=np.int64)
        for i, ((tokens, values), target) in enumerate(pairs):
            row = tokens + target
            x[i, : len(row)] = row
            v[i, : len(values)] = np.asarray(values, dtype=np.float32)
            for j, token in enumerate(target):
                y[i, len(tokens) - 1 + j] = token
        return torch.from_numpy(x), torch.from_numpy(v), torch.from_numpy(y)

    def loss(self, pairs: list[tuple[Row, list[int]]]) -> Any:
        x, v, y = self.batch(pairs)
        logits = self.net(x, v)
        return self.torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)

    def update(self, pairs: list[tuple[Row, list[int]]]) -> float:
        torch = self.torch
        loss = self.loss(pairs)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), controls.CLIP)
        self.opt.step()
        length = max(len(p[0]) + len(t) for p, t in pairs)
        self.work["updates"] += 1
        self.work["macs"] += 3 * len(pairs) * self.net.macs(length)
        return float(loss.item())

    def say(self, prompt: Row, end: int, most: int = 6) -> list[int]:
        torch = self.torch
        tokens, values = list(prompt[0]), list(prompt[1])
        out: list[int] = []
        with torch.no_grad():
            for _ in range(most):
                v = torch.from_numpy(np.asarray(values, dtype=np.float32)[None])
                logits = self.net(torch.tensor([tokens]), v)[0, -1]
                token = int(logits.argmax())
                self.work["macs"] += self.net.macs(len(tokens))
                if token == end:
                    break
                out.append(token)
                tokens.append(token)
                values.append(np.zeros(READINGS))
        return out


def run(seed: int, model: str, mode: str, parameters: int, **given: Any) -> dict[str, Any]:
    """One control's life on the founder's script; the readings by gate, its parameters and
    work."""
    import torch
    import words

    torch.set_num_threads(1)
    lex = world.lexicon(seed)
    tok = Tokens(lex)
    items = sequences(seed, **given)
    width = matched_width(model, parameters, len(tok))
    control = Control(model, width, len(tok), seed, controls.RATES[mode])
    began = time.monotonic()
    train = [pairs for kind, pairs, _ in items if kind == "train"]
    if mode == "online":
        for pairs in train:
            control.update(pairs)
    else:
        flat = [p for pairs in train for p in pairs]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 326, 9]))
        order = rng.permutation(len(flat))
        aside = [flat[i] for i in order[: len(flat) // 10]]
        fit = [flat[i] for i in order[len(flat) // 10 :]]
        best, best_state, since, since_halved = math.inf, None, 0, 0
        rate = controls.RATES[mode]
        for _ in range(controls.EPOCHS):
            shuffled = rng.permutation(len(fit))
            for start in range(0, len(fit), controls.BATCH):
                control.update([fit[int(i)] for i in shuffled[start : start + controls.BATCH]])
            with torch.no_grad():
                held = float(np.mean([control.loss(aside[i : i + controls.BATCH]).item()
                                      for i in range(0, len(aside), controls.BATCH)]))
            if held < best - 1e-6:
                best, since, since_halved = held, 0, 0
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
    found: dict[str, list[bool]] = {}
    for kind, first, second in items:
        if kind == "new":
            control.update(first)
        elif kind == "ask":
            gate, (prompt, q) = first, second
            said = [tok.names[t] for t in control.say(prompt, tok("END"))]
            ok = world.right(q, said)
            found.setdefault(gate, []).append(ok)
            if gate.startswith("W1") and q.about == "out_of_view":
                found.setdefault("W4_out_of_view", []).append(ok)
    out = words.readings_of(seed, {}, found, [], None,
                            {"train": trained_at - began, "test": time.monotonic() - trained_at})
    out["C2_room"] = float(np.mean(found.get("C2_colour", []) + found.get("C2_where", [])))
    out.update({"model": model, "mode": mode, "width": width, "parameters": count(control.net),
                "updates": control.work["updates"], "macs": control.work["macs"]})
    return out


if __name__ == "__main__":
    import json

    told = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    print(json.dumps(run(int(sys.argv[1]), sys.argv[2], sys.argv[3],
                         int(sys.argv[4]) if len(sys.argv) > 4 else 56864, **told),
                     default=float))
