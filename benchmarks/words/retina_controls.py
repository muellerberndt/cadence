"""The retina chamber's matched controls: a convolutional network and a vision transformer
trained by gradient on the same images and episodes as the brain (cadence#330, R5).

**What a control is given.** What the parrot's eyes see, image for image: the images the brain
saw at each moment of a room (``retina.render``, the same receptor noise in the same order),
each receptor's red, green, blue and nearness (one over its distance), and at each moment where
the head faced, as the cosine and sine of its turn from where it faces at the episode's end. An
episode's context is the room's look-around (nine moments) and the moment of the latest turn.
Then, as tokens:

- a naming: ``NAME``, where the teacher points (the cosine and sine of the thing's direction
  from where the head faces, and its distance), ``SAY``, and the heard word and ``END``;
- a question of practice or of the test: ``ASK`` and its words, ``SAY``, and the answer's words
  and ``END``.

The loss is the cross-entropy of the tokens after ``SAY``; words are tokens, one a nonce form of
the founder.

**The networks.** ``cnn``: a convolutional encoder of each eye's image (three layers along the
receptors, the second and third with stride 2, then the mean and the most of each channel over
the receptors), the two eyes' codes and the head's turn into one input a moment, and an LSTM over
the episode. ``vit``: each eye's image cut into patches of 16 receptors, each patch a token
(a linear map of its receptors, with a learned position of its eye and patch and the head's turn
of its moment), with the words' tokens in one causal transformer (two layers, two heads).

**Modes and work** as the words chamber's controls (``controls.py``): ``online`` one pass in the
brain's order, ``offline`` shuffled epochs with early stopping on a tenth held aside. The width
is the one whose parameter count is nearest the brain's.
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
import retina  # noqa: E402
import world  # noqa: E402

MARKERS = ("PAD", "IMAGE", "POINT", "NAME", "ASK", "SAY", "END")
PATCH = 16
MOMENTS = 10  # an episode's moments: the look-around and the latest turn


class Tokens:
    """A founder's vocabulary: the markers and the founder's nonce words."""

    def __init__(self, lex: world.Lexicon):
        self.names = list(MARKERS)
        self.words = (lex.kind + lex.colour + lex.place + list(lex.question.values())
                      + [lex.none] + lex.foils)
        self.names += self.words
        self.index = {n: i for i, n in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def __call__(self, name: str) -> int:
        return self.index[name]


def episodes(seed: int, **given: Any) -> tuple[np.ndarray, list[tuple[str, Any, Any]]]:
    """The founder's images and items. ``images`` is ``(moments, eyes, receptors, 4)``; an
    item is ``("train", (context, prompt, target))`` for a naming or a practice question,
    ``("new", ...)`` for a new word's naming, or ``("ask", gate, ((context, prompt), q))``. A
    context is ``(image numbers, headings)``, its moments' headings relative to the latest; a
    prompt is its tokens, a pointer's three numbers after ``POINT``."""
    from cadence.words.retina import parrot_eyes

    lex = world.lexicon(seed)
    tok = Tokens(lex)
    eyes = parrot_eyes()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 330, 2]))
    bank: list[np.ndarray] = []
    out: list[tuple[str, Any, Any]] = []
    r: world.Room | None = None
    around: list[tuple[int, float]] = []
    latest: tuple[int, float] | None = None

    def moment(room: world.Room, heading: float) -> tuple[int, float]:
        image = retina.render(seed, room, eyes, np.zeros(2), heading, rng=rng)
        rgb, depth = image[..., :3], image[..., 3:]
        bank.append(np.concatenate([rgb, 1.0 / np.maximum(depth, 1e-3)], axis=-1).astype(
            np.float32))
        return len(bank) - 1, heading

    def context() -> tuple[list[int], list[float]]:
        assert r is not None
        moments = around + ([latest] if latest is not None else [around[-1]])
        return [m for m, _ in moments], [h - r.heading for _, h in moments]

    for event in retina.script(seed, **given):
        kind = event[0]
        if kind == "room":
            r = event[1]
            around = [moment(r, r.heading + (i % 8) * math.pi / 4 if i < 8 else r.heading)
                      for i in range(9)]
            latest = None
        elif kind == "turn":
            assert r is not None
            r.heading = event[1]
            latest = moment(r, r.heading)
        elif kind == "name":
            assert r is not None
            place, word = event[1], event[2]
            a = math.atan2(place[1], place[0]) - r.heading
            point = [math.cos(a), math.sin(a), float(np.linalg.norm(place))]
            prompt = [tok("NAME"), tok("POINT"), tok("SAY")]
            out.append(("train" if event[3] == "naming" else "new",
                        (context(), (prompt, point), [tok(word), tok("END")]), None))
        elif kind == "practice":
            q = event[1]
            prompt = [tok("ASK")] + [tok(w) for w in q.text.split()] + [tok("SAY")]
            out.append(("train", (context(), (prompt, None),
                                  [tok(w) for w in q.answer] + [tok("END")]), None))
        elif kind == "ask":
            q = event[2]
            prompt = [tok("ASK")] + [tok(w) for w in q.text.split()] + [tok("SAY")]
            out.append(("ask", event[1], ((context(), (prompt, None)), q)))
    return np.stack(bank), out


def build(model: str, width: int, vocabulary: int, receptors: int, seed: int) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    patches = math.ceil(receptors / PATCH)

    class Inputs(nn.Module):  # type: ignore[misc]
        """Tokens, a pointer's numbers and each moment's turn, as rows of ``width``."""

        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocabulary, width)
            self.point = nn.Linear(3, width)
            self.turn = nn.Linear(2, width)

    class CNN(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            c = max(width // 2, 2)
            self.inputs = Inputs()
            self.conv = nn.Sequential(
                nn.Conv1d(4, c, 5, padding=2), nn.ReLU(),
                nn.Conv1d(c, c, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv1d(c, c, 5, stride=2, padding=2), nn.ReLU())
            self.image = nn.Linear(4 * c, width)  # two eyes, the mean and the most of each
            self.cell = nn.LSTM(width, width, batch_first=True)
            self.out = nn.Linear(width, vocabulary)
            self.c = c

        def encode(self, images: Any, turns: Any) -> Any:
            # images (moments, eyes, receptors, 4) -> (moments, width)
            m, e, n, _ = images.shape
            h = self.conv(images.reshape(m * e, n, 4).transpose(1, 2))
            pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1).reshape(m, -1)
            return self.image(pooled) + self.inputs.turn(turns)

        def forward(self, rows: Any) -> Any:
            h, _ = self.cell(rows)
            return self.out(h)

        def macs(self, moments: int, length: int) -> int:
            c = self.c
            n2, n3 = math.ceil(receptors / 2), math.ceil(receptors / 4)
            conv = 2 * (receptors * 4 * c * 5 + n2 * c * c * 5 + n3 * c * c * 5) + 4 * c * width
            return moments * conv + length * (4 * width * 2 * width + width * vocabulary)

    class ViT(nn.Module):  # type: ignore[misc]
        LAYERS, HEADS = 2, 2

        def __init__(self) -> None:
            super().__init__()
            self.inputs = Inputs()
            self.patch = nn.Linear(PATCH * 4, width)
            self.where = nn.Parameter(torch.zeros(2 * patches, width))
            layer = nn.TransformerEncoderLayer(width, self.HEADS, 2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, self.LAYERS, enable_nested_tensor=False)
            self.out = nn.Linear(width, vocabulary)
            length = MOMENTS * 2 * patches + 32
            position = torch.arange(length)[:, None]
            scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
            table = torch.zeros(length, width)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : width // 2]
            self.register_buffer("position", table)

        def encode(self, images: Any, turns: Any) -> Any:
            # images (moments, eyes, receptors, 4) -> (moments * eyes * patches, width)
            m, e, n, _ = images.shape
            padded = torch.nn.functional.pad(images, (0, 0, 0, patches * PATCH - n))
            cut = padded.reshape(m, e * patches, PATCH * 4)
            rows = self.patch(cut) + self.where[None] + self.inputs.turn(turns)[:, None, :]
            return rows.reshape(m * e * patches, width)

        def forward(self, rows: Any) -> Any:
            n = rows.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            return self.out(self.body(rows + self.position[:n], mask=mask, is_causal=True))

        def macs(self, moments: int, length: int) -> int:
            tokens = moments * 2 * patches + length
            per = self.LAYERS * 8 * width * width + width * vocabulary
            return (moments * 2 * patches * PATCH * 4 * width + tokens * per
                    + self.LAYERS * width * tokens * (tokens + 1))

    return CNN() if model == "cnn" else ViT()


def matched_width(model: str, target: int, vocabulary: int, receptors: int) -> int:
    step = 2 if model == "vit" else 1
    best = min(range(4, 129, step), key=lambda w: (abs(controls.count(
        build(model, w, vocabulary, receptors, 0)) - target), w))
    return int(best)


class Control:
    def __init__(self, model: str, width: int, vocabulary: int, images: np.ndarray, seed: int,
                 rate: float):
        import torch

        self.torch = torch
        self.images = torch.from_numpy(images)
        self.net = build(model, width, vocabulary, images.shape[2], seed)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=rate)
        self.work = {"updates": 0, "macs": 0, "seconds": 0.0}

    def rows(self, context: Any, prompt: Any, target: list[int]) -> Any:
        """One sequence's input rows: the context's moments, the prompt (its ``POINT`` token
        followed by the pointer's row) and the target but its last token."""
        torch = self.torch
        numbers, headings = context
        turns = torch.tensor([[math.cos(h), math.sin(h)] for h in headings],
                             dtype=torch.float32)
        seen = self.net.encode(self.images[numbers], turns)
        tokens, point = prompt
        inputs = self.net.inputs
        said = inputs.embed(torch.tensor(tokens + target[:-1], dtype=torch.long))
        if point is not None:
            at = tokens.index(MARKERS.index("POINT"))
            pointer = inputs.point(torch.tensor(point, dtype=torch.float32))[None]
            said = torch.cat([said[: at + 1] + 0.0, pointer, said[at + 1:]], dim=0)
        return torch.cat([seen, said], dim=0)

    def loss(self, batch: list[tuple[Any, Any, list[int]]]) -> Any:
        torch = self.torch
        rows = [self.rows(c, p, t) for c, p, t in batch]
        n = max(len(r) for r in rows)
        x = torch.zeros(len(rows), n, rows[0].shape[1])
        y = torch.full((len(rows), n), -100, dtype=torch.long)
        for i, ((_, _, target), r) in enumerate(zip(batch, rows, strict=True)):
            x[i, : len(r)] = r
            start = len(r) - len(target)
            y[i, start: start + len(target)] = torch.tensor(target)
        logits = self.net(x)
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)

    def update(self, batch: list[tuple[Any, Any, list[int]]]) -> float:
        torch = self.torch
        started = time.monotonic()
        loss = self.loss(batch)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), controls.CLIP)
        self.opt.step()
        self.work["updates"] += 1
        for c, p, t in batch:
            self.work["macs"] += 3 * self.net.macs(len(c[0]), len(p[0]) + len(t) + 1)
        self.work["seconds"] += time.monotonic() - started
        return float(loss.item())

    def say(self, context: Any, prompt: Any, end: int, most: int = 6) -> list[int]:
        torch = self.torch
        out: list[int] = []
        with torch.no_grad():
            for _ in range(most):
                rows = self.rows(context, prompt, out + [0])
                token = int(self.net(rows[None])[0, -1].argmax())
                self.work["macs"] += self.net.macs(len(context[0]), len(prompt[0]) + len(out)
                                                   + 1)
                if token == end:
                    break
                out.append(token)
        return out


def run(seed: int, model: str, mode: str, parameters: int, **given: Any) -> dict[str, Any]:
    """One control's life on the founder's images and episodes; the readings by gate, its
    parameters and work."""
    import torch
    import words

    torch.set_num_threads(1)
    lex = world.lexicon(seed)
    tok = Tokens(lex)
    images, items = episodes(seed, **given)
    width = matched_width(model, parameters, len(tok), images.shape[2])
    control = Control(model, width, len(tok), images, seed, controls.RATES[mode])
    began = time.monotonic()
    train = [first for kind, first, _ in items if kind == "train"]
    if mode == "online":
        for item in train:
            control.update([item])
    else:
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 330, 9]))
        order = rng.permutation(len(train))
        aside = [train[i] for i in order[: len(train) // 10]]
        fit = [train[i] for i in order[len(train) // 10:]]
        best, best_state, since, since_halved = math.inf, None, 0, 0
        rate = controls.RATES[mode]
        for _ in range(controls.EPOCHS):
            shuffled = rng.permutation(len(fit))
            for start in range(0, len(fit), controls.BATCH):
                control.update([fit[int(i)] for i in shuffled[start: start + controls.BATCH]])
            with torch.no_grad():
                held = float(np.mean([control.loss(aside[i: i + controls.BATCH]).item()
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
    readings: dict[str, list[bool]] = {}
    for kind, first, second in items:
        if kind == "new":
            control.update([first])
        elif kind == "ask":
            gate, ((context, prompt), q) = first, second
            said = [tok.names[t] for t in control.say(context, prompt, tok("END"))
                    if t < len(tok)]
            ok = world.right(q, said)
            readings.setdefault(gate, []).append(ok)
            if gate.startswith("W1") and q.about == "out_of_view":
                readings.setdefault("W4_out_of_view", []).append(ok)
    out = words.readings_of(seed, {}, readings, [], None,
                            {"train": trained_at - began, "test": time.monotonic() - trained_at})
    out.update({"model": model, "mode": mode, "width": width,
                "parameters": controls.count(control.net), "updates": control.work["updates"],
                "macs": control.work["macs"]})
    return out


if __name__ == "__main__":
    import json

    seed, model, mode, parameters = (int(sys.argv[1]), sys.argv[2], sys.argv[3],
                                     int(sys.argv[4]))
    told = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
    print(json.dumps(run(seed, model, mode, parameters, **told), indent=1, default=float))
