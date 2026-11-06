"""Attention ablations: the matched transformer control under the recall memory's restrictions.

The recall memory differs from transformer attention in five ways (the memory guide, #260):
its keys and values are the observation code and never learned; its read is held, addressed
before the settle from the current input and the previous moment's cortex, one read per
moment; the read's value reaches the motor only through the association, the copy pathway a
fixed identity around it; its heads are copies of the held read on the same keys and values,
competing for slots, not learned subspaces mixed by an output projection; and no credit reaches
keys or values. Each row here puts one restriction (or a combination) on the grammar chamber's
transformer control ([controls.py](controls.py): two layers, two heads, pre-norm, sinusoidal
positions), re-matches its width so that its learned parameter count is the nearest to the
brain's, and reads it on the chamber's panels with the same sentences, lessons and scorer. The
restriction that drops the control toward the brain's numbers names the gap.

- ``none``: the control transformer, rebuilt here with explicit query, key, value and output
  maps: ``controls.py``'s function at the same weights (a test copies them across), drawn from
  another initialization (``nn.Linear``'s maps, random biases and two separately drawn layers,
  where torch's encoder draws xavier maps, zero biases and two cloned layers). It is the
  baseline of every row, and not the control's life.
- ``A`` (``frozen_kv``): the key and value maps of every layer are frozen at their random
  initialization; queries, output maps and the MLP blocks learn; credit still flows through
  the frozen maps into the layer below.
- ``B`` (``single``): attention in the first layer only, its query from the token's embedding;
  the second layer keeps its MLP block. One read per moment, nothing stacked on a read.
- ``Bp`` (``single, recurrent``): the single read's query also reads the previous position's
  layer output through a learned map, the brain's address ``x W_x + c W_c`` with ``c`` the
  cortex of the moment before; the positions are computed in turn.
- ``C`` (``raw_values``): the values are the raw moments (no value map); the read enters the
  residual through a learned map from the moment's code (the recall synapses) and the logits
  through a fixed identity on the word rows (the copy pathway); no output map. The read is
  linear in the values, so that map is a learned value map of the raw moment: this row removes
  the contextual values and the mix and adds the identity, and keeps a learned value path.
- ``Cs`` (``raw_values, copy_only``): the read reaches the logits through the identity alone,
  no learned path from the read at all. Concept moments have no word rows.
- ``D1`` (``heads=1``): one head. ``D2`` (``no_mix``): two heads, each with values at full
  width, summed into the residual without an output map.
- ``E`` (``detach_kv``): no gradient into the keys and values (detached, so their maps never
  move and nothing below learns through them); queries learn. ``AE`` is ``E`` by construction.
- ``H1b``, ``H2b`` (``brain_heads``): one or two heads that share one key and one value map and
  differ only by an address map of their own and a competition for the slots: a head takes of
  every slot at most what the heads before left, keeping the shape of its own attention where
  the slots allow (a water-filling of its mass); the reads are summed, no output map. Against
  the subspace heads ``D1`` (one) and ``none`` (two).
- ``AB``, ``BC``, ``all``: combinations; ``all`` is ``A B' C D1 E`` at once.
- ``lstmA`` (``frozen_ih``): the LSTM control with its input-to-hidden weights frozen random.
- ``ctl``, ``lstm``: ``controls.py``'s transformer and LSTM themselves, here so that they take the
  rows' initializations on the rows' machine.

``A`` and ``E`` still read their keys from a learned embedding and, in the second layer, from
what the first attention found, and ``B`` bundles three differences. These rows separate them.
``code`` is a fixed random map of the raw moment plus the position table, ``R x + position``,
with ``R`` scaled so that a one-hot moment's code has the position's norm; it holds neither the
embedding nor any layer's output. ``e`` is the learned embedding, ``h`` the layer's input.

| Row | First layer: query, keys and values | Second layer: query | Second layer: keys and values |
| --- | --- | --- | --- |
| ``none`` | ``e``, ``e`` | ``h`` | ``h`` |
| ``Ar`` | ``e``, ``code`` | ``h`` | ``code`` |
| ``S1`` | ``e``, ``e`` | ``h`` | ``code`` |
| ``S2`` | ``e``, ``e`` | ``e`` | ``h`` |
| ``S3`` | ``e``, ``e`` | ``e`` | ``e`` |
| ``S3r`` | ``e``, ``code`` | ``e`` | ``code`` |

Keys and values of ``code`` are frozen maps of a frozen code, so no credit reaches them or passes
through them: these rows hold ``E``'s restriction wherever they hold ``code``. ``allr`` is ``all``
with its keys from ``code``; its values stay the raw moment, which the identity needs, and carry
no position. ``allrn`` is ``allr`` without the identity: the recipe brain has no copy pathway.
Neither carries the brain's next port or its working trace (``Bp``'s state reaches the next moment
through the query alone).

Modes and the life as in ``controls.py``: ``online`` one pass in the brain's order, ``offline``
shuffled epochs to convergence. Work: updates, tokens, analytic multiply-accumulates (every
weight once per token, the attention's scores and mixing per pair) and wall time.

    python benchmarks/grammar/ablations.py --rows none A B Bp C D1 D2 E lstmA \\
        --modes online offline --founders 0 1 2 --out DIR/ablations.jsonl
    python benchmarks/grammar/ablations.py --table DIR/ablations.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections.abc import Sequence
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import controls as ct
import grammar as gr
import numpy as np

ROWS: dict[str, dict[str, Any]] = {
    "none": {},
    "A": {"frozen_kv": True},
    "B": {"single": True},
    "Bp": {"single": True, "recurrent": True},
    "C": {"raw_values": True},
    "D1": {"heads": 1},
    "D2": {"no_mix": True},
    "E": {"detach_kv": True},
    "H1b": {"heads": 1, "brain_heads": True},
    "H2b": {"heads": 2, "brain_heads": True},
    "AB": {"frozen_kv": True, "single": True},
    "AE": {"frozen_kv": True, "detach_kv": True},
    "BC": {"single": True, "raw_values": True},
    "all": {"frozen_kv": True, "single": True, "recurrent": True, "raw_values": True,
            "heads": 1, "detach_kv": True},
    "Ar": {"raw_kv": [0, 1]},
    "S1": {"raw_kv": [1]},
    "S2": {"embed_q": True},
    "S3": {"embed_q": True, "embed_kv": True},
    "S3r": {"raw_kv": [0, 1], "embed_q": True},
    "allr": {"raw_kv": [0], "single": True, "recurrent": True, "raw_values": True, "heads": 1},
    "allrn": {"raw_kv": [0], "single": True, "recurrent": True, "raw_values": True, "heads": 1,
              "no_copy": True},
    "Cs": {"raw_values": True, "copy_only": True},
    "lstmA": {"model": "lstm", "frozen_ih": True},
    "ctl": {"model": "control"},
    "lstm": {"model": "lstm"},
}
FOUNDER_PARAMETERS = 24384  # the recipe brain's ``parameters()`` (the chamber's README)
SINGLE = ("none", "A", "B", "Bp", "C", "D1", "D2", "E", "lstmA")
PANELS = gr.G1_PANELS


def build(model: str, width: int, seed: int, *, heads: int = 2, frozen_kv: bool = False,
          single: bool = False, recurrent: bool = False, raw_values: bool = False,
          no_mix: bool = False, brain_heads: bool = False, detach_kv: bool = False,
          frozen_ih: bool = False, raw_kv: Sequence[int] = (), embed_q: bool = False,
          embed_kv: bool = False, copy_only: bool = False, no_copy: bool = False) -> Any:
    """The control under the named restrictions (see the module's docstring)."""
    import torch
    from torch import nn
    from torch.nn import functional as fn

    if model == "control":
        return ct.build("transformer", width, seed)
    if model == "lstm":
        net = ct.build("lstm", width, seed)
        if frozen_ih:
            net.cell.weight_ih_l0.requires_grad_(False)
        return net

    torch.manual_seed(int(seed))
    w = width
    if not brain_heads and w % heads:
        raise ValueError("the width must divide into the heads")
    dk = w if brain_heads else w // heads
    LAYERS = 2

    def freeze(module: Any) -> None:
        for p in module.parameters():
            p.requires_grad_(False)

    class Attention(nn.Module):
        def __init__(self, fixed: bool) -> None:
            super().__init__()
            if brain_heads:  # one key and one value map, an address map per head
                self.q: Any = nn.ModuleList([nn.Linear(w, w) for _ in range(heads)])
            else:
                self.q = nn.Linear(w, w)
            self.k: Any = nn.Linear(w, w)
            self.v: Any = None
            if not raw_values:
                self.v = nn.Linear(w, w * heads) if no_mix and not brain_heads else nn.Linear(w, w)
            mixed = not (raw_values or no_mix or brain_heads)
            self.o: Any = nn.Linear(w, w) if mixed else None
            learned_read = raw_values and not copy_only
            self.r: Any = nn.Linear(gr.INPUTS, w) if learned_read else None  # recall synapses
            self.c: Any = nn.Linear(w, w, bias=False) if recurrent else None  # the cortex before
            if frozen_kv or detach_kv or fixed:
                freeze(self.k)
                if self.v is not None:
                    freeze(self.v)

        @property
        def dv(self) -> int:
            if raw_values:
                return gr.INPUTS
            return w if (no_mix or brain_heads) else dk

        def keys_values(self, u: Any, x: Any) -> tuple[Any, Any]:
            k = self.k(u)
            v = x if raw_values else self.v(u)
            if detach_kv:
                k, v = k.detach(), v.detach()
            return k, v

        def queries(self, u: Any, state: Any = None) -> Any:
            """``(batch, heads, positions, dk)``; ``state`` adds the cortex of the moment
            before, the recurrent address."""
            b, n, _ = u.shape
            if brain_heads:
                q = torch.stack([lin(u) for lin in self.q], dim=1)
            else:
                q = self.q(u).view(b, n, heads, dk).transpose(1, 2)
            if state is not None:
                q = q + self.c(state).view(b, -1, heads, dk).transpose(1, 2)
            return q

        def attend(self, q: Any, k: Any, v: Any, mask: Any) -> tuple[Any, Any, Any]:
            """The reads: what enters the residual, what is copied to the logits, and the
            attention ``(batch, heads, positions, slots)``."""
            b, _, n, _ = q.shape
            m = k.shape[1]
            kh = k[:, None] if brain_heads else k.view(b, m, heads, dk).transpose(1, 2)
            scores = q @ kh.transpose(-1, -2) / math.sqrt(dk)
            if mask is not None:
                scores = scores + mask
            p = fn.softmax(scores, dim=-1)
            if brain_heads and heads > 1:
                p = compete(p)
            if raw_values or brain_heads:
                vh = v[:, None]
            elif no_mix:
                vh = v.view(b, m, heads, w).transpose(1, 2)
            else:
                vh = v.view(b, m, heads, dk).transpose(1, 2)
            read = p @ vh
            if raw_values:
                raw = read.sum(dim=1)
                add = 0.0 if self.r is None else self.r(raw)
                return add, None if no_copy else raw[..., : gr.ACTIONS], p
            if self.o is None:
                return read.sum(dim=1), None, p
            return self.o(read.transpose(1, 2).reshape(b, n, w)), None, p

    def compete(p: Any) -> Any:
        """Each head takes of every slot at most what the heads before left, its own
        attention scaled up where the slots allow (a water-filling, 24 bisection steps)."""
        rem = torch.ones_like(p[:, 0])
        out = [p[:, 0]]
        for h in range(1, p.shape[1]):
            rem = (rem - out[-1]).clamp(min=0.0)
            ph = p[:, h]
            target = rem.sum(dim=-1, keepdim=True).clamp(max=1.0)
            lo, hi = torch.ones_like(target), torch.full_like(target, 1e4)
            for _ in range(24):
                mid = (lo + hi) / 2
                mass = torch.minimum(mid * ph, rem).sum(dim=-1, keepdim=True)
                short = mass < target
                lo, hi = torch.where(short, mid, lo), torch.where(short, hi, mid)
            out.append(torch.minimum(hi * ph, rem))
        return torch.stack(out, dim=1)

    class Block(nn.Module):
        def __init__(self, attention: bool, fixed: bool) -> None:
            super().__init__()
            self.norm1: Any = nn.LayerNorm(w) if attention else None
            self.attn: Any = Attention(fixed) if attention else None
            self.norm2 = nn.LayerNorm(w)
            self.mlp = nn.Sequential(nn.Linear(w, 2 * w), nn.ReLU(), nn.Linear(2 * w, w))

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Linear(gr.INPUTS, w)
            self.layers = nn.ModuleList([Block(i == 0 or not single, i in raw_kv)
                                         for i in range(LAYERS)])
            self.out = nn.Linear(w, gr.ACTIONS)
            position = torch.arange(ct.MAX_LENGTH)[:, None]
            scale = torch.exp(torch.arange(0, w, 2) * (-math.log(10000.0) / w))
            table = torch.zeros(ct.MAX_LENGTH, w)
            table[:, 0::2] = torch.sin(position * scale)
            table[:, 1::2] = torch.cos(position * scale)[:, : w // 2]
            self.register_buffer("position", table)
            self.attention: list[Any] = []  # the last forward's attention per layer
            self.code: Any = None
            if raw_kv:  # drawn last: the rest of the net is as it is without it
                self.code = nn.Linear(gr.INPUTS, w, bias=False)
                nn.init.normal_(self.code.weight, std=math.sqrt(0.5))
                freeze(self.code)

        def forward(self, x: Any) -> Any:
            n = x.shape[1]
            mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
            table: Any = self.position
            e = self.inp(x) + table[:n]
            h = e
            code = None if self.code is None else self.code(x) + table[:n]
            copy: Any = 0.0
            self.attention = []
            block: Any
            for i, block in enumerate(self.layers):
                fixed = code if i in raw_kv else None
                if block.attn is not None and recurrent:  # the block's MLP runs inside
                    h, to_logits = self.recur(block, h, x, fixed)
                    copy = copy + to_logits
                    continue
                if block.attn is not None:
                    u = block.norm1(h)
                    plain = block.norm1(e) if embed_q or embed_kv else u
                    source = fixed if fixed is not None else plain if embed_kv else u
                    k, v = block.attn.keys_values(source, x)
                    q = block.attn.queries(plain if embed_q else u)
                    add, to_logits, p = block.attn.attend(q, k, v, mask)
                    h = h + add
                    copy = copy + (0.0 if to_logits is None else to_logits)
                    self.attention.append(p.detach())
                h = h + block.mlp(block.norm2(h))
            return self.out(h) + copy

        def recur(self, block: Any, h: Any, x: Any, fixed: Any = None) -> tuple[Any, Any]:
            """The positions in turn: each query reads the layer's output at the position
            before (the cortex of the moment before), as the recall memory's address does."""
            b, n, _ = h.shape
            u = block.norm1(h)
            k, v = block.attn.keys_values(u if fixed is None else fixed, x)
            q0 = block.attn.queries(u)
            state = torch.zeros(b, 1, w)
            outs, copies, ps = [], [], []
            for t in range(n):
                q = q0[:, :, t: t + 1] + block.attn.c(state).view(b, 1, heads, dk).transpose(1, 2)
                add, to_logits, p = block.attn.attend(q, k[:, : t + 1], v[:, : t + 1], None)
                ht = h[:, t: t + 1] + add
                ht = ht + block.mlp(block.norm2(ht))
                state = ht
                outs.append(ht)
                copies.append(to_logits)
                ps.append(p[:, :, 0])
            self.attention.append(ps)
            stacked = torch.cat(outs, dim=1)
            return stacked, (torch.cat(copies, dim=1) if raw_values and not no_copy else 0.0)

        def macs(self, length: int) -> int:
            per_token = sum(int(m.weight.numel()) for m in self.modules()
                            if isinstance(m, nn.Linear))
            pairs = length * (length + 1) // 2
            blocks: list[Any] = list(self.layers)
            attention = sum((heads * dk + heads * block.attn.dv) * pairs
                            for block in blocks if block.attn is not None)
            return int(length * per_token + attention)

    return Transformer()


def count(net: Any, trainable: bool = True) -> int:
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad or not trainable))


def make(row: str, width: int, seed: int) -> Any:
    settings = dict(ROWS[row])
    model = settings.pop("model", "transformer")
    return build(model, width, seed, **settings)


def matched_width(row: str, target: int) -> int:
    """The width whose learned parameter count is nearest to ``target``."""
    best = None
    for width in range(4, 257, 2):
        n = count(make(row, width, 0))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (width, n)
    assert best is not None
    return best[0]


class Ablated(ct.Learner):
    """The control's learner on an ablated network; Adam over its learned parameters."""

    def __init__(self, row: str, width: int, seed: int, rate: float) -> None:
        import torch

        self.torch = torch
        self.net = make(row, width, seed)
        self.opt = torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad],
                                    lr=rate)
        self.work = {"updates": 0, "tokens": 0, "macs": 0, "seconds": 0.0}


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life: ``row``, ``mode``, ``founder``, ``phases`` as the controls'; ``init_seed`` draws
    another initialization on the founder's split, lessons and panels (default: the founder)."""
    import torch

    torch.set_num_threads(1)
    row_name, mode, founder = spec["row"], spec["mode"], int(spec["founder"])
    phases = [(str(g), int(n)) for g, n in spec.get("phases", [["g1", 4000]])]
    target = int(spec.get("parameters") or FOUNDER_PARAMETERS)
    width = int(spec.get("width") or matched_width(row_name, target))
    started = time.monotonic()
    init_seed = int(spec.get("init_seed", founder))
    learner = Ablated(row_name, width, init_seed, ct.RATES[mode])
    split = gr.Split.of(founder)
    tests = gr.panels(founder, int(spec.get("panel_batches", 4)))
    names = tuple(spec.get("panels") or PANELS)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    row: dict[str, Any] = {
        "row": row_name, "settings": ROWS[row_name], "mode": mode, "founder": founder,
        "init_seed": init_seed, "width": width, "parameters": count(learner.net),
        "parameters_total": count(learner.net, trainable=False), "brain_parameters": target,
        "phases": [list(p) for p in phases], "rate": ct.RATES[mode], "readings": [],
        "torch": torch.__version__,
    }
    for index, (grammar, n) in enumerate(phases):
        batches = [gr.train_batch(rng, split, grammar) for _ in range(n)]
        before = dict(learner.work)
        losses: list[float] = []
        if mode == "online":
            losses = [learner.update(b) for b in batches]
            epochs = 1
        else:
            order = np.random.default_rng(np.random.SeedSequence([founder, 99, index]))
            best, stale, epochs = math.inf, 0, 0
            while epochs < ct.EPOCHS and stale < ct.PATIENCE:
                epoch = [learner.update(batches[int(i)]) for i in order.permutation(n)]
                epochs += 1
                mean = float(np.mean(epoch))
                losses.append(mean)
                stale = 0 if mean < best - 1e-4 else stale + 1
                best = min(best, mean)
        work = {k: learner.work[k] - before[k] for k in learner.work}
        row["readings"].append({
            "phase": index, "grammar": grammar, "batches": n, "epochs": epochs,
            "loss_last": float(np.mean(losses[-50:])) if mode == "online" else losses[-1],
            "work": work, "panels": ct.reading(learner, tests, names)})
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


# ------------------------------------------------------------------------------- crossing


def crossing(rng: np.random.Generator, n: int = gr.BATCH) -> list[gr.Row]:
    """#228's non-additive addressing in the chamber's words: a context (an open bracket
    ``X_c``), two facts heard as words (``A near V0 end``, ``B near V1 end``, the subjects two
    nouns, the values any noun words), a question (a close bracket ``Q_k``); at ``GO`` the
    answer is the value of the fact ``c xor k``, then ``end``. A separate small panel: the
    chamber's panels never ask it. A reader that ignores the crossing says 9/16 right (the two
    values equal 1/8 of the time, the right fact half of the rest). Episodes are drawn from
    3,072, so a life's lessons cover them: the panel reads whether the crossing can be learned,
    not whether it generalizes."""
    rows = []
    for _ in range(n):
        c, k = int(rng.integers(2)), int(rng.integers(2))
        a, b = (int(i) for i in rng.permutation(gr.NOUNS)[:2])
        values = [int(v) for v in rng.integers(2 * gr.NOUNS, size=2)]
        pre = [gr.heard(gr.OPENS[c])]
        for subject, value in zip((a, b), values, strict=True):
            pre += [gr.heard(subject), gr.heard(gr.NEAR), gr.heard(value), gr.heard(gr.END)]
        pre.append(gr.heard(gr.MATCH[gr.OPENS[k]]))
        rows.append(gr.Row("crossing", pre, [values[c ^ k], gr.END], {"c": c, "k": k}))
    return rows


def run_crossing(spec: dict[str, Any]) -> dict[str, Any]:
    """One online life on the crossing alone: ``row``, ``founder``, ``batches`` (3,000 of 16),
    read on four fresh batches, exact and per context-question pair."""
    import torch

    torch.set_num_threads(1)
    row_name, founder = spec["row"], int(spec["founder"])
    n = int(spec.get("batches", 3000))
    target = int(spec.get("parameters") or FOUNDER_PARAMETERS)
    width = int(spec.get("width") or matched_width(row_name, target))
    started = time.monotonic()
    init_seed = int(spec.get("init_seed", founder))
    learner = Ablated(row_name, width, init_seed, ct.RATES["online"])
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"], 228]))
    losses = [learner.update(crossing(rng)) for _ in range(n)]
    test = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["test"], 228]))
    batches = [crossing(test) for _ in range(int(spec.get("panel_batches", 4)))]
    panel = ct.reading(learner, {"crossing": batches}, ("crossing",))["crossing"]
    learner.net.eval()
    pairs: dict[str, list[float]] = {}
    for batch in batches:
        for r, s in zip(batch, learner.speak(batch), strict=True):
            pairs.setdefault(f"c{r.meta['c']}k{r.meta['k']}", []).append(float(s == r.target))
    learner.net.train()
    panel["conditions"] = {key: round(float(np.mean(v)), 4) for key, v in sorted(pairs.items())}
    return {
        "row": row_name, "settings": ROWS[row_name], "mode": "crossing", "founder": founder,
        "init_seed": init_seed, "width": width, "parameters": count(learner.net),
        "parameters_total": count(learner.net, trainable=False), "brain_parameters": target,
        "phases": [["crossing", n]], "rate": ct.RATES["online"], "torch": torch.__version__,
        "readings": [{"phase": 0, "grammar": "crossing", "batches": n, "epochs": 1,
                      "loss_last": float(np.mean(losses[-50:])), "work": dict(learner.work),
                      "panels": {"crossing": panel}}],
        "seconds": round(time.monotonic() - started, 1),
    }


def run_crossing_brain(spec: dict[str, Any]) -> dict[str, Any]:
    """The recipe brain on the crossing: the chamber's life (a lesson on each word from ``GO``)
    on the same batches as the rows', read on the same panel."""
    import warnings

    warnings.simplefilter("ignore")
    founder, n = int(spec["founder"]), int(spec.get("batches", 3000))
    started = time.monotonic()
    brain = gr.make_brain("recipe", founder, module=int(spec.get("module", 64)))
    work = gr.new_work()
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"], 228]))
    for _ in range(n):
        gr.speak(brain, crossing(rng), teach=True, work=work)
    work["seconds"] = round(time.monotonic() - started, 1)
    test = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["test"], 228]))
    batches = [crossing(test) for _ in range(int(spec.get("panel_batches", 4)))]
    panel = gr.read_panel(brain, batches)
    pairs: dict[str, list[float]] = {}
    fork = copy.deepcopy(brain)
    for batch in batches:
        for r, s in zip(batch, gr.speak(fork, batch, teach=False), strict=True):
            pairs.setdefault(f"c{r.meta['c']}k{r.meta['k']}", []).append(float(s == r.target))
    panel["conditions"] = {key: round(float(np.mean(v)), 4) for key, v in sorted(pairs.items())}
    parameters = int(brain.parameters())
    return {
        "row": "recipe", "settings": gr.ARMS["recipe"], "mode": "crossing", "founder": founder,
        "init_seed": founder, "width": int(spec.get("module", 64)), "parameters": parameters,
        "parameters_total": parameters, "brain_parameters": parameters,
        "phases": [["crossing", n]],
        "readings": [{"phase": 0, "grammar": "crossing", "batches": n, "epochs": 1,
                      "loss_last": None, "work": work, "panels": {"crossing": panel}}],
        "seconds": round(time.monotonic() - started, 1),
    }


def _crossing_life(spec: dict[str, Any]) -> dict[str, Any]:
    return run_crossing_brain(spec) if spec["row"] == "recipe" else run_crossing(spec)


# ---------------------------------------------------------------------------------- table

CROSSING = (("crossing", "exact"), ("crossing", "conditions.c0k0"), ("crossing", "conditions.c0k1"),
            ("crossing", "conditions.c1k0"), ("crossing", "conditions.c1k1"),
            ("crossing", "next_word"))
CROSSING_HEADERS = ("exact", "c0 k0", "c0 k1", "c1 k0", "c1 k1", "next word")
COLUMNS = (
    ("clause", "exact"), ("clause", "conditions.L0"), ("clause", "conditions.L1"),
    ("clause", "conditions.L2"), ("clause_held", "exact"), ("clause_held", "agree"),
    ("clause_long", "exact"), ("roles", "both"), ("dyck", "exact"), ("dyck_deep", "exact"),
    ("path", "exact"), ("path", "conditions.hops1"), ("path", "conditions.hops2"),
    ("path_long", "exact"), ("clause", "next_word"),
)
HEADERS = ("clause", "L0", "L1", "L2", "held", "held agree", "long", "roles both", "dyck",
           "dyck deep", "path", "1 hop", "2 hops", "path long", "next word")


def _get(panels: dict[str, Any], panel: str, key: str) -> float | None:
    node: Any = panels.get(panel)
    for part in key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return None if node is None else float(node)


def _fmt(values: list[float | None]) -> str:
    kept = [v for v in values if v is not None]
    if not kept:
        return ""
    mean = f"{np.mean(kept):.2f}".lstrip("0") or "0"
    if len(kept) < 2:
        return mean
    lo, hi = (f"{v:.2f}".lstrip("0") for v in (min(kept), max(kept)))
    return mean if lo == hi else f"{mean} ({lo}-{hi})"


def table(path: Path, rows: tuple[str, ...] = (), per_founder: bool = False) -> str:
    """Markdown: every row and mode, mean over founders (least to greatest in brackets)."""
    lives = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    named = list(ROWS) + sorted({life["row"] for life in lives} - set(ROWS))
    order = [r for r in named if not rows or r in rows]
    modes = ("crossing",) if any(life["mode"] == "crossing" for life in lives) else ct.MODES
    columns, headers = (CROSSING, CROSSING_HEADERS) if "crossing" in modes else (COLUMNS, HEADERS)
    lines = ["| row | mode | width | learned (all) | founders | epochs | seconds | "
             + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in range(7 + len(headers)))
             + " |"]
    for name in order:
        for mode in modes:
            group = sorted((life for life in lives if life["row"] == name and life["mode"] == mode),
                           key=lambda life: life["founder"])
            if not group:
                continue
            sets = [[life] for life in group] if per_founder else [group]
            for lives_ in sets:
                founders = ",".join(str(life["founder"]) for life in lives_)
                panels = [life["readings"][-1]["panels"] for life in lives_]
                cells = [_fmt([_get(p, panel, key) for p in panels]) for panel, key in columns]
                epochs = _fmt([float(life["readings"][-1]["epochs"]) for life in lives_])
                seconds = int(np.mean([life["seconds"] for life in lives_]))
                lines.append(f"| {name} | {mode} | {lives_[0]['width']} | "
                             f"{lives_[0]['parameters']:,} ({lives_[0]['parameters_total']:,}) | "
                             f"{founders} | {epochs} | {seconds} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def spread(paths: list[Path], panel: str = "clause_held", bar: float = 0.95) -> str:
    """Markdown: the panel over every life of a row and mode (founders and initializations):
    its bimodal failures read as the share of lives at or above ``bar``."""
    lives = [json.loads(line) for path in paths for line in path.read_text().splitlines()
             if line.strip()]
    lines = [f"| row | mode | lives | at {bar:.2f} or above | mean | least | exact, per life |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in list(ROWS) + sorted({life["row"] for life in lives} - set(ROWS)):
        for mode in ct.MODES:
            group = [life for life in lives if life["row"] == name and life["mode"] == mode]
            group.sort(key=lambda life: (life["founder"], life.get("init_seed", life["founder"]),
                                         life.get("replica", 0)))
            exact = [_get(life["readings"][0]["panels"], panel, "exact") for life in group]
            kept = [v for v in exact if v is not None]
            if not kept:
                continue
            at = sum(v >= bar for v in kept)
            lines.append(f"| {name} | {mode} | {len(kept)} | {at} of {len(kept)} | "
                         f"{np.mean(kept):.2f} | {min(kept):.2f} | "
                         + " ".join(f"{v:.2f}" for v in kept) + " |")
    return "\n".join(lines)


def _key(spec: dict[str, Any]) -> tuple[str, str, int, int]:
    founder = int(spec["founder"])
    return (spec["row"], spec.get("mode", "crossing"), founder,
            int(spec.get("init_seed", founder)))


def pending(specs: list[dict[str, Any]], out: Path) -> list[dict[str, Any]]:
    """The specifications whose lives ``out`` does not hold yet."""
    done = {_key(json.loads(line)) for line in out.read_text().splitlines()
            if line.strip()} if out.exists() else set()
    return [spec for spec in specs if _key(spec) not in done]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rows", nargs="*", default=list(SINGLE),
                        choices=[*ROWS, "recipe"])
    parser.add_argument("--modes", nargs="*", default=["online"], choices=ct.MODES)
    parser.add_argument("--founders", nargs="*", type=int, default=list(gr.FOUNDERS))
    parser.add_argument("--phases", type=json.loads, default=[["g1", 4000]])
    parser.add_argument("--parameters", type=int, default=FOUNDER_PARAMETERS,
                        help="the brain's parameter count to match")
    parser.add_argument("--crossing", action="store_true",
                        help="the crossing panel alone (online), instead of the chamber")
    parser.add_argument("--crossing-batches", type=int, default=3000)
    parser.add_argument("--init-seeds", nargs="*", type=int, default=[],
                        help="initializations per founder, -1 for the founder's own, first "
                             "over every row (default: the founder's own alone)")
    parser.add_argument("--resume", action="store_true",
                        help="skip the lives the results file already holds")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--table", type=Path, help="print the table of this results file")
    parser.add_argument("--per-founder", action="store_true")
    parser.add_argument("--spread", nargs="*", type=Path, default=[],
                        help="print the held-out spread over the lives of these files")
    args = parser.parse_args(argv)
    if args.spread:
        print(spread(args.spread))
        return 0
    if args.table is not None:
        print(table(args.table, tuple(args.rows) if args.rows != list(SINGLE) else (),
                    args.per_founder))
        return 0
    if args.out is None:
        parser.error("--out is required")
    seeds: list[dict[str, Any]] = [{} if s < 0 else {"init_seed": s}
                                   for s in args.init_seeds] or [{}]
    if args.crossing:
        specs = [{"row": r, "founder": f, "parameters": args.parameters,
                  "batches": args.crossing_batches, **seed}
                 for seed in seeds for r in args.rows for f in args.founders]
    else:
        specs = [{"row": r, "mode": mode, "founder": f, "phases": args.phases,
                  "parameters": args.parameters, **seed}
                 for seed in seeds for mode in args.modes for r in args.rows
                 for f in args.founders]
    if args.resume:
        specs = pending(specs, args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    life = _crossing_life if args.crossing else run_life
    if not specs:
        return 0
    with args.out.open("a") as sink:
        results: Any = map(life, specs)
        if args.workers > 1:
            pool = get_context("spawn").Pool(min(args.workers, len(specs)))
            results = pool.imap_unordered(life, specs, chunksize=1)
        for result in results:
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            panels = result["readings"][-1]["panels"]
            said = " ".join(f"{name} {panel['exact']:.2f}" for name, panel in panels.items()
                            if name in ("clause", "clause_held", "dyck", "path", "crossing"))
            print(f"{result['row']} {result['mode']} f{result['founder']} width {result['width']}"
                  f" learned {result['parameters']} {said} {result['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
