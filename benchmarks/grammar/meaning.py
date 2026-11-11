"""Where a sentence's meaning lives (cadence#286): the grammar chamber read for a held meaning.

The chamber gives the brain a meaning as concept moments and asks for the sentence at ``GO``.
This instrument asks where the brain keeps that meaning between hearing it and saying it: in
the recall memory's slots, which the read looks up word by word, or in the state the brain
carries from one settle to the next (the settled association activity and the working trace).

``where`` is the diagnosis, with no library change. One life per founder of the default
composition (``Brain.compose`` with the recall memory) is taught the first grammar, and then:

- **Whole sentences under an erasure at ``GO``** (``CONDITIONS``). ``state`` empties every recall
  slot written before ``GO`` (keys, values and nexts set to zero), so the sentence can come from
  the carried state alone; ``silent`` holds the read's drive at zero from ``GO`` to ``end``;
  ``no_trace`` empties the working trace at ``GO``, so the sentence comes from the read alone;
  ``neither`` does both, the floor. Paths also read with the start's slot alone erased
  (``start``) and with the facts' slots alone erased (``facts``).
- **Probes** (``probe``). A linear read-out per component of the meaning (the agent's noun and
  number, the verb, the patient, the count and content of the distractors, the bracket stack, a
  path's start and first hop), fitted offline on one part of fresh sentences and scored on the
  other, from three places at ``GO``, at the verb and at the last word: ``carried`` (the
  working trace entering the settle, which is the association's settled activity of the moment
  before), ``settled`` (the association's activity after the settle) and ``read`` (what the
  recall read holds on its ports). The probes read arrays and change nothing.
- **The controls' states.** The chamber's LSTM and transformer at the brain's parameter count,
  one online pass on the same sentences, read on the same panels and probed at the same
  moments. The LSTM has no attention, so its state is a held meaning by construction: it is
  the reference. The LSTM with its state emptied at ``GO`` and the transformer with the
  meaning's positions masked from ``GO`` on are their floors.

    python benchmarks/grammar/meaning.py where --founders 0 1 2 --out DIR/where.jsonl --workers 3
    python benchmarks/grammar/meaning.py read DIR/where.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import grammar as gr  # noqa: E402

FOUNDERS = (0, 1, 2, 3, 4, 5)  # development
BATCHES = 4000
SEEDS = {"probe": 55}
WHOLE = ("clause", "clause_held", "roles", "dyck", "path")  # whole sentences at trained shapes
LONG = ("clause_long", "dyck_deep", "path_long")
CONDITIONS = ("intact", "state", "silent", "no_trace", "neither")
PATH_CONDITIONS = ("start", "facts")
PLACES = ("carried", "settled", "read")
PROBE_BATCHES = {"clause": 40, "dyck": 90, "path": 60}  # per shape: clause per L, path per hops
HELD_SHARE = 0.3  # of the probe sentences, scored and never fitted


def make_brain(founder: int, settings: dict[str, Any] | None = None, module: int = 64,
               actions: int = gr.ACTIONS) -> Any:
    """The default composition: what ``Brain.compose`` gives a new brain with the recall
    memory, nothing named but the backend and ``settings``. ``"parts": True`` among the
    settings says each word by its parts (``parts_brain``)."""
    import cadence as cd

    settings = dict(settings or {})
    if settings.pop("tied_message", False):
        # one set of synapses from every block of the message onto the association
        groups = len(settings.get("message_roles", ())) + 1 if settings.get(
            "message_roles") else 1
        blocks = groups * int(settings.get("message_phases", 4))
        settings["tied"] = {("message", "association"): (blocks, 1)}
    if settings.pop("parts", False):
        brain = cd.Brain.compose(gr.INPUTS, sum(PART_SLOTS), modules=(module,),
                                 seed=int(founder), slots=PART_SLOTS,
                                 **{"backend": "cpu", "recall": True, **settings})
        return parts_brain(brain)
    return cd.Brain.compose(gr.INPUTS, actions, modules=(module,), seed=int(founder),
                            **{"backend": "cpu", "recall": True, **settings})


# --------------------------------------------------------------------- a word by its parts

# A word is said as two motor slots that settle together: its stem (the four nouns, the two
# verbs, near, the four brackets, end) and its form (singular, plural, none). A noun's form is its
# number and a verb's the number its agent calls for; every lesson that says a word teaches both
# slots, so a stem and a form no lesson said together are said from parts each lesson taught.
PART_SLOTS = (12, 3)
SINGULAR, PLURAL, NO_FORM = 0, 1, 2


def word_parts(word: int) -> tuple[int, int]:
    """A word's stem and form."""
    w = int(word)
    if w < 2 * gr.NOUNS:
        return w % gr.NOUNS, PLURAL if w >= gr.NOUNS else SINGULAR
    if w < 12:  # sees see chases chase: the stem the verb, the form the agent's number
        return gr.NOUNS + (w - 8) // 2, (w - 8) % 2
    return gr.NOUNS + 2 + (w - 12), NO_FORM


def parts_word(stem: int, form: int) -> int:
    """The word a stem and a form say; a stem without a number says its one word."""
    stem, form = int(stem), int(form)
    if stem < gr.NOUNS:
        return stem + (gr.NOUNS if form == PLURAL else 0)
    if stem < gr.NOUNS + 2:
        return 8 + 2 * (stem - gr.NOUNS) + (1 if form == PLURAL else 0)
    return 12 + stem - gr.NOUNS - 2


def parts_brain(brain: Any) -> Any:
    """``brain`` (composed with ``PART_SLOTS``) saying and taught words: ``act`` returns the
    word its two slots say, and ``lesson`` teaches a word's stem and form."""

    class PartsBrain(type(brain)):  # type: ignore[misc]
        def act(self, observations: Any, **kwargs: Any) -> np.ndarray:
            chosen = np.asarray(super().act(observations, **kwargs)).reshape(-1, 2)
            return np.array([parts_word(a, b) for a, b in chosen], dtype=np.int64)

        def lesson(self, observations: Any, labels: Any = None, **kwargs: Any) -> Any:
            if labels is not None:  # a word, or ``-(stem + 1)`` for a stem without a form
                labels = np.array([word_parts(w) if w >= 0 else (-int(w) - 1, NO_FORM)
                                   for w in np.asarray(labels).ravel()], dtype=np.int64)
            return super().lesson(observations, labels, **kwargs)

    PartsBrain.__name__ = PartsBrain.__qualname__ = "PartsBrain"
    brain.__class__ = PartsBrain
    return brain


# ------------------------------------------------------------------------------- erasures


def _erase_slots(brain: Any, keep: Any) -> None:
    """Empty the recall slots whose age ``keep`` refuses: their keys and values, their own
    nexts, and the next of each slot before an emptied one (which holds the emptied value)."""
    recall = brain.recall
    if recall is None or not recall.slots:
        return
    gone = ~np.asarray([bool(keep(int(a))) for a in recall.ages()])  # the oldest slot first
    before = np.concatenate([gone[1:], [False]])
    recall.keys = np.where(gone[None, :, None], 0.0, recall.keys)
    recall.values = np.where(gone[None, :, None], 0.0, recall.values)
    recall.nexts = np.where((gone | before)[None, :, None], 0.0, recall.nexts)


def intervention(condition: str) -> Any:
    """What happens at ``GO``, before its act, on a whole sentence (``CONDITIONS`` and
    ``PATH_CONDITIONS``)."""

    def act(brain: Any, rows: list[gr.Row]) -> None:
        if condition in ("state", "neither"):
            _erase_slots(brain, lambda age: False)  # every slot written before GO
        if condition == "start":
            _erase_slots(brain, lambda age: age > 0)  # the start is the latest moment heard
        if condition == "facts":
            _erase_slots(brain, lambda age: age == 0)
        if condition == "silent" and brain.recall is not None:
            brain.recall.amplitude = 0.0  # the read's drive, from GO to the sentence's end
        if condition in ("no_trace", "neither") and brain.working_memory is not None:
            brain.working_memory.trace = np.zeros_like(brain.working_memory.trace)

    return None if condition == "intact" else act


def read_condition(brain: Any, batches: list[list[gr.Row]], condition: str,
                   twins: bool = False) -> dict[str, Any]:
    """A panel's whole sentences, closed loop, on a fork of ``brain`` under ``condition``."""
    fork = copy.deepcopy(brain)
    amplitude = None if fork.recall is None else fork.recall.amplitude

    def speaker(rows: list[gr.Row]) -> list[list[int]]:
        if amplitude is not None:
            fork.recall.amplitude = amplitude  # a silenced read is heard again while listening
        return gr.speak(fork, rows, teach=False, intervene=intervention(condition))

    out = gr.read_speaker(speaker, None, batches, twins=twins)
    return {k: v for k, v in out.items() if k != "examples"} | {"examples": out["examples"][:2]}


def sentences(brain: Any, tests: dict[str, list[list[gr.Row]]]) -> dict[str, Any]:
    """Every whole-sentence panel under every erasure, and the chamber's context twins with
    their own erasures (``grammar.reading``)."""
    out: dict[str, Any] = {}
    for condition in CONDITIONS:
        out[condition] = {name: read_condition(brain, tests[name], condition, name in gr.TWINS)
                          for name in WHOLE + LONG}
    for condition in PATH_CONDITIONS:
        out[condition] = {name: read_condition(brain, tests[name], condition)
                          for name in ("path", "path_long")}
    out["context"] = gr.reading(brain, tests, gr.CONTEXT_PANELS, components=False)
    return out


# --------------------------------------------------------------------------------- probes


def components(row: gr.Row) -> dict[str, int]:
    """The parts of a row's meaning a probe reads, each a class index."""
    m = row.meta
    if row.family == "clause":
        out = {"agent_noun": m["agent"][0], "agent_number": m["agent"][1], "verb": m["verb"],
               "patient_noun": m["patient"][0], "patient_number": m["patient"][1],
               "distractors": m["L"],
               # the verb's said form: its stem and the agent's number together
               "verb_form": 2 * m["verb"] + m["agent"][1]}
        for i, (n, plural) in enumerate(m["mods"][:2], start=1):
            out[f"distractor{i}_noun"] = n
            out[f"distractor{i}_number"] = plural
        return out
    if row.family == "dyck":
        stack: list[int] = []
        for x in row.pre:
            if x[gr.C["CLOSE"]]:
                stack.pop()
            else:
                stack.append(int(x[gr.C["OPEN_SQUARE"]]))
        out = {"open": len(stack) - 1, "innermost": stack[-1], "outermost": stack[0]}
        if len(stack) > 1:
            out["second_innermost"] = stack[-2]
        return out
    chain = m["chain"]
    return {"start": chain[0], "first_hop": chain[1], "hops": m["hops"] - 1}


def positions(row: gr.Row) -> dict[str, int]:
    """The said words a probe is read at, by name: the index into ``target`` of the settle that
    says it. ``go`` is the settle on ``GO``, which says the first word."""
    last = len(row.target) - 1
    if row.family == "clause":
        return {"go": 0, "verb": next(i for i, w in enumerate(row.target) if 8 <= w <= 11),
                "end": last}
    return {"go": 0, "end": last}


def probe_batches(founder: int, scale: float = 1.0) -> dict[str, list[list[gr.Row]]]:
    """Fresh sentences for the probes, any combination (held-out ones at their natural share),
    at trained shapes; ``scale`` shrinks the counts for a test."""
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["probe"]]))

    def count(family: str) -> int:
        return max(2, int(round(PROBE_BATCHES[family] * scale)))

    out: dict[str, list[list[gr.Row]]] = {"clause": [], "dyck": [], "path": []}
    for L in gr.TRAIN_L:
        out["clause"] += [[gr.clause(rng, L) for _ in range(gr.BATCH)]
                          for _ in range(count("clause"))]
    for _ in range(count("dyck")):
        pattern = gr.dyck_pattern(rng, int(rng.integers(1, 7)), gr.TRAIN_DEPTH)
        out["dyck"].append([gr.dyck(rng, pattern) for _ in range(gr.BATCH)])
    for hops in gr.TRAIN_HOPS:
        out["path"] += [[gr.path(rng, hops) for _ in range(gr.BATCH)]
                        for _ in range(count("path"))]
    return out


def split_batches(founder: int, per_shape: int = 16) -> dict[str, list[list[gr.Row]]]:
    """Clauses at trained shapes, ``trained`` of the combinations the lessons teach and
    ``held`` of the held-out ones, ``per_shape`` batches of each per shape."""
    split = gr.Split.of(int(founder))
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["probe"], 2]))
    out: dict[str, list[list[gr.Row]]] = {"trained": [], "held": []}
    for L in gr.TRAIN_L:
        for _ in range(per_shape):
            out["trained"].append([gr.trained_clause(rng, split, L, False)
                                   for _ in range(gr.BATCH)])
            rows = []
            while len(rows) < gr.BATCH:
                row = gr.clause(rng, L)
                m = row.meta
                if split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"]):
                    rows.append(row)
            out["held"].append(rows)
    return out


def long_batches(founder: int, per_shape: int = 8) -> list[list[gr.Row]]:
    """Clauses of trained combinations at the longer shapes no lesson shows (``TEST_L``)."""
    split = gr.Split.of(int(founder))
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["probe"], 3]))
    return [[gr.trained_clause(rng, split, L, False) for _ in range(gr.BATCH)]
            for L in gr.TEST_L for _ in range(per_shape)]


def transfer_probes(founder: int, trained: list[tuple[gr.Row, dict[str, Any]]],
                    held: list[tuple[gr.Row, dict[str, Any]]],
                    places: tuple[str, ...]) -> dict[str, Any]:
    """Probes fitted on the trained combinations' states and scored on the held-out ones':
    whether a reading the lessons could set up reads new combinations. Per component, position
    and place, the held-out accuracy and its chance."""
    items = list(trained) + list(held)
    scored = np.arange(len(items)) >= len(trained)
    labels = [components(row) for row, _ in items]
    names = sorted({k for label in labels for k in label})
    spots = sorted({k for _, at in items for k in at})
    table: dict[str, Any] = {}
    for name in names:
        has = np.array([name in label for label in labels])
        for spot in spots:
            there = has & np.array([spot in at for _, at in items])
            if (there & scored).sum() < 20 or (there & ~scored).sum() < 20:
                continue
            y = np.array([labels[i][name] for i in np.flatnonzero(there)])
            rows = np.flatnonzero(there)
            for place in places:
                if any(place not in items[i][1][spot] for i in rows):
                    continue  # a spot that holds no such place (a control's heard moment)
                x = np.stack([items[i][1][spot][place] for i in rows])
                table.setdefault(name, {}).setdefault(spot, {})[place] = fit_probe(
                    x, y, scored[there])
    return table


def capture(brain: Any, rows: list[gr.Row], condition: str = "intact") -> list[dict[str, Any]]:
    """One batch, teacher-forced, on ``brain`` (a fork): for every said word, what each place
    holds at its settle. ``carried`` is the working trace entering the settle, ``settled`` the
    association's activity after it, ``read`` what the recall read holds on its ports at
    amplitude one."""
    recall, trace = brain.recall, brain.working_memory
    act = intervention(condition)
    brain.reset()
    for t in range(len(rows[0].pre)):
        brain.act(np.stack([r.pre[t] for r in rows]), greedy=True)
    association = brain.association_index
    out = []
    for j in range(len(rows[0].target)):
        x = (np.tile(gr.concept("GO"), (len(rows), 1)) if j == 0
             else np.stack([gr.spoken(r.target[j - 1]) for r in rows]))
        if j == 0 and act is not None:
            act(brain, rows)
        carried = (np.array(trace.trace, dtype=float)
                   if trace is not None and len(trace.trace) == len(rows)
                   else np.zeros((len(rows), len(association))))
        action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
        settled = np.asarray(brain.basal_ganglia.state.activation)[:, association]
        if recall is not None and getattr(recall, "drive_gain", False):
            read = np.array(recall.trace, dtype=float)
        elif recall is not None:
            ports = np.concatenate([np.asarray(brain.connectome.populations[name])
                                    for port in recall.ports(recall.heads, recall.next)
                                    for name in port])
            read = np.asarray(brain.basal_ganglia._drive)[:, ports] / max(recall.amplitude, 1e-9)
        else:
            read = np.zeros((len(rows), 1))
        out.append({"carried": carried, "settled": np.array(settled, dtype=float),
                    "read": read, "said": action})
    return out


def fit_probe(x: np.ndarray, y: np.ndarray, held: np.ndarray, steps: int = 300,
              rate: float = 0.05, decay: float = 1e-3) -> dict[str, float]:
    """A linear probe: softmax regression of class ``y`` on features ``x``, fitted on the rows
    ``held`` leaves out and scored on the rows it names. ``chance`` is the share of the fitted
    rows' most frequent class among the held rows."""
    classes = int(y.max()) + 1
    fit_x, fit_y, test_x, test_y = x[~held], y[~held], x[held], y[held]
    if classes < 2 or not len(test_y) or not len(fit_y):
        return {"n": int(len(test_y)), "accuracy": float("nan"), "chance": float("nan")}
    mean, spread = fit_x.mean(axis=0), fit_x.std(axis=0) + 1e-6
    a, b = (fit_x - mean) / spread, (test_x - mean) / spread
    onehot = np.eye(classes)[fit_y]
    w, bias = np.zeros((a.shape[1], classes)), np.zeros(classes)
    moments = [np.zeros_like(w), np.zeros_like(w), np.zeros_like(bias), np.zeros_like(bias)]
    for step in range(1, steps + 1):  # full-batch Adam, deterministic
        logits = a @ w + bias
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        gradient = (a.T @ (p - onehot)) / len(a) + decay * w
        gradient_bias = (p - onehot).mean(axis=0)
        for k, g in ((0, gradient), (2, gradient_bias)):
            moments[k] = 0.9 * moments[k] + 0.1 * g
            moments[k + 1] = 0.999 * moments[k + 1] + 0.001 * g * g
            update = rate * (moments[k] / (1 - 0.9 ** step)) / (
                np.sqrt(moments[k + 1] / (1 - 0.999 ** step)) + 1e-8)
            if k == 0:
                w -= update
            else:
                bias -= update
    majority = int(np.bincount(fit_y, minlength=classes).argmax())
    return {"n": int(len(test_y)),
            "accuracy": round(float((np.argmax(b @ w + bias, axis=1) == test_y).mean()), 4),
            "chance": round(float((test_y == majority).mean()), 4)}


def _held(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, SEEDS["probe"], 1]))
    return rng.random(n) < HELD_SHARE


def probe_states(founder: int, states: dict[str, list[tuple[gr.Row, dict[str, Any]]]],
                 places: tuple[str, ...]) -> dict[str, Any]:
    """Probes over captured states. ``states[family]`` holds, per sentence, the row and a
    mapping from a position's name (``positions``) to ``{place: vector}``. Returns, per family,
    component, position and place, the held-out accuracy and its chance."""
    out: dict[str, Any] = {}
    for family, items in states.items():
        if not items:
            continue
        held = _held(len(items), founder)
        labels = [components(row) for row, _ in items]
        names = sorted({k for label in labels for k in label})
        spots = sorted({k for _, at in items for k in at})
        table: dict[str, Any] = {}
        for name in names:
            has = np.array([name in label for label in labels])
            for spot in spots:
                there = has & np.array([spot in at for _, at in items])
                if there.sum() < 20:
                    continue
                y = np.array([labels[i][name] for i in np.flatnonzero(there)])
                for place in places:
                    if any(place not in items[i][1][spot] for i in np.flatnonzero(there)):
                        continue  # a spot that holds no such place
                    x = np.stack([items[i][1][spot][place] for i in np.flatnonzero(there)])
                    table.setdefault(name, {}).setdefault(spot, {})[place] = fit_probe(
                        x, y, held[there])
        out[family] = table
    return out


def brain_states(brain: Any, batches: dict[str, list[list[gr.Row]]], condition: str = "intact",
                 steps: bool = True) -> dict[str, list[tuple[gr.Row, dict[str, Any]]]]:
    """Every probe sentence's states at its named positions, on a fork; with ``steps`` the
    clauses of two distractors also hold every said word as ``step<j>``."""
    fork = copy.deepcopy(brain)
    amplitude = None if fork.recall is None else fork.recall.amplitude
    out: dict[str, list[tuple[gr.Row, dict[str, Any]]]] = {}
    for family, groups in batches.items():
        items = []
        for rows in groups:
            if amplitude is not None:
                fork.recall.amplitude = amplitude  # a silenced read is heard while listening
            moments = capture(fork, rows, condition)
            for i, row in enumerate(rows):
                at = {name: {place: moments[j][place][i] for place in PLACES}
                      for name, j in positions(row).items()}
                if steps and family == "clause" and row.meta["L"] == 2:
                    for j in range(len(row.target)):
                        at[f"step{j}"] = {place: moments[j][place][i] for place in PLACES}
                items.append((row, at))
        out[family] = items
    return out


# ------------------------------------------------------------------------------- controls


def _controls() -> Any:
    return gr._module("controls", HERE / "controls.py")


def lstm_states(net: Any, rows: list[gr.Row], erase: bool = False) -> np.ndarray:
    """The LSTM's state ``[h, c]`` after every moment, ``(batch, moments, 2 * width)``,
    teacher-forced; with ``erase`` the state is emptied before ``GO``."""
    import torch

    x, _ = _controls().encode(rows)
    state = None
    out = []
    with torch.no_grad():
        for t in range(x.shape[1]):
            if erase and t == len(rows[0].pre):
                state = None
            _, state = net.cell(torch.from_numpy(x[:, t:t + 1]), state)
            out.append(torch.cat([state[0][0], state[1][0]], dim=1).numpy())
    return np.stack(out, axis=1)


def lstm_speak(net: Any, rows: list[gr.Row], erase: bool = False, cap: int = 4) -> list[list[int]]:
    """The LSTM's sentences, closed loop and greedy, stepped one moment at a time so that its
    state can be emptied before ``GO`` (``erase``): the floor of a speaker whose meaning is its
    state."""
    import torch

    n = len(rows)
    said: list[list[int]] = [[] for _ in rows]
    state = None
    with torch.no_grad():
        for t in range(len(rows[0].pre)):
            x = torch.from_numpy(np.stack([r.pre[t] for r in rows]).astype(np.float32))
            _, state = net.cell(x[:, None], state)
        if erase:
            state = None
        done = [False] * n
        x = np.tile(gr.concept("GO"), (n, 1))
        for j in range(max(len(r.target) for r in rows) + cap):
            h, state = net.cell(torch.from_numpy(x.astype(np.float32))[:, None], state)
            action = net.out(h[:, 0]).argmax(dim=1).numpy()
            words = []
            for i, r in enumerate(rows):
                said[i].append(int(action[i]))
                done[i] = done[i] or (j >= r.force and action[i] == gr.END)
                given = j + 1 <= r.force and j < len(r.target)
                words.append(r.target[j] if given else int(action[i]))
            if all(done):
                break
            x = np.stack([gr.spoken(w) for w in words])
    out = []
    for i, r in enumerate(rows):
        words = said[i][r.force:]
        out.append(words[:words.index(gr.END) + 1] if gr.END in words else words)
    return out


def transformer_forward(net: Any, x: Any, go: int | None = None) -> tuple[Any, Any]:
    """The transformer's logits and its last layer's output per position; with ``go`` no
    position from ``go`` on attends to a position before it (the meaning masked)."""
    import torch

    n = x.shape[1]
    mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1)
    if go is not None:
        mask[go:, :go] = float("-inf")
    h = net.body(net.inp(x) + net.position[:n], mask=mask)
    return net.out(h), h


def transformer_speak(net: Any, rows: list[gr.Row], masked: bool = False,
                      cap: int = 4) -> list[list[int]]:
    import torch

    n = len(rows)
    go = len(rows[0].pre) if masked else None
    said: list[list[int]] = [[] for _ in rows]
    done = [False] * n
    moments = [[*r.pre, gr.concept("GO")] for r in rows]
    with torch.no_grad():
        for j in range(max(len(r.target) for r in rows) + cap):
            x = torch.from_numpy(np.stack([np.stack(m) for m in moments]).astype(np.float32))
            action = transformer_forward(net, x, go)[0][:, -1].argmax(dim=1).numpy()
            for i, r in enumerate(rows):
                said[i].append(int(action[i]))
                done[i] = done[i] or (j >= r.force and action[i] == gr.END)
                given = j + 1 <= r.force and j < len(r.target)
                moments[i].append(gr.spoken(r.target[j] if given else int(action[i])))
            if all(done):
                break
    out = []
    for i, r in enumerate(rows):
        words = said[i][r.force:]
        out.append(words[:words.index(gr.END) + 1] if gr.END in words else words)
    return out


def control_states(model: str, net: Any, batches: dict[str, list[list[gr.Row]]],
                   ) -> dict[str, list[tuple[gr.Row, dict[str, Any]]]]:
    """A control's state at the probes' positions. The LSTM's ``carried`` is its state before
    the moment and ``settled`` after it; the transformer has no state between positions, so
    its ``settled`` is its last layer's output at the position and it has no ``carried``."""
    import torch

    out: dict[str, list[tuple[gr.Row, dict[str, Any]]]] = {}
    for family, groups in batches.items():
        items = []
        for rows in groups:
            start = len(rows[0].pre)
            if model == "lstm":
                states = lstm_states(net, rows)
            else:
                x, _ = _controls().encode(rows)
                with torch.no_grad():
                    states = transformer_forward(net, torch.from_numpy(x))[1].numpy()
            for i, row in enumerate(rows):
                at = {}
                for name, j in positions(row).items():
                    at[name] = {"settled": states[i, start + j]}
                    if model == "lstm":
                        at[name]["carried"] = states[i, start + j - 1]
                at["heard"] = {"settled": states[i, start - 1]}  # after the last heard moment
                items.append((row, at))
        out[family] = items
    return out


def control_life(model: str, founder: int, parameters: int, batches: int,
                 tests: dict[str, list[list[gr.Row]]],
                 probes: dict[str, list[list[gr.Row]]] | None) -> dict[str, Any]:
    """One control, one online pass at the brain's exposure: its panels, its floor and its
    probed states."""
    import torch

    torch.set_num_threads(1)
    ct = _controls()
    width = ct.matched_width(model, parameters)
    learner = ct.Learner(model, width, founder, ct.RATES["online"])
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    started = time.monotonic()
    for _ in range(batches):
        learner.update(gr.train_batch(rng, split, "g1"))
    learner.net.eval()
    names = WHOLE + LONG
    row: dict[str, Any] = {
        "model": model, "width": width, "parameters": ct.count(learner.net),
        "seconds": round(time.monotonic() - started, 1), "macs": learner.work["macs"],
        "panels": {"intact": {name: gr.read_speaker(learner.speak, None, tests[name],
                                                    twins=name in gr.TWINS) for name in names}},
    }
    if model == "lstm":
        floor = lambda rows: lstm_speak(learner.net, rows, erase=True)  # noqa: E731
        check = lambda rows: lstm_speak(learner.net, rows)  # noqa: E731
    else:
        floor = lambda rows: transformer_speak(learner.net, rows, masked=True)  # noqa: E731
        check = lambda rows: transformer_speak(learner.net, rows)  # noqa: E731
    row["panels"]["floor"] = {name: gr.read_speaker(floor, None, tests[name],
                                                    twins=name in gr.TWINS) for name in names}
    # the stepped speakers say what the control's own speaker says
    row["stepped_agrees"] = all(check(rows) == learner.speak(rows) for rows in tests["clause"])
    for panels in row["panels"].values():
        for panel in panels.values():
            panel.pop("examples", None)
    if probes is not None:
        places = ("carried", "settled") if model == "lstm" else ("settled",)
        row["probes"] = probe_states(founder, control_states(model, learner.net, probes), places)
    return row


# ---------------------------------------------------------------------------------- lives


def teach(brain: Any, founder: int, batches: int) -> dict[str, float]:
    """The first grammar's lessons, the chamber's stream and order."""
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    work = gr.new_work()
    started = time.monotonic()
    for _ in range(batches):
        gr.speak(brain, gr.train_batch(rng, split, "g1"), teach=True, work=work)
    work["seconds"] = round(time.monotonic() - started, 1)
    return work


def where_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One founder's diagnosis: ``founder``, ``batches``, ``module``, ``settings`` over the
    default composition, ``panel_batches``, ``probe_scale``, ``controls`` (model names) and
    ``save`` (a folder for the taught brain)."""
    warnings.simplefilter("ignore")
    import cadence as cd

    founder = int(spec["founder"])
    batches = int(spec.get("batches", BATCHES))
    started = time.monotonic()
    brain = make_brain(founder, spec.get("settings"), int(spec.get("module", 64)))
    tests = gr.panels(founder, int(spec.get("panel_batches", 4)))
    row: dict[str, Any] = {
        "kind": "where", "founder": founder, "batches": batches, "tag": spec.get("tag"),
        "settings": spec.get("settings", {}), "module": int(spec.get("module", 64)),
        "parameters": int(brain.parameters()), "cadence": cd.__version__,
    }
    row["work"] = teach(brain, founder, batches)
    if spec.get("save"):
        folder = Path(spec["save"])
        folder.mkdir(parents=True, exist_ok=True)
        row["saved"] = str(brain.save(folder / f"default-f{founder}"))
    row["gains"] = dict(brain.carried_gains) if brain.hearing is not None else {}
    row["sentences"] = sentences(brain, tests)
    row["located"] = {name: gr.locate(brain, tests[name]) for name in ("clause", "dyck", "path")}
    scale = float(spec.get("probe_scale", 1.0))
    probes = probe_batches(founder, scale) if scale > 0 else None
    if probes is not None:
        row["probes"] = {
            condition: probe_states(founder, brain_states(brain, probes, condition), PLACES)
            for condition in ("intact", "state")}
    row["controls"] = [control_life(model, founder, row["parameters"], batches, tests, probes)
                       for model in spec.get("controls", ("lstm", "transformer"))]
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


# ---------------------------------------------------------------------------- the message


class Message:
    """A message as the instrument writes it: an emulation of a held region, with no library
    change. Every heard moment's observation is given a fixed sparse code (``units`` units, the
    ``winners`` largest of a random projection drawn from the seed set to one; with
    ``code="and"`` every unit reads ``fan`` inputs at weight one, so the units that read every
    active input win; ``code="separator"`` is the library's code, ``cadence.Message``'s), and
    each block holds the stream's episode in its own way:

    - ``bag``: the codes summed, each at one; ``set``: their union.
    - ``primacy:g``: the ``t``-th moment at ``g ** t``; ``recent:d``: decaying by ``d`` per
      moment.
    - ``pairs``: the codes of each moment joined to the moment before it, summed (which
      moment followed which).
    - ``phase:k``: ``k`` unions of ``units`` each, the ``t``-th moment's code joining the union
      ``t mod k``, so neighbouring moments lie in neighbouring blocks and what a block holds
      is present or absent, with no magnitude to compare.
    - ``slots:k``: the last ``k`` codes, the latest first (an ordered buffer of codes).
    - ``register:k``: the last ``k`` observations as heard, the latest first, ``k`` times the
      observation's width (an ordered buffer with no code, the reference).

    With ``again`` the code also reads, per input, whether the episode held it before.

    The write is fixed, as the working trace's and the heard history's are; what the brain
    makes of the message is learned. With ``hold`` nothing is written from ``GO`` on. A
    ``sham`` drives the same neurons without the stream's meaning: ``"present"`` holds the
    present moment's code alone, ``"shuffled"`` gives each stream another stream's message."""

    def __init__(self, design: dict[str, Any], seed: int) -> None:
        self.units = int(design.get("units", 70))
        self.winners = int(design.get("winners", 5))
        self.blocks = tuple(design.get("blocks", ("bag", "primacy:0.7")))
        defaults = {"bag": 1.0 / 3.0, "pairs": 1.0 / 3.0, "register": 1.0 / 3.0}
        self.scales = tuple(float(v) for v in design.get(
            "scales", [defaults.get(kind.partition(":")[0], 1.0) for kind in self.blocks]))
        self.hold = bool(design.get("hold", False))
        # two shams of the same neurons: the present moment's code alone, and another
        # stream's message
        self.sham = design.get("sham")
        if self.sham not in (None, "present", "shuffled"):
            raise ValueError(f"unknown sham: {self.sham!r}")
        self.kind = str(design.get("code", "sum"))
        self.again = bool(design.get("again", False))
        self.library: Any = None
        if self.kind == "separator":  # the library's code (``cadence.Message``)
            import cadence as cd

            self.library = cd.Message(
                gr.INPUTS, phases=1, units=self.units, winners=self.winners,
                fan_in=2 * int(design.get("fan", 8)), seed=int(seed))
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 286]))
        heard = (2 if self.again else 1) * gr.INPUTS
        self.weights = self._projection(rng, heard, int(design.get("fan", 8)))
        self.pair_weights = self._projection(rng, 2 * gr.INPUTS, 2 * int(design.get("fan", 8)))
        self.parts: list[np.ndarray] = []
        self.before: np.ndarray | None = None
        self.seen: np.ndarray | None = None
        self.told = 0

    def _projection(self, rng: np.random.Generator, inputs: int, fan: int) -> np.ndarray:
        if self.kind in ("sum", "separator"):
            return np.asarray(rng.normal(size=(inputs, self.units)))
        if self.kind != "and":
            raise ValueError(f"unknown code: {self.kind!r}")
        # every unit reads ``fan`` inputs at one, with a small fixed preference to break ties
        w = np.zeros((inputs, self.units))
        for u in range(self.units):
            w[rng.choice(inputs, size=min(fan, inputs), replace=False), u] = 1.0
        return np.asarray(w + 1e-3 * rng.random((inputs, self.units)))

    def _width(self, kind: str) -> int:
        name, _, value = kind.partition(":")
        if name == "register":
            return int(value) * gr.INPUTS
        return int(value) * self.units if name in ("phase", "slots") else self.units

    @property
    def width(self) -> int:
        return sum(self._width(kind) for kind in self.blocks)

    def code(self, x: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """The sparse code of observations: one at each row's ``winners`` largest sums."""
        if self.library is not None and weights is None:
            return np.asarray(self.library.code(np.asarray(x, dtype=float)))
        z = np.asarray(x, dtype=float) @ (self.weights if weights is None else weights)
        out = np.zeros_like(z)
        top = np.argpartition(-z, self.winners - 1, axis=1)[:, :self.winners]
        np.put_along_axis(out, top, 1.0, axis=1)
        return np.asarray(out * (np.abs(x).sum(axis=1, keepdims=True) > 0))

    def reset(self, rows: int) -> None:
        self.parts = [np.zeros((rows, self._width(kind))) for kind in self.blocks]
        self.before = None
        self.seen = np.zeros((rows, gr.INPUTS))
        self.told = 0

    def hear(self, x: np.ndarray, speaking: bool = False) -> None:
        """One heard moment joins every block; with ``hold`` a moment from ``GO`` on joins
        none."""
        if speaking and self.hold:
            return
        if self.sham == "present":
            self.reset(len(x))
        x = np.asarray(x, dtype=float)
        assert self.seen is not None
        c = self.code(np.concatenate([x, x * self.seen], axis=1) if self.again else x)
        self.seen = np.maximum(self.seen, (x != 0).astype(float))
        for i, kind in enumerate(self.blocks):
            name, _, value = kind.partition(":")
            if name == "bag":
                self.parts[i] = self.parts[i] + c
            elif name == "set":
                self.parts[i] = np.maximum(self.parts[i], c)
            elif name == "primacy":
                self.parts[i] = self.parts[i] + float(value) ** self.told * c
            elif name == "recent":
                self.parts[i] = float(value) * self.parts[i] + c
            elif name == "pairs":
                before = np.zeros_like(x) if self.before is None else self.before
                self.parts[i] = self.parts[i] + self.code(
                    np.concatenate([before, x], axis=1), self.pair_weights)
            elif name == "phase":
                at = (self.told % int(value)) * self.units
                self.parts[i][:, at:at + self.units] = np.maximum(
                    self.parts[i][:, at:at + self.units], c)
            elif name == "slots":
                self.parts[i] = np.concatenate([c, self.parts[i][:, :-self.units]], axis=1)
            elif name == "register":
                self.parts[i] = np.concatenate([x, self.parts[i][:, :-gr.INPUTS]], axis=1)
            else:
                raise ValueError(f"unknown block: {kind!r}")
        self.before = x.copy()
        self.told += 1

    def state(self, scaled: bool = True) -> np.ndarray:
        """The message, ``(rows, width)``; ``scaled`` as the region is driven."""
        scales = self.scales if scaled else (1.0,) * len(self.blocks)
        out = np.concatenate([k * part for k, part in zip(scales, self.parts, strict=True)],
                             axis=1)
        return np.roll(out, 1, axis=0) if self.sham == "shuffled" and scaled else out


def carrier_blocks(design: dict[str, Any] | None) -> int:
    """The heard-history blocks of sensory width that carry a message of ``design``."""
    return 0 if design is None else -(-Message(design, 0).width // gr.INPUTS)


def message_brain(founder: int, design: dict[str, Any] | None,
                  settings: dict[str, Any] | None = None, module: int = 64) -> Any:
    """The default composition with the neurons that carry the message: a heard-history
    region the instrument writes (``install``), read through the history's plastic
    projection. Without a design it is ``make_brain``."""
    blocks = carrier_blocks(design)
    extra = {"history_decays": (0.0,) * blocks} if blocks else {}
    return make_brain(founder, {**extra, **(settings or {})}, module)


def install(brain: Any, message: Message | None) -> None:
    """The message onto its carrier, replacing what the history heard."""
    if message is None or brain.history is None:
        return
    state = message.state()
    held = np.zeros((len(state), len(brain.history.glow)))
    held[:, :state.shape[1]] = state
    brain.history.trace = held


def bar_meaning(brain: Any, rows: list[gr.Row]) -> None:
    """The read barred from the meaning's moments: their slots emptied at ``GO``. A clause's
    and a bracket sentence's meaning is every moment heard; a path's is its start, and the
    world's facts stay. ``GO`` and the words given before the continuation stay too."""
    given = rows[0].force  # GO and the given words are the latest slots: they stay
    if rows[0].family == "path":
        _erase_slots(brain, lambda age: age != given)
    else:
        _erase_slots(brain, lambda age: age < given)


def echo_labels(rows: list[gr.Row], t: int) -> np.ndarray | None:
    """The word each row's ``t``-th heard moment names, for a lesson at that moment (an echo):
    a heard word itself, a noun concept's word in its number, a verb concept's stem (coded
    ``-(stem + 1)``: its form waits for the agent). None where a row's moment names no word."""
    out = []
    for r in rows:
        if r.family != "clause":  # brackets and paths are heard without a lesson
            return None
        x = r.pre[t]
        words = np.flatnonzero(x[:gr.ACTIONS])
        if len(words) == 1:
            out.append(int(words[0]))
            continue
        on = {name for name, i in gr.C.items() if x[i]}
        noun = [i for i, name in enumerate(gr.CONCEPTS[:gr.NOUNS]) if name in on]
        if noun and on & {"AGENT", "PATIENT", "MOD"}:
            out.append(gr.noun(noun[0], "PLURAL" in on))
        elif on & {"SEE", "CHASE"}:
            out.append(-(gr.NOUNS + (0 if "SEE" in on else 1) + 1))
        else:
            return None
    return np.array(out, dtype=np.int64)


def speak(brain: Any, rows: list[gr.Row], *, teach: bool, message: Message | None = None,
          bar: bool = False, work: dict[str, float] | None = None, cap: int = 4,
          forced: bool = False, watch: Any = None, intervene: Any = None,
          echo: bool = False) -> list[list[int]]:
    """``grammar.speak`` with a message and a barred read. After every act the moment joins
    ``message``, which is installed for the next settle; with ``bar`` the meaning's slots are
    emptied at the continuation's start (``bar_meaning``), in lessons and in readings alike.
    ``forced`` gives every previous word; ``watch(j, brain, action)`` is called after each said
    word; ``intervene(brain, rows)`` runs once at the continuation's start."""
    work = work if work is not None else gr.new_work()
    n = len(rows)
    if len({len(r.pre) for r in rows}) != 1:
        raise ValueError("a batch's rows hear the same number of moments")
    if (teach or forced) and len({len(r.target) for r in rows}) != 1:
        raise ValueError("a taught batch's sentences have one length")
    brain.reset()
    if message is not None:
        message.reset(n)
    for t in range(len(rows[0].pre)):
        x = np.stack([r.pre[t] for r in rows])
        named = echo_labels(rows, t) if teach and echo else None
        if named is not None:  # a lesson at the heard moment: say the word it names
            try:
                report = brain.lesson(x, named)
                work["lessons"] += 1
                work["lesson_row_sweeps"] += report.get("total_row_sweeps", 0.0)
            except RuntimeError:
                work["refused"] += 1
        gr._act(brain, x, work)
        if message is not None:
            message.hear(x)
            install(brain, message)
    said: list[list[int]] = [[] for _ in rows]
    done = [False] * n
    longest = max(len(r.target) for r in rows) + (0 if teach or forced else cap)
    for j in range(longest):
        if j == 0:
            x = np.tile(gr.concept("GO"), (n, 1))
        else:
            words = []
            for i, r in enumerate(rows):
                given = teach or forced or j <= r.force
                words.append(r.target[j - 1] if given and j - 1 < len(r.target)
                             else said[i][j - 1])
            x = np.stack([gr.spoken(w) if w >= 0 else gr.concept("SPEAK") for w in words])
        if bar and j == rows[0].force:
            bar_meaning(brain, rows)
        if intervene is not None and j == rows[0].force:
            intervene(brain, rows)
        if teach:
            labels = np.array([r.target[j] for r in rows])
            try:
                report = brain.lesson(x, labels)
                work["lessons"] += 1
                work["rows"] += report.get("attempted_presentations", 0.0)
                work["agreed_rows"] += report.get("agreed_presentations", 0.0)
                work["lesson_row_sweeps"] += report.get("total_row_sweeps", 0.0)
                work["free_budget_exhausted"] += report.get("free_budget_exhausted", 0.0)
            except RuntimeError:
                work["refused"] += 1
        action = gr._act(brain, x, work)
        if watch is not None:
            watch(j, brain, action)
        if message is not None:
            message.hear(x, speaking=True)
            install(brain, message)
        for i, r in enumerate(rows):
            said[i].append(int(action[i]))
            if j >= r.force and action[i] == gr.END:
                done[i] = True
        if not (teach or forced) and all(done):
            break
    out = []
    for i, r in enumerate(rows):
        words = said[i][r.force:]
        out.append(words[:words.index(gr.END) + 1] if gr.END in words else words)
    return out


def said_by_kind(brain: Any, batches: list[list[gr.Row]], message: Message | None,
                 bar: bool) -> dict[str, float]:
    """Each word said right with every previous word given, by the word's kind
    (``grammar._kind``, and ``near`` and ``end`` by family): where a sentence is lost."""
    fork = copy.deepcopy(brain)
    groups: dict[str, list[int]] = {}
    for rows in batches:
        hits: list[np.ndarray] = []
        speak(fork, rows, teach=False, message=message, bar=bar, forced=True,
              watch=lambda j, b, action, hits=hits: hits.append(np.array(action)))
        for i, r in enumerate(rows):
            for j, word in enumerate(r.target):
                kind = (f"{r.family}_{gr.WORDS[word]}" if word in (gr.NEAR, gr.END)
                        else gr._kind(r, j))
                g = groups.setdefault(kind, [0, 0])
                g[0] += int(hits[j][i] == word)
                g[1] += 1
    return {k: round(v[0] / v[1], 4) for k, v in sorted(groups.items())}


def message_panels(brain: Any, tests: dict[str, list[list[gr.Row]]], message: Message | None,
                   bar: bool, names: tuple[str, ...], silent: bool = False) -> dict[str, Any]:
    """Whole sentences on a fork, closed loop, with the life's message and bar. ``silent``
    holds the brain's own message at no drive through the reading (its amplitudes at zero)."""
    fork = copy.deepcopy(brain)
    if silent:
        fork.message.amplitude = fork.message.read_amplitude = 0.0
    out = {}
    for name in names:
        panel = gr.read_speaker(
            lambda rows: speak(fork, rows, teach=False, message=message, bar=bar), None,
            tests[name], twins=name in gr.TWINS)
        panel["examples"] = panel["examples"][:2]
        out[name] = panel
    return out


def message_states(brain: Any, batches: dict[str, list[list[gr.Row]]],
                   message: Message | None, bar: bool,
                   ) -> dict[str, list[tuple[gr.Row, dict[str, Any]]]]:
    """Every probe sentence's states at its named positions, teacher-forced, with the life's
    message and bar: ``message`` (the region as written, unscaled), ``carried`` (the working
    trace entering the settle) and ``settled`` (the association after it)."""
    fork = copy.deepcopy(brain)
    association = fork.association_index
    out: dict[str, list[tuple[gr.Row, dict[str, Any]]]] = {}
    for family, groups in batches.items():
        items = []
        for rows in groups:
            moments: list[dict[str, np.ndarray]] = []
            carried = {"trace": np.zeros((len(rows), len(association)))}

            def watch(j: int, b: Any, action: np.ndarray, moments: Any = moments,
                      carried: Any = carried) -> None:
                state = np.asarray(b.basal_ganglia.state.activation)[:, association]
                moment = {"carried": carried["trace"], "settled": np.array(state, dtype=float)}
                if message is not None:
                    moment["message"] = message.state(scaled=False).copy()
                elif getattr(b, "message", None) is not None:
                    # the brain's own message, the settle's moment joined
                    moment["message"] = np.array(b.message.state, dtype=float)
                moments.append(moment)
                carried["trace"] = np.array(b.working_memory.trace, dtype=float)

            heard: dict[str, np.ndarray] = {}

            # the trace entering GO is the one the last heard moment left; the settled state
            # after the last heard moment is the sentence as the brain holds it (``heard``)
            def before(b: Any, carried: Any = carried, heard: Any = heard) -> None:
                carried["trace"] = np.array(b.working_memory.trace, dtype=float)
                state = np.asarray(b.basal_ganglia.state.activation)[:, association]
                heard["settled"] = np.array(state, dtype=float)
                heard["carried"] = carried["trace"]
                if message is not None:
                    heard["message"] = message.state(scaled=False).copy()
                elif getattr(b, "message", None) is not None:
                    heard["message"] = np.array(b.message.state, dtype=float)

            _speak_watched(fork, rows, message, bar, before, watch)
            for i, row in enumerate(rows):
                at = {name: {place: moments[j][place][i] for place in moments[j]}
                      for name, j in positions(row).items()}
                if heard:
                    at["heard"] = {place: heard[place][i] for place in heard}
                for spot in at.values():  # the store read with the association (cadence#308)
                    if "message" in spot:
                        spot["joint"] = np.concatenate([spot["message"], spot["settled"]])
                items.append((row, at))
        out[family] = items
    return out


def _speak_watched(brain: Any, rows: list[gr.Row], message: Message | None, bar: bool,
                   before: Any, watch: Any) -> None:
    """One batch teacher-forced with the life's message and bar: ``before(brain)`` once the
    meaning is heard, ``watch(j, brain, action)`` after each said word."""
    n = len(rows)
    brain.reset()
    if message is not None:
        message.reset(n)
    work = gr.new_work()
    for t in range(len(rows[0].pre)):
        x = np.stack([r.pre[t] for r in rows])
        gr._act(brain, x, work)
        if message is not None:
            message.hear(x)
            install(brain, message)
    if brain.working_memory is not None and len(brain.working_memory.trace) == n:
        before(brain)
    for j in range(len(rows[0].target)):
        x = (np.tile(gr.concept("GO"), (n, 1)) if j == 0
             else np.stack([gr.spoken(r.target[j - 1]) for r in rows]))
        if bar and j == 0:
            bar_meaning(brain, rows)
        action = gr._act(brain, x, work)
        watch(j, brain, action)
        if message is not None:
            message.hear(x, speaking=True)
            install(brain, message)


def twin_panels(brain: Any, tests: dict[str, list[list[gr.Row]]], message: Message | None,
                bar: bool) -> dict[str, Any]:
    """The chamber's context twins with the life's message and bar: intact, with the context
    removed, and with it erased at the continuation's start from the recall memory, from the
    working trace or from the message."""

    def empty_message(b: Any, rows: list[gr.Row]) -> None:
        if message is not None:
            message.reset(len(rows))
            install(b, message)
        elif getattr(b, "message", None) is not None:  # the brain's own
            b.message.trace = np.zeros_like(b.message.trace)
            b.message.told = np.zeros_like(b.message.told)

    erasures = {"recall": gr.erase("recall"), "trace": gr.erase("trace"),
                "message": empty_message}
    out: dict[str, Any] = {}

    def read(name: str, intervene: Any = None) -> dict[str, Any]:
        fork = copy.deepcopy(brain)
        panel = gr.read_speaker(
            lambda rows: speak(fork, rows, teach=False, message=message, bar=bar,
                               intervene=intervene), None, tests[name], twins=True)
        panel.pop("examples", None)
        return panel

    for name in gr.ERASED:
        out[name] = read(name)
        out[name + "_removed"] = read(name + "_removed")
        for where, act in erasures.items():
            if where == "message" and message is None and getattr(brain, "message", None) is None:
                continue
            out[f"{name}_erased_{where}"] = read(name, act)
    return out


def invariance(brain: Any, founder: int, message: Message | None, bar: bool,
               pairs: int = 64) -> dict[str, Any]:
    """One meaning heard in the canonical and in the inverted order (the agent before the
    patient, or after it), on combinations that training shows: how close the two held states
    are beside the states of different meanings, and whether the same sentence is said.
    ``closer`` is the share of meanings whose
    two orders are more alike (cosine) than the canonical order is to the nearest other
    meaning of the same shape; ``same`` the mean cosine of the two orders, ``other`` of
    different meanings. Read at ``GO`` on the message as written, on its key (the union over
    its phases) and on the settled association."""
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["probe"], 3]))
    split = gr.Split.of(founder)
    fork = copy.deepcopy(brain)
    association = fork.association_index
    places: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    identical: list[np.ndarray] = []  # per batch, the rows that hold one meaning
    said_same, both_right, n = 0, 0, 0
    for index in range(pairs // gr.BATCH):
        L = gr.TRAIN_L[index % len(gr.TRAIN_L)]
        base = [gr.twin(gr.trained_clause(rng, split, L, False), inverted=False)
                for _ in range(gr.BATCH)]
        other = [gr.twin(r, inverted=True) for r in base]
        identical.append(np.array([[a.target == b.target for b in base] for a in base]))
        states = []
        for rows in (base, other):
            seen: dict[str, np.ndarray] = {}

            def watch(j: int, b: Any, action: np.ndarray, seen: Any = seen) -> None:
                if j == 0:
                    seen["settled"] = np.array(
                        np.asarray(b.basal_ganglia.state.activation)[:, association])

            def before(b: Any, seen: Any = seen) -> None:
                seen["carried"] = np.array(b.working_memory.trace, dtype=float)
                if message is not None:
                    state = message.state(scaled=False)
                    seen["message"] = state.copy()
                    if len(message.blocks) == 1 and message.blocks[0].startswith("phase"):
                        seen["key"] = state.reshape(len(state), -1, message.units).max(axis=1)
                elif getattr(b, "message", None) is not None:  # the brain's own
                    seen["message"] = np.array(b.message.state, dtype=float)
                    seen["key"] = np.array(b.message.key(), dtype=float)

            _speak_watched(fork, rows, message, bar, before, watch)
            states.append(seen)
        for place in states[0]:
            places.setdefault(place, []).append((states[0][place], states[1][place]))
        said = [speak(fork, rows, teach=False, message=message, bar=bar) for rows in (base, other)]
        for a, b, row in zip(said[0], said[1], base, strict=True):
            said_same += int(a == b)
            both_right += int(a == row.target and b == row.target)
            n += 1
    out: dict[str, Any] = {"n": n, "said_same": round(said_same / n, 4),
                           "both_right": round(both_right / n, 4)}

    def unit(x: np.ndarray) -> np.ndarray:
        return np.asarray(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12))

    for place, batches in places.items():
        closer, same, different = [], [], []
        for (canonical, inverted), alike in zip(batches, identical, strict=True):
            a, b = unit(canonical), unit(inverted)
            own = (a * b).sum(axis=1)
            cross = np.where(alike, -np.inf, a @ a.T)  # different meanings only
            closer += list(own > cross.max(axis=1))
            same += list(own)
            different += list(cross[np.isfinite(cross)])
        out[place] = {"closer": round(float(np.mean(closer)), 4),
                      "same": round(float(np.mean(same)), 4),
                      "other": round(float(np.mean(different)), 4)}
    return out


CAPACITY_L = (0, 1, 2, 3, 4)  # distractors in a capacity life's clauses


def capacity_batch(rng: np.random.Generator, split: gr.Split) -> list[gr.Row]:
    """A capacity life's lesson batch: clauses of zero to four distractors, combinations that
    training shows."""
    L = int(rng.choice(CAPACITY_L))
    return [gr.trained_clause(rng, split, L, False) for _ in range(gr.BATCH)]


def capacity_panels(founder: int, batches: int = 4) -> dict[str, list[list[gr.Row]]]:
    """Clauses of each number of distractors, ``batches`` batches each, combinations that
    training shows."""
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), gr.SEEDS["test"], 5]))
    return {f"L{L}": [[gr.trained_clause(rng, split, L, False) for _ in range(gr.BATCH)]
                      for _ in range(batches)] for L in CAPACITY_L}


def paraphrase(row: gr.Row) -> gr.Row:
    """A paraphrase episode of a clause: the world says the first grammar's sentence (heard
    words, no flag), a ``DIALECT`` moment asks for the second grammar, and the sentence to say
    is the second grammar's, the patient before the verb. No concept moment is given."""
    said = gr.twin(row, dialect=True)
    pre = [gr.heard(w) for w in gr.twin(row, dialect=False).target] + [gr.concept("DIALECT")]
    return gr.Row("clause", pre, said.target, {**said.meta, "paraphrase": True})


def sayback(row: gr.Row) -> gr.Row:
    """A say-back episode of a clause: the sentence heard as its words, then said at ``GO``.
    No concept moment is given."""
    return gr.Row("clause", [gr.heard(w) for w in row.target], row.target,
                  {**row.meta, "sayback": True})


def sayback_batch(rng: np.random.Generator, split: gr.Split) -> list[gr.Row]:
    """A say-back life's lesson batch: the first grammar's mix (one half) and clauses of
    trained combinations heard as words and said back (one half)."""
    if rng.random() < 0.5:
        return gr.train_batch(rng, split, "g1")
    L = int(rng.choice(gr.TRAIN_L))
    return [sayback(gr.trained_clause(rng, split, L, False)) for _ in range(gr.BATCH)]


def sayback_panels(founder: int, batches: int = 4) -> dict[str, list[list[gr.Row]]]:
    """Say back at trained lengths (``back`` of combinations that training shows,
    ``back_held`` each with a held-out triple or the held-out pair) and at the longer ones
    (``back_long``)."""
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), gr.SEEDS["test"], 308]))

    def held(L: int) -> gr.Row:
        while True:
            row = gr.clause(rng, L)
            m = row.meta
            if split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"]):
                return row

    shape = [gr.TRAIN_L[i % len(gr.TRAIN_L)] for i in range(batches)]
    return {
        "back": [[sayback(gr.trained_clause(rng, split, L, False)) for _ in range(gr.BATCH)]
                 for L in shape],
        "back_held": [[sayback(held(L)) for _ in range(gr.BATCH)] for L in shape],
        "back_long": [[sayback(gr.trained_clause(rng, split, gr.TEST_L[i % len(gr.TEST_L)],
                                                 False)) for _ in range(gr.BATCH)]
                      for i in range(batches)],
    }


def paraphrase_batch(rng: np.random.Generator, split: gr.Split) -> list[gr.Row]:
    """A paraphrase life's lesson batch: the first grammar's mix (one half), the second
    grammar's clauses from concepts (one quarter) and paraphrases (one quarter), combinations
    that training shows."""
    kind = rng.random()
    if kind < 0.5:
        return gr.train_batch(rng, split, "g1")
    L = int(rng.choice(gr.TRAIN_L))
    rows = [gr.trained_clause(rng, split, L, True) for _ in range(gr.BATCH)]
    return rows if kind < 0.75 else [paraphrase(r) for r in rows]


def paraphrase_panels(founder: int, batches: int = 4) -> dict[str, list[list[gr.Row]]]:
    """Paraphrases at trained lengths: ``para`` of combinations that training shows,
    ``para_held`` each with a held-out triple or the held-out pair; ``g2`` the second
    grammar's clauses from concepts."""
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), gr.SEEDS["test"], 9]))

    def held(L: int) -> gr.Row:
        while True:
            row = gr.clause(rng, L, dialect=True)
            m = row.meta
            if split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"]):
                return row

    shape = [gr.TRAIN_L[i % len(gr.TRAIN_L)] for i in range(batches)]
    return {
        "para": [[paraphrase(gr.trained_clause(rng, split, L, True)) for _ in range(gr.BATCH)]
                 for L in shape],
        "para_held": [[paraphrase(held(L)) for _ in range(gr.BATCH)] for L in shape],
        "g2": [[gr.trained_clause(rng, split, L, True) for _ in range(gr.BATCH)] for L in shape],
    }


def retrieval(founder: int, design: dict[str, Any] | None = None,
              stored: tuple[int, ...] = (16, 64, 256, 1024), cues: int = 256) -> dict[str, Any]:
    """Retrieval by meaning, on the code alone: ``cadence.Message`` holds each stored clause
    and each cue, and a cue retrieves the stored record whose key it overlaps most (cosine; a
    tie for the best counts as a miss). ``key`` records are the union over the phases,
    ``state`` records the phased message. Every store holds each clause beside its
    role-swapped twin (agent and patient exchanged), the nearest different meaning. Cues:

    - ``reordered``: the stored clause's concept moments in the other order (a paraphrase).
    - ``swapped``: the role-swapped twin's concept moments in the other order: right where it
      finds the twin's record and not the original's.
    - ``partial``: the stored clause's concepts without its patient: right where the record
      found agrees with the cue on the agent, the verb and the distractors.
    - ``words``: the clause is stored as the first grammar's sentence, heard as words, and the
      cue is the second grammar's sentence of the same meaning (a paraphrase in other words'
      order).
    - ``crossed``: the first grammar's sentence, heard as words, asks the store of concept
      moments (the form the code does not join)."""
    import cadence as cd

    design = design or {}
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["probe"], 7]))

    def held(moments: list[list[np.ndarray]]) -> dict[str, np.ndarray]:
        keys, states = [], []
        for heard in moments:
            m = cd.Message(gr.INPUTS, seed=int(founder), **design)
            for x in heard:
                m.hear(x[None])
            keys.append(m.key()[0])
            states.append(m.state[0].copy())
        return {"key": np.stack(keys), "state": np.stack(states)}

    def unit(x: np.ndarray) -> np.ndarray:
        return np.asarray(x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12))

    def concepts(r: gr.Row, drop: str | None = None) -> list[np.ndarray]:
        return [x for x in r.pre if drop is None or not x[gr.C[drop]]]

    def words(r: gr.Row, dialect: bool = False) -> list[np.ndarray]:
        return [gr.heard(w) for w in gr.twin(r, dialect=dialect).target]

    def swap(r: gr.Row, **change: Any) -> gr.Row:
        return gr.twin(r, agent=tuple(r.meta["patient"]), patient=tuple(r.meta["agent"]),
                       **change)

    def content(r: gr.Row) -> tuple[Any, ...]:
        m = r.meta
        return (tuple(m["agent"]), m["verb"], tuple(map(tuple, m["mods"])))

    out: dict[str, Any] = {}
    for count in stored:
        meanings: dict[tuple[Any, ...], gr.Row] = {}
        while len(meanings) < count // 2:
            r = gr.clause(rng, int(rng.choice(gr.TRAIN_L)), inverted=False)
            m = r.meta
            name = (tuple(m["agent"]), tuple(m["patient"]), m["verb"],
                    tuple(map(tuple, m["mods"])))
            twin_name = (name[1], name[0], name[2], name[3])
            if name[0] != name[1] and twin_name not in meanings:
                meanings.setdefault(name, r)
        originals = list(meanings.values())
        records = originals + [swap(r) for r in originals]
        by_concepts = held([concepts(r) for r in records])
        by_words = held([words(r) for r in records])
        chosen = rng.permutation(len(originals))[:min(cues, len(originals))]
        asked = [originals[int(i)] for i in chosen]
        table: dict[str, Any] = {}
        for cue, moments, store, answer in (
            ("reordered", [concepts(gr.twin(r, inverted=True)) for r in asked], by_concepts,
             chosen),
            ("swapped", [concepts(swap(r, inverted=True)) for r in asked], by_concepts,
             chosen + len(originals)),
            ("partial", [concepts(r, "PATIENT") for r in asked], by_concepts, None),
            ("words", [words(r, dialect=True) for r in asked], by_words, chosen),
            ("crossed", [words(r) for r in asked], by_concepts, chosen),
        ):
            asks = held(moments)
            for kind in ("key", "state"):
                scores = unit(asks[kind]) @ unit(store[kind]).T
                found = np.argmax(scores, axis=1)
                if answer is None:  # any record that agrees with the cue's content
                    hit = np.array([content(records[int(f)]) == content(r)
                                    for f, r in zip(found, asked, strict=True)])
                else:
                    best = scores.max(axis=1)
                    unique = (scores >= best[:, None] - 1e-12).sum(axis=1) == 1
                    hit = (found == answer) & unique
                table.setdefault(cue, {})[kind] = round(float(hit.mean()), 4)
        out[str(len(records))] = table
    return out


ARMS: dict[str, dict[str, Any]] = {
    # the default composition, the read free: the reference
    "default": {},
    # the read barred from the meaning's moments and nothing added: the floor
    "barred": {"bar": True},
    # the message, the read barred from the meaning: the state must carry it
    "message_barred": {"message": {}, "bar": True},
    # the message and the read together
    "message": {"message": {}},
}


def message_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life of an arm: ``arm`` (``ARMS``) or its parts given outright (``message``, a
    design or None; ``bar``, the share of lesson sentences taught with the read barred from the
    meaning, True for every one; ``settings`` over the default composition), ``founder``,
    ``batches``, ``module``, ``panel_batches``, ``probe_scale``, ``tag``. A life barred on every
    sentence is read barred; any other is read with the read free and again barred. ``twins``
    reads the context twins and ``invariance`` the two orders of a meaning; ``after`` continues
    the life through further phases (``[grammar, batches]``), each read; ``"train": "capacity"``
    teaches clauses of zero to four distractors alone and reads each number apart;
    ``"train": "paraphrase"`` teaches the first grammar's mix, the second grammar and
    paraphrases (``paraphrase_batch``) and reads the paraphrase panels as well."""
    warnings.simplefilter("ignore")
    import cadence as cd

    arm = dict(ARMS.get(spec.get("arm", ""), {}))
    for key in ("message", "bar", "settings"):
        if key in spec:
            arm[key] = spec[key]
    founder = int(spec["founder"])
    batches = int(spec.get("batches", BATCHES))
    design = arm.get("message")
    rate = float(arm.get("bar", 0.0))  # the share of lesson sentences taught with the read barred
    bar = rate >= 1.0
    barred = np.random.default_rng(np.random.SeedSequence([founder, 286, 1]))
    started = time.monotonic()
    brain = message_brain(founder, design, arm.get("settings"), int(spec.get("module", 64)))
    message = None if design is None else Message(design, founder)
    split = gr.Split.of(founder)
    tests = gr.panels(founder, int(spec.get("panel_batches", 4)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    row: dict[str, Any] = {
        "kind": "message", "arm": spec.get("arm"), "tag": spec.get("tag"), "founder": founder,
        "batches": batches, "message": design, "bar": rate, "settings": arm.get("settings", {}),
        "module": int(spec.get("module", 64)), "parameters": int(brain.parameters()),
        "cadence": cd.__version__, "curve": [],
    }
    work = gr.new_work()
    checks = set(int(c) for c in spec.get("checkpoints", ()))
    capacity = spec.get("train") == "capacity"
    paraphrases = spec.get("train") == "paraphrase"
    saysback = spec.get("train") == "sayback"

    def lessons(grammar: str, count: int, work: dict[str, float]) -> None:
        for b in range(1, count + 1):
            batch = (capacity_batch(rng, split) if capacity
                     else paraphrase_batch(rng, split) if paraphrases
                     else sayback_batch(rng, split) if saysback
                     else gr.train_batch(rng, split, grammar))
            speak(brain, batch, teach=True, message=message,
                  bar=bool(barred.random() < rate), work=work, echo=bool(spec.get("echo")))
            if b in checks and grammar == "g1" and not capacity and not paraphrases:
                lean = message_panels(brain, tests, message, bar, ("clause", "dyck", "path"))
                row["curve"].append({"batches": b, **{k: v["exact"] for k, v in lean.items()}})

    lessons("g1", batches, work)
    work["seconds"] = round(time.monotonic() - started, 1)
    row["work"] = work
    row["synapses"] = int(brain.describe()["layout"]["synapses"])
    row["gains"] = dict(brain.carried_gains) if brain.hearing is not None else {}
    if capacity:
        panels = capacity_panels(founder, int(spec.get("panel_batches", 4)))
        row["capacity"] = {name: panel["exact"] for name, panel in message_panels(
            brain, panels, message, bar, tuple(panels)).items()}
        row["seconds"] = round(time.monotonic() - started, 1)
        return row
    row["panels"] = message_panels(brain, tests, message, bar, WHOLE + LONG)
    if saysback:
        panels = sayback_panels(founder, int(spec.get("panel_batches", 4)))
        row["sayback"] = message_panels(brain, panels, message, bar, tuple(panels))
    if paraphrases:
        panels = paraphrase_panels(founder, int(spec.get("panel_batches", 4)))
        row["paraphrase"] = message_panels(brain, panels, message, bar, tuple(panels))
    if (message is None and getattr(brain, "message", None) is None and rate == 0.0
            and brain.recall is not None):
        row["located"] = {name: gr.locate(brain, tests[name])
                          for name in ("clause", "clause_long", "dyck", "dyck_deep", "path",
                                       "path_long")}
    if spec.get("twins"):
        row["twins"] = twin_panels(brain, tests, message, bar)
    if spec.get("invariance"):
        row["invariance"] = invariance(brain, founder, message, bar)
    if not bar:  # a life taught with the read free, read with it barred
        row["panels_barred"] = message_panels(brain, tests, message, True, WHOLE)
        if getattr(brain, "message", None) is not None:  # and with its own message silent
            row["panels_silent"] = message_panels(brain, tests, message, False,
                                                  WHOLE + ("path_long",), silent=True)
    scale = float(spec.get("probe_scale", 0.5))
    fresh = probe_batches(founder, scale) if scale > 0 else None
    # the readout by word kind: on the probes' fresh sentences where a life has them
    row["said"] = said_by_kind(
        brain, [b for name in ("clause", "dyck", "path")
                for b in (tests if fresh is None else fresh)[name]], message, bar)
    if spec.get("held_probes"):
        # where held-out combinations are lost: each word kind said on them with the previous
        # words given, and probes fitted on trained combinations, scored on held-out ones
        both = split_batches(founder, int(spec.get("held_probes")))
        row["said_held"] = said_by_kind(brain, both["held"], message, bar)
        own = getattr(brain, "message", None) is not None
        places = ("message", "carried", "settled", "joint") if message is not None or own else (
            "carried", "settled")
        states = message_states(brain, {"trained": both["trained"], "held": both["held"]},
                                message, bar)
        row["probes_held"] = transfer_probes(founder, states["trained"], states["held"], places)
        if own:  # the store as a record's key: its width and how much of it a sentence fills
            held = np.stack([at["heard"]["message"] for _, at in states["trained"]
                             if "heard" in at])
            row["store"] = {
                "width": int(held.shape[1]),
                "nonzero": round(float(np.mean(held != 0.0)), 4),
                "above_tenth": round(float(np.mean(np.abs(held) > 0.1)), 4),
                "blocks_filled": round(float(np.mean(np.any(held.reshape(
                    len(held), brain.message.blocks, -1) != 0.0, axis=2))), 4),
            }
        if saysback:  # the same clauses heard as words
            words = {k: [[sayback(r) for r in rows] for rows in both[k]] for k in both}
            states = message_states(brain, words, message, bar)
            row["probes_back"] = transfer_probes(founder, states["trained"], states["held"],
                                                 places)
        if spec.get("silent_reading") and getattr(brain, "message", None) is not None:
            # the same readings with the message silent: what the association holds and says
            # by itself, the recall read barred as in the life's lessons
            quiet = copy.deepcopy(brain)
            quiet.message.amplitude = quiet.message.read_amplitude = 0.0
            row["said_held_silent"] = said_by_kind(quiet, both["held"], message, bar)
            states = message_states(quiet, {"trained": both["trained"], "held": both["held"]},
                                    message, bar)
            row["probes_held_silent"] = transfer_probes(founder, states["trained"],
                                                        states["held"], places)
            row["panels_held_silent"] = message_panels(brain, tests, message, bar,
                                                       ("clause", "clause_held", "roles"),
                                                       silent=True)
    if spec.get("long_probes"):
        # where longer and deeper sentences are lost: each word kind said on the long panels
        # with the previous words given, and probes fitted on trained shapes, scored on longer
        row["said_long"] = said_by_kind(brain, [b for name in LONG for b in tests[name]],
                                        message, bar)
        own = getattr(brain, "message", None) is not None
        places = ("message", "carried", "settled") if message is not None or own else (
            "carried", "settled")
        count = int(spec.get("long_probes"))
        states = message_states(
            brain, {"trained": split_batches(founder, count)["trained"],
                    "long": long_batches(founder, count)}, message, bar)
        row["probes_long"] = transfer_probes(founder, states["trained"], states["long"], places)
    if fresh is not None:
        own = getattr(brain, "message", None) is not None
        places = ("message", "carried", "settled") if message is not None or own else (
            "carried", "settled")
        row["probes"] = probe_states(founder, message_states(brain, fresh, message, bar), places)
    # persistence: further phases of lessons (the second grammar, then the first again), each
    # read on the first grammar's panels and the second's
    row["persistence"] = []
    for grammar, count in spec.get("after", ()):
        more = gr.new_work()
        clock = time.monotonic()
        lessons(str(grammar), int(count), more)
        more["seconds"] = round(time.monotonic() - clock, 1)
        row["persistence"].append({
            "grammar": grammar, "batches": int(count), "work": more,
            "panels": message_panels(brain, tests, message, bar,
                                     ("clause", "clause_held", "roles", "dyck", "path", "g2"))})
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


# -------------------------------------------------------------------------------- reading


def _mean(values: list[float | None]) -> str:
    kept = [v for v in values if v is not None and v == v]
    if not kept:
        return "  -  "
    return f"{np.mean(kept):.2f} ({min(kept):.2f}-{max(kept):.2f})"


def _short(values: list[float | None]) -> str:
    kept = [v for v in values if v is not None and v == v]
    return f"{np.mean(kept):.2f}" if kept else "-"


def _score(panel: dict[str, Any], name: str) -> float | None:
    return panel.get("both") if name == "roles" else panel.get("exact")


def read(rows: list[dict[str, Any]]) -> list[str]:
    """The diagnosis as tables: sentences per erasure, the controls, and the probes."""
    rows = [r for r in rows if r.get("kind") == "where"]
    lines = [f"lives: {len(rows)} (founders {sorted(r['founder'] for r in rows)})", ""]
    names = WHOLE + LONG
    lines.append("whole sentences said exactly (roles: both twins), mean (least-greatest)")
    lines.append("| Speaker | " + " | ".join(names) + " |")
    lines.append("| --- |" + " --- |" * len(names))
    for condition in CONDITIONS + PATH_CONDITIONS:
        cells = []
        for name in names:
            cells.append(_mean([_score(r["sentences"][condition][name], name)
                                if name in r["sentences"].get(condition, {}) else None
                                for r in rows]))
        lines.append(f"| brain, {condition} | " + " | ".join(cells) + " |")
    for model in ("lstm", "transformer"):
        for condition in ("intact", "floor"):
            cells = []
            for name in names:
                cells.append(_mean([_score(c["panels"][condition][name], name)
                                    for r in rows for c in r["controls"] if c["model"] == model]))
            lines.append(f"| {model}, {condition} | " + " | ".join(cells) + " |")
    lines += ["", "context twins, both right / differ"]
    for name in gr.ERASED:
        for suffix in ("", "_erased_recall", "_erased_trace", "_removed"):
            key = name + suffix
            both = _mean([r["sentences"]["context"][key]["both"] for r in rows])
            differ = _mean([r["sentences"]["context"][key]["differ"] for r in rows])
            lines.append(f"  {key:26s} {both} / {differ}")
    lines += ["", "probes, held-out accuracy, mean over lives"]

    def brain_probe(family: str, name: str, spot: str, condition: str, place: str,
                    key: str = "accuracy") -> str:
        return _short([r["probes"][condition][family][name][spot][place][key] for r in rows
                       if spot in r.get("probes", {}).get(condition, {}).get(family, {}).get(
                           name, {})])

    def control_probe(model: str, family: str, name: str, spot: str, place: str) -> str:
        return _short([c["probes"][family][name][spot][place]["accuracy"]
                       for r in rows for c in r["controls"]
                       if c["model"] == model and place in c.get("probes", {}).get(
                           family, {}).get(name, {}).get(spot, {})])

    for family in ("clause", "dyck", "path"):
        table = rows[0].get("probes", {}).get("intact", {}).get(family, {})
        for spot in ("go", "verb", "end"):
            if not any(spot in table[name] for name in table):
                continue
            lines += ["", f"{family} at {spot}",
                      "| Component | Chance | Carried | Settled | Read | Carried, slots emptied "
                      "| Settled, slots emptied | LSTM carried | LSTM settled | Transformer |",
                      "| --- |" + " --- |" * 9]
            for name in table:
                if spot not in table[name]:
                    continue
                cells = [brain_probe(family, name, spot, "intact", "settled", "chance"),
                         *(brain_probe(family, name, spot, "intact", place) for place in PLACES),
                         brain_probe(family, name, spot, "state", "carried"),
                         brain_probe(family, name, spot, "state", "settled"),
                         control_probe("lstm", family, name, spot, "carried"),
                         control_probe("lstm", family, name, spot, "settled"),
                         control_probe("transformer", family, name, spot, "settled")]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


# ------------------------------------------------------------- the message on any chamber


def message_class(design: dict[str, Any]) -> Any:
    """``cadence.Brain`` with a ``cadence.Message`` on any chamber's own life loop: a class
    whose ``compose`` adds the carrier neurons (heard-history blocks) and whose brains write
    the message after every act and empty it at ``reset``. ``design`` holds the message's
    ``phases``, ``units``, ``winners`` and ``fan_in`` and the ``scale`` the carrier is driven
    at (times the history's amplitude of 3). The emulation of ``speak`` for chambers that
    build and run their own brains; the library holds no such option."""
    import cadence as cd

    code = {k: int(design[k]) for k in ("phases", "units", "winners", "fan_in") if k in design}
    scale = float(design.get("scale", 0.15))

    class MessageBrain(cd.Brain):  # type: ignore[misc]
        held: Any = None

        @classmethod
        def compose(cls, inputs: Any, actions: int, **options: Any) -> Any:
            width = cd.Message(int(inputs), **code).width
            blocks = -(-width // int(inputs))
            brain = super().compose(inputs, actions,
                                    **{**options, "history_decays": (0.0,) * blocks})
            brain.held = cd.Message(int(inputs), seed=int(options.get("seed", 0)), **code)
            return brain

        def _choose(self, x: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
            action = super()._choose(x, *args, **kwargs)
            self.held.hear(np.asarray(x, dtype=float))
            carried = np.zeros((len(x), len(self.history.glow)))
            carried[:, :self.held.width] = scale * self.held.state
            self.history.trace = carried
            return np.asarray(action)

        def reset(self, **kwargs: Any) -> None:
            super().reset(**kwargs)
            if self.held is not None:
                self.held.reset(0)

    return MessageBrain


def knock_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life of another chamber on the default composition, with a message or without:
    ``chamber`` (``"battery"``, the sequence battery, with ``task`` and ``task_args``; or
    ``"routes"``, the goal-route chamber where its file is present, with ``speaker``),
    ``founder``, ``message`` (a design for ``message_class``, or None), and the chamber's own
    settings (``lessons``, ``tests``, ``batches``, ``module``)."""
    warnings.simplefilter("ignore")
    import cadence as cd

    design = spec.get("message")
    brain_class = cd.Brain if design is None else message_class(design)
    founder = int(spec["founder"])

    def composed(inputs: int, actions: int, module: int, extra: dict[str, Any]) -> Any:
        return brain_class.compose(inputs, actions, modules=(module,), seed=founder,
                                   **{"backend": "cpu", "recall": True, **extra})

    if spec["chamber"] == "battery":
        battery = gr.BATTERY
        battery.make_brain = lambda task, arm, f, settings=None, module=64: composed(
            task.inputs, task.actions, module, settings or {})
        row = battery.run_life({k: v for k, v in spec.items()
                                if k not in ("chamber", "message", "tag")} | {"arm": "table"})
        row["settings"] = spec.get("settings", {})
    elif spec["chamber"] == "routes":
        routes = gr._module("routes", HERE / "routes.py")
        routes.make_brain = lambda speaker, f, settings=None, module=64: composed(
            routes.INPUTS, routes.ACTIONS, module, settings or {})
        row = routes.run_life({k: v for k, v in spec.items()
                               if k not in ("chamber", "message", "tag")})
    else:
        raise ValueError(f"unknown chamber: {spec['chamber']!r}")
    return {**row, "kind": "knock", "chamber": spec["chamber"], "tag": spec.get("tag"),
            "message": design, "arm": "message" if design else "default"}


# ------------------------------------------------------------------------- restated notes

SILENT = gr.ACTIONS  # the word slot's action at a moment the brain listens
NOTES = ("agent", "verb", "patient", "distractor1", "distractor2")
NOTE_SIZES = {"agent": 9, "verb": 3, "patient": 9}  # eight noun forms or two verbs, and empty


def note_sizes(notes: tuple[str, ...] = NOTES) -> tuple[int, ...]:
    """The motor slots of a brain that restates its message: the word (every word and
    silence), then one slot per note."""
    return (gr.ACTIONS + 1, *(NOTE_SIZES.get(name, 9) for name in notes))


def note_targets(row: gr.Row, heard: int, notes: tuple[str, ...] = NOTES) -> list[int]:
    """What each note holds once the first ``heard`` moments of the row's meaning are heard:
    a clause's agent, verb, patient and distractors in the order heard, each a filler or empty
    (the slot's last action). Bracket sentences and paths leave every note empty."""
    out = {name: NOTE_SIZES.get(name, 9) - 1 for name in notes}
    if row.family == "clause":
        mods = 0
        for x in row.pre[:heard]:
            on = {name for name, i in gr.C.items() if x[i]}
            if on & {"SEE", "CHASE"}:
                out["verb"] = 0 if "SEE" in on else 1
                continue
            filler = gr.noun(gr.CONCEPTS.index(next(n for n in on if n in gr.CONCEPTS[:gr.NOUNS])),
                             "PLURAL" in on)
            if "AGENT" in on:
                out["agent"] = filler
            elif "PATIENT" in on:
                out["patient"] = filler
            else:
                mods += 1
                if f"distractor{mods}" in out:
                    out[f"distractor{mods}"] = filler
    return [out[name] for name in notes]


def notes_brain(founder: int, settings: dict[str, Any] | None = None, module: int = 64,
                notes: tuple[str, ...] = NOTES) -> Any:
    """The default composition with one motor slot per note beside the word's, and the
    efference copy that carries every slot's command to the next moment."""
    sizes = note_sizes(notes)
    return make_brain(founder, {"slots": sizes, "efference_amplitude": 3.0,
                                "efference_decay": 0.0, **(settings or {})}, module,
                      actions=sum(sizes))


def speak_notes(brain: Any, rows: list[gr.Row], *, teach: bool, bar: bool = False,
                force_notes: bool = True, notes: tuple[str, ...] = NOTES,
                work: dict[str, float] | None = None, cap: int = 4,
                record: list[np.ndarray] | None = None) -> list[list[int]]:
    """One batch of a brain that restates its message. At every moment the brain settles the
    word and every note together. With ``teach`` a lesson names them all before every act: at
    a heard moment silence and the notes so far, from ``GO`` on the target word and the whole
    notes. The efference copy carries the brain's own notes to the next moment; with
    ``force_notes`` a taught moment carries the lesson's notes, as it hears the lesson's word.
    With ``bar`` a clause's meaning slots are emptied at ``GO`` (``bar_meaning``). ``record``
    takes each moment's issued command, ``(rows, slots)``."""
    work = work if work is not None else gr.new_work()
    n = len(rows)
    sizes = note_sizes(notes)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    brain.reset()
    said: list[list[int]] = [[] for _ in rows]
    done = [False] * n
    heard = len(rows[0].pre)
    longest = max(len(r.target) for r in rows) + (0 if teach else cap)

    def moment(x: np.ndarray, labels: np.ndarray | None) -> np.ndarray:
        if labels is not None:
            try:
                report = brain.lesson(x, labels)
                work["lessons"] += 1
                work["rows"] += report.get("attempted_presentations", 0.0)
                work["agreed_rows"] += report.get("agreed_presentations", 0.0)
                work["lesson_row_sweeps"] += report.get("total_row_sweeps", 0.0)
            except RuntimeError:
                work["refused"] += 1
        try:
            action = np.asarray(brain.act(x, greedy=True)).reshape(n, len(sizes))
            work["act_sweeps"] += float((brain.last_settlement or {}).get("steps", 0))
        except RuntimeError:
            work["exhausted"] += 1
            action = np.full((n, len(sizes)), -1)
        work["acts"] += 1
        if labels is not None and force_notes and brain.efference is not None:
            command = np.zeros((n, int(sum(sizes))))
            command[np.arange(n)[:, None], labels + offsets[None, :]] = 1.0
            brain.efference.trace = command
            brain.efference.last = command.copy()
        if record is not None:
            record.append(action.copy())
        return action

    for t in range(heard):
        x = np.stack([r.pre[t] for r in rows])
        labels = (np.array([[SILENT, *note_targets(r, t + 1, notes)] for r in rows])
                  if teach else None)
        moment(x, labels)
    for j in range(longest):
        if j == 0:
            x = np.tile(gr.concept("GO"), (n, 1))
        else:
            words = []
            for i, r in enumerate(rows):
                given = teach or j <= r.force
                words.append(r.target[j - 1] if given and j - 1 < len(r.target)
                             else said[i][j - 1])
            x = np.stack([gr.spoken(w) if 0 <= w < gr.ACTIONS else gr.concept("SPEAK")
                          for w in words])
        if bar and j == rows[0].force and rows[0].family == "clause":
            bar_meaning(brain, rows)
        labels = (np.array([[r.target[j], *note_targets(r, heard, notes)] for r in rows])
                  if teach else None)
        action = moment(x, labels)
        for i, r in enumerate(rows):
            word = int(action[i, 0])
            said[i].append(word if word < gr.ACTIONS else -1)  # silence is no word
            if j >= r.force and word == gr.END:
                done[i] = True
        if not teach and all(done):
            break
    out = []
    for i, r in enumerate(rows):
        words = said[i][r.force:]
        out.append(words[:words.index(gr.END) + 1] if gr.END in words else words)
    return out


def notes_held(brain: Any, batches: list[list[gr.Row]], bar: bool,
               notes: tuple[str, ...] = NOTES) -> dict[str, float]:
    """Closed loop on a fork: the share of sentences whose every note is right at ``GO`` and
    at the last word said, and each note's share at ``GO``."""
    fork = copy.deepcopy(brain)
    at_go, at_end, each, n = 0, 0, np.zeros(len(notes)), 0
    for rows in batches:
        record: list[np.ndarray] = []
        said = speak_notes(fork, rows, teach=False, bar=bar, notes=notes, record=record)
        start = len(rows[0].pre)
        for i, r in enumerate(rows):
            want = np.array(note_targets(r, start, notes))
            go = record[start][i, 1:]
            last = record[min(start + max(len(said[i]) - 1, 0), len(record) - 1)][i, 1:]
            at_go += int((go == want).all())
            at_end += int((last == want).all())
            each += go == want
            n += 1
    return {"all_at_go": round(at_go / n, 4), "all_at_end": round(at_end / n, 4),
            **{name: round(float(v) / n, 4) for name, v in zip(notes, each, strict=True)}}


def notes_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life of a brain that restates its message: ``founder``, ``batches``, ``module``,
    ``settings`` over the default composition, ``bar`` (the read barred from a clause's meaning
    in lessons and readings), ``force_notes``, ``panel_batches``, ``tag``."""
    warnings.simplefilter("ignore")
    import cadence as cd

    founder = int(spec["founder"])
    batches = int(spec.get("batches", BATCHES))
    bar = bool(spec.get("bar", True))
    force = bool(spec.get("force_notes", True))
    started = time.monotonic()
    brain = notes_brain(founder, spec.get("settings"), int(spec.get("module", 64)))
    split = gr.Split.of(founder)
    tests = gr.panels(founder, int(spec.get("panel_batches", 4)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    row: dict[str, Any] = {
        "kind": "notes", "tag": spec.get("tag"), "founder": founder, "batches": batches,
        "bar": bar, "force_notes": force, "settings": spec.get("settings", {}),
        "module": int(spec.get("module", 64)), "parameters": int(brain.parameters()),
        "cadence": cd.__version__,
    }
    work = gr.new_work()
    for _ in range(batches):
        speak_notes(brain, gr.train_batch(rng, split, "g1"), teach=True, bar=bar,
                    force_notes=force, work=work)
    work["seconds"] = round(time.monotonic() - started, 1)
    row["work"] = work
    row["synapses"] = int(brain.describe()["layout"]["synapses"])
    row["gains"] = dict(brain.carried_gains) if brain.hearing is not None else {}
    fork = copy.deepcopy(brain)
    row["panels"] = {}
    for name in WHOLE + LONG:
        panel = gr.read_speaker(lambda rows: speak_notes(fork, rows, teach=False, bar=bar),
                                None, tests[name], twins=name in gr.TWINS)
        panel["examples"] = panel["examples"][:2]
        row["panels"][name] = panel
    row["notes"] = {name: notes_held(brain, tests[name], bar)
                    for name in ("clause", "clause_held", "roles")}
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


# --------------------------------------------------------------------- the declared run

DECLARED = tuple(range(9628600, 9628606))  # fresh founders (MEANING_PROTOCOL.md)
PHASED = {"code": "separator", "blocks": ["phase:4"], "winners": 5}
BESIDE = {**PHASED, "units": 70, "scales": [0.15]}  # the message beside a free read
ALONE = {**PHASED, "units": 280, "scales": [0.5]}  # the message the sentence is said from
STATE = {"module": 128, "settings": {"learning_surprise": False}}  # the state-only lives
LATER = [["g2", 1500], ["g1", 500]]  # persistence and recovery
KNOCK = (  # the sequence battery's tasks read with the message and without
    ("lookback", {"lengths": [5, 12], "test_lengths": [24, 60]}),
    ("repeat", {"lengths": [3, 6], "test_lengths": [8, 12]}),
    ("repeat", {"unique": False, "lengths": [3, 6], "test_lengths": [8, 12]}),
    ("induction", {"lengths": [8, 16], "test_lengths": [24, 60]}),
    ("nback", {}),
    ("twohop", {}),
    ("periodic", {"periods": [40]}),
)


def declared_plan(founders: tuple[int, ...] = DECLARED) -> dict[str, list[dict[str, Any]]]:
    """The declared run's lives: ``brains`` (this module's ``lives``), ``controls``
    (``controls.py``'s plan, each control at one brain arm's parameter count), ``knock`` (the
    other chambers, with the message and without), and the protocol's amendments
    (``amendments``, and ``amendments_knock`` for the other chambers at 256 association
    neurons)."""
    full = {"twins": True, "invariance": True, "after": LATER, "probe_scale": 1.0}
    arms: dict[str, dict[str, Any]] = {
        "default": {"arm": "default", **full},
        "barred": {"arm": "barred", "probe_scale": 0.0},
        "message": {"arm": "message", "message": BESIDE, **full},
        "message_wide": {"arm": "message", "message": {**ALONE, "scales": [0.05]},
                         "probe_scale": 0.0},
        "sham": {"arm": "message", "message": {**BESIDE, "sham": "present"},
                 "probe_scale": 0.0},
        "state": {"arm": "message_barred", "message": ALONE, **STATE, **full,
                  "probe_scale": 3.0},
        "state_default": {"arm": "message_barred", "message": ALONE, "probe_scale": 0.0},
        "restated": {"kind": "notes", "module": 128},
        "paraphrase": {"arm": "message_barred", **STATE, "train": "paraphrase",
                       "batches": 6000, "probe_scale": 0.0,
                       "message": {**PHASED, "blocks": ["phase:8"], "units": 140,
                                   "scales": [0.5]}},
    }
    arms["paraphrase_read"] = {"arm": "default", "train": "paraphrase", "batches": 6000,
                               "probe_scale": 0.0}
    for units in (70, 140, 280):
        arms[f"capacity_u{units}"] = {"arm": "message_barred", **STATE, "train": "capacity",
                                      "message": {**ALONE, "units": units}}
    for module in (64, 256):
        arms[f"capacity_m{module}"] = {"arm": "message_barred", "module": module,
                                       "settings": STATE["settings"], "train": "capacity",
                                       "message": ALONE}
    brains = [{**spec, "founder": f, "tag": tag} for tag, spec in arms.items() for f in founders]
    controls = []
    for tag in ("default", "message", "state"):
        spec = arms[tag]
        for f in founders:
            parameters = int(message_brain(f, spec.get("message"), spec.get("settings"),
                                           int(spec.get("module", 64))).parameters())
            for model in ("lstm", "transformer"):
                controls.append({"model": model, "mode": "online", "founder": f,
                                 "phases": [["g1", BATCHES], *LATER], "parameters": parameters})
                controls.append({"model": model, "mode": "offline", "founder": f,
                                 "phases": [["g1", BATCHES]], "parameters": parameters})
    knock = [{"kind": "knock", "chamber": "battery", "task": task, "task_args": args,
              "founder": f, "message": design, "tag": "message" if design else "default"}
             for task, args in KNOCK for f in founders
             for design in (None, {"units": 70, "scale": 0.15})]
    if (HERE / "routes.py").exists():
        knock += [{"kind": "knock", "chamber": "routes", "speaker": "word", "founder": f,
                   "batches": 8000, "message": design,
                   "tag": "message" if design else "default"}
                  for f in founders for design in (None, {"units": 70, "scale": 0.15})]
    # the protocol's amendments, each declared before its lives: the moment before as heard,
    # the state alone at 256 association neurons, and the wide message where nothing is barred
    wide = {"units": ALONE["units"], "scale": ALONE["scales"][0]}
    later: dict[str, dict[str, Any]] = {
        "heard": {"arm": "default", "settings": {"history_decays": [0.0]}, "twins": True,
                  "probe_scale": 1.0},
        "state_256": {"arm": "message_barred", "message": ALONE, "module": 256,
                      "settings": STATE["settings"], "twins": True, "invariance": True,
                      "probe_scale": 1.0},
        "wide_256": {"arm": "message", "message": ALONE, "module": 256, "probe_scale": 0.0},
        "plain_256": {"arm": "default", "module": 256, "probe_scale": 0.0},
    }
    amendments = [{**spec, "founder": f, "tag": tag} for tag, spec in later.items()
                  for f in founders]
    amendments_knock = [{**spec, "module": 256, "message": design,
                         "tag": "message" if design else "default"}
                        for spec in knock if spec["message"] is None
                        for design in (None, wide)]
    return {"brains": brains, "controls": controls, "knock": knock, "amendments": amendments,
            "amendments_knock": amendments_knock}


# ------------------------------------------------- the brain's own message (the second run)

OWN = tuple(range(9728600, 9728606))  # fresh founders (MEANING_PROTOCOL_2.md)
EVERY = {"learning_surprise": False}  # every row taught
HEARD = {"history_decays": [0.0]}  # the previous-word control: the moment before, as heard
OWN_MESSAGE = {"message": 280}  # the message the sentence is said from
# the message at the default width: quiet, and read one block at a time
OWN_DIRECT = {"message": 70, "message_amplitude": 0.45}
OWN_READ = {**OWN_DIRECT, "message_read": True}
OWN_QUIET = {"message": 70, "message_amplitude": 0.15}  # beside a free read
FIXED = {"carried_gain": 0.0}  # no learned gain on any carried route


def wide_width(model: str, target: int, limit: int = 640) -> int:
    """``controls.matched_width`` past its 256: the width whose parameter count is the nearest
    to ``target``."""
    ct = _controls()
    best = None
    for width in range(4, limit + 1, 2):
        n = ct.count(ct.build(model, width, 0))
        if best is None or abs(n - target) < abs(best[1] - target):
            best = (width, n)
        elif n > target:
            break
    assert best is not None
    return best[0]


def own_plan(founders: tuple[int, ...] = OWN) -> dict[str, list[dict[str, Any]]]:
    """The second declared run's lives, the message the brain's own (``Brain.compose``'s
    ``message``): ``brains`` and ``controls`` as ``declared_plan`` gives them. Every arm that
    holds a message has its previous-word control beside it, the same life with
    ``history_decays=(0.0,)`` in the message's place."""
    wide = {"arm": "barred", "module": 256}
    arms: dict[str, dict[str, Any]] = {
        "default": {"arm": "default", "twins": True, "probe_scale": 0.0},
        "heard": {"arm": "default", "settings": HEARD, "twins": True, "probe_scale": 0.0},
        "barred": {"arm": "barred", "probe_scale": 0.0},
        # said from the state alone
        "state_256": {**wide, "settings": {**OWN_MESSAGE, **EVERY}, "twins": True,
                      "invariance": True, "probe_scale": 3.0},
        "state_128": {**wide, "module": 128, "settings": {**OWN_MESSAGE, **EVERY},
                      "probe_scale": 0.0},
        "heard_256": {**wide, "settings": {**HEARD, **EVERY}, "probe_scale": 0.0},
        # at the default composition's width
        "read_64": {"arm": "barred", "settings": {**OWN_READ, **EVERY}, "twins": True,
                    "invariance": True, "probe_scale": 1.0},
        "direct_64": {"arm": "barred", "settings": {**OWN_DIRECT, **EVERY}, "probe_scale": 0.0},
        "heard_64": {"arm": "barred", "settings": {**HEARD, **EVERY}, "probe_scale": 0.0},
        # nothing barred: what the message costs beside a free read
        "message_free": {"arm": "default", "settings": OWN_MESSAGE, "probe_scale": 0.0},
        "message_quiet": {"arm": "default", "settings": OWN_QUIET, "probe_scale": 0.0},
        "message_fixed": {"arm": "default", "settings": {**OWN_QUIET, **FIXED},
                          "probe_scale": 0.0},
        "default_fixed": {"arm": "default", "settings": FIXED, "probe_scale": 0.0},
        # paraphrase and capacity, each beside its control
        "paraphrase": {**wide, "train": "paraphrase", "batches": 6000, "probe_scale": 0.0,
                       "settings": {"message": 140, "message_phases": 8, **EVERY}},
        "paraphrase_heard": {**wide, "train": "paraphrase", "batches": 6000,
                             "probe_scale": 0.0, "settings": {**HEARD, **EVERY}},
        "paraphrase_read": {"arm": "default", "train": "paraphrase", "batches": 6000,
                            "probe_scale": 0.0},
        "capacity_256": {**wide, "train": "capacity", "settings": {**OWN_MESSAGE, **EVERY}},
        "capacity_128": {**wide, "module": 128, "train": "capacity",
                         "settings": {**OWN_MESSAGE, **EVERY}},
        "capacity_heard": {**wide, "train": "capacity", "settings": {**HEARD, **EVERY}},
    }
    brains = [{**spec, "founder": f, "tag": tag} for tag, spec in arms.items() for f in founders]
    controls = []
    for tag in ("default", "read_64", "state_256"):
        spec = arms[tag]
        parameters = int(make_brain(founders[0], spec.get("settings"),
                                    int(spec.get("module", 64))).parameters())
        for model in ("lstm", "transformer"):
            width = wide_width(model, parameters)
            for f in founders:
                for mode in ("online", "offline"):
                    controls.append({"model": model, "mode": mode, "founder": f,
                                     "phases": [["g1", BATCHES]], "parameters": parameters,
                                     "width": width})
    return {"brains": brains, "controls": controls}


def own_verdict(brains: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[str]:
    """The gates of MEANING_PROTOCOL_2.md, read from the run's rows."""
    arms: dict[str, list[dict[str, Any]]] = {}
    for r in brains:
        arms.setdefault(str(r.get("tag")), []).append(r)
    for lives in arms.values():
        lives.sort(key=lambda r: r["founder"])

    def per(tag: str, get: Any) -> list[float | None]:
        return [get(r) for r in arms.get(tag, [])]

    def gate(name: str, values: list[float | None], bar: float) -> str:
        count, held = passes(values, bar)
        shown = " ".join("-" if v is None else f"{v:.2f}" for v in values)
        return f"  {name}: {shown}  ({count} of {len(values)} at {bar:.2f}: " + (
            "holds" if held else "fails") + ")"

    def less(tag: str, base: str, get: Any, floor: float = 0.05) -> str:
        theirs = {r["founder"]: r for r in arms.get(base, [])}
        lives = [r for r in arms.get(tag, []) if r["founder"] in theirs]
        if not lives:
            return "-"
        mean, error = paired([get(r) for r in lives], [get(theirs[r["founder"]]) for r in lives])
        mark = "*" if abs(mean) >= floor and abs(mean) > 2 * error else ""
        return f"{mean:+.2f} ({error:.2f}){mark}"

    shown = WHOLE + LONG
    lines = ["whole sentences said exactly (roles: both twins), mean (least-greatest)", "",
             "| Arm | Parameters | Lives | " + " | ".join(shown) + " |",
             "| --- | --- | --- | " + " | ".join("---" for _ in shown) + " |"]
    for tag, lives in arms.items():
        if "panels" in lives[0]:
            lines.append(f"| `{tag}` | {lives[0]['parameters']} | {len(lives)} | " + " | ".join(
                _mean([_score(r["panels"][n], n) for r in lives]) for n in shown) + " |")
    ranked: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for c in controls:
        ranked.setdefault((int(c["brain_parameters"]), c["model"], c["mode"]), []).append(c)
    for (_size, model, mode), lives in sorted(ranked.items()):
        first = [c["readings"][0] for c in lives]
        lines.append(f"| {model}, {mode} | {lives[0]['parameters']} | {len(lives)} | "
                     + " | ".join(_mean([_score(rd["panels"][n], n) for rd in first])
                                  for n in shown) + " |")
    lines.append("")
    for title, tag in (("H1, said from the state", "state_256"),
                       ("H1w, at the default width with a read of one block", "read_64"),
                       ("the same message without the read", "direct_64"),
                       ("the wide message at 128 association neurons", "state_128")):
        lines.append(f"{title} ({tag})")
        for name in ("clause", "clause_held", "roles"):
            lines.append(gate(name, per(tag, lambda r, n=name: _score(r["panels"][n], n)), 0.90))
    for tag in ("state_256", "read_64"):
        lines.append(f"H2, held and used ({tag})")
        for spot in ("go", "end"):
            def least(r: dict[str, Any], spot: str = spot) -> float | None:
                table = r.get("probes", {}).get("clause", {})
                found = [table[c][spot]["message"]["accuracy"] for c in table
                         if "message" in table[c].get(spot, {})]
                return min(found) if found else None
            lines.append(gate(f"least probe from the message at {spot}", per(tag, least), 0.95))
        lines.append(gate("least word kind said", per(tag, lambda r: min(
            v for k, v in r["said"].items() if k.startswith("clause"))), 0.95))
        lines.append(f"H3, one meaning in two orders ({tag})")
        lines.append(gate("key closer", per(tag, lambda r: r["invariance"]["key"]["closer"]),
                          0.95))
        lines.append(gate("both said right", per(tag, lambda r: r["invariance"]["both_right"]),
                          0.90))
        lines.append(f"context twins ({tag}): " + ", ".join(
            f"{n} " + _short([r["twins"][n]["both"] for r in arms.get(tag, [])])
            + " (message emptied " + _short([r["twins"][n + "_erased_message"]["both"]
                                             for r in arms.get(tag, [])]) + ")"
            for n in gr.ERASED))
    lines.append("H4, paraphrase")
    for tag in ("paraphrase", "paraphrase_heard", "paraphrase_read"):
        for name in ("para", "para_held"):
            lines.append(gate(f"{tag} {name}", per(
                tag, lambda r, n=name: r["paraphrase"][n]["exact"]), 0.80))
    lines.append("H5, capacity: clauses said exactly at 0 to 4 distractors (mean)")
    for tag in sorted(t for t in arms if t.startswith("capacity")):
        lines.append(f"  {tag} ({arms[tag][0]['parameters']} parameters): " + " ".join(
            _short([r["capacity"][n] for r in arms[tag]]) for n in arms[tag][0]["capacity"]))
    lines.append("C, the message less its previous-word control, paired (mean, standard error)")
    for tag, base in (("state_256", "heard_256"), ("read_64", "heard_64"),
                      ("read_64", "direct_64")):
        lines.append(f"  {tag} less {base}: " + ", ".join(
            f"{n} " + less(tag, base, lambda r, n=n: _score(r["panels"][n], n))
            for n in ("clause", "clause_held", "roles")))
    lines.append("  paraphrase less paraphrase_heard: " + ", ".join(
        f"{n} " + less("paraphrase", "paraphrase_heard",
                       lambda r, n=n: r["paraphrase"][n]["exact"])
        for n in ("para", "para_held")))
    lines.append("  capacity_256 less capacity_heard: " + ", ".join(
        f"{n} " + less("capacity_256", "capacity_heard", lambda r, n=n: r["capacity"][n])
        for n in ("L0", "L1", "L2", "L3", "L4")))
    lines.append("F, nothing barred: the arm less default, paired (the bar: paths and three "
                 "hops no more than .03 under)")
    for tag, base in (("message_free", "default"), ("message_quiet", "default"),
                      ("message_fixed", "default_fixed"), ("heard", "default")):
        lines.append(f"  {tag} less {base}: " + ", ".join(
            f"{n} " + less(tag, base, lambda r, n=n: _score(r["panels"][n], n), 0.03)
            for n in WHOLE + ("path_long",)))
    lines.append("  the same lives read with the message silent, then with the read barred")
    for tag in ("message_free", "message_quiet", "message_fixed"):
        for key in ("panels_silent", "panels_barred"):
            lives = [r for r in arms.get(tag, []) if key in r]
            if lives:
                lines.append(f"  {tag}, {key[7:]}: " + ", ".join(
                    f"{n} " + _mean([_score(r[key][n], n) for r in lives])
                    for n in WHOLE + ("path_long",) if n in lives[0][key]))
    lines.append("A1 lines on default and heard (lives at .90)")
    for tag in ("default", "heard"):
        cells = [f"{n} {passes(per(tag, lambda r, n=n: _score(r['panels'][n], n)), 0.9)[0]}"
                 for n in ("clause_held", "roles")]
        cells += [f"{n} {passes(per(tag, lambda r, n=n: r['twins'][n]['both']), 0.9)[0]}"
                  for n in gr.ERASED]
        lines.append(f"  {tag}: " + ", ".join(cells))
    lines.append("controls: whole sentences after the first grammar, by parameter count")
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for c in controls:
        groups.setdefault((int(c["brain_parameters"]), c["model"], c["mode"]), []).append(c)
    for (size, model, mode), lives in sorted(groups.items()):
        first = [c["readings"][0] for c in lives]
        lines.append(f"  {model} {mode} at {size} ({lives[0]['parameters']}): " + ", ".join(
            f"{n} " + _mean([_score(rd["panels"][n], n) for rd in first])
            for n in WHOLE + ("path_long",))
            + f"; {np.mean([rd['work']['macs'] for rd in first]):.2e} multiply-accumulates")
    lines.append("work: multiply-accumulates and seconds per life")
    for tag, lives in arms.items():
        work = [macs(r) for r in lives]
        lines.append(f"  {tag}: " + (f"{np.mean([w for w in work if w]):.2e}" if any(work)
                                     else "-")
                     + f", {np.mean([r['work']['seconds'] for r in lives]):.0f} s")
    return lines


# ------------------------------------------- the third run: the cost beside a free read, roles

THIRD = tuple(range(9828600, 9828606))  # fresh founders (MEANING_PROTOCOL_3.md)
GO_CHANNEL = int(np.flatnonzero(gr.concept("GO"))[0])  # the boundary the brain can see
SIX = WHOLE + ("path_long",)  # the six panels of the third run's first part
WIDE_OWN = {"arm": "barred", "module": 256, "settings": {**OWN_MESSAGE, **EVERY},
            "probe_scale": 0.0}  # the brain's own message, said from the state at 256


def third_plan(founders: tuple[int, ...] = THIRD,
               located: tuple[int, ...] = OWN) -> dict[str, list[dict[str, Any]]]:
    """The third declared run. ``free``: the quiet message beside a free read, the arms that test
    the located cost (the phase count scaled to the longest episode and past it, a message held
    or cleared at ``GO``, the amplitude a notch lower), each beside ``default`` and ``heard``.
    ``roles``: the brain's own message at 256 association neurons beside the instrument's
    message on the same founders (``located``, the second run's) and the brain's own message
    heard through one gain per 35 neurons, the instrument's read; then the same on fresh
    founders."""
    quiet = OWN_QUIET
    free: dict[str, dict[str, Any]] = {
        "default": {"arm": "default", "probe_scale": 0.0},
        "heard": {"arm": "default", "settings": HEARD, "probe_scale": 0.0},
        "quiet": {"arm": "default", "settings": quiet, "probe_scale": 0.0},
        "quiet_24": {"arm": "default", "settings": {**quiet, "message_phases": 24},
                     "probe_scale": 0.0},
        "quiet_32": {"arm": "default", "settings": {**quiet, "message_phases": 32},
                     "probe_scale": 0.0},
        "quiet_hold": {"arm": "default", "probe_scale": 0.0,
                       "settings": {**quiet, "message_boundary": GO_CHANNEL,
                                    "message_boundary_holds": True}},
        "quiet_clear": {"arm": "default", "probe_scale": 0.0,
                        "settings": {**quiet, "message_boundary": GO_CHANNEL}},
        "quiet_10": {"arm": "default", "settings": {**quiet, "message_amplitude": 0.10},
                     "probe_scale": 0.0},
    }
    roles: dict[str, dict[str, Any]] = {
        "own_256": WIDE_OWN,
        "instrument_256": {"arm": "message_barred", "message": ALONE, "module": 256,
                           "settings": EVERY, "probe_scale": 0.0},
        "fine_256": {**WIDE_OWN, "settings": {**WIDE_OWN["settings"], "message_gains": 8}},
    }
    return {
        "free": [{**spec, "founder": f, "tag": tag} for tag, spec in free.items()
                 for f in founders],
        "roles_located": [{**spec, "founder": f, "tag": tag} for tag, spec in roles.items()
                          for f in located],
        "roles": [{**spec, "founder": f, "tag": tag} for tag, spec in roles.items()
                  for f in founders if tag != "instrument_256"],
    }


CONFIRM = tuple(range(9928600, 9928612))  # twelve fresh founders (protocol 3, Amendment 1)
HOLD = {**OWN_QUIET, "message_boundary": GO_CHANNEL, "message_boundary_holds": True}


def third_confirm_plan(founders: tuple[int, ...] = CONFIRM) -> list[dict[str, Any]]:
    """Protocol 3, Amendment 1: the message held from ``GO`` on twelve founders, beside the
    default composition, the previous-word signal, and both together."""
    arms: dict[str, dict[str, Any]] = {
        "default": {"arm": "default", "probe_scale": 0.0},
        "heard": {"arm": "default", "settings": HEARD, "probe_scale": 0.0},
        "quiet_hold": {"arm": "default", "settings": HOLD, "probe_scale": 0.0},
        "quiet_hold_heard": {"arm": "default", "settings": {**HOLD, **HEARD}, "probe_scale": 0.0},
    }
    return [{**spec, "founder": f, "tag": tag} for tag, spec in arms.items() for f in founders]


def third_confirm_verdict(rows: list[dict[str, Any]], floor: float = 0.03) -> list[str]:
    """Gate F3c: for each of the six panels the paired mean difference against ``default`` is at
    or above ``-floor`` and the default is not above the arm by two standard errors; a lift
    counts where it clears two standard errors. ``quiet_hold_heard`` is read against ``heard``
    as well."""
    arms = _by_tag(rows)
    lines = ["whole sentences said exactly (roles: both twins), mean (least-greatest)", ""]
    lines += _panel_table(arms, SIX) + [""]
    for base, tags in (("default", ("quiet_hold", "quiet_hold_heard", "heard")),
                       ("heard", ("quiet_hold_heard",))):
        theirs = {r["founder"]: r for r in arms.get(base, [])}
        lines.append(f"each arm less `{base}`, paired by founder: mean (standard error); "
                     "* a lift beyond two standard errors, - a cost beyond them")
        for tag in tags:
            lives = [r for r in arms.get(tag, []) if r["founder"] in theirs]
            if not lives:
                continue
            cells, held = [], True
            for n in SIX:
                mean, error = paired(
                    [_score(r["panels"][n], n) or 0.0 for r in lives],
                    [_score(theirs[r["founder"]]["panels"][n], n) or 0.0 for r in lives])
                cost = mean < 0 and -mean > 2 * error
                lift = mean > 0 and mean > 2 * error
                held = held and mean >= -floor and not cost
                cells.append(f"{n} {mean:+.2f} ({error:.2f})" + ("*" if lift else "-" if cost
                                                                  else ""))
            verdict = f"  -> F3c {'holds' if held else 'fails'}" if base == "default" else ""
            lines.append(f"  {tag}: " + ", ".join(cells) + verdict)
        lines.append("")
    lines.append("by founder: path / path_long")
    for tag, lives in arms.items():
        lines.append(f"  {tag}: " + " ".join(
            f"{_score(r['panels']['path'], 'path'):.2f}" for r in lives) + " / " + " ".join(
            f"{_score(r['panels']['path_long'], 'path_long'):.2f}" for r in lives))
    lines.append("learned gains (mean over founders)")
    for tag, lives in arms.items():
        names = list(lives[0].get("gains", {}))
        if names:
            lines.append(f"  {tag}: " + ", ".join(
                f"{k} {np.mean([r['gains'][k] for r in lives]):.2f}" for k in names))
    lines.append("work: seconds per life")
    lines.append("  " + ", ".join(f"{tag} {np.mean([r['seconds'] for r in lives]):.0f} s"
                                  for tag, lives in arms.items()))
    return lines


FOURTH = tuple(range(10028600, 10028612))  # twelve fresh founders (MEANING_PROTOCOL_4.md)


def fourth_plan(founders: tuple[int, ...] = FOURTH) -> list[dict[str, Any]]:
    """Protocol 4, part 1: the brain's own message at 256 association neurons, the read barred,
    every row taught, with the read of one block, the previous word, or both beside it."""
    base = {"arm": "barred", "module": 256, "probe_scale": 0.0, "held_probes": 8}
    arms: dict[str, dict[str, Any]] = {
        "own_256": {**OWN_MESSAGE, **EVERY},
        "read_256": {**OWN_MESSAGE, "message_read": True, **EVERY},
        "heard_own_256": {**OWN_MESSAGE, **HEARD, **EVERY},
        "both_256": {**OWN_MESSAGE, "message_read": True, **HEARD, **EVERY},
    }
    return [{**base, "settings": settings, "founder": f, "tag": tag}
            for tag, settings in arms.items() for f in founders]


def fourth_verdict(rows: list[dict[str, Any]], floor: float = 0.03, of: int = 10,
                   base: str = "own_256") -> list[str]:
    """Gate H1c: ``clause_held`` at .90 on ``of`` founders of twelve, ``clause`` and ``roles``
    at .90 on as many, and on ``dyck``, ``path`` and three hops the paired mean against
    ``base`` at or above ``-floor`` and not under it by two standard errors."""
    arms = _by_tag(rows)
    shown = ("clause", "clause_held", "roles", "dyck", "path", "path_long")
    lines = ["whole sentences said from the state (roles: both twins), mean (least-greatest)", "",
             "| Arm | Parameters | Multiply-accumulates | Seconds | Lives | "
             + " | ".join(shown) + " |",
             "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in shown) + " |"]
    for tag, lives in arms.items():
        work = [macs(r) or 0.0 for r in lives]
        lines.append(f"| `{tag}` | {lives[0]['parameters']} | {np.mean(work):.2e} | "
                     f"{np.mean([r['seconds'] for r in lives]):.0f} | {len(lives)} | "
                     + " | ".join(_mean([_score(r["panels"][n], n) for r in lives])
                                  for n in shown) + " |")
    lines.append("")
    theirs = {r["founder"]: r for r in arms.get(base, [])}
    for tag, lives in arms.items():
        cells, held = [], True
        for n in ("clause", "clause_held", "roles"):
            values = [_score(r["panels"][n], n) for r in lives]
            count, _ = passes(values, 0.9, of)
            held = held and count >= of
            cells.append(f"{n} {count} of {len(values)} at .90 ("
                         + " ".join(f"{v:.2f}" for v in values if v is not None) + ")")
        paired_lives = [r for r in lives if r["founder"] in theirs]
        if tag != base and paired_lives:
            for n in shown:
                mean, error = paired(
                    [_score(r["panels"][n], n) or 0.0 for r in paired_lives],
                    [_score(theirs[r["founder"]]["panels"][n], n) or 0.0
                     for r in paired_lives])
                cost = mean < 0 and -mean > 2 * error
                lift = mean > 0 and mean > 2 * error
                if n in ("dyck", "path", "path_long"):
                    held = held and mean >= -floor and not cost
                cells.append(f"{n} less {base} {mean:+.2f} ({error:.2f})"
                             + ("*" if lift else "-" if cost else ""))
        lines.append(f"  {tag}: " + "; ".join(cells) + f"  -> H1c {'holds' if held else 'fails'}")
    lines.append("")
    lines.append("held-out clauses with the previous words given, by word kind (least founder)")
    for tag, lives in arms.items():
        said = [r["said_held"] for r in lives if "said_held" in r]
        if said:
            kinds = sorted(k for k in said[0] if k.startswith("clause"))
            lines.append(f"  {tag}: " + ", ".join(
                f"{k[7:]} {np.mean([d[k] for d in said]):.2f} ({min(d[k] for d in said):.2f})"
                for k in kinds))
    lines.append("probes fitted on trained combinations, scored on held-out ones, at GO: "
                 "the least component per place, by founder")
    for tag, lives in arms.items():
        for place in ("message", "settled"):
            cells = []
            for r in lives:
                table = r.get("probes_held", {})
                found = [(table[c]["go"][place]["accuracy"], c) for c in table
                         if place in table[c].get("go", {})]
                if found:
                    low = min(found)
                    cells.append(f"{low[0]:.2f} {low[1]}")
            if cells:
                lines.append(f"  {tag} {place}: " + "; ".join(cells))
    return lines


FOURTH_LONG = tuple(range(10128600, 10128606))  # six fresh founders (protocol 4, Amendment 1)


def fourth_long_plan(founders: tuple[int, ...] = FOURTH_LONG) -> dict[str, list[dict[str, Any]]]:
    """Protocol 4, Amendment 1: longer clauses, deeper brackets and longer paths from the state
    on ``both_256``, the default composition with its read free beside it, and the chamber's
    LSTM and transformer at ``both_256``'s parameter count."""
    winner = {**OWN_MESSAGE, "message_read": True, **HEARD, **EVERY}
    arms: dict[str, dict[str, Any]] = {
        "both_256": {"arm": "barred", "module": 256, "settings": winner, "probe_scale": 0.0,
                     "long_probes": 8},
        "default": {"arm": "default", "probe_scale": 0.0, "long_probes": 8},
    }
    brains = [{**spec, "founder": f, "tag": tag} for tag, spec in arms.items() for f in founders]
    parameters = int(make_brain(founders[0], winner, 256).parameters())
    controls = [{"model": model, "mode": mode, "founder": f, "phases": [["g1", BATCHES]],
                 "parameters": parameters, "width": wide_width(model, parameters)}
                for model in ("lstm", "transformer") for mode in ("online", "offline")
                for f in founders]
    return {"long": brains, "long_controls": controls}


def fourth_long_verdict(rows: list[dict[str, Any]], controls: list[dict[str, Any]],
                        bar: float = 0.8, of: int = 5) -> list[str]:
    """Gate A2s: ``clause_long``, ``dyck_deep`` and ``path_long`` at ``bar`` on ``of`` founders
    of six, said from the state; the controls and the located terms beside it."""
    arms = _by_tag(rows)
    shown = ("clause", "clause_held", "roles") + LONG
    lines = ["whole sentences (roles: both twins), mean (least-greatest)", "",
             "| Learner | Parameters | Multiply-accumulates | Lives | " + " | ".join(shown) + " |",
             "| --- | --- | --- | --- | " + " | ".join("---" for _ in shown) + " |"]
    for tag, lives in arms.items():
        lines.append(f"| `{tag}` | {lives[0]['parameters']} | "
                     f"{np.mean([macs(r) or 0.0 for r in lives]):.2e} | {len(lives)} | "
                     + " | ".join(_mean([_score(r["panels"][n], n) for r in lives])
                                  for n in shown) + " |")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in controls:
        groups.setdefault((c["model"], c["mode"]), []).append(c)
    for (model, mode), lives in sorted(groups.items()):
        first = [c["readings"][0] for c in lives]
        lines.append(f"| {model}, {mode} | {lives[0]['parameters']} | "
                     f"{np.mean([rd['work']['macs'] for rd in first]):.2e} | {len(lives)} | "
                     + " | ".join(_mean([_score(rd["panels"][n], n) for rd in first])
                                  for n in shown) + " |")
    lines.append("")
    for tag, lives in arms.items():
        cells, held = [], True
        for n in LONG:
            values = [_score(r["panels"][n], n) for r in lives]
            count, _ = passes(values, bar, of)
            held = held and count >= of
            cells.append(f"{n} {count} of {len(values)} at {bar:.2f}")
        lines.append(f"  {tag}: " + ", ".join(cells) + f"  -> A2s {'holds' if held else 'fails'}")
    lines.append("the long panels with the previous words given, by word kind: mean (least)")
    for tag, lives in arms.items():
        said = [r["said_long"] for r in lives if "said_long" in r]
        if said:
            lines.append(f"  {tag}: " + ", ".join(
                f"{k} {np.mean([d[k] for d in said]):.2f} ({min(d[k] for d in said):.2f})"
                for k in sorted(said[0])))
    lines.append("probes fitted on trained shapes, scored on longer clauses, at GO "
                 "(message / settled), mean")
    for tag, lives in arms.items():
        tables = [r["probes_long"] for r in lives if r.get("probes_long")]
        if tables:
            cells = []
            for c in sorted(tables[0]):
                at = [t[c]["go"] for t in tables if c in t and "go" in t[c]]
                if at:
                    cells.append(f"{c} " + " / ".join(
                        f"{np.mean([a[place]['accuracy'] for a in at]):.2f}"
                        for place in ("message", "settled") if place in at[0]))
            lines.append(f"  {tag}: " + ", ".join(cells))
    lines.append("the default composition's read on the long panels (delivered, said when "
                 "delivered), mean")
    for r_tag, lives in arms.items():
        found = [r["located"] for r in lives if "located" in r]
        if found:
            for name in ("clause_long", "dyck_deep", "path_long"):
                terms = sorted(found[0][name])
                lines.append(f"  {r_tag} {name}: " + ", ".join(
                    f"{t} " + _short([f[name][t]["delivered"] for f in found]) + " "
                    + _short([f[name][t]["said_when_delivered"] for f in found])
                    for t in terms))
    return lines


WINNERS = tuple(range(10228600, 10228612))  # twelve fresh founders (protocol 4, Amendment 2)
FIFTH = tuple(range(10328600, 10328606))  # six fresh founders (MEANING_PROTOCOL_5.md)
BOTH = {**OWN_MESSAGE, "message_read": True, **HEARD, **EVERY}  # the fourth run's both_256


def winners_plan(founders: tuple[int, ...] = WINNERS) -> list[dict[str, Any]]:
    """Protocol 4, Amendment 2: ``both_256`` at 10 winners a moment beside 5."""
    base = {"arm": "barred", "module": 256, "probe_scale": 0.0}
    arms = {"both_256": BOTH, "both_256_w10": {**BOTH, "message_winners": 10}}
    return [{**base, "settings": settings, "founder": f, "tag": tag}
            for tag, settings in arms.items() for f in founders]


def fifth_plan(founders: tuple[int, ...] = FIFTH) -> list[dict[str, Any]]:
    """Protocol 5: ``both_256`` with as many blocks as a four-distractor clause holds heard
    moments (7), one past it (8), and as many as its whole episode (16), beside four."""
    base = {"arm": "barred", "module": 256, "probe_scale": 0.0, "long_probes": 8}
    arms = {"both_256": BOTH, "both_p7": {**BOTH, "message_phases": 7},
            "both_p8": {**BOTH, "message_phases": 8},
            "both_p16": {**BOTH, "message_phases": 16}}
    return [{**base, "settings": settings, "founder": f, "tag": tag}
            for tag, settings in arms.items() for f in founders]


def fifth_verdict(rows: list[dict[str, Any]], floor: float = 0.03, of: int = 5,
                  base: str = "both_256") -> list[str]:
    """Gate A2c: ``clause_long`` at .80 on ``of`` founders of six, ``clause``,
    ``clause_held`` and ``roles`` at .90 on as many, and on ``dyck``, ``path`` and three hops
    the paired mean against ``base`` at or above ``-floor`` and not under it by two standard
    errors. Deeper brackets and three hops are read and located beside it."""
    arms = _by_tag(rows)
    shown = ("clause", "clause_held", "roles", "dyck", "path", "path_long", "clause_long",
             "dyck_deep")
    lines = ["whole sentences said from the state (roles: both twins), mean (least-greatest)", "",
             "| Arm | Parameters | Multiply-accumulates | Seconds | Lives | "
             + " | ".join(shown) + " |",
             "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in shown) + " |"]
    for tag, lives in arms.items():
        lines.append(f"| `{tag}` | {lives[0]['parameters']} | "
                     f"{np.mean([macs(r) or 0.0 for r in lives]):.2e} | "
                     f"{np.mean([r['seconds'] for r in lives]):.0f} | {len(lives)} | "
                     + " | ".join(_mean([_score(r["panels"][n], n) for r in lives])
                                  for n in shown) + " |")
    lines.append("")
    theirs = {r["founder"]: r for r in arms.get(base, [])}
    for tag, lives in arms.items():
        cells, held = [], True
        for n, bar in (("clause_long", 0.8), ("clause", 0.9), ("clause_held", 0.9),
                       ("roles", 0.9)):
            values = [_score(r["panels"][n], n) for r in lives]
            count, _ = passes(values, bar, of)
            held = held and count >= of
            cells.append(f"{n} {count} of {len(values)} at {bar:.2f} ("
                         + " ".join(f"{v:.2f}" for v in values if v is not None) + ")")
        paired_lives = [r for r in lives if r["founder"] in theirs]
        if tag != base and paired_lives:
            for n in shown:
                mean, error = paired(
                    [_score(r["panels"][n], n) or 0.0 for r in paired_lives],
                    [_score(theirs[r["founder"]]["panels"][n], n) or 0.0
                     for r in paired_lives])
                cost = mean < 0 and -mean > 2 * error
                lift = mean > 0 and mean > 2 * error
                if n in ("dyck", "path", "path_long"):
                    held = held and mean >= -floor and not cost
                cells.append(f"{n} less {base} {mean:+.2f} ({error:.2f})"
                             + ("*" if lift else "-" if cost else ""))
        lines.append(f"  {tag}: " + "; ".join(cells) + f"  -> A2c {'holds' if held else 'fails'}")
    lines.append("")
    lines.append("the long panels with the previous words given, by word kind: mean (least)")
    for tag, lives in arms.items():
        said = [r["said_long"] for r in lives if "said_long" in r]
        if said:
            lines.append(f"  {tag}: " + ", ".join(
                f"{k} {np.mean([d[k] for d in said]):.2f} ({min(d[k] for d in said):.2f})"
                for k in sorted(said[0])))
    lines.append("probes fitted on trained shapes, scored on longer clauses, at GO "
                 "(message / settled), mean")
    for tag, lives in arms.items():
        tables = [r["probes_long"] for r in lives if r.get("probes_long")]
        if tables:
            cells = []
            for c in sorted(tables[0]):
                at = [t[c]["go"] for t in tables if c in t and "go" in t[c]]
                if at:
                    cells.append(f"{c} " + " / ".join(
                        f"{np.mean([a[place]['accuracy'] for a in at]):.2f}"
                        for place in ("message", "settled") if place in at[0]))
            lines.append(f"  {tag}: " + ", ".join(cells))
    lines.append("learned gains (mean over founders)")
    for tag, lives in arms.items():
        names = list(lives[0].get("gains", {}))
        lines.append(f"  {tag}: " + ", ".join(
            f"{k} {np.mean([r['gains'][k] for r in lives]):.2f}" for k in names))
    return lines


SIXTH = tuple(range(10428600, 10428624))  # 24 fresh founders (MEANING_PROTOCOL_6.md)
# a moment's role chooses its blocks: the agent, the patient, a distractor, the verb, GO and the
# brain's own words, each a group of channels; brackets and paths join the last group
ROLES = tuple((gr.C[name],) for name in ("AGENT", "PATIENT", "MOD")) + (
    (gr.C["SEE"], gr.C["CHASE"]), (gr.C["GO"],), (gr.C["SPEAK"],))


def sixth_plan(founders: tuple[int, ...] = SIXTH) -> dict[str, list[dict[str, Any]]]:
    """Protocol 6 (cadence#301): the message with ``GO`` in a block of its own (8 blocks) on the
    first twelve founders, and the message keyed by role on all 24, each beside ``both_256``;
    the chamber's LSTM and transformer at the two arms' parameter counts on the first six."""
    base = {"arm": "barred", "module": 256, "probe_scale": 0.0, "held_probes": 4,
            "long_probes": 4}
    arms: dict[str, tuple[dict[str, Any], tuple[int, ...]]] = {
        "both_256": (BOTH, founders),
        "both_p8": ({**BOTH, "message_phases": 8}, founders[:12]),
        "role_256": ({**BOTH, "message_roles": [list(g) for g in ROLES]}, founders),
    }
    brains = [{**base, "settings": settings, "founder": f, "tag": tag}
              for tag, (settings, chosen) in arms.items() for f in chosen]
    controls = []
    for tag in ("both_p8", "role_256"):
        parameters = int(make_brain(founders[0], arms[tag][0], 256).parameters())
        for model in ("lstm", "transformer"):
            width = wide_width(model, parameters, limit=1200)
            for mode in ("online", "offline"):
                controls += [{"model": model, "mode": mode, "founder": f,
                              "phases": [["g1", BATCHES]], "parameters": parameters,
                              "width": width} for f in founders[:6]]
    return {"sixth": brains, "sixth_controls": controls}


def sixth_verdict(rows: list[dict[str, Any]], controls: list[dict[str, Any]],
                  floor: float = 0.03) -> list[str]:
    """Protocol 6's gates. G8: ``both_p8`` at .90 on 10 of 12 for clauses, held-out
    combinations and role twins. R1: ``role_256`` at .90 on 20 of 24 for the same three. R2:
    ``role_256``'s ``clause_long`` at .80 on 5 of the first 6. Each arm's brackets, paths and
    three hops against ``both_256``, paired: at or above ``-floor`` and not under it by two
    standard errors."""
    arms = _by_tag(rows)
    shown = ("clause", "clause_held", "roles", "dyck", "path", "path_long", "clause_long",
             "dyck_deep")
    lines = ["whole sentences said from the state (roles: both twins), mean (least-greatest)", "",
             "| Learner | Parameters | Multiply-accumulates | Seconds | Lives | "
             + " | ".join(shown) + " |",
             "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in shown) + " |"]
    for tag, lives in arms.items():
        lines.append(f"| `{tag}` | {lives[0]['parameters']} | "
                     f"{np.mean([macs(r) or 0.0 for r in lives]):.2e} | "
                     f"{np.mean([r['seconds'] for r in lives]):.0f} | {len(lives)} | "
                     + " | ".join(_mean([_score(r["panels"][n], n) for r in lives])
                                  for n in shown) + " |")
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for c in controls:
        groups.setdefault((int(c["brain_parameters"]), c["model"], c["mode"]), []).append(c)
    for (size, model, mode), lives in sorted(groups.items()):
        first = [c["readings"][0] for c in lives]
        lines.append(f"| {model}, {mode} at {size} | {lives[0]['parameters']} | "
                     f"{np.mean([rd['work']['macs'] for rd in first]):.2e} | "
                     f"{np.mean([c['seconds'] for c in lives]):.0f} | {len(lives)} | "
                     + " | ".join(_mean([_score(rd["panels"][n], n) for rd in first])
                                  for n in shown) + " |")
    lines.append("")
    base = {r["founder"]: r for r in arms.get("both_256", [])}
    first_six = sorted(base)[:6]
    for tag, need in (("both_256", None), ("both_p8", 10), ("role_256", 20)):
        lives = arms.get(tag, [])
        if not lives:
            continue
        held, cells = True, []
        for n in ("clause", "clause_held", "roles"):
            values = [_score(r["panels"][n], n) for r in lives]
            count = sum(v is not None and v >= 0.9 for v in values)
            held = held and need is not None and count >= need
            cells.append(f"{n} {count} of {len(values)} at .90")
        paired_lives = [r for r in lives if r["founder"] in base]
        if tag != "both_256" and paired_lives:
            for n in shown:
                mean, error = paired(
                    [_score(r["panels"][n], n) or 0.0 for r in paired_lives],
                    [_score(base[r["founder"]]["panels"][n], n) or 0.0 for r in paired_lives])
                cost = mean < 0 and -mean > 2 * error
                lift = mean > 0 and mean > 2 * error
                if n in ("dyck", "path", "path_long"):
                    held = held and mean >= -floor and not cost
                cells.append(f"{n} less both_256 {mean:+.2f} ({error:.2f})"
                             + ("*" if lift else "-" if cost else ""))
        gate = {"both_p8": "G8", "role_256": "R1"}.get(tag)
        lines.append(f"  {tag}: " + "; ".join(cells)
                     + (f"  -> {gate} {'holds' if held else 'fails'}" if gate else ""))
    for tag in ("both_256", "both_p8", "role_256"):
        lives = [r for r in arms.get(tag, []) if r["founder"] in first_six]
        if lives:
            values = [_score(r["panels"]["clause_long"], "clause_long") for r in lives]
            count = sum(v is not None and v >= 0.8 for v in values)
            lines.append(f"  {tag} clause_long on the first six: "
                         + " ".join(f"{v:.2f}" for v in values if v is not None)
                         + f" ({count} of {len(values)} at .80)"
                         + ("  -> R2 " + ("holds" if count >= 5 else "fails")
                            if tag == "role_256" else ""))
    lines.append("")
    lines.append("the long panels with the previous words given, by word kind: mean (least)")
    for tag, lives in arms.items():
        said = [r["said_long"] for r in lives if "said_long" in r]
        if said:
            lines.append(f"  {tag}: " + ", ".join(
                f"{k} {np.mean([d[k] for d in said]):.2f} ({min(d[k] for d in said):.2f})"
                for k in sorted(said[0]) if k.startswith("clause")))
    lines.append("held-out clauses with the previous words given, by word kind (least founder)")
    for tag, lives in arms.items():
        said = [r["said_held"] for r in lives if "said_held" in r]
        if said:
            lines.append(f"  {tag}: " + ", ".join(
                f"{k[7:]} {np.mean([d[k] for d in said]):.2f} ({min(d[k] for d in said):.2f})"
                for k in sorted(said[0]) if k.startswith("clause")))
    for title, key in (("longer clauses", "probes_long"), ("held-out combinations",
                                                          "probes_held")):
        lines.append(f"probes fitted on trained shapes, scored on {title}, at GO "
                     "(message / settled), mean")
        for tag, lives in arms.items():
            tables = [r[key] for r in lives if r.get(key)]
            if tables:
                cells = []
                for c in sorted(tables[0]):
                    at = [t[c]["go"] for t in tables if c in t and "go" in t[c]]
                    if at:
                        cells.append(f"{c} " + " / ".join(
                            f"{np.mean([a[place]['accuracy'] for a in at]):.2f}"
                            for place in ("message", "settled") if place in at[0]))
                lines.append(f"  {tag}: " + ", ".join(cells))
    return lines


SEVENTH = tuple(range(10528600, 10528612))  # twelve fresh founders (MEANING_PROTOCOL_7.md)


def seventh_plan(founders: tuple[int, ...] = SEVENTH) -> dict[str, list[dict[str, Any]]]:
    """Protocol 7 (cadence#302): a word said by its parts. From the held state (the read
    barred, 256 association neurons): words, parts, and words at ten message winners, each also
    read with the message silent. With the read free (the 0.97.0 default at 64): words and
    parts. The chamber's LSTM and transformer at ``parts_256``'s count on the first six."""
    held = {"arm": "barred", "module": 256, "probe_scale": 0.0, "held_probes": 4,
            "silent_reading": True}
    free = {"arm": "default", "probe_scale": 0.0, "held_probes": 4}
    arms: dict[str, dict[str, Any]] = {
        "words_256": {**held, "settings": BOTH},
        "parts_256": {**held, "settings": {**BOTH, "parts": True}},
        "w10_256": {**held, "settings": {**BOTH, "message_winners": 10}},
        "words_free": {**free, "settings": {}},
        "parts_free": {**free, "settings": {"parts": True}},
    }
    brains = [{**spec, "founder": f, "tag": tag} for tag, spec in arms.items() for f in founders]
    parameters = int(make_brain(founders[0], arms["parts_256"]["settings"], 256).parameters())
    controls = [{"model": model, "mode": mode, "founder": f, "phases": [["g1", BATCHES]],
                 "parameters": parameters, "width": wide_width(model, parameters)}
                for model in ("lstm", "transformer") for mode in ("online", "offline")
                for f in founders[:6]]
    return {"seventh": brains, "seventh_controls": controls}


def _verb_said(said: dict[str, float]) -> float:
    """The verb's form said on held-out clauses with the previous words given, over shapes."""
    return float(np.mean([v for k, v in said.items() if k.startswith("clause") and
                          k.endswith("_verb")]))


def _verb_probe(table: dict[str, Any]) -> float:
    """The settled association's probe of the verb and of the agent's number at the settle
    that says the verb, the lesser of the two."""
    return float(min(table[c]["verb"]["settled"]["accuracy"] for c in ("verb", "agent_number")))


def seventh_verdict(rows: list[dict[str, Any]], controls: list[dict[str, Any]],
                    floor: float = 0.03, of: int = 10) -> list[str]:
    """Protocol 7's gates. P1: ``parts_256`` says the verb's form on held-out clauses at .95 on
    ``of`` of twelve, its mean within ``floor`` of the probe's. P2: ``parts_256``'s
    ``clause_held`` at .90 on ``of`` of twelve. P3: no panel more than ``floor`` under the
    reference and not under it by two standard errors, paired: ``parts_256`` against
    ``words_256`` and ``parts_free`` against ``words_free``."""
    arms = _by_tag(rows)
    shown = ("clause", "clause_held", "roles", "dyck", "path", "path_long", "clause_long")
    lines = ["whole sentences (roles: both twins), mean (least-greatest)", "",
             "| Learner | Parameters | Multiply-accumulates | Seconds | Lives | "
             + " | ".join(shown) + " |",
             "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in shown) + " |"]
    for tag, lives in arms.items():
        lines.append(f"| `{tag}` | {lives[0]['parameters']} | "
                     f"{np.mean([macs(r) or 0.0 for r in lives]):.2e} | "
                     f"{np.mean([r['seconds'] for r in lives]):.0f} | {len(lives)} | "
                     + " | ".join(_mean([_score(r["panels"][n], n) for r in lives])
                                  for n in shown) + " |")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in controls:
        groups.setdefault((c["model"], c["mode"]), []).append(c)
    for (model, mode), lives in sorted(groups.items()):
        first = [c["readings"][0] for c in lives]
        lines.append(f"| {model}, {mode} | {lives[0]['parameters']} | "
                     f"{np.mean([rd['work']['macs'] for rd in first]):.2e} | "
                     f"{np.mean([c['seconds'] for c in lives]):.0f} | {len(lives)} | "
                     + " | ".join(_mean([_score(rd["panels"][n], n) for rd in first])
                                  for n in shown) + " |")
    lines.append("")
    lines.append("the verb's form on held-out clauses, previous words given (said / probe, by "
                 "founder)")
    for tag, lives in arms.items():
        cells = [f"{_verb_said(r['said_held']):.2f}/{_verb_probe(r['probes_held']):.2f}"
                 for r in lives if "said_held" in r and r.get("probes_held")]
        if cells:
            lines.append(f"  {tag}: " + " ".join(cells))
    parts = arms.get("parts_256", [])
    if parts:
        said = [_verb_said(r["said_held"]) for r in parts]
        probe = [_verb_probe(r["probes_held"]) for r in parts]
        count = sum(v >= 0.95 for v in said)
        near = abs(float(np.mean(said)) - float(np.mean(probe))) <= floor
        lines.append(f"  P1: {count} of {len(said)} at .95, said {np.mean(said):.2f} against "
                     f"the probe's {np.mean(probe):.2f} -> "
                     + ("holds" if count >= of and near else "fails"))
        values = [_score(r["panels"]["clause_held"], "clause_held") for r in parts]
        count = sum(v is not None and v >= 0.9 for v in values)
        lines.append(f"  P2: clause_held {count} of {len(values)} at .90 -> "
                     + ("holds" if count >= of else "fails"))
    for tag, base in (("parts_256", "words_256"), ("w10_256", "words_256"),
                      ("parts_free", "words_free")):
        theirs = {r["founder"]: r for r in arms.get(base, [])}
        lives = [r for r in arms.get(tag, []) if r["founder"] in theirs]
        if not lives:
            continue
        held, cells = True, []
        for n in shown:
            mean, error = paired([_score(r["panels"][n], n) or 0.0 for r in lives],
                                 [_score(theirs[r["founder"]]["panels"][n], n) or 0.0
                                  for r in lives])
            cost = mean < 0 and -mean > 2 * error
            lift = mean > 0 and mean > 2 * error
            if n != "clause_held":
                held = held and mean >= -floor and not cost
            cells.append(f"{n} {mean:+.2f} ({error:.2f})" + ("*" if lift else "-" if cost
                                                              else ""))
        gate = "  -> P3 " + ("holds" if held else "fails") if tag != "w10_256" else ""
        lines.append(f"  {tag} less {base}: " + ", ".join(cells) + gate)
    lines.append("with the message silent at the reading (the association's own state)")
    for tag, lives in arms.items():
        quiet = [r for r in lives if "panels_held_silent" in r]
        if quiet:
            lines.append(f"  {tag}: " + ", ".join(
                f"{n} " + _mean([_score(r["panels_held_silent"][n], n) for r in quiet])
                for n in ("clause", "clause_held", "roles"))
                + f"; verb said {np.mean([_verb_said(r['said_held_silent']) for r in quiet]):.2f}"
                + "; the form's probe " + _mean([
                    r["probes_held_silent"]["verb_form"]["verb"]["settled"]["accuracy"]
                    for r in quiet]))
    lines.append("held-out clauses: agreement and the share of wrong forms a distractor drew")
    for tag, lives in arms.items():
        lines.append(f"  {tag}: agree " + _mean([r["panels"]["clause_held"].get("agree")
                                                for r in lives])
                     + ", attracted " + _mean([r["panels"]["clause_held"].get("attracted")
                                               for r in lives]))
    return lines


# ---------------------------------------- the sentence in the hidden layers (cadence#308)

FLOORS = tuple(range(10628600, 10628606))  # six fresh founders (SENTENCE_PROTOCOL.md)
SAID = ("clause", "clause_held", "roles", "dyck", "path", "clause_long")
BACK = ("back", "back_held", "back_long")


def sentence_control(spec: dict[str, Any]) -> dict[str, Any]:
    """One control of the sentence instrument (``"kind": "sentence_control"``): the chamber's
    LSTM or transformer at ``width``, taught the say-back mix online (one pass) or offline (to
    the controls' stopping rule), its panels said, and its hidden state probed as the brain's
    is: after the last heard moment, at ``GO`` and at the last word, fitted on trained
    combinations and scored on held-out ones, on clauses heard as concepts and as words."""
    import torch

    torch.set_num_threads(1)
    ct = _controls()
    model, mode, founder = str(spec["model"]), str(spec["mode"]), int(spec["founder"])
    count = int(spec.get("batches", BATCHES))
    learner = ct.Learner(model, int(spec["width"]), founder, ct.RATES[mode])
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    lessons = [sayback_batch(rng, split) for _ in range(count)]
    started = time.monotonic()
    epochs = 1
    if mode == "online":
        for batch in lessons:
            learner.update(batch)
    else:
        order = np.random.default_rng(np.random.SeedSequence([founder, 99, 0]))
        best, stale, epochs = float("inf"), 0, 0
        while epochs < ct.EPOCHS and stale < ct.PATIENCE:
            mean = float(np.mean([learner.update(lessons[int(i)])
                                  for i in order.permutation(count)]))
            epochs += 1
            stale = 0 if mean < best - 1e-4 else stale + 1
            best = min(best, mean)
    learner.net.eval()
    batches = int(spec.get("panel_batches", 4))
    tests = {**gr.panels(founder, batches), **sayback_panels(founder, batches)}
    row: dict[str, Any] = {
        "kind": "sentence_control", "model": model, "mode": mode, "founder": founder,
        "width": int(spec["width"]), "parameters": ct.count(learner.net),
        "brain_parameters": int(spec.get("parameters", 0)), "epochs": epochs,
        "macs": float(learner.work["macs"]),
        "panels": {name: gr.read_speaker(learner.speak, None, tests[name],
                                         twins=name in gr.TWINS) for name in SAID + BACK},
    }
    for panel in row["panels"].values():
        panel.pop("examples", None)
    both = split_batches(founder, int(spec.get("held_probes", 4)))
    places = ("carried", "settled") if model == "lstm" else ("settled",)
    states = control_states(model, learner.net, both)
    row["probes_held"] = transfer_probes(founder, states["trained"], states["held"], places)
    words = {k: [[sayback(r) for r in rows] for rows in both[k]] for k in both}
    states = control_states(model, learner.net, words)
    row["probes_back"] = transfer_probes(founder, states["trained"], states["held"], places)
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def floors_plan(founders: tuple[int, ...] = FLOORS) -> dict[str, list[dict[str, Any]]]:
    """The sentence instrument's floors (cadence#308, step 1): what the settled state holds
    and says of a sentence heard as concepts and as words, with the recall read barred and no
    message; the held message as the reference; the controls probed the same way."""
    base = {"arm": "barred", "train": "sayback", "probe_scale": 0.0, "held_probes": 4}
    wide = {**base, "module": 256}
    arms: dict[str, dict[str, Any]] = {
        "default_64": {**base, "settings": {"parts": True}},
        "plain_256": {**wide, "settings": {**EVERY, "parts": True}},
        "slow_256": {**wide, "settings": {**EVERY, "parts": True,
                                          "working_memory_decay": 0.8}},
        "both_256": {**wide, "settings": {**BOTH, "parts": True}, "silent_reading": True},
    }
    brains = [{**spec, "founder": f, "tag": tag} for tag, spec in arms.items() for f in founders]
    controls = []
    for tag in ("plain_256", "both_256"):
        parameters = int(make_brain(founders[0], arms[tag]["settings"], 256).parameters())
        for model in ("lstm", "transformer"):
            width = wide_width(model, parameters)
            for mode in ("online", "offline"):
                controls += [{"kind": "sentence_control", "model": model, "mode": mode,
                              "founder": f, "parameters": parameters, "width": width}
                             for f in founders[:6]]
    return {"floors": brains, "floors_controls": controls}


def _least(table: dict[str, Any], spot: str, place: str) -> float | None:
    """The least component's probe at a spot and place: the agent, the verb, the patient, their
    numbers and each distractor (the count of distractors and the verb's form, a conjunction of
    two of them, apart)."""
    found = [table[c][spot][place]["accuracy"] for c in table
             if c not in ("distractors", "verb_form") and place in table[c].get(spot, {})]
    return float(min(found)) if found else None


def sentence_table(rows: list[dict[str, Any]], controls: list[dict[str, Any]],
                   place: str = "settled") -> list[str]:
    """The sentence instrument's table: sentences said from the state heard as concepts
    (``SAID``) and as words (``BACK``), and the least clause component's probe after the last
    heard moment (fitted on trained combinations, scored on held-out ones): ``place`` for the
    brain (``joint`` reads the store with the association), the hidden state for a control."""
    arms = _by_tag(rows)
    shown = SAID + BACK
    lines = ["| Learner | Parameters | Multiply-accumulates | Lives | " + " | ".join(shown)
             + " | held (concepts) | held (words) |",
             "| --- | --- | --- | --- | " + " | ".join("---" for _ in shown) + " | --- | --- |"]

    def cell(lives: list[dict[str, Any]], name: str) -> str:
        values = [_score((r["panels"] if name in SAID else r.get("sayback", r["panels"]))[name],
                         name) for r in lives
                  if name in (r["panels"] if name in SAID else r.get("sayback", r["panels"]))]
        return _short(values) if values else "-"

    def held(lives: list[dict[str, Any]], key: str, where: str) -> str:
        values = [_least(r[key], "heard", where) for r in lives if r.get(key)]
        return _short(values) if values else "-"

    for tag, lives in arms.items():
        where = place if any(place in spot for r in lives[:1] for c in r.get(
            "probes_held", {}).values() for spot in c.values()) else "settled"
        lines.append(f"| `{tag}` | {lives[0]['parameters']} | "
                     f"{np.mean([macs(r) or 0.0 for r in lives]):.2e} | {len(lives)} | "
                     + " | ".join(cell(lives, n) for n in shown)
                     + f" | {held(lives, 'probes_held', where)} ({where})"
                     + f" | {held(lives, 'probes_back', where)} |")
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for c in controls:
        groups.setdefault((int(c["brain_parameters"]), c["model"], c["mode"]), []).append(c)
    for (_size, model, mode), lives in sorted(groups.items()):
        lines.append(f"| {model}, {mode} | {lives[0]['parameters']} | "
                     f"{np.mean([c['macs'] for c in lives]):.2e} | {len(lives)} | "
                     + " | ".join(cell(lives, n) for n in shown)
                     + f" | {held(lives, 'probes_held', 'settled')}"
                     + f" | {held(lives, 'probes_back', 'settled')} |")
    return lines


def floors_verdict(rows: list[dict[str, Any]], controls: list[dict[str, Any]]) -> list[str]:
    """The floors' reading: the table, each component's probe after the last heard moment,
    and the message arms read with the message silent."""
    arms = _by_tag(rows)
    lines = ["sentences said from the state, and the least component held after the last heard "
             "moment (mean over founders)", ""]
    lines += sentence_table(rows, controls, "joint") + [""]
    lines.append("each component after the last heard moment, held-out combinations "
                 "(concepts / words), from the settled association")
    everyone = [(tag, lives) for tag, lives in arms.items()] + [
        (f"{model} {mode} at {size}", lives) for (size, model, mode), lives in sorted(
            {(int(c["brain_parameters"]), c["model"], c["mode"]): [
                d for d in controls if (int(d["brain_parameters"]), d["model"], d["mode"]) == (
                    int(c["brain_parameters"]), c["model"], c["mode"])] for c in controls}.items())]
    for tag, lives in everyone:
        cells = []
        for c in sorted(lives[0].get("probes_held", {})):
            pair = []
            for key in ("probes_held", "probes_back"):
                found = [r[key][c]["heard"]["settled"]["accuracy"] for r in lives
                         if c in r.get(key, {}) and "heard" in r[key][c]]
                pair.append(f"{np.mean(found):.2f}" if found else "-")
            cells.append(f"{c} {pair[0]}/{pair[1]}")
        lines.append(f"  {tag}: " + ", ".join(cells))
    lines.append("with the message silent at the reading")
    for tag, lives in arms.items():
        quiet = [r for r in lives if "panels_held_silent" in r]
        if quiet:
            lines.append(f"  {tag}: " + ", ".join(
                f"{n} " + _mean([_score(r["panels_held_silent"][n], n) for r in quiet])
                for n in ("clause", "clause_held", "roles")))
    return lines


def _by_tag(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    arms: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        arms.setdefault(str(r.get("tag")), []).append(r)
    for lives in arms.values():
        lives.sort(key=lambda r: r["founder"])
    return arms


def _panel_table(arms: dict[str, list[dict[str, Any]]], shown: tuple[str, ...]) -> list[str]:
    lines = ["| Arm | Parameters | Lives | " + " | ".join(shown) + " |",
             "| --- | --- | --- | " + " | ".join("---" for _ in shown) + " |"]
    for tag, lives in arms.items():
        if "panels" in lives[0]:
            lines.append(f"| `{tag}` | {lives[0]['parameters']} | {len(lives)} | " + " | ".join(
                _mean([_score(r["panels"][n], n) for r in lives]) for n in shown) + " |")
    return lines


def third_free_verdict(rows: list[dict[str, Any]], floor: float = 0.03,
                       of: int = 5) -> list[str]:
    """The third run's first part: each message arm less ``default`` and less ``heard``, paired
    by founder, on the six panels; the gate holds for an arm where every panel's paired
    difference is at least ``-floor`` on ``of`` founders of six."""
    arms = _by_tag(rows)
    lines = ["whole sentences said exactly (roles: both twins), mean (least-greatest)", ""]
    lines += _panel_table(arms, SIX) + [""]
    for base in ("default", "heard"):
        theirs = {r["founder"]: r for r in arms.get(base, [])}
        lines.append(f"each arm less `{base}`, paired by founder: mean (standard error), "
                     f"founders at or above -{floor:.2f}")
        for tag, lives in arms.items():
            if tag in ("default", "heard") or "panels" not in lives[0]:
                continue
            paired_lives = [r for r in lives if r["founder"] in theirs]
            cells, held = [], True
            for n in SIX:
                diff = [(_score(r["panels"][n], n) or 0.0)
                        - (_score(theirs[r["founder"]]["panels"][n], n) or 0.0)
                        for r in paired_lives]
                mean, error = paired([d for d in diff], [0.0 for _ in diff])
                count = sum(d >= -floor for d in diff)
                held = held and count >= of
                cells.append(f"{n} {mean:+.2f} ({error:.2f}) {count}/{len(diff)}")
            verdict = ("holds" if held else "fails") if base == "default" else ""
            lines.append(f"  {tag}: " + ", ".join(cells) + (f"  -> {verdict}" if verdict else ""))
        lines.append("")
    lines.append("by founder: path / path_long")
    for tag, lives in arms.items():
        if "panels" in lives[0]:
            lines.append(f"  {tag}: " + " ".join(
                f"{_score(r['panels']['path'], 'path'):.2f}" for r in lives) + " / " + " ".join(
                f"{_score(r['panels']['path_long'], 'path_long'):.2f}" for r in lives))
    lines.append("the same lives read with the message silent, then with the read barred")
    for tag, lives in arms.items():
        for key in ("panels_silent", "panels_barred"):
            with_it = [r for r in lives if key in r]
            if with_it:
                lines.append(f"  {tag}, {key[7:]}: " + ", ".join(
                    f"{n} " + _mean([_score(r[key][n], n) for r in with_it])
                    for n in SIX if n in with_it[0][key]))
    lines.append("learned gains (mean over founders)")
    for tag, lives in arms.items():
        names = list(lives[0].get("gains", {}))
        if names:
            lines.append(f"  {tag}: " + ", ".join(
                f"{k} {np.mean([r['gains'][k] for r in lives]):.2f}" for k in names[:8])
                + (" ..." if len(names) > 8 else ""))
    return lines


def third_roles_verdict(located: list[dict[str, Any]], fresh: list[dict[str, Any]],
                        earlier: list[dict[str, Any]] | None = None) -> list[str]:
    """The third run's second part. ``located``: the three arms on the second run's founders
    (``earlier`` rows of that run's ``state_256`` beside them); ``fresh``: the brain's own
    message with one gain per block and per 35 neurons on fresh founders. The gate: roles .90
    on 5 of 6 with clause and held-out combinations kept at .90 on 5 of 6."""
    lines = []
    for title, rows in (("on the second run's founders", located), ("on fresh founders", fresh)):
        arms = _by_tag(rows)
        before = sorted([r for r in earlier or [] if r.get("tag") == "state_256"],
                        key=lambda r: r["founder"])
        if before and title.startswith("on the second"):
            arms = {"state_256 (second run)": before, **arms}
        lines += [f"{title}: whole sentences, mean (least-greatest)", ""]
        lines += _panel_table(arms, ("clause", "clause_held", "roles", "dyck", "path")) + [""]
        for tag, lives in arms.items():
            if not lives:
                continue
            counts = []
            for n in ("clause", "clause_held", "roles"):
                values = [_score(r["panels"][n], n) for r in lives]
                count, _ = passes(values, 0.9)
                counts.append(f"{n} " + " ".join(f"{v:.2f}" for v in values if v is not None)
                              + f" ({count} of {len(values)})")
            lines.append(f"  {tag}: " + "; ".join(counts))
            kinds = {k: v for k, v in lives[0]["said"].items() if k.startswith("clause")}
            weakest = min(lives, key=lambda r: _score(r["panels"]["roles"], "roles") or 0.0)
            lines.append(f"    weakest founder {weakest['founder']}: " + ", ".join(
                f"{k} {weakest['said'][k]:.2f}" for k in kinds if weakest["said"][k] < 0.95)
                or "    every word kind at .95")
            gains = lives[0].get("gains", {})
            if gains:
                values = [np.mean([r["gains"][k] for r in lives]) for k in gains]
                lines.append(f"    gains: {len(gains)} routes, trace "
                             f"{np.mean([r['gains'].get('trace', 0) for r in lives]):.2f}, "
                             f"the message's least {min(values[1:] or [0]):.2f} and greatest "
                             f"{max(values[1:] or [0]):.2f}")
        base = arms.get("own_256", [])
        theirs = {r["founder"]: r for r in base}
        for tag in ("instrument_256", "fine_256"):
            lives = [r for r in arms.get(tag, []) if r["founder"] in theirs]
            if lives:
                cells = []
                for n in ("clause", "clause_held", "roles"):
                    mean, error = paired(
                        [_score(r["panels"][n], n) or 0.0 for r in lives],
                        [_score(theirs[r["founder"]]["panels"][n], n) or 0.0 for r in lives])
                    cells.append(f"{n} {mean:+.2f} ({error:.2f})")
                lines.append(f"  {tag} less own_256, paired: " + ", ".join(cells))
        lines.append("")
    fine = [r for r in fresh if r.get("tag") == "fine_256"]
    if fine:
        held = all(passes([_score(r["panels"][n], n) for r in fine], 0.9)[1]
                   for n in ("clause", "clause_held", "roles"))
        lines.append("gate, fine_256 on fresh founders (roles .90 on 5 of 6, clause and held-out "
                     "kept): " + ("holds" if held else "fails"))
    return lines


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life of a plan: a message life, or by ``kind`` a diagnosis life (``"where"``) or a
    life that restates its message (``"notes"``) or a life of another chamber
    (``"knock"``)."""
    kind = spec.get("kind")
    return (where_life if kind == "where" else notes_life if kind == "notes"
            else knock_life if kind == "knock"
            else sentence_control if kind == "sentence_control" else message_life)(spec)


def macs(row: dict[str, Any]) -> float | None:
    """A life's analytic multiply-accumulates through its first phase: every row sweep of a
    lesson and every act's sweeps times its rows, each over every synapse once (the count
    ``exposure.py`` takes; an act whose rows stop early counts at the batch's longest, and the
    memory's reads are left out)."""
    work, synapses = row.get("work"), row.get("synapses")
    if not work or not synapses:
        return None
    return float((work["lesson_row_sweeps"] + work["act_sweeps"] * gr.BATCH) * synapses)


def passes(values: list[float | None], bar: float, of: int = 5) -> tuple[int, bool]:
    """How many lives reach ``bar``, and whether at least ``of`` do."""
    count = sum(1 for v in values if v is not None and v >= bar)
    return count, count >= of


def read_lives(rows: list[dict[str, Any]]) -> list[str]:
    """Message lives by tag (or arm): whole sentences, the words lost, the probes, and where a
    life holds them the context twins, the two orders of a meaning, the later phases and the
    capacity."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        if r.get("kind") in ("message", "notes"):
            groups.setdefault(str(r.get("tag") or r.get("arm")), []).append(r)
    for lives in groups.values():
        lives.sort(key=lambda r: r["founder"])
    names = WHOLE + LONG
    spoken = {tag: lives for tag, lives in groups.items() if "panels" in lives[0]}
    lines = ["whole sentences said exactly (roles: both twins), mean (least-greatest)",
             "| Arm | Lives | " + " | ".join(names) + " |", "| --- | --- |" + " --- |" * len(names)]
    for tag, lives in spoken.items():
        cells = [_mean([_score(r["panels"][name], name) for r in lives]) for name in names]
        lines.append(f"| {tag} | {len(lives)} | " + " | ".join(cells) + " |")
        if "panels_barred" in lives[0]:
            cells = [_mean([_score(r["panels_barred"][name], name) for r in lives])
                     if name in lives[0]["panels_barred"] else "-" for name in names]
            lines.append(f"| {tag}, read barred at the reading | {len(lives)} | "
                         + " | ".join(cells) + " |")
    lines += ["", "by founder: clause / clause_held / roles (lives at .90)"]
    for tag, lives in spoken.items():
        cells = []
        for name in ("clause", "clause_held", "roles"):
            values = [_score(r["panels"][name], name) for r in lives]
            count, _ = passes(values, 0.90)
            cells.append(" ".join(f"{v:.2f}" for v in values) + f" ({count} of {len(values)})")
        lines.append(f"  {tag}: " + " / ".join(cells))
    lines += ["", "words said right with every previous word given"]
    for tag, lives in spoken.items():
        if "said" not in lives[0]:
            continue
        kinds = sorted({k for r in lives for k in r["said"]})
        lines.append(f"  {tag}: " + ", ".join(
            f"{k} {_short([r['said'].get(k) for r in lives])}" for k in kinds))
    lines += ["", "probes, held-out accuracy (message / carried / settled) at GO, then at the "
              "last word"]
    for tag, lives in spoken.items():
        for family in ("clause", "dyck", "path"):
            table = lives[0].get("probes", {}).get(family, {})
            cells = []
            for name in table:
                parts = []
                for spot in ("go", "end"):
                    parts.append("/".join(
                        _short([r["probes"][family][name][spot][place]["accuracy"]
                                for r in lives
                                if place in r.get("probes", {}).get(family, {}).get(
                                    name, {}).get(spot, {})])
                        for place in ("message", "carried", "settled")))
                cells.append(f"{name} " + ", ".join(parts))
            if cells:
                lines.append(f"  {tag} {family}: " + "; ".join(cells))
    if any("notes" in lives[0] for lives in groups.values()):
        lines += ["", "notes held, closed loop (every note right at GO / at the last word)"]
        for tag, lives in groups.items():
            if "notes" in lives[0]:
                lines.append(f"  {tag}: " + ", ".join(
                    f"{panel} {_short([r['notes'][panel]['all_at_go'] for r in lives])}/"
                    f"{_short([r['notes'][panel]['all_at_end'] for r in lives])}"
                    for panel in lives[0]["notes"]))
    if any("twins" in lives[0] for lives in groups.values()):
        lines += ["", "context twins, both right / differ"]
        for tag, lives in groups.items():
            if "twins" not in lives[0]:
                continue
            for key in lives[0]["twins"]:
                values = [r["twins"][key]["both"] for r in lives]
                lines.append(f"  {tag} {key:26s} {_mean(values)} / "
                             f"{_mean([r['twins'][key]['differ'] for r in lives])}"
                             f"  ({passes(values, 0.90)[0]} of {len(values)} at .90)")
    if any("invariance" in lives[0] for lives in groups.values()):
        lines += ["", "one meaning in two orders: closer than another meaning / cosine of the "
                  "two orders / of different meanings"]
        for tag, lives in groups.items():
            if "invariance" not in lives[0]:
                continue
            cells = [f"said the same {_short([r['invariance']['said_same'] for r in lives])}",
                     f"both right {_short([r['invariance']['both_right'] for r in lives])}"]
            for place in ("message", "key", "carried", "settled"):
                if place in lives[0]["invariance"]:
                    cells.append(f"{place} " + "/".join(
                        _short([r["invariance"][place][k] for r in lives])
                        for k in ("closer", "same", "other")))
            lines.append(f"  {tag}: " + ", ".join(cells))
    if any(lives[0].get("persistence") for lives in groups.values()):
        lines += ["", "persistence: clause / dyck / path / g2 after each later phase"]
        for tag, lives in groups.items():
            for k, phase in enumerate(lives[0].get("persistence") or ()):
                cells = [_short([_score(r["persistence"][k]["panels"][name], name)
                                 for r in lives]) for name in ("clause", "dyck", "path", "g2")]
                lines.append(f"  {tag} after {phase['grammar']} x{phase['batches']}: "
                             + " / ".join(cells))
    if any("paraphrase" in lives[0] for lives in groups.values()):
        lines += ["", "paraphrase: the first grammar heard, the second said (by founder)"]
        for tag, lives in groups.items():
            if "paraphrase" in lives[0]:
                for name in lives[0]["paraphrase"]:
                    values = [r["paraphrase"][name]["exact"] for r in lives]
                    lines.append(f"  {tag} {name}: {_mean(values)}  "
                                 + " ".join(f"{v:.2f}" for v in values)
                                 + f" ({passes(values, 0.80)[0]} of {len(values)} at .80)")
    if any("capacity" in lives[0] for lives in groups.values()):
        lines += ["", "capacity: clauses said exactly by distractors (0 to 4)"]
        for tag, lives in groups.items():
            if "capacity" in lives[0]:
                lines.append(f"  {tag} ({lives[0]['parameters']} parameters): " + " ".join(
                    _short([r["capacity"][name] for r in lives]) for name in lives[0]["capacity"]))
    lines += ["", "work"]
    for tag, lives in groups.items():
        work = [macs(r) for r in lives]
        lines.append(f"  {tag}: parameters {lives[0]['parameters']}, seconds "
                     + _mean([r["work"]["seconds"] for r in lives]) + ", multiply-accumulates "
                     + (f"{np.mean([w for w in work if w]):.2e}" if any(work) else "-")
                     + ", gains " + json.dumps(
                         {k: round(float(np.mean([r["gains"].get(k, np.nan) for r in lives])), 2)
                          for k in lives[0]["gains"]}))
    return lines


def read_retrieval(rows: list[dict[str, Any]]) -> list[str]:
    """Retrieval by the code: per key width and cue, the share found by key and by state at
    each number of stored records, mean over founders."""
    lines = ["retrieval by the code (key / state), mean over founders"]
    for units in sorted({r["units"] for r in rows}):
        lives = [r["found"] for r in rows if r["units"] == units]
        for cue in lives[0][next(iter(lives[0]))]:
            cells = [f"{count}: " + "/".join(
                _short([f[count][cue][kind] for f in lives]) for kind in ("key", "state"))
                for count in lives[0]]
            lines.append(f"  {units} units, {cue}: " + ", ".join(cells))
    return lines


def paired(with_it: list[float], without: list[float]) -> tuple[float, float]:
    """The mean of the paired differences and its standard error."""
    d = np.asarray(with_it, dtype=float) - np.asarray(without, dtype=float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0


def verdict(brains: list[dict[str, Any]], controls: list[dict[str, Any]],
            knock: list[dict[str, Any]]) -> list[str]:
    """The declared gates of MEANING_PROTOCOL.md, read from the run's rows."""
    arms: dict[str, list[dict[str, Any]]] = {}
    for r in brains:
        arms.setdefault(str(r.get("tag")), []).append(r)
    for lives in arms.values():
        lives.sort(key=lambda r: r["founder"])

    def per(tag: str, get: Any) -> list[float | None]:
        return [get(r) for r in arms.get(tag, [])]

    def gate(name: str, values: list[float | None], bar: float) -> str:
        count, held = passes(values, bar)
        shown = " ".join("-" if v is None else f"{v:.2f}" for v in values)
        return f"  {name}: {shown}  ({count} of {len(values)} at {bar:.2f}: " + (
            "holds" if held else "fails") + ")"

    lines = ["H1, said from the state (state)"]
    for name in ("clause", "clause_held", "roles"):
        lines.append(gate(name, per("state", lambda r, n=name: _score(r["panels"][n], n)), 0.90))
    lines.append("H2, held: the least clause component's probe from the message")
    for spot in ("go", "end"):
        def least(r: dict[str, Any], spot: str = spot) -> float | None:
            table = r.get("probes", {}).get("clause", {})
            found = [table[c][spot]["message"]["accuracy"] for c in table
                     if "message" in table[c].get(spot, {})]
            return min(found) if found else None
        lines.append(gate(f"at {spot}", per("state", least), 0.95))
    lines.append("H2u, used: the least clause word kind said right")
    lines.append(gate("said", per("state", lambda r: min(
        v for k, v in r["said"].items() if k.startswith("clause"))), 0.95))
    lines.append("H3, one meaning in two orders (state)")
    lines.append(gate("key closer", per("state", lambda r: r["invariance"]["key"]["closer"]),
                      0.95))
    lines.append(gate("both said right", per("state", lambda r: r["invariance"]["both_right"]),
                      0.90))
    lines.append("H4, paraphrase")
    for tag in ("paraphrase", "paraphrase_read"):
        for name in ("para", "para_held"):
            lines.append(gate(f"{tag} {name}", per(
                tag, lambda r, n=name: r["paraphrase"][n]["exact"]), 0.80))
    lines.append("H5, capacity: clauses said exactly at 0 to 4 distractors (mean)")
    for tag in sorted(t for t in arms if t.startswith("capacity")):
        lines.append(f"  {tag} ({arms[tag][0]['parameters']} parameters): " + " ".join(
            _short([r["capacity"][n] for r in arms[tag]]) for n in arms[tag][0]["capacity"]))
    lines.append("A1, held-out combinations and context twins")
    for tag in ("default", "message", "state"):
        for name in ("clause_held", "roles"):
            lines.append(gate(f"{tag} {name}", per(
                tag, lambda r, n=name: _score(r["panels"][n], n)), 0.90))
        for name in gr.ERASED:
            lines.append(gate(f"{tag} {name}", per(
                tag, lambda r, n=name: r["twins"][n]["both"]), 0.90))
            fallen = [f"{suffix} " + _short([r["twins"][name + "_" + suffix]["both"]
                                             for r in arms.get(tag, [])
                                             if name + "_" + suffix in r["twins"]])
                      for suffix in ("removed", "erased_recall", "erased_trace",
                                     "erased_message")]
            lines.append(f"    {name}: " + ", ".join(fallen))
    lines.append("A2, held-out lengths and depths")
    for tag in arms:
        if "panels" in arms[tag][0]:
            for name in LONG:
                lines.append(gate(f"{tag} {name}", per(
                    tag, lambda r, n=name: r["panels"][n]["exact"]), 0.80))
    lines.append("A3, persistence: the reading before the second grammar, after it, after 500 "
                 "batches of the first; lives within .05 after the second")
    for tag in ("default", "message", "state"):
        for name in ("clause", "dyck", "path"):
            lives = [r for r in arms.get(tag, []) if r.get("persistence")]
            before = [_score(r["panels"][name], name) for r in lives]
            after = [r["persistence"][0]["panels"][name]["exact"] for r in lives]
            back = [r["persistence"][1]["panels"][name]["exact"] for r in lives]
            kept = sum(1 for a, b in zip(after, before, strict=True) if a >= b - 0.05)
            lines.append(f"  {tag} {name}: {_short(before)} / {_short(after)} / {_short(back)}"
                         f"  ({kept} of {len(lives)})")
        lines.append(f"  {tag} g2 after the second grammar: " + _short(
            [r["persistence"][0]["panels"]["g2"]["exact"] for r in arms.get(tag, [])
             if r.get("persistence")]))
    lines.append("M1, beside the read: the arm less default, paired (mean, standard error)")
    base = {r["founder"]: r for r in arms.get("default", [])}
    for tag in ("message", "message_wide", "sham", "barred", "state", "state_default",
                "restated"):
        cells = []
        for name in WHOLE + LONG:
            lives = [r for r in arms.get(tag, []) if r["founder"] in base]
            if not lives:
                continue
            mean, error = paired([_score(r["panels"][name], name) for r in lives],
                                 [_score(base[r["founder"]]["panels"][name], name)
                                  for r in lives])
            mark = "*" if abs(mean) >= 0.05 and abs(mean) > 2 * error else ""
            cells.append(f"{name} {mean:+.2f} ({error:.2f}){mark}")
        lines.append(f"  {tag}: " + ", ".join(cells))
    lines.append("controls: whole sentences after the first grammar, by parameter count")
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for c in controls:
        groups.setdefault((c["model"], c["mode"], int(c["brain_parameters"])), []).append(c)
    for (model, mode, size), lives in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0])):
        first = [c["readings"][0] for c in lives]
        cells = [f"{n} " + _short([_score(rd["panels"][n], n) for rd in first])
                 for n in WHOLE + LONG]
        twins = [f"{n} " + _short([rd["panels"][n]["both"] for rd in first]) for n in gr.ERASED]
        work = f"{np.mean([rd['work']['macs'] for rd in first]):.2e} multiply-accumulates, " \
               f"{np.mean([rd['work']['seconds'] for rd in first]):.0f} s"
        lines.append(f"  {model} {mode} at {size} ({lives[0]['parameters']}): "
                     + ", ".join(cells + twins) + f"; {work}")
        if mode == "online" and len(lives[0]["readings"]) > 1:
            after = [c["readings"][1] for c in lives]
            lines.append("    after the second grammar: " + ", ".join(
                f"{n} " + _short([rd["panels"][n]["exact"] for rd in after])
                for n in ("clause", "dyck", "path", "g2")))
    lines.append("K, other chambers: with the message less without, paired")
    tasks: dict[str, dict[str, dict[int, float]]] = {}
    for r in knock:
        if r["chamber"] == "battery":
            name = r["task"] + ("" if r["task_settings"].get("unique", True) else " repeated")
            value = r["test"]["accuracy"]
        else:
            name, value = "routes first hop, held-out", r["readings"][-1]["panels"]["held"][
                "first_hop"]
        tasks.setdefault(name, {}).setdefault(r["arm"], {})[int(r["founder"])] = float(value)
    for name, by_arm in tasks.items():
        shared = sorted(set(by_arm.get("message", {})) & set(by_arm.get("default", {})))
        if not shared:
            continue
        with_it = [by_arm["message"][f] for f in shared]
        without = [by_arm["default"][f] for f in shared]
        mean, error = paired(with_it, without)
        mark = " *" if mean <= -0.03 and abs(mean) > 2 * error else ""
        lines.append(f"  {name}: {np.mean(with_it):.3f} against {np.mean(without):.3f}, "
                     f"{mean:+.3f} ({error:.3f}){mark}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    where = sub.add_parser("where", help="the diagnosis lives")
    where.add_argument("--founders", nargs="*", type=int, default=list(FOUNDERS))
    where.add_argument("--batches", type=int, default=BATCHES)
    where.add_argument("--module", type=int, default=64)
    where.add_argument("--settings", type=json.loads, default={})
    where.add_argument("--probe-scale", type=float, default=1.0)
    where.add_argument("--controls", nargs="*", default=["lstm", "transformer"])
    where.add_argument("--save", type=Path)
    where.add_argument("--tag")
    where.add_argument("--out", type=Path, required=True)
    where.add_argument("--workers", type=int, default=1)
    lives = sub.add_parser("lives", help="message lives from a plan (a JSON list of specs)")
    lives.add_argument("--plan", type=Path, required=True)
    lives.add_argument("--out", type=Path, required=True)
    lives.add_argument("--workers", type=int, default=1)
    planner = sub.add_parser("plan", help="write the declared run's plans into a folder")
    planner.add_argument("--out", type=Path, required=True)
    planner.add_argument("--founders", nargs="*", type=int)
    planner.add_argument("--own", action="store_true",
                         help="the second run: the brain's own message")
    planner.add_argument("--third", action="store_true",
                         help="the third run: the cost beside a free read, and roles")
    planner.add_argument("--fourth", action="store_true",
                         help="protocol 4: held-out combinations from the state at 256")
    planner.add_argument("--long", action="store_true",
                         help="protocol 4, Amendment 1: lengths and depths from the state")
    planner.add_argument("--winners", action="store_true",
                         help="protocol 4, Amendment 2: ten winners a moment beside five")
    planner.add_argument("--fifth", action="store_true",
                         help="protocol 5: blocks enough for the longest clause")
    planner.add_argument("--floors", action="store_true",
                         help="cadence#308, step 1: the sentence instrument's floors")
    planner.add_argument("--seventh", action="store_true",
                         help="protocol 7: a word said by its parts")
    planner.add_argument("--sixth", action="store_true",
                         help="protocol 6: GO in a block of its own, and a message keyed by role")
    planner.add_argument("--confirm", action="store_true",
                         help="protocol 3, Amendment 1: the held message on twelve founders")
    reader = sub.add_parser("read", help="print the tables of a results file")
    reader.add_argument("results", type=Path)
    reader.add_argument("--controls", type=Path, help="with --knock: the declared gates")
    reader.add_argument("--knock", type=Path)
    reader.add_argument("--own", action="store_true", help="the second run's gates")
    reader.add_argument("--fourth", action="store_true", help="protocol 4's gate")
    reader.add_argument("--winners", action="store_true",
                        help="protocol 4, Amendment 2's gate, against both_256")
    reader.add_argument("--fifth", action="store_true", help="protocol 5's gate")
    reader.add_argument("--floors", action="store_true",
                        help="the sentence instrument's floors; --controls beside it")
    reader.add_argument("--seventh", action="store_true",
                        help="protocol 7's gates; --controls beside it")
    reader.add_argument("--sixth", action="store_true",
                        help="protocol 6's gates; --controls beside it")
    reader.add_argument("--long", action="store_true",
                        help="protocol 4, Amendment 1's gate; --controls beside it")
    reader.add_argument("--third", choices=("free", "roles", "confirm"),
                        help="the third run's gates: its first part, its second, or Amendment 1")
    reader.add_argument("--fresh", type=Path, help="with --third roles: the fresh founders' rows")
    reader.add_argument("--earlier", type=Path,
                        help="with --third roles: the second run's brain rows")
    finder = sub.add_parser("retrieval", help="retrieval by the code, no brain")
    finder.add_argument("--founders", nargs="*", type=int, default=list(DECLARED))
    finder.add_argument("--units", nargs="*", type=int, default=[70, 280])
    finder.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "retrieval":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("a") as sink:
            for f in args.founders:
                for units in args.units:
                    sink.write(json.dumps({"kind": "retrieval", "founder": f, "units": units,
                                           "found": retrieval(f, {"units": units})}) + "\n")
                    sink.flush()
        return 0
    if args.command == "plan":
        warnings.simplefilter("ignore")
        args.out.mkdir(parents=True, exist_ok=True)
        founders = tuple(args.founders or (
            FLOORS if args.floors else SEVENTH if args.seventh else SIXTH if args.sixth
            else WINNERS if args.winners else FIFTH if args.fifth
            else FOURTH_LONG if args.long else FOURTH if args.fourth else CONFIRM if args.confirm
            else THIRD if args.third else OWN if args.own else DECLARED))
        plans = (floors_plan(founders) if args.floors
                 else seventh_plan(founders) if args.seventh
                 else sixth_plan(founders) if args.sixth
                 else {"winners": winners_plan(founders)} if args.winners
                 else {"fifth": fifth_plan(founders)} if args.fifth
                 else fourth_long_plan(founders) if args.long
                 else {"fourth": fourth_plan(founders)} if args.fourth
                 else {"confirm": third_confirm_plan(founders)} if args.confirm
                 else third_plan(founders) if args.third else own_plan(founders) if args.own
                 else declared_plan(founders))
        for name, specs in plans.items():
            (args.out / f"{name}.json").write_text(json.dumps(specs))
            print(f"{name}: {len(specs)} lives")
        return 0
    if args.command == "read":
        rows = [json.loads(line) for line in args.results.read_text().splitlines() if line]
        if any(r.get("kind") == "retrieval" for r in rows):
            print("\n".join(read_retrieval(rows)))
        elif args.third == "free":
            print("\n".join(third_free_verdict(rows)))
        elif args.long:
            print("\n".join(fourth_long_verdict(rows, [
                json.loads(line) for line in args.controls.read_text().splitlines() if line]
                if args.controls is not None else [])))
        elif args.fourth:
            print("\n".join(fourth_verdict(rows)))
        elif args.winners:
            print("\n".join(fourth_verdict(rows, base="both_256")))
        elif args.fifth:
            print("\n".join(fifth_verdict(rows)))
        elif args.floors:
            print("\n".join(floors_verdict(rows, [
                json.loads(line) for line in args.controls.read_text().splitlines() if line]
                if args.controls is not None else [])))
        elif args.seventh:
            print("\n".join(seventh_verdict(rows, [
                json.loads(line) for line in args.controls.read_text().splitlines() if line]
                if args.controls is not None else [])))
        elif args.sixth:
            print("\n".join(sixth_verdict(rows, [
                json.loads(line) for line in args.controls.read_text().splitlines() if line]
                if args.controls is not None else [])))
        elif args.third == "confirm":
            print("\n".join(third_confirm_verdict(rows)))
        elif args.third == "roles":
            others = [[json.loads(line) for line in path.read_text().splitlines() if line]
                      if path is not None else [] for path in (args.fresh, args.earlier)]
            print("\n".join(third_roles_verdict(rows, *others)))
        elif args.controls is not None:
            others = [[json.loads(line) for line in path.read_text().splitlines() if line]
                      if path is not None else [] for path in (args.controls, args.knock)]
            print("\n".join(own_verdict(rows, others[0]) if args.own
                            else verdict(rows, *others)))
        else:
            print("\n".join(read(rows) if any(r.get("kind") == "where" for r in rows)
                            else read_lives(rows)))
        return 0
    if args.command == "lives":
        specs, life = json.loads(args.plan.read_text()), run_life
    else:
        life = where_life
        specs = [{"founder": f, "batches": args.batches, "module": args.module,
                  "settings": args.settings, "probe_scale": args.probe_scale,
                  "controls": args.controls, "tag": args.tag,
                  "save": str(args.save) if args.save else None} for f in args.founders]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            for result in map(life, specs):
                sink.write(json.dumps(result) + "\n")
                sink.flush()
        else:
            with get_context("spawn").Pool(min(args.workers, len(specs))) as pool:
                for result in pool.imap_unordered(life, specs, chunksize=1):
                    sink.write(json.dumps(result) + "\n")
                    sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
