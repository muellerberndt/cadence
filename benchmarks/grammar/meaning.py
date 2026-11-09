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
    memory, nothing named but the backend and ``settings``."""
    import cadence as cd

    return cd.Brain.compose(gr.INPUTS, actions, modules=(module,), seed=int(founder),
                            **{"backend": "cpu", "recall": True, **(settings or {})})


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
               "distractors": m["L"]}
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


def speak(brain: Any, rows: list[gr.Row], *, teach: bool, message: Message | None = None,
          bar: bool = False, work: dict[str, float] | None = None, cap: int = 4,
          forced: bool = False, watch: Any = None, intervene: Any = None) -> list[list[int]]:
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
                   bar: bool, names: tuple[str, ...]) -> dict[str, Any]:
    """Whole sentences on a fork, closed loop, with the life's message and bar."""
    fork = copy.deepcopy(brain)
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
                moments.append(moment)
                carried["trace"] = np.array(b.working_memory.trace, dtype=float)

            # the trace entering GO is the one the last heard moment left
            def before(b: Any, carried: Any = carried) -> None:
                carried["trace"] = np.array(b.working_memory.trace, dtype=float)

            _speak_watched(fork, rows, message, bar, before, watch)
            for i, row in enumerate(rows):
                at = {name: {place: moments[j][place][i] for place in moments[j]}
                      for name, j in positions(row).items()}
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
            if where == "message" and message is None:
                continue
            out[f"{name}_erased_{where}"] = read(name, act)
    return out


def invariance(brain: Any, founder: int, message: Message | None, bar: bool,
               pairs: int = 64) -> dict[str, Any]:
    """One meaning heard in the canonical and in the inverted order (the agent before the
    patient, or after it): how close the two held states are beside the states of different
    meanings, and whether the same sentence is said. ``closer`` is the share of meanings whose
    two orders are more alike (cosine) than the canonical order is to the nearest other
    meaning of the same shape; ``same`` the mean cosine of the two orders, ``other`` of
    different meanings. Read at ``GO`` on the message as written, on its key (the union over
    its phases) and on the settled association."""
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["probe"], 3]))
    fork = copy.deepcopy(brain)
    association = fork.association_index
    places: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    identical: list[np.ndarray] = []  # per batch, the rows that hold one meaning
    said_same, both_right, n = 0, 0, 0
    for index in range(pairs // gr.BATCH):
        L = gr.TRAIN_L[index % len(gr.TRAIN_L)]
        base = [gr.clause(rng, L, inverted=False) for _ in range(gr.BATCH)]
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
    teaches clauses of zero to four distractors alone and reads each number apart."""
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

    def lessons(grammar: str, count: int, work: dict[str, float]) -> None:
        for b in range(1, count + 1):
            batch = (capacity_batch(rng, split) if capacity
                     else gr.train_batch(rng, split, grammar))
            speak(brain, batch, teach=True, message=message,
                  bar=bool(barred.random() < rate), work=work)
            if b in checks and grammar == "g1" and not capacity:
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
    if spec.get("twins"):
        row["twins"] = twin_panels(brain, tests, message, bar)
    if spec.get("invariance"):
        row["invariance"] = invariance(brain, founder, message, bar)
    if not bar:  # a life taught with the read free, read with it barred
        row["panels_barred"] = message_panels(brain, tests, message, True, WHOLE)
    row["said"] = said_by_kind(brain, [b for name in ("clause", "dyck", "path")
                                       for b in tests[name]], message, bar)
    scale = float(spec.get("probe_scale", 0.5))
    if scale > 0:
        places = ("message", "carried", "settled") if message is not None else (
            "carried", "settled")
        row["probes"] = probe_states(
            founder, message_states(brain, probe_batches(founder, scale), message, bar), places)
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


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life of a plan: a message life, or by ``kind`` a diagnosis life (``"where"``) or a
    life that restates its message (``"notes"``)."""
    kind = spec.get("kind")
    return (where_life if kind == "where" else notes_life if kind == "notes"
            else message_life)(spec)


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
    reader = sub.add_parser("read", help="print the tables of a results file")
    reader.add_argument("results", type=Path)
    args = parser.parse_args(argv)
    if args.command == "read":
        rows = [json.loads(line) for line in args.results.read_text().splitlines() if line]
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
