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


def make_brain(founder: int, settings: dict[str, Any] | None = None, module: int = 64) -> Any:
    """The default composition: what ``Brain.compose`` gives a new brain with the recall
    memory, nothing named but the backend and ``settings``."""
    import cadence as cd

    return cd.Brain.compose(gr.INPUTS, gr.ACTIONS, modules=(module,), seed=int(founder),
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
    reader = sub.add_parser("read", help="print the tables of a results file")
    reader.add_argument("results", type=Path)
    args = parser.parse_args(argv)
    if args.command == "read":
        rows = [json.loads(line) for line in args.results.read_text().splitlines() if line]
        print("\n".join(read(rows)))
        return 0
    specs = [{"founder": f, "batches": args.batches, "module": args.module,
              "settings": args.settings, "probe_scale": args.probe_scale,
              "controls": args.controls, "tag": args.tag,
              "save": str(args.save) if args.save else None} for f in args.founders]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            for result in map(where_life, specs):
                sink.write(json.dumps(result) + "\n")
                sink.flush()
        else:
            with get_context("spawn").Pool(min(args.workers, len(specs))) as pool:
                for result in pool.imap_unordered(where_life, specs, chunksize=1):
                    sink.write(json.dumps(result) + "\n")
                    sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
