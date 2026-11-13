"""A sentence held in queues of settled activity that the brain's own role slot operates
(cadence#308).

The chamber's brain says a word by its parts (a stem slot and a form slot, ``meaning.py``). This
instrument adds a third motor slot, the role slot, and a store the role slot operates:

- **The store** holds ``GROUPS`` groups of ``depth`` blocks, each block one unit per association
  neuron. A block holds the association's settled activity of one moment, as the settle left it:
  no code is drawn and nothing is written but what the brain's own settle made.
- **The role slot** says, at every moment, what the moment is to the sentence (``OPS``): the
  agent, the verb, the patient, a distractor, an open bracket, a close, a path's start, a
  distractor said, a bracket said, a close said, or none. Each choice names fixed operations:
  ``set`` puts the moment's settled activity into a group's one block, ``write`` appends it to a
  group, ``shift`` moves a group's head to its next block, ``pop`` removes a group's last block.
- **The heads.** Before every settle each group shows one block on its own neurons, read by the
  association through plastic synapses: a queue its first block not yet shifted past, a stack
  (``STACKS``) its last block. A group with no such block shows silence.

So the sentence is held in the brain's settled activity, a word's place in the store is the
brain's own settled choice from the words heard, and a longer clause's third and fourth
distractors pass the same head, read through the synapses every lesson taught. The lessons name
the role beside the word, at the heard moments (an echo: the word the moment names and its role)
and at the said ones. In a lesson the store follows the taught role; in a reading it follows the
brain's own.

The store is an emulation with no library change: the instrument keeps the blocks and writes the
heads onto a ``message`` region composed for the purpose (``install``), whose own fixed code is
overwritten at every moment. Every reading has the recall slots of the heard sentence emptied at
``GO`` (``meaning.bar_meaning``).

    python benchmarks/grammar/sentence.py plan --dev --out DIR/plans
    python benchmarks/grammar/sentence.py lives --plan DIR/plans/dev.json --out DIR/dev.jsonl
    python benchmarks/grammar/sentence.py read DIR/dev.jsonl --controls DIR/controls.jsonl
"""

from __future__ import annotations

import argparse
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
import meaning as mn  # noqa: E402

# ----------------------------------------------------------------------------- the store

GROUPS = ("agent", "verb", "patient", "distractor", "sequence", "stack", "start")
AGENT, VERB, PATIENT, DISTRACTOR, SEQUENCE, STACK, START = range(len(GROUPS))
STACKS = (STACK,)  # groups whose head is their last block
# the role slot's choices: a name and the operations it names, each ``(operation, group)``
OPS: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = (
    ("none", ()),
    ("agent", (("set", AGENT),)),
    ("verb", (("set", VERB),)),
    ("patient", (("set", PATIENT),)),
    ("distractor", (("write", DISTRACTOR),)),
    ("said_distractor", (("shift", DISTRACTOR),)),
    ("open", (("write", SEQUENCE), ("write", STACK))),
    ("close", (("write", SEQUENCE), ("pop", STACK))),
    ("said_bracket", (("shift", SEQUENCE),)),
    ("said_close", (("pop", STACK),)),
    ("start", (("set", START),)),
)
OP = {name: i for i, (name, _) in enumerate(OPS)}
SLOTS = (*mn.PART_SLOTS, len(OPS))  # a word's stem, its form, and the moment's role
DEPTH = 8  # blocks a group holds


class Store:
    """Groups of blocks of settled activity, per stream: ``apply`` runs the operations each
    stream's role names, ``heads`` is what the groups show, ``state`` every block."""

    def __init__(self, units: int, depth: int = DEPTH, groups: int = len(GROUPS),
                 stacks: tuple[int, ...] = STACKS) -> None:
        self.units, self.depth, self.groups = int(units), int(depth), int(groups)
        self.stacks = tuple(stacks)
        self.reset(0)

    def reset(self, rows: int) -> None:
        self.blocks = np.zeros((rows, self.groups, self.depth, self.units))
        self.written = np.zeros((rows, self.groups), dtype=np.int64)  # blocks appended
        self.passed = np.zeros((rows, self.groups), dtype=np.int64)  # blocks shifted past

    def apply(self, roles: np.ndarray, activity: np.ndarray) -> None:
        """Each stream's role runs its operations with the stream's settled ``activity``."""
        for i, role in enumerate(np.asarray(roles, dtype=np.int64)):
            if not 0 <= role < len(OPS):
                continue
            for operation, g in OPS[int(role)][1]:
                w, q = int(self.written[i, g]), int(self.passed[i, g])
                if operation == "set":
                    self.blocks[i, g] = 0.0
                    self.blocks[i, g, 0] = activity[i]
                    self.written[i, g], self.passed[i, g] = 1, 0
                elif operation == "write":
                    self.blocks[i, g, w % self.depth] = activity[i]
                    self.written[i, g] = w + 1
                elif operation == "shift":
                    if q < w:
                        self.passed[i, g] = q + 1
                elif operation == "pop":
                    if w > q:
                        self.blocks[i, g, (w - 1) % self.depth] = 0.0
                        self.written[i, g] = w - 1
                else:
                    raise ValueError(f"unknown operation: {operation!r}")

    def heads(self) -> np.ndarray:
        """What every group shows, ``(streams, groups * units)``: a queue its first block not
        shifted past, a stack its last block, silence where a group holds none."""
        rows = len(self.blocks)
        out = np.zeros((rows, self.groups, self.units))
        for g in range(self.groups):
            w, q = self.written[:, g], self.passed[:, g]
            at = ((w - 1) if g in self.stacks else q) % self.depth
            out[:, g] = np.where((w > q)[:, None], self.blocks[np.arange(rows), g, at], 0.0)
        return out.reshape(rows, -1)

    def state(self) -> np.ndarray:
        """Every block, ``(streams, groups * depth * units)``."""
        return self.blocks.reshape(len(self.blocks), -1)


def make_brain(founder: int, settings: dict[str, Any] | None = None, module: int = 256) -> Any:
    """The chamber's brain with the role slot and the region the heads are shown on: every row
    taught, the previous observation heard (``history_decays=(0.0,)``), the recall memory of
    the default composition."""
    import cadence as cd

    settings = {"learning_surprise": False, "history_decays": [0.0], "message_amplitude": 3.0,
                **(settings or {})}
    return cd.Brain.compose(
        gr.INPUTS, sum(SLOTS), modules=(module,), seed=int(founder), slots=SLOTS,
        **{"backend": "cpu", "recall": True, "message": module, "message_phases": len(GROUPS),
           **settings})


def install(brain: Any, store: Store) -> None:
    """The heads onto the region the association reads them from, for the next settle."""
    brain.message.trace = store.heads()


def settled(brain: Any, rows: int) -> np.ndarray:
    """The association's settled activity of the moment just acted on."""
    state = brain.basal_ganglia.state
    if state is None:
        return np.zeros((rows, len(brain.association_index)))
    return np.array(np.asarray(state.activation)[:, brain.association_index], dtype=float)


# ---------------------------------------------------------------------------- the lessons


def _word_role(row: gr.Row, j: int) -> str | None:
    """The role of a clause's ``j``-th word: agent, verb, patient, distractor, or None for
    ``near`` and ``end``."""
    if row.target[j] in (gr.NEAR, gr.END):
        return None
    kind = gr._kind(row, j).rsplit("_", 1)[1]
    return "distractor" if kind.startswith("mod") else kind


def heard_label(row: gr.Row, t: int) -> tuple[int, int, int] | None:
    """What a lesson at a row's ``t``-th heard moment names (an echo): the word's stem and form
    and the moment's role. A clause's moment names its word (a verb concept its stem alone: the
    form waits for the agent) and its role; a bracket moment its bracket (a close the bracket it
    closes) and ``open`` or ``close``; a path's start its noun and ``start``. None where no
    lesson is given: the facts a path hears."""
    x = row.pre[t]
    if row.family == "dyck":
        return (*mn.word_parts(row.target[t]), OP["close" if x[gr.C["CLOSE"]] else "open"])
    if row.family == "path":
        if t != len(row.pre) - 1:
            return None
        return (*mn.word_parts(row.meta["chain"][0]), OP["start"])
    words = np.flatnonzero(x[:gr.ACTIONS])
    if len(words) == 1:  # a heard word: its role by its place in the sentence
        role = _word_role(row, t)
        return (*mn.word_parts(int(words[0])), OP[role or "none"])
    on = {name for name, i in gr.C.items() if x[i]}
    noun = [i for i, name in enumerate(gr.CONCEPTS[:gr.NOUNS]) if name in on]
    if on & {"SEE", "CHASE"}:
        return (gr.NOUNS + (0 if "SEE" in on else 1), mn.NO_FORM, OP["verb"])
    for flag, role in (("AGENT", "agent"), ("PATIENT", "patient"), ("MOD", "distractor")):
        if noun and flag in on:
            return (*mn.word_parts(gr.noun(noun[0], "PLURAL" in on)), OP[role])
    return None


def said_label(row: gr.Row, j: int) -> tuple[int, int, int]:
    """What a lesson at a row's ``j``-th said word names: the word's parts and the role, which
    is none but where the word leaves the store (a distractor said, a bracket of the prefix
    said, a closing bracket said)."""
    if j >= len(row.target):
        return (*mn.word_parts(gr.END), OP["none"])
    role = "none"
    if row.family == "clause":
        if _word_role(row, j) == "distractor":
            role = "said_distractor"
    elif row.family == "dyck":
        if j < row.meta["prefix"]:
            role = "said_bracket"
        elif j < len(row.target) - 1:
            role = "said_close"
    return (*mn.word_parts(row.target[j]), OP[role])


def _lesson(brain: Any, x: np.ndarray, labels: np.ndarray, work: dict[str, float],
            said: bool) -> None:
    try:
        report = brain.lesson(x, labels)
        work["lessons"] += 1
        work["lesson_row_sweeps"] += report.get("total_row_sweeps", 0.0)
        if said:
            work["rows"] += report.get("attempted_presentations", 0.0)
            work["agreed_rows"] += report.get("agreed_presentations", 0.0)
            work["free_budget_exhausted"] += report.get("free_budget_exhausted", 0.0)
    except RuntimeError:
        work["refused"] += 1


def _act(brain: Any, x: np.ndarray, work: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """One greedy act: the word each stream says and the role it names."""
    try:
        raw = np.asarray(brain.act(x, greedy=True)).reshape(len(x), len(SLOTS))
        work["act_sweeps"] += float((brain.last_settlement or {}).get("steps", 0))
        words = np.array([mn.parts_word(a, b) for a, b in raw[:, :2]], dtype=np.int64)
        roles = np.array(raw[:, 2], dtype=np.int64)
    except RuntimeError:
        work["exhausted"] += 1
        words = np.full(len(x), -1, dtype=np.int64)
        roles = np.zeros(len(x), dtype=np.int64)
    work["acts"] += 1
    return words, roles


def speak(brain: Any, rows: list[gr.Row], store: Store, *, teach: bool = False,
          bar: bool = True, forced: bool = False, work: dict[str, float] | None = None,
          cap: int = 4, watch: Any = None, echo: bool = True,
          listen: bool = False) -> list[list[int]]:
    """One batch on fresh streams. Every heard moment is acted on and the store follows the
    role; from ``GO`` each moment hears the previous word and the brain says the next. With
    ``teach`` a lesson comes before every act (at a heard moment the echo, where
    ``heard_label`` names one) and the store follows the taught roles. ``forced`` gives every
    previous word and every taught role without a lesson. In a reading the words and the roles
    are the brain's own. ``watch(when, index, words, roles, labels)`` is called after each act
    (``when`` is ``"heard"`` or ``"said"``; ``labels`` None where no lesson names the moment),
    and ``watch("held", -1, None, None, None)`` once the last heard moment is acted on; with
    ``listen`` the batch ends there."""
    work = work if work is not None else gr.new_work()
    n = len(rows)
    if len({len(r.pre) for r in rows}) != 1:
        raise ValueError("a batch's rows hear the same number of moments")
    if (teach or forced) and len({len(r.target) for r in rows}) != 1:
        raise ValueError("a taught batch's sentences have one length")
    brain.reset()
    store.reset(n)
    for t in range(len(rows[0].pre)):
        x = np.stack([r.pre[t] for r in rows])
        named = [heard_label(r, t) for r in rows]
        labels = None if any(v is None for v in named) else np.array(named, dtype=np.int64)
        if teach and echo and labels is not None:
            _lesson(brain, x, labels, work, said=False)
        words, roles = _act(brain, x, work)
        given = labels is not None and (teach or forced)
        store.apply(labels[:, 2] if given and labels is not None else roles, settled(brain, n))
        install(brain, store)
        if watch is not None:
            watch("heard", t, words, roles, labels)
    if watch is not None:
        watch("held", -1, None, None, None)
    if listen:
        return [[] for _ in rows]
    said: list[list[int]] = [[] for _ in rows]
    done = [False] * n
    longest = max(len(r.target) for r in rows) + (0 if teach or forced else cap)
    for j in range(longest):
        if j == 0:
            x = np.tile(gr.concept("GO"), (n, 1))
        else:
            before = []
            for i, r in enumerate(rows):
                given = teach or forced or j <= r.force
                before.append(r.target[j - 1] if given and j - 1 < len(r.target)
                              else said[i][j - 1])
            x = np.stack([gr.spoken(w) if w >= 0 else gr.concept("SPEAK") for w in before])
        if bar and j == rows[0].force:
            mn.bar_meaning(brain, rows)
        labels = np.array([said_label(r, j) for r in rows], dtype=np.int64)
        if teach:
            _lesson(brain, x, labels, work, said=True)
        words, roles = _act(brain, x, work)
        store.apply(labels[:, 2] if teach or forced else roles, settled(brain, n))
        install(brain, store)
        if watch is not None:
            watch("said", j, words, roles, labels)
        for i, r in enumerate(rows):
            said[i].append(int(words[i]))
            if j >= r.force and words[i] == gr.END:
                done[i] = True
        if not (teach or forced) and all(done):
            break
    out = []
    for i, r in enumerate(rows):
        words_said = said[i][r.force:]
        out.append(words_said[:words_said.index(gr.END) + 1] if gr.END in words_said
                   else words_said)
    return out


# --------------------------------------------------------------------------- the readings


def panels(brain: Any, store: Store, tests: dict[str, list[list[gr.Row]]],
           names: tuple[str, ...], bar: bool = True) -> dict[str, Any]:
    """Whole sentences, closed loop: the words and the roles the brain's own."""
    out = {}
    for name in names:
        panel = gr.read_speaker(lambda rows: speak(brain, rows, store, bar=bar), None,
                                tests[name], twins=name in gr.TWINS)
        panel["examples"] = panel["examples"][:2]
        out[name] = panel
    return out


def by_kind(brain: Any, store: Store, batches: list[list[gr.Row]]) -> dict[str, Any]:
    """With every previous word and every taught role given: each word said right by its kind
    (``words``), and the role named right at the heard moments a lesson names and at the said
    ones (``roles``): where a sentence is lost."""
    groups: dict[str, list[int]] = {}
    roles: dict[str, list[int]] = {}
    for rows in batches:
        def watch(when: str, index: int, words: Any, named: Any, labels: Any,
                  rows: list[gr.Row] = rows) -> None:
            if labels is None or when == "held":
                return
            for i, r in enumerate(rows):
                g = roles.setdefault(f"{r.family}_{when}_{OPS[int(labels[i, 2])][0]}", [0, 0])
                g[0] += int(named[i] == labels[i, 2])
                g[1] += 1
                if when == "said" and index < len(r.target):
                    word = r.target[index]
                    kind = (f"{r.family}_{gr.WORDS[word]}" if word in (gr.NEAR, gr.END)
                            else gr._kind(r, index))
                    g = groups.setdefault(kind, [0, 0])
                    g[0] += int(words[i] == word)
                    g[1] += 1

        speak(brain, rows, store, forced=True, watch=watch)
    return {"words": {k: round(v[0] / v[1], 4) for k, v in sorted(groups.items())},
            "roles": {k: round(v[0] / v[1], 4) for k, v in sorted(roles.items())}}


def held_states(brain: Any, store: Store, batches: list[list[gr.Row]],
                ) -> list[tuple[gr.Row, dict[str, Any]]]:
    """Each sentence's state after its last heard moment, the roles the brain's own: the
    settled association (``settled``), every block of the store (``message``) and the two
    together (``joint``)."""
    items: list[tuple[gr.Row, dict[str, Any]]] = []
    for rows in batches:
        held: dict[str, np.ndarray] = {}

        def watch(when: str, index: int, words: Any, named: Any, labels: Any,
                  held: dict[str, np.ndarray] = held, rows: list[gr.Row] = rows) -> None:
            if when == "held":
                held["settled"] = settled(brain, len(rows)).astype(np.float32)
                held["message"] = store.state().astype(np.float32)

        speak(brain, rows, store, watch=watch, listen=True)
        for i, row in enumerate(rows):
            spot = {"settled": held["settled"][i], "message": held["message"][i]}
            spot["joint"] = np.concatenate([spot["message"], spot["settled"]])
            items.append((row, {"heard": spot}))
    return items


def probes(brain: Any, store: Store, founder: int, both: dict[str, list[list[gr.Row]]],
           ) -> dict[str, Any]:
    """Probes of every clause component after the last heard moment, fitted on trained
    combinations and scored on held-out ones (``meaning.transfer_probes``)."""
    return mn.transfer_probes(founder, held_states(brain, store, both["trained"]),
                              held_states(brain, store, both["held"]),
                              ("settled", "message", "joint"))


# -------------------------------------------------------------------------------- a life

LEAN = ("clause", "clause_long", "dyck", "path", "back", "back_long")  # a checkpoint's panels


def life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life (``"kind": "store"``): ``founder``, ``batches`` of the say-back mix
    (``meaning.sayback_batch``), ``module``, ``depth``, ``settings`` over ``make_brain``'s,
    ``echo`` (lessons at the heard moments, on unless false), ``panel_batches``,
    ``held_probes`` and ``long_probes`` (batches per shape), ``checkpoints`` (batch counts at
    which the lean panels are read), ``tag``."""
    warnings.simplefilter("ignore")
    import cadence as cd

    founder = int(spec["founder"])
    batches = int(spec.get("batches", mn.BATCHES))
    module = int(spec.get("module", 256))
    echo = bool(spec.get("echo", True))
    started = time.monotonic()
    brain = make_brain(founder, spec.get("settings"), module)
    store = Store(module, int(spec.get("depth", DEPTH)))
    split = gr.Split.of(founder)
    count = int(spec.get("panel_batches", 4))
    tests = {**gr.panels(founder, count), **mn.sayback_panels(founder, count)}
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    row: dict[str, Any] = {
        "kind": "store", "tag": spec.get("tag"), "founder": founder, "batches": batches,
        "module": module, "depth": store.depth, "settings": spec.get("settings", {}),
        "echo": echo, "parameters": int(brain.parameters()), "cadence": cd.__version__,
        "curve": [],
    }
    work = gr.new_work()
    checks = set(int(c) for c in spec.get("checkpoints", ()))
    for b in range(1, batches + 1):
        speak(brain, mn.sayback_batch(rng, split), store, teach=True, work=work, echo=echo)
        if b in checks:
            lean = panels(brain, store, {k: v[:2] for k, v in tests.items()}, LEAN)
            row["curve"].append({"batches": b, "seconds": round(time.monotonic() - started, 1),
                                 **{k: mn._score(v, k) for k, v in lean.items()}})
    work["seconds"] = round(time.monotonic() - started, 1)
    row["work"] = work
    row["synapses"] = int(brain.describe()["layout"]["synapses"])
    row["gains"] = dict(brain.carried_gains) if brain.hearing is not None else {}
    row["panels"] = panels(brain, store, tests, mn.SAID)
    row["sayback"] = panels(brain, store, tests, mn.BACK)
    both = mn.split_batches(founder, int(spec.get("held_probes", 4)))
    words = {k: [[mn.sayback(r) for r in rows] for rows in both[k]] for k in both}
    row["kinds"] = by_kind(brain, store, [b for name in ("clause", "dyck", "path")
                                          for b in tests[name]])
    row["kinds_held"] = by_kind(brain, store, both["held"][:12])
    row["kinds_back"] = by_kind(brain, store, words["held"][:12])
    row["kinds_long"] = by_kind(brain, store, tests["clause_long"] + tests["back_long"])
    row["said_long"] = row["kinds_long"]["words"]
    row["probes_held"] = probes(brain, store, founder, both)
    row["probes_back"] = probes(brain, store, founder, words)
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """A plan's life: a store life, or one of ``meaning.py``'s (the controls)."""
    return life(spec) if spec.get("kind") == "store" else mn.run_life(spec)


# ------------------------------------------------------------------------------- reading


def read(rows: list[dict[str, Any]], controls: list[dict[str, Any]], of: int = 10) -> list[str]:
    """The table and the gates (``meaning.sentence_verdict``), then where sentences are lost:
    each word kind with the previous words and roles given, and the roles named."""
    lines = mn.sentence_verdict(rows, controls, of=of)
    arms = mn._by_tag(rows)
    for key, title in (("kinds", "the panels' sentences"), ("kinds_held", "held-out clauses"),
                       ("kinds_back", "held-out clauses heard as words"),
                       ("kinds_long", "longer clauses")):
        lines += ["", f"{title}, the previous words and roles given: mean"]
        for tag, lives in arms.items():
            for part in ("words", "roles"):
                found = [r[key][part] for r in lives if key in r]
                if found:
                    names = sorted({k for d in found for k in d})
                    lines.append(f"  {tag} {part}: " + ", ".join(
                        f"{k} {np.mean([d[k] for d in found if k in d]):.2f}" for k in names))
    lines += ["", "curves (batches: lean panels), mean"]
    for tag, lives in arms.items():
        points = sorted({p["batches"] for r in lives for p in r.get("curve", ())})
        for b in points:
            found = [p for r in lives for p in r.get("curve", ()) if p["batches"] == b]
            lines.append(f"  {tag} at {b}: " + ", ".join(
                f"{k} {np.mean([p[k] for p in found]):.2f}" for k in LEAN))
    lines += ["", "seconds a life, mean"]
    for tag, lives in arms.items():
        lines.append(f"  {tag}: {np.mean([r['seconds'] for r in lives]):.0f} "
                     f"(lessons {np.mean([r['work']['seconds'] for r in lives]):.0f})")
    return lines


# --------------------------------------------------------------------------------- plans

DEV = (0, 1, 2, 3, 4, 5)  # development founders


def dev_plan(founders: tuple[int, ...] = DEV, batches: int = mn.BATCHES,
             arms: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Development lives: the store at its settings, and what ``arms`` names beside it."""
    arms = arms or {"store": {}}
    base = {"kind": "store", "batches": batches, "held_probes": 8,
            "checkpoints": [batches // 4, batches // 2, 3 * batches // 4]}
    return [{**base, **spec, "founder": f, "tag": tag}
            for tag, spec in arms.items() for f in founders]


def control_plan(founders: tuple[int, ...], parameters: int, held_probes: int = 4,
                 modes: tuple[str, ...] = ("offline",)) -> list[dict[str, Any]]:
    """The chamber's LSTM and transformer at ``parameters``, taught the same say-back mix
    (``meaning.sentence_control``)."""
    out = []
    for model in ("lstm", "transformer"):
        width = mn.wide_width(model, parameters, limit=1200)
        for mode in modes:
            out += [{"kind": "sentence_control", "model": model, "mode": mode, "founder": f,
                     "parameters": parameters, "width": width, "held_probes": held_probes}
                    for f in founders]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    planner = sub.add_parser("plan", help="write plans into a folder")
    planner.add_argument("--out", type=Path, required=True)
    planner.add_argument("--dev", action="store_true", help="development lives")
    planner.add_argument("--founders", nargs="*", type=int)
    planner.add_argument("--batches", type=int, default=mn.BATCHES)
    planner.add_argument("--arms", type=json.loads, help="tag to spec, over the store's")
    planner.add_argument("--controls", type=int, help="controls at this parameter count")
    lives = sub.add_parser("lives", help="lives from a plan (a JSON list of specs)")
    lives.add_argument("--plan", type=Path, required=True)
    lives.add_argument("--out", type=Path, required=True)
    lives.add_argument("--workers", type=int, default=1)
    reader = sub.add_parser("read", help="print the tables of a results file")
    reader.add_argument("results", type=Path)
    reader.add_argument("--controls", type=Path)
    reader.add_argument("--of", type=int, default=10)
    args = parser.parse_args(argv)
    if args.command == "plan":
        warnings.simplefilter("ignore")
        args.out.mkdir(parents=True, exist_ok=True)
        founders = tuple(args.founders or DEV)
        plans = {"dev": dev_plan(founders, args.batches, args.arms)}
        if args.controls:
            plans["controls"] = control_plan(founders, args.controls)
        for name, specs in plans.items():
            (args.out / f"{name}.json").write_text(json.dumps(specs))
            print(f"{name}: {len(specs)} lives")
        return 0
    if args.command == "read":
        rows = [json.loads(line) for line in args.results.read_text().splitlines() if line]
        controls = ([json.loads(line) for line in args.controls.read_text().splitlines() if line]
                    if args.controls is not None else [])
        controls += [r for r in rows if r.get("kind") == "sentence_control"]
        rows = [r for r in rows if r.get("kind") == "store"]
        print("\n".join(read(rows, controls, of=args.of)))
        return 0
    specs = json.loads(args.plan.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as sink:
        if args.workers <= 1:
            for result in map(run_life, specs):
                sink.write(json.dumps(result) + "\n")
                sink.flush()
        else:
            with get_context("spawn").Pool(min(args.workers, len(specs))) as pool:
                for result in pool.imap_unordered(run_life, specs, chunksize=1):
                    sink.write(json.dumps(result) + "\n")
                    sink.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
