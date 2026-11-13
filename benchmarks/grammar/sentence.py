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

GROUPS = ("agent", "verb", "patient", "distractor", "sequence", "stack", "start", "task")
AGENT, VERB, PATIENT, DISTRACTOR, SEQUENCE, STACK, START, TASK = range(len(GROUPS))
STACKS = (STACK,)  # groups whose head is their last block
BASE = (7, 11)  # the base inventory's groups and roles: the sentences of the say-back mix
SILENCE = mn.PART_SLOTS[0]  # a stem for a moment that names no word
DEPTH = 8  # blocks a group holds
_CLAUSE = (
    ("none", ()),
    ("agent", (("set", AGENT),)),
    ("verb", (("set", VERB),)),
    ("patient", (("set", PATIENT),)),
    ("distractor", (("write", DISTRACTOR),)),
    ("said_distractor", (("shift", DISTRACTOR),)),
)
# the full inventory's roles: what is asked of the sentence (the second grammar, a question)
_TASKS = (
    ("verb_dialect", (("set", VERB), ("set", TASK))),
    ("dialect", (("set", TASK),)),
    ("question", (("set", TASK),)),
)


class Design:
    """A store's design: the role slot's choices, each a name and the operations it names, and
    what a block takes (``alone``: the settle of the moment's observation alone, in place of
    the settle in its context). With ``facts`` a path's heard facts are echoed with the role
    ``none`` (each heard word named, nothing written), with ``start_said`` a path's start
    leaves the store once said (``said_start``), and with ``clears`` a path's start empties
    every other group: the facts heard before it, words like a clause's, may have been written
    as a clause's parts."""

    def __init__(self, name: str, brackets: tuple[tuple[str, tuple[Any, ...]], ...],
                 alone: bool, facts: bool = False, start_said: bool = False,
                 clears: bool = False) -> None:
        self.name, self.alone, self.facts = name, alone, facts
        said = (("said_start", (("shift", START),)),) if start_said else ()
        cleared = tuple(("clear", g) for g in range(len(GROUPS)) if g != START) if clears else ()
        self.roles = (*_CLAUSE, *brackets, ("start", (("set", START), *cleared)), *said,
                      *_TASKS)
        self.base = len(self.roles) - len(_TASKS)  # the say-back mix's roles
        assert self.base == BASE[1] + len(said)
        self.op = {role: i for i, (role, _) in enumerate(self.roles)}
        # a bracket's echo names its word only where the heard moment settles it
        self.silent = alone

    def inventory(self, full: bool) -> tuple[int, tuple[int, int, int]]:
        """The store's groups and the motor's slots (a word's stem, its form, the moment's
        role): the base inventory, or the full one with the task group and its roles. A
        design with a silent stem holds it in either, and one whose start clears every group
        holds the task group in either."""
        stems = mn.PART_SLOTS[0] + (1 if full or self.silent else 0)
        if full:
            return len(GROUPS), (stems, mn.PART_SLOTS[1], len(self.roles))
        groups = 1 + max(g for _, steps in self.roles[:self.base] for _, g, *_ in steps)
        return max(BASE[0], groups), (stems, mn.PART_SLOTS[1], self.base)


# ``context``: a block takes the moment's settle in its context; a heard close is settled to the
# bracket it closes (the stack's head is read at the heard moment) and the prefix is said again
# from the sequence. ``alone``: a block takes the settle of the observation alone; the heard
# brackets join the sequence as heard, and saying them works the stack (an open said moves from
# the sequence onto the stack, a close said takes the stack's head)
_ALONE = (
    ("bracket", (("write", SEQUENCE),)),
    ("said_open", (("move", SEQUENCE, STACK),)),
    ("said_match", (("shift", SEQUENCE), ("pop", STACK))),
    ("said_close", (("pop", STACK),)),
)
DESIGNS = {
    "context": Design("context", (
        ("open", (("write", SEQUENCE), ("write", STACK))),
        ("close", (("write", SEQUENCE), ("pop", STACK))),
        ("said_bracket", (("shift", SEQUENCE),)),
        ("said_close", (("pop", STACK),)),
    ), alone=False),
    "alone": Design("alone", _ALONE, alone=True),
    # ``alone`` with a path's facts echoed (role ``none``) and its start leaving once said
    "alone_path": Design("alone_path", _ALONE, alone=True, facts=True, start_said=True),
    "alone_facts": Design("alone_facts", _ALONE, alone=True, facts=True),  # the facts echoed
    "alone_clean": Design("alone_clean", _ALONE, alone=True, clears=True),  # the start clears
}


class Store:
    """Groups of blocks of settled activity, per stream: ``apply`` runs the operations each
    stream's role names (``roles``, a design's), ``heads`` is what the groups show, ``state``
    every block."""

    def __init__(self, units: int, roles: tuple[tuple[str, tuple[Any, ...]], ...],
                 depth: int = DEPTH, groups: int = BASE[0],
                 stacks: tuple[int, ...] = STACKS) -> None:
        self.units, self.depth, self.groups = int(units), int(depth), int(groups)
        self.roles, self.stacks = roles, tuple(stacks)
        self.reset(0)

    def reset(self, rows: int) -> None:
        self.blocks = np.zeros((rows, self.groups, self.depth, self.units))
        self.written = np.zeros((rows, self.groups), dtype=np.int64)  # blocks appended
        self.passed = np.zeros((rows, self.groups), dtype=np.int64)  # blocks shifted past

    def apply(self, roles: np.ndarray, activity: np.ndarray) -> None:
        """Each stream's role runs its operations with the stream's settled ``activity``."""
        for i, role in enumerate(np.asarray(roles, dtype=np.int64)):
            if not 0 <= role < len(self.roles):
                continue
            for operation, g, *to in self.roles[int(role)][1]:
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
                elif operation == "clear":
                    self.blocks[i, g] = 0.0
                    self.written[i, g], self.passed[i, g] = 0, 0
                elif operation == "move":
                    if w > q:  # the head leaves its group and joins the other's end
                        at = ((w - 1) if g in self.stacks else q) % self.depth
                        there = int(self.written[i, to[0]])
                        self.blocks[i, to[0], there % self.depth] = self.blocks[i, g, at]
                        self.written[i, to[0]] = there + 1
                        if g in self.stacks:
                            self.blocks[i, g, at] = 0.0
                            self.written[i, g] = w - 1
                        else:
                            self.passed[i, g] = q + 1
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
        return out.reshape(rows, self.groups * self.units)

    def state(self) -> np.ndarray:
        """Every block, ``(streams, groups * depth * units)``."""
        return self.blocks.reshape(len(self.blocks), self.groups * self.depth * self.units)


def make_brain(founder: int, settings: dict[str, Any] | None = None, module: int = 256,
               depth: int | None = DEPTH, full: bool = False,
               design: Design | None = None) -> Any:
    """The chamber's brain with the role slot and the store: every row taught, the previous
    observation heard (``history_decays=(0.0,)``), the recall memory of the default
    composition. ``depth`` blocks a group is the library's store (``message_settled`` with
    ``message_operations``); None composes the region alone, for the instrument's own
    ``Store`` to write its heads onto (``install``). ``full`` is the full inventory."""
    import cadence as cd

    design = design or DESIGNS["context"]
    groups, slots = design.inventory(full)
    settings = {"learning_surprise": False, "history_decays": [0.0], "message_amplitude": 3.0,
                **(settings or {})}
    store = ({"message_phases": groups} if depth is None else
             {"message_phases": int(depth), "message_settled": True,
              "message_operations": [list(steps) for _, steps in design.roles[:slots[2]]],
              "message_stacks": list(STACKS),
              **({"message_alone": True} if design.alone else {})})
    return cd.Brain.compose(
        gr.INPUTS, sum(slots), modules=(module,), seed=int(founder), slots=slots,
        **{"backend": "cpu", "recall": True, "message": module, **store, **settings})


def install(brain: Any, store: Store | None) -> None:
    """The heads onto the region the association reads them from, for the next settle; the
    library's store shows its own."""
    if store is not None:
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


def _heard_clause(row: gr.Row) -> gr.Row:
    """The clause whose words a row hears: its own, the other wording's (``wording``) or the
    clause a question is asked of (``question``)."""
    if "question" in row.meta:
        return gr.twin(row, dialect=False)
    if "heard_dialect" in row.meta:
        return gr.twin(row, dialect=bool(row.meta["heard_dialect"]))
    return row


# a question is one concept moment after the sentence: a role's flag alone, or both verbs
QUESTIONS = {"agent": ("AGENT",), "patient": ("PATIENT",), "verb": ("SEE", "CHASE")}


def wording(row: gr.Row, heard: bool, said: bool) -> gr.Row:
    """One meaning in two wordings: the clause heard as the words of one grammar (``heard``
    true for the second) and said in the other or the same (``said``). A ``DIALECT`` moment
    before ``GO`` asks for the second grammar; without it the first is said."""
    target = gr.twin(row, dialect=said)
    pre = [gr.heard(w) for w in gr.twin(row, dialect=heard).target]
    if said:
        pre.append(gr.concept("DIALECT"))
    return gr.Row("clause", pre, target.target, {**target.meta, "heard_dialect": heard})


def question(row: gr.Row, ask: str, words: bool) -> gr.Row:
    """A question about a clause: the clause heard (as its words, or as concept moments), one
    question moment (``QUESTIONS``), and at ``GO`` the answer said as one word and ``end``:
    the agent, the patient, or the verb in the agent's number."""
    clause = gr.twin(row, dialect=False)
    m = clause.meta
    pre = [gr.heard(w) for w in clause.target] if words else list(clause.pre)
    answer = {"agent": gr.noun(*m["agent"]), "patient": gr.noun(*m["patient"]),
              "verb": gr.verb(m["verb"], m["agent"][1])}[ask]
    return gr.Row("clause", [*pre, gr.concept(*QUESTIONS[ask])], [answer, gr.END],
                  {**m, "question": ask, "heard_words": words})


def heard_label(row: gr.Row, t: int, design: Design) -> tuple[int, int, int] | None:
    """What a lesson at a row's ``t``-th heard moment names (an echo): the word's stem and form
    and the moment's role. A clause's moment names its word (a verb concept its stem alone: the
    form waits for the agent) and its role; a bracket moment its bracket (a close the bracket it
    closes) and ``open`` or ``close``; a path's start its noun and ``start``. Where a block takes
    the observation alone (``design.alone``) a bracket moment is a ``bracket`` and a close
    names no word: the bracket it closes is settled when it is said. None where no lesson is
    given: the facts a path hears."""
    x = row.pre[t]
    OP = design.op
    if row.family == "dyck":
        close = bool(x[gr.C["CLOSE"]])
        if design.alone:
            word = (SILENCE, mn.NO_FORM) if close else mn.word_parts(row.target[t])
            return (*word, OP["bracket"])
        return (*mn.word_parts(row.target[t]), OP["close" if close else "open"])
    if row.family == "path":
        if t != len(row.pre) - 1:
            if design.facts:  # a fact's heard word, named with no role
                return (*mn.word_parts(int(np.argmax(x[:gr.ACTIONS]))), OP["none"])
            return None
        return (*mn.word_parts(row.meta["chain"][0]), OP["start"])
    words = np.flatnonzero(x[:gr.ACTIONS])
    if len(words) == 1:  # a heard word: its role by its place in the sentence it is heard in
        role = _word_role(_heard_clause(row), t)
        return (*mn.word_parts(int(words[0])), OP[role or "none"])
    on = {name for name, i in gr.C.items() if x[i]}
    noun = [i for i, name in enumerate(gr.CONCEPTS[:gr.NOUNS]) if name in on]
    if on == {"DIALECT"}:  # the second grammar asked for: no word, the task
        return (SILENCE, mn.NO_FORM, OP["dialect"])
    if "question" in row.meta and t == len(row.pre) - 1:
        return (SILENCE, mn.NO_FORM, OP["question"])
    if on & {"SEE", "CHASE"}:
        return (gr.NOUNS + (0 if "SEE" in on else 1), mn.NO_FORM,
                OP["verb_dialect" if "DIALECT" in on else "verb"])
    for flag, role in (("AGENT", "agent"), ("PATIENT", "patient"), ("MOD", "distractor")):
        if noun and flag in on:
            return (*mn.word_parts(gr.noun(noun[0], "PLURAL" in on)), OP[role])
    return None


def said_label(row: gr.Row, j: int, design: Design) -> tuple[int, int, int]:
    """What a lesson at a row's ``j``-th said word names: the word's parts and the role, which
    is none but where the word leaves the store (a distractor said, a bracket of the prefix
    said, a closing bracket said)."""
    OP = design.op
    if j >= len(row.target):
        return (*mn.word_parts(gr.END), OP["none"])
    role = "none"
    if "question" in row.meta:
        pass  # an answer takes nothing from the store
    elif row.family == "clause":
        if _word_role(row, j) == "distractor":
            role = "said_distractor"
    elif row.family == "path":
        if j == 0 and "said_start" in OP:
            role = "said_start"
    elif row.family == "dyck":
        if j < row.meta["prefix"] and design.alone:
            role = "said_open" if row.target[j] in gr.OPENS else "said_match"
        elif j < row.meta["prefix"]:
            role = "said_bracket"
        elif j < len(row.target) - 1:
            role = "said_close"
    return (*mn.word_parts(row.target[j]), OP[role])


_READ_RATES = ("address", "position_rate", "slope")  # what steps the recall read's address


def _lesson(brain: Any, x: np.ndarray, labels: np.ndarray, work: dict[str, float],
            said: bool, quiet: bool = False) -> bool:
    """One lesson; whether it was given (a refused one names no role for the store). With
    ``quiet`` the recall read's address and position terms take no step (their rates held at
    zero for the lesson): an echo teaches the word and its role, and the moment it names is
    not yet in the slots, so the read's lesson there would step toward another moment."""
    recall = brain.recall if quiet else None
    rates = {name: getattr(recall, name) for name in _READ_RATES} if recall is not None else {}
    try:
        if recall is not None:
            for name in _READ_RATES:
                setattr(recall, name, 0.0)
        report = brain.lesson(x, labels)
        work["lessons"] += 1
        work["lesson_row_sweeps"] += report.get("total_row_sweeps", 0.0)
        if said:
            work["rows"] += report.get("attempted_presentations", 0.0)
            work["agreed_rows"] += report.get("agreed_presentations", 0.0)
            work["free_budget_exhausted"] += report.get("free_budget_exhausted", 0.0)
    except RuntimeError:
        work["refused"] += 1
        return False
    finally:
        for name, rate in rates.items():
            setattr(recall, name, rate)
    return True


def _act(brain: Any, x: np.ndarray, work: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """One greedy act: the word each stream says and the role it names."""
    try:
        raw = np.asarray(brain.act(x, greedy=True)).reshape(len(x), 3)
        work["act_sweeps"] += float((brain.last_settlement or {}).get("steps", 0))
        words = np.array([-1 if a == SILENCE else mn.parts_word(a, b) for a, b in raw[:, :2]],
                         dtype=np.int64)
        roles = np.array(raw[:, 2], dtype=np.int64)
    except RuntimeError:
        work["exhausted"] += 1
        words = np.full(len(x), -1, dtype=np.int64)
        roles = np.zeros(len(x), dtype=np.int64)
    work["acts"] += 1
    return words, roles


def speak(brain: Any, rows: list[gr.Row], store: Store | None, design: Design, *,
          teach: bool = False, bar: bool = True, forced: bool = False,
          work: dict[str, float] | None = None, cap: int = 4, watch: Any = None,
          echo: bool = True, listen: bool = False, quiet: bool = False) -> list[list[int]]:
    """One batch on fresh streams. Every heard moment is acted on and the store follows the
    role; from ``GO`` each moment hears the previous word and the brain says the next. With
    ``teach`` a lesson comes before every act (at a heard moment the echo, where
    ``heard_label`` names one) and the store follows the taught roles. ``forced`` gives every
    previous word and every taught role without a lesson. In a reading the words and the roles
    are the brain's own. ``store`` is the instrument's own, or None for the library's (the
    brain's ``message``). ``watch(when, index, words, roles, labels)`` is called after each act
    (``when`` is ``"heard"`` or ``"said"``; ``labels`` None where no lesson names the moment),
    and ``watch("held", -1, None, None, None)`` once the last heard moment is acted on; with
    ``listen`` the batch ends there. With ``quiet`` an echo leaves the recall read's address as
    it is (``_lesson``)."""
    work = work if work is not None else gr.new_work()
    n = len(rows)
    if len({len(r.pre) for r in rows}) != 1:
        raise ValueError("a batch's rows hear the same number of moments")
    if (teach or forced) and len({len(r.target) for r in rows}) != 1:
        raise ValueError("a taught batch's sentences have one length")
    brain.reset()
    if store is not None:
        store.reset(n)

    def follow(labels: np.ndarray | None, roles: np.ndarray, x: np.ndarray) -> None:
        # the instrument's store follows the given role, or the brain's own
        if store is not None:
            run = roles if labels is None else labels[:, 2]
            takes = any(step[0] in ("set", "write") for r in run if 0 <= r < len(design.roles)
                        for step in design.roles[int(r)][1])
            held = (settled(brain, n) if not design.alone
                    else brain._settled_alone(x, brain._code(x)) if takes
                    else np.zeros((n, store.units)))
            store.apply(run, held)
            install(brain, store)

    def give(labels: np.ndarray) -> None:
        # the library's store is given a role no lesson named (a lesson names its own)
        if store is None:
            brain.message.name(labels[:, 2])

    for t in range(len(rows[0].pre)):
        x = np.stack([r.pre[t] for r in rows])
        named = [heard_label(r, t, design) for r in rows]
        labels = None if any(v is None for v in named) else np.array(named, dtype=np.int64)
        taught = teach and echo and labels is not None
        if taught and labels is not None:
            taught = _lesson(brain, x, labels, work, said=False, quiet=quiet)
        given = labels is not None and (teach or forced)
        if given and not taught and labels is not None:
            give(labels)
        words, roles = _act(brain, x, work)
        follow(labels if given else None, roles, x)
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
        labels = np.array([said_label(r, j, design) for r in rows], dtype=np.int64)
        if not (teach and _lesson(brain, x, labels, work, said=True)) and (teach or forced):
            give(labels)
        words, roles = _act(brain, x, work)
        follow(labels if teach or forced else None, roles, x)
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


def panels(brain: Any, store: Store | None, design: Design,
           tests: dict[str, list[list[gr.Row]]], names: tuple[str, ...],
           bar: bool = True) -> dict[str, Any]:
    """Whole sentences, closed loop: the words and the roles the brain's own."""
    out = {}
    for name in names:
        panel = gr.read_speaker(lambda rows: speak(brain, rows, store, design, bar=bar), None,
                                tests[name], twins=name in gr.TWINS)
        panel["examples"] = panel["examples"][:2]
        out[name] = panel
    return out


def by_kind(brain: Any, store: Store | None, design: Design,
            batches: list[list[gr.Row]]) -> dict[str, Any]:
    """With every previous word and every taught role given: each word said right by its kind
    (``words``), and the role named right at the heard moments a lesson names and at the said
    ones (``roles``): where a sentence is lost."""
    groups: dict[str, list[int]] = {}
    roles: dict[str, list[int]] = {}
    own: dict[str, int] = {}  # the roles the brain names where no lesson names one
    reads: dict[str, list[Any]] = {}  # with the read of one head: its mean attention by kind
    told = brain.message if store is None else None
    for rows in batches:
        def watch(when: str, index: int, words: Any, named: Any, labels: Any,
                  rows: list[gr.Row] = rows) -> None:
            if when == "held":
                return
            if labels is None:
                for i, r in enumerate(rows):
                    name = design.roles[int(named[i])][0]
                    own[f"{r.family}_{name}"] = own.get(f"{r.family}_{name}", 0) + 1
                return
            for i, r in enumerate(rows):
                name = design.roles[int(labels[i, 2])][0]
                g = roles.setdefault(f"{r.family}_{when}_{name}", [0, 0])
                g[0] += int(named[i] == labels[i, 2])
                g[1] += 1
                if when == "said" and index < len(r.target):
                    word = r.target[index]
                    if "question" in r.meta:
                        kind = "question_" + ("end" if word == gr.END else r.meta["question"])
                    else:
                        kind = (f"{r.family}_{gr.WORDS[word]}" if word in (gr.NEAR, gr.END)
                                else gr._kind(r, index))
                    g = groups.setdefault(kind, [0, 0])
                    g[0] += int(words[i] == word)
                    g[1] += 1
                    if told is not None and told.reads and told.last is not None:
                        p = np.asarray(told.last[2])
                        if len(p) == len(rows):
                            seen = reads.setdefault(kind, [np.zeros(p.shape[1]), 0])
                            seen[0] = seen[0] + p[i]
                            seen[1] += 1

        speak(brain, rows, store, design, forced=True, watch=watch)
    return {"words": {k: round(v[0] / v[1], 4) for k, v in sorted(groups.items())},
            "roles": {k: round(v[0] / v[1], 4) for k, v in sorted(roles.items())},
            **({"own": dict(sorted(own.items()))} if own else {}),
            **({"reads": {k: [round(float(a), 3) for a in v[0] / v[1]]
                          for k, v in sorted(reads.items())}} if reads else {})}


def held_states(brain: Any, store: Store | None, design: Design,
                batches: list[list[gr.Row]]) -> list[tuple[gr.Row, dict[str, Any]]]:
    """Each sentence's state after its last heard moment, the roles the brain's own: the
    settled association (``settled``) and every block of the store read with it
    (``joint``)."""
    items: list[tuple[gr.Row, dict[str, Any]]] = []
    for rows in batches:
        held: dict[str, np.ndarray] = {}

        def watch(when: str, index: int, words: Any, named: Any, labels: Any,
                  held: dict[str, np.ndarray] = held, rows: list[gr.Row] = rows) -> None:
            if when == "held":
                held["settled"] = settled(brain, len(rows))
                held["message"] = np.array(
                    brain.message.state if store is None else store.state(), dtype=float)

        speak(brain, rows, store, design, watch=watch, listen=True)
        for i, row in enumerate(rows):
            items.append((row, {"heard": {
                "settled": held["settled"][i],
                "joint": np.concatenate([held["message"][i], held["settled"][i]])}}))
    return items


def probes(brain: Any, store: Store | None, design: Design, founder: int,
           both: dict[str, list[list[gr.Row]]]) -> dict[str, Any]:
    """Probes of every clause component after the last heard moment, fitted on trained
    combinations and scored on held-out ones (``meaning.transfer_probes``)."""
    return mn.transfer_probes(founder, held_states(brain, store, design, both["trained"]),
                              held_states(brain, store, design, both["held"]),
                              ("settled", "joint"))


# ------------------------------------------------------------------- the lessons' mixes

WORDINGS = {"para_12": (False, True), "para_21": (True, False), "para_22": (True, True)}
ASKED = ("ask_concepts", "ask_words")
SAME = tuple(WORDINGS) + ("g2",)  # the panels of one meaning in two wordings
MORE = tuple(name + tail for name in SAME + ASKED for tail in ("", "_held"))


def full_batch(rng: np.random.Generator, split: gr.Split) -> list[gr.Row]:
    """A lesson batch of the full mix, combinations that training shows: the first grammar's
    mix (four tenths), clauses heard as words and said back (two tenths), the second grammar
    from concepts (one tenth), one wording said from the other (``WORDINGS``, three
    twentieths) and questions (three twentieths)."""
    kind = rng.random()
    if kind < 0.4:
        return gr.train_batch(rng, split, "g1")
    L = int(rng.choice(gr.TRAIN_L))
    if kind < 0.6:
        return [mn.sayback(gr.trained_clause(rng, split, L, False)) for _ in range(gr.BATCH)]
    if kind < 0.7:
        return [gr.trained_clause(rng, split, L, True) for _ in range(gr.BATCH)]
    if kind < 0.85:
        heard, said = list(WORDINGS.values())[int(rng.integers(len(WORDINGS)))]
        return [wording(gr.trained_clause(rng, split, L, False), heard, said)
                for _ in range(gr.BATCH)]
    ask = list(QUESTIONS)[int(rng.integers(len(QUESTIONS)))]
    words = bool(rng.integers(2))
    return [question(gr.trained_clause(rng, split, L, False), ask, words)
            for _ in range(gr.BATCH)]


def lesson_batch(train: str, rng: np.random.Generator, split: gr.Split) -> list[gr.Row]:
    """A life's lesson batch: the say-back mix (``meaning.sayback_batch``) or the full mix."""
    return full_batch(rng, split) if train == "all" else mn.sayback_batch(rng, split)


def more_panels(founder: int, batches: int = 4) -> dict[str, list[list[gr.Row]]]:
    """The full mix's panels at trained lengths, each of combinations that training shows and,
    as ``_held``, of held-out ones: one wording said from the other (``para_12`` the second
    grammar from the first's words, ``para_21`` the first from the second's, ``para_22`` the
    second said back), the second grammar from concepts (``g2``), and questions about a clause
    heard as concepts and as words (``ask_concepts``, ``ask_words``), the three questions in
    turn."""
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([int(founder), gr.SEEDS["test"], 3080]))

    def clause(L: int, held: bool, dialect: bool = False) -> gr.Row:
        while True:
            row = gr.clause(rng, L, dialect=dialect)
            m = row.meta
            if split.held(tuple(m["agent"]), tuple(m["patient"]), m["verb"]) == held:
                return row

    shape = [gr.TRAIN_L[i % len(gr.TRAIN_L)] for i in range(batches)]
    out: dict[str, list[list[gr.Row]]] = {}
    for held, tail in ((False, ""), (True, "_held")):
        for name, (heard, said) in WORDINGS.items():
            out[name + tail] = [[wording(clause(L, held), heard, said) for _ in range(gr.BATCH)]
                                for L in shape]
        out["g2" + tail] = [[clause(L, held, True) for _ in range(gr.BATCH)] for L in shape]
        asks = list(QUESTIONS)
        for name, words in (("ask_concepts", False), ("ask_words", True)):
            out[name + tail] = [[question(clause(L, held), asks[i % len(asks)], words)
                                 for _ in range(gr.BATCH)] for i, L in enumerate(shape)]
    return out


def cross_batches(founder: int, per_shape: int) -> dict[str, list[list[gr.Row]]]:
    """Clauses heard as the words of each grammar, for the probes across wordings: trained
    combinations and held-out ones in the first grammar's words (``g1``, ``g1_held``) and in
    the second's (``g2``, ``g2_held``)."""
    both = mn.split_batches(founder, per_shape)
    out = {}
    for heard, name in ((False, "g1"), (True, "g2")):
        out[name] = [[wording(r, heard, False) for r in rows] for rows in both["trained"]]
        out[name + "_held"] = [[wording(r, heard, False) for r in rows] for rows in both["held"]]
    return out


# -------------------------------------------------------------------------------- a life

# a checkpoint's panels
LEAN = ("clause", "clause_held", "roles", "clause_long", "dyck", "path", "back", "back_long")
LEAN_MORE = ("para_12_held", "para_21_held", "g2_held", "ask_concepts_held", "ask_words_held")


def life(spec: dict[str, Any]) -> dict[str, Any]:
    """One life (``"kind": "store"``): ``founder``, ``batches`` of the say-back mix
    (``meaning.sayback_batch``) or with ``"train": "all"`` of the full mix (``full_batch``, the
    full inventory and its panels), ``module``, ``depth``, ``settings`` over ``make_brain``'s,
    ``echo`` (lessons at the heard moments, on unless false), ``quiet`` (an echo leaves the
    recall read's address as it is), ``panel_batches``,
    ``emulated`` (the instrument's own store in place of the library's), ``held_probes``
    (batches per shape), ``checkpoints`` (batch counts at which the lean
    panels are read, appended to the file ``progress`` names), ``save`` (a folder the taught
    brain is saved in, for a later reading), ``tag``."""
    warnings.simplefilter("ignore")
    import cadence as cd

    founder = int(spec["founder"])
    batches = int(spec.get("batches", mn.BATCHES))
    module = int(spec.get("module", 256))
    echo = bool(spec.get("echo", True))
    quiet = bool(spec.get("quiet", False))
    started = time.monotonic()
    depth = int(spec.get("depth", DEPTH))
    emulated = bool(spec.get("emulated", False))
    train = str(spec.get("train", "sayback"))
    full = bool(spec.get("full", train == "all"))
    design = DESIGNS[str(spec.get("design", "context"))]
    brain = make_brain(founder, spec.get("settings"), module, None if emulated else depth, full,
                       design)
    store = (Store(module, design.roles, depth, design.inventory(full)[0]) if emulated
             else None)
    split = gr.Split.of(founder)
    count = int(spec.get("panel_batches", 4))
    tests = {**gr.panels(founder, count), **mn.sayback_panels(founder, count)}
    if train == "all":
        tests.update(more_panels(founder, count))
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    row: dict[str, Any] = {
        "kind": "store", "tag": spec.get("tag"), "founder": founder, "batches": batches,
        "module": module, "depth": depth, "emulated": emulated, "train": train, "full": full,
        "design": design.name,
        "settings": spec.get("settings", {}),
        "echo": echo, "quiet": quiet, "parameters": int(brain.parameters()),
        "cadence": cd.__version__,
        "curve": [],
    }
    work = gr.new_work()
    checks = set(int(c) for c in spec.get("checkpoints", ()))
    for b in range(1, batches + 1):
        speak(brain, lesson_batch(train, rng, split), store, design, teach=True, work=work,
              echo=echo, quiet=quiet)
        if b in checks:
            lean = panels(brain, store, design, {k: v[:2] for k, v in tests.items()},
                          LEAN + (LEAN_MORE if train == "all" else ()))
            row["curve"].append({"batches": b, "seconds": round(time.monotonic() - started, 1),
                                 **{k: mn._score(v, k) for k, v in lean.items()}})
            if spec.get("progress"):  # a file the checkpoints are appended to as they are read
                with Path(spec["progress"]).open("a") as sink:
                    sink.write(json.dumps({"tag": row["tag"], "founder": founder,
                                           **row["curve"][-1]}) + "\n")
    work["seconds"] = round(time.monotonic() - started, 1)
    row["work"] = work
    if spec.get("save"):  # the taught brain kept for a later reading (a folder per life)
        folder = Path(spec["save"])
        folder.mkdir(parents=True, exist_ok=True)
        row["saved"] = str(brain.save(folder / f"{row['tag']}_{founder}"))
    row["synapses"] = int(brain.describe()["layout"]["synapses"])
    row["gains"] = dict(brain.carried_gains) if brain.hearing is not None else {}
    row["panels"] = panels(brain, store, design, tests, mn.SAID)
    row["sayback"] = panels(brain, store, design, tests, mn.BACK)
    if train == "all":
        row["more"] = panels(brain, store, design, tests, MORE)
        row["kinds_more"] = by_kind(brain, store, design,
                                    [b for name in MORE if name.endswith("_held")
                                     for b in tests[name]])
        cross = cross_batches(founder, int(spec.get("held_probes", 4)))
        states = {name: held_states(brain, store, design, cross[name]) for name in cross}
        places = ("settled", "joint")
        # fitted on one wording's trained combinations, scored on the other's held-out ones
        row["probes_cross"] = {
            "g1_to_g2": mn.transfer_probes(founder, states["g1"], states["g2_held"], places),
            "g2_to_g1": mn.transfer_probes(founder, states["g2"], states["g1_held"], places)}
        del states
    both = mn.split_batches(founder, int(spec.get("held_probes", 4)))
    words = {k: [[mn.sayback(r) for r in rows] for rows in both[k]] for k in both}
    row["kinds"] = by_kind(brain, store, design, [b for name in ("clause", "dyck", "path")
                                                  for b in tests[name]])
    row["kinds_held"] = by_kind(brain, store, design, both["held"][:12])
    row["kinds_back"] = by_kind(brain, store, design, words["held"][:12])
    row["kinds_long"] = by_kind(brain, store, design,
                                tests["clause_long"] + tests["back_long"])
    row["said_long"] = row["kinds_long"]["words"]
    row["probes_held"] = probes(brain, store, design, founder, both)
    row["probes_back"] = probes(brain, store, design, founder, words)
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def _taught_roles(learner: Any, model: str, design: Design) -> Any:
    """A control's update with the brain's lessons in full: beside each said word, the word an
    echo names at a heard moment and the role of every moment a lesson names, the roles read
    from the control's hidden state by a linear head of their own. Returns the update."""
    import torch

    ct = mn._controls()
    net = learner.net
    head = torch.nn.Linear(net.out.in_features, len(design.roles))
    learner.opt.add_param_group({"params": head.parameters()})
    learner.role_head = head

    def update(rows: list[gr.Row]) -> float:
        x, y = ct.encode(rows)
        roles = np.full_like(y, -100)
        for i, r in enumerate(rows):
            for t in range(len(r.pre)):
                label = heard_label(r, t, design)
                if label is not None:
                    roles[i, t] = label[2]
                    verb = gr.NOUNS <= label[0] < gr.NOUNS + 2
                    # an echo's word, where it names a whole one (a verb concept names a stem)
                    if label[0] != SILENCE and not (verb and label[1] == mn.NO_FORM):
                        y[i, t] = mn.parts_word(label[0], label[1])
            for j in range(len(r.target)):
                roles[i, len(r.pre) + j] = said_label(r, j, design)[2]
        tensor = torch.from_numpy(x)
        if model == "lstm":
            hidden, _ = net.cell(tensor)
            logits = net.out(hidden)
        else:
            logits, hidden = mn.transformer_forward(net, tensor)
        entropy = torch.nn.functional.cross_entropy
        loss = entropy(logits.reshape(-1, gr.ACTIONS), torch.from_numpy(y).reshape(-1),
                       ignore_index=-100)
        loss = loss + entropy(head(hidden).reshape(-1, len(design.roles)),
                              torch.from_numpy(roles).reshape(-1), ignore_index=-100)
        learner.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*net.parameters(), *head.parameters()], ct.CLIP)
        learner.opt.step()
        learner.work["updates"] += 1
        learner.work["tokens"] += x.shape[0] * x.shape[1]
        learner.work["macs"] += 3 * x.shape[0] * net.macs(x.shape[1])
        return float(loss.item())

    return update


def control_life(spec: dict[str, Any]) -> dict[str, Any]:
    """One control (``"kind": "store_control"``): the chamber's LSTM or transformer at
    ``width``, taught the same lesson batches (``train``) online (one pass) or offline (to the
    controls' stopping rule), its panels said and its hidden state probed as the brain's is,
    with the full mix the probes across wordings too. With ``roles`` (a design's name) its
    lessons are the brain's in full: the echoes and every moment's role (``_taught_roles``).
    Its row reads as one of ``meaning.sentence_control``."""
    import torch

    torch.set_num_threads(1)
    ct = mn._controls()
    model, mode, founder = str(spec["model"]), str(spec["mode"]), int(spec["founder"])
    train = str(spec.get("train", "all"))
    count = int(spec.get("batches", mn.BATCHES))
    learner = ct.Learner(model, int(spec["width"]), founder, ct.RATES[mode])
    teach = (_taught_roles(learner, model, DESIGNS[str(spec["roles"])]) if spec.get("roles")
             else learner.update)
    split = gr.Split.of(founder)
    rng = np.random.default_rng(np.random.SeedSequence([founder, gr.SEEDS["train"]]))
    lessons = [lesson_batch(train, rng, split) for _ in range(count)]
    started = time.monotonic()
    epochs = 1
    if mode == "online":
        for batch in lessons:
            teach(batch)
    else:
        order = np.random.default_rng(np.random.SeedSequence([founder, 99, 0]))
        best, stale, epochs = float("inf"), 0, 0
        while epochs < ct.EPOCHS and stale < ct.PATIENCE:
            mean = float(np.mean([teach(lessons[int(i)])
                                  for i in order.permutation(count)]))
            epochs += 1
            stale = 0 if mean < best - 1e-4 else stale + 1
            best = min(best, mean)
    learner.net.eval()
    batches = int(spec.get("panel_batches", 4))
    tests = {**gr.panels(founder, batches), **mn.sayback_panels(founder, batches)}
    names = mn.SAID + mn.BACK
    if train == "all":
        tests.update(more_panels(founder, batches))
        names += MORE
    taught = bool(spec.get("roles"))
    row: dict[str, Any] = {
        "kind": "sentence_control", "model": model + (" with roles" if taught else ""),
        "mode": mode, "founder": founder, "roles": spec.get("roles"),
        "train": train, "width": int(spec["width"]),
        "parameters": ct.count(learner.net) + (ct.count(learner.role_head) if taught else 0),
        "brain_parameters": int(spec.get("parameters", 0)), "epochs": epochs,
        "macs": float(learner.work["macs"]),
        "panels": {name: gr.read_speaker(learner.speak, None, tests[name],
                                         twins=name in gr.TWINS) for name in names},
    }
    for panel in row["panels"].values():
        panel.pop("examples", None)
    per_shape = int(spec.get("held_probes", 4))
    both = mn.split_batches(founder, per_shape)
    places = ("carried", "settled") if model == "lstm" else ("settled",)
    states = mn.control_states(model, learner.net, both)
    row["probes_held"] = mn.transfer_probes(founder, states["trained"], states["held"], places)
    words = {k: [[mn.sayback(r) for r in rows] for rows in both[k]] for k in both}
    states = mn.control_states(model, learner.net, words)
    row["probes_back"] = mn.transfer_probes(founder, states["trained"], states["held"], places)
    if train == "all":
        states = mn.control_states(model, learner.net, cross_batches(founder, per_shape))
        row["probes_cross"] = {
            "g1_to_g2": mn.transfer_probes(founder, states["g1"], states["g2_held"], places),
            "g2_to_g1": mn.transfer_probes(founder, states["g2"], states["g1_held"], places)}
    row["seconds"] = round(time.monotonic() - started, 1)
    return row


def run_life(spec: dict[str, Any]) -> dict[str, Any]:
    """A plan's life: a store life, a control of the full mix, or one of ``meaning.py``'s
    (the say-back mix's controls)."""
    kind = spec.get("kind")
    return (life if kind == "store" else control_life if kind == "store_control"
            else mn.run_life)(spec)


# ------------------------------------------------------------------------------- reading


def read(rows: list[dict[str, Any]], controls: list[dict[str, Any]], of: int = 10) -> list[str]:
    """The table and the gates (``meaning.sentence_verdict``), then where sentences are lost:
    each word kind with the previous words and roles given, and the roles named."""
    lines = mn.sentence_verdict(rows, controls, of=of)
    arms = mn._by_tag(rows)
    taught = sorted({c["model"] for c in controls if c.get("roles")})
    if taught:
        lines += ["", "each arm less the controls taught the echoes and roles, offline, paired "
                  "by founder: mean (s.e.)"]
    for tag, lives in arms.items():
        for model in taught:
            theirs = {c["founder"]: c for c in controls
                      if c["model"] == model and c["mode"] == "offline"}
            pairs = [r for r in lives if r["founder"] in theirs]
            cells = []
            for n in mn.SAID + mn.BACK:
                mine = [mn._score((r["panels"] if n in mn.SAID else r["sayback"])[n], n) or 0.0
                        for r in pairs]
                other = [mn._score(theirs[r["founder"]]["panels"][n], n) or 0.0 for r in pairs]
                if pairs:
                    mean, error = mn.paired(mine, other)
                    cells.append(f"{n} {mean:+.2f} ({error:.2f})")
            if cells:
                lines.append(f"  {tag} less {model}: " + ", ".join(cells))
    for key, title in (("kinds", "the panels' sentences"), ("kinds_held", "held-out clauses"),
                       ("kinds_back", "held-out clauses heard as words"),
                       ("kinds_long", "longer clauses")):
        lines += ["", f"{title}, the previous words and roles given: mean"]
        for tag, lives in arms.items():
            for part in ("words", "roles", "own"):
                found = [r[key][part] for r in lives if key in r and part in r[key]]
                if found:
                    names = sorted({k for d in found for k in d})
                    lines.append(f"  {tag} {part}: " + ", ".join(
                        f"{k} {np.mean([d[k] for d in found if k in d]):.2f}" for k in names))
    lines += more_verdict(rows, controls, of)
    lines += ["", "curves (batches: lean panels), mean"]
    for tag, lives in arms.items():
        points = sorted({p["batches"] for r in lives for p in r.get("curve", ())})
        for b in points:
            found = [p for r in lives for p in r.get("curve", ()) if p["batches"] == b]
            lines.append(f"  {tag} at {b}: " + ", ".join(
                f"{k} {np.mean([p[k] for p in found if k in p]):.2f}"
                for k in LEAN + LEAN_MORE if any(k in p for p in found)))
    lines += ["", "seconds a life, mean"]
    for tag, lives in arms.items():
        lines.append(f"  {tag}: {np.mean([r['seconds'] for r in lives]):.0f} "
                     f"(lessons {np.mean([r['work']['seconds'] for r in lives]):.0f})")
    return lines


def more_verdict(rows: list[dict[str, Any]], controls: list[dict[str, Any]],
                 of: int = 10) -> list[str]:
    """The full mix's gates, for the lives that hold its panels. H3: one wording said from
    the other (``para_12`` and ``para_21``, trained and held-out combinations) at .90 on ``of``
    lives, and the least clause component's probe across wordings (fitted on one wording's
    trained combinations, scored on the other's held-out ones, the store read with the
    association) at .95 on ``of``. H4: the questions (heard as concepts and as words, trained
    and held-out) at .90 on ``of``. Then every panel beside the matched controls offline."""
    arms = {tag: lives for tag, lives in mn._by_tag(rows).items()
            if any("more" in r for r in lives)}
    if not arms:
        return []
    lines = ["", "the full mix's panels, mean", "",
             "| Learner | Parameters | Lives | " + " | ".join(MORE) + " | across (1 to 2) | "
             "across (2 to 1) |",
             "| --- | --- | --- | " + " | ".join("---" for _ in MORE) + " | --- | --- |"]

    def across(lives: list[dict[str, Any]], way: str, place: str) -> list[float | None]:
        return [mn._least(r["probes_cross"][way], "heard", place) for r in lives
                if r.get("probes_cross")]

    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for c in controls:
        if "more" in c or any(name in c.get("panels", {}) for name in MORE):
            groups.setdefault((int(c["brain_parameters"]), c["model"], c["mode"]), []).append(c)
    for tag, lives in arms.items():
        lines.append(f"| `{tag}` | {lives[0]['parameters']} | {len(lives)} | " + " | ".join(
            mn._short([mn._score(r["more"][n], n) for r in lives]) for n in MORE)
            + f" | {mn._short(across(lives, 'g1_to_g2', 'joint'))}"
            + f" | {mn._short(across(lives, 'g2_to_g1', 'joint'))} |")
    for (_size, model, mode), lives in sorted(groups.items()):
        lines.append(f"| {model}, {mode} | {lives[0]['parameters']} | {len(lives)} | " + " | ".join(
            mn._short([mn._score(c["panels"][n], n) for c in lives]) for n in MORE)
            + f" | {mn._short(across(lives, 'g1_to_g2', 'settled'))}"
            + f" | {mn._short(across(lives, 'g2_to_g1', 'settled'))} |")
    lines.append("")
    for tag, lives in arms.items():
        counts = {n: sum((mn._score(r["more"][n], n) or 0.0) >= 0.9 for r in lives)
                  for n in MORE}
        probes = {way: sum(v is not None and v >= 0.95 for v in across(lives, way, "joint"))
                  for way in ("g1_to_g2", "g2_to_g1")}
        h3 = all(counts[n] >= of for n in ("para_12", "para_12_held", "para_21", "para_21_held")
                 ) and all(c >= of for c in probes.values())
        h4 = all(counts[n + tail] >= of for n in ASKED for tail in ("", "_held"))
        lines.append(
            f"  {tag}: H3 said " + ", ".join(
                f"{n} {counts[n]}" for n in ("para_12", "para_12_held", "para_21",
                                              "para_21_held"))
            + "; held across wordings " + ", ".join(f"{w} {c}" for w, c in probes.items())
            + f" of {len(lives)} -> " + ("holds" if h3 else "fails")
            + "; H4 " + ", ".join(f"{n + t} {counts[n + t]}" for n in ASKED
                                  for t in ("", "_held"))
            + f" of {len(lives)} -> " + ("holds" if h4 else "fails"))
        size = int(lives[0]["parameters"])
        for model in ("lstm", "transformer"):
            theirs = {c["founder"]: c for (s, m, mode), found in groups.items() for c in found
                      if m == model and mode == "offline" and abs(s - size) <= 0.02 * size}
            paired_lives = [r for r in lives if r["founder"] in theirs]
            if not paired_lives:
                continue
            cells = []
            for n in MORE:
                mine = [mn._score(r["more"][n], n) or 0.0 for r in paired_lives]
                other = [mn._score(theirs[r["founder"]]["panels"][n], n) or 0.0
                         for r in paired_lives]
                mean, error = mn.paired(mine, other)
                cells.append(f"{n} {mean:+.2f} ({error:.2f})")
            lines.append(f"  {tag} less {model}: " + ", ".join(cells))
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
                 modes: tuple[str, ...] = ("offline",), train: str = "sayback",
                 batches: int = mn.BATCHES, roles: tuple[str | None, ...] = (None,),
                 ) -> list[dict[str, Any]]:
    """The chamber's LSTM and transformer at ``parameters``, taught the same lesson batches
    (``control_life``); ``roles`` names for each further set a design whose echoes and roles
    the controls are taught as well."""
    out = []
    for model in ("lstm", "transformer"):
        width = mn.wide_width(model, parameters, limit=1200)
        for mode in modes:
            for design in roles:
                out += [{"kind": "store_control", "model": model, "mode": mode, "founder": f,
                         "train": train, "parameters": parameters, "width": width,
                         "held_probes": held_probes, "batches": batches,
                         **({"roles": design} if design else {})} for f in founders]
    return out


SECOND = tuple(range(10828600, 10828612))  # twelve fresh founders (SENTENCE_PROTOCOL_2.md)
# the declared arm: a block takes the observation alone, lessons where the answer differs (the
# library's default for a new brain), the full mix
STORE = {"kind": "store", "design": "alone", "train": "all", "held_probes": 16,
         "settings": {"learning_surprise": True},
         "checkpoints": [1000, 2000, 3000]}


def second_plan(founders: tuple[int, ...] = SECOND) -> dict[str, list[dict[str, Any]]]:
    """The second declared run of cadence#308 (``SENTENCE_PROTOCOL_2.md``): the store's arm on
    the full mix, and the chamber's LSTM and transformer taught the same batches at its
    parameter count, online and offline, and offline with the echoes and roles as well."""
    brains = [{**STORE, "founder": f, "tag": "store"} for f in founders]
    parameters = int(make_brain(founders[0], STORE["settings"], 256, DEPTH, True,
                                DESIGNS["alone"]).parameters())
    controls = control_plan(founders, parameters, held_probes=16, train="all",
                            modes=("offline", "online"))
    controls += control_plan(founders, parameters, held_probes=16, train="all",
                             modes=("offline",), roles=("alone",))
    return {"second": brains, "second_controls": controls}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    planner = sub.add_parser("plan", help="write plans into a folder")
    planner.add_argument("--out", type=Path, required=True)
    planner.add_argument("--dev", action="store_true", help="development lives")
    planner.add_argument("--second", action="store_true",
                         help="the second declared run (SENTENCE_PROTOCOL_2.md)")
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
        if args.second:
            plans = second_plan(tuple(args.founders or SECOND))
        else:
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
