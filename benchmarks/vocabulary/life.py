"""One continuing life of the composed brain in the grammar chamber at a vocabulary size.

The life, the lessons and the scorer are the grammar chamber's (``benchmarks/grammar/``): a
meaning arrives as concept moments, at ``GO`` the brain says the sentence word by word, a lesson
comes before the act on every word it is taught, panels are read on forks, greedy, closed loop.
``Language(4, 2)`` lives the chamber's life to the bit (a test holds it). Each checkpoint writes
a line at once, so a life cut short keeps what it read.

    python benchmarks/vocabulary/life.py --words 200 --founder 0 --arm Fd \\
        --checkpoints 500 2000 8000 --out DIR/raw200.jsonl
    python benchmarks/vocabulary/life.py --summary DIR/raw200.jsonl
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import resource
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from language import EVERYTHING, LEAN, SEEDS, TWINS, Language, Row, by_size
from pieces import PieceLanguage

RECIPE: dict[str, Any] = {
    "recall": True, "recall_recency": 0.0, "learning_surprise": True, "recall_boundary": True,
    "recall_anchor": True, "recall_slope": 0.1, "recall_position_rate": 0.01,
    "recall_relative": True,
}
SEQUENCE: dict[str, Any] = {"episodic": False, "teaching": "normalized"}
ARMS: dict[str, dict[str, Any]] = {
    "recipe": {**SEQUENCE, **RECIPE},
    # the read's rule: the deep teaching point, no apical (the address lesson is the library's
    # contrast)
    "Fd": {**SEQUENCE, **RECIPE, "teaching": "deep"},
}


def make_brain(lang: Any, arm: str, founder: int, settings: dict[str, Any] | None = None,
               module: int = 64) -> Any:
    import cadence as cd

    options = {**ARMS[arm], **(settings or {})}
    options.setdefault("backend", "cpu")  # one backend for every life and size
    if options.get("recall_tokens") == "words":  # action j and observation j are one token
        options["recall_tokens"] = list(range(lang.actions))
    scale = options.pop("eta_scale", None)  # the teaching point's steps, scaled, as an expert
    map_code = options.pop("map_code", None)  # the read's maps on a fixed code (``mapcode``)
    gain = options.pop("init_gain", None)  # ``both``, ``in`` or ``motor``: see ``chamber_gain``
    sharp = options.pop("address_scale", None)  # the lesson's temperature follows the key width
    cortex = options.pop("cortex", None)  # how the maps take the cortex (``install_cortex``)
    if gain is not None:
        chamber_gain(lang, str(gain), module, options)
    brain = cd.Brain.compose(lang.inputs, lang.actions, modules=(module,), seed=int(founder),
                             **options)
    if scale is not None:
        config = brain.learner.config
        brain.retune(learning=dataclasses.replace(config, eta=config.eta * float(scale),
                                                  eta_bias=config.eta_bias * float(scale)))
    if sharp is not None:  # an emulation: a slot's share of the read-back falls with the width
        brain.recall.address_temperature = 0.1 * float(np.sqrt(35 / lang.inputs))
    if map_code is not None:
        import mapcode

        mapcode.install(brain, int(map_code), int(founder))
    if cortex is not None:
        install_cortex(brain, cortex)
    return brain


CORTEX_MAPS = ("head_maps", "position_maps", "anchor_maps", "relative_maps", "slope_maps",
               "grid_maps")


def install_cortex(brain: Any, how: dict[str, Any]) -> None:
    """A development emulation of how the read's maps take the cortex of the moment before, on
    one brain and its forks: ``unit`` reads it at unit norm per stream, ``scale`` times a
    number, and ``rate`` scales each lesson's step on every map's cortex columns. The brain's
    class becomes a subclass that does this around ``act`` and ``lesson``."""
    unit, scale = bool(how.get("unit")), how.get("scale")
    rate = how.get("rate")

    def read(recall: Any) -> None:
        if unit:
            raw = np.atleast_2d(np.asarray(recall.previous, dtype=float))
            norm = np.linalg.norm(raw, axis=1, keepdims=True)
            recall.previous = np.where(norm > 0, raw / np.where(norm > 0, norm, 1.0), raw)
        if scale is not None:
            recall.previous = float(scale) * np.asarray(recall.previous)

    base = type(brain)

    class Balanced(base):  # type: ignore[misc, valid-type]
        def act(self, *args: Any, **kwargs: Any) -> Any:
            read(self.recall)
            return base.act(self, *args, **kwargs)

        def lesson(self, *args: Any, **kwargs: Any) -> Any:
            read(self.recall)
            before = None if rate is None else {
                name: [np.array(h[1]) for h in getattr(self.recall, name, None) or []]
                for name in CORTEX_MAPS}
            try:
                return base.lesson(self, *args, **kwargs)
            finally:
                if before is not None:
                    for name, columns in before.items():
                        maps = getattr(self.recall, name, None) or []
                        setattr(self.recall, name, [
                            (h[0], old + float(rate) * (np.asarray(h[1]) - old), h[2])
                            for h, old in zip(maps, columns, strict=True)])

    Balanced.__name__ = base.__name__
    Balanced.__qualname__ = base.__qualname__
    brain.__class__ = Balanced


def chamber_gain(lang: Any, gain: str, module: int, options: dict[str, Any]) -> None:
    """A development emulation, one process per life: a projection's initial synapses are scaled
    by its fan, ``sqrt(6 / (pre + post))``, which suits dense activity, and at a one-hot word a
    large vocabulary's single active neuron then drives far less than at the chamber's size. This
    sets the sensory, recall and next projections (``in``) and the association's with the motor
    region (``motor``) to the per-synapse magnitude they have at 18 words: their scales times
    ``sqrt((pre + post) / (pre + post at 18 words))``."""
    from cadence import generic

    if gain in ("both", "in"):
        factor = float(np.sqrt((lang.inputs + module) / (35 + module)))
        options["scales"] = {**options.get("scales", {}), "sensory": factor}
        generic._RECALL_SCALE = 1.2 * factor
    if gain in ("both", "motor"):
        factor = float(np.sqrt((lang.actions + module) / (18 + module)))
        original = generic.Projection

        def scaled(pre: str, post: str, **kw: Any) -> Any:
            if (pre, post) == ("association", "motor"):
                kw["scale"] = factor
            return original(pre, post, **kw)

        generic.Projection = scaled


def new_work() -> dict[str, float]:
    return {"lessons": 0, "rows": 0, "agreed_rows": 0, "lesson_row_sweeps": 0, "acts": 0,
            "act_sweeps": 0, "refused": 0, "exhausted": 0, "free_budget_exhausted": 0}


def speak(lang: Language, brain: Any, rows: list[Row], *, teach: bool, closed: bool = True,
          work: dict[str, float] | None = None, cap: int = 4,
          intervene: Any = None) -> list[list[int]]:
    """One batch on fresh streams (``grammar.speak``). Heard moments are acted on greedily and
    never taught. From ``GO`` each moment hears the previous word: the target's while teaching,
    within a row's ``force`` and when ``closed`` is off; else the brain's own. With ``teach`` a
    lesson on the target word comes before every act from ``GO`` on."""
    work = work if work is not None else new_work()
    n = len(rows)
    lengths = {len(r.pre) for r in rows}
    if len(lengths) != 1:
        raise ValueError("a batch's rows hear the same number of moments")
    if teach and len({len(r.target) for r in rows}) != 1:
        raise ValueError("a taught batch's sentences have one length")
    brain.reset()
    for t in range(lengths.pop()):
        act(brain, lang.dense([r.pre[t] for r in rows]), work)
    said: list[list[int]] = [[] for _ in rows]
    done = [False] * n
    longest = max(len(r.target) for r in rows) + (0 if teach else cap)
    for j in range(longest):
        if j == 0:
            x = lang.dense([(lang.go,)] * n)
        else:
            moments: list[tuple[int, ...]] = []
            for i, r in enumerate(rows):
                given = teach or not closed or j <= r.force
                w = r.target[j - 1] if given and j - 1 < len(r.target) else said[i][j - 1]
                moments.append(lang.spoken(w) if w >= 0 else (lang.speak,))
            x = lang.dense(moments)
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
        action = act(brain, x, work)
        for i, r in enumerate(rows):
            said[i].append(int(action[i]))
            if j >= r.force and action[i] == lang.end:
                done[i] = True
        if not teach and all(done):
            break
    out = []
    for i, r in enumerate(rows):
        words = said[i][r.force:]
        out.append(words[:words.index(lang.end) + 1] if lang.end in words else words)
    return out


def act(brain: Any, x: np.ndarray, work: dict[str, float]) -> np.ndarray:
    try:
        action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
        settled = brain.last_settlement or {}
        work["act_sweeps"] += float(settled.get("steps", 0))
    except RuntimeError:
        work["exhausted"] += 1
        action = np.full(len(x), -1)
    work["acts"] += 1
    return action


def erase(where: str) -> Any:
    """``grammar.erase``: the earlier context emptied from the recall slots, or the trace."""

    def apply(brain: Any, rows: list[Row]) -> None:
        if where == "trace":
            brain.working_memory.trace = np.zeros_like(brain.working_memory.trace)
            return
        recall = brain.recall
        if recall is None:
            return
        written = len(rows[0].pre) + rows[0].force
        ages = np.asarray(recall.ages())
        meaning = ages >= written - len(rows[0].pre) - 2
        recall.keys = np.where(meaning[None, :, None], 0.0, recall.keys)
        recall.values = np.where(meaning[None, :, None], 0.0, recall.values)
        recall.nexts = np.where(meaning[None, :, None], 0.0, recall.nexts)

    return apply


def forced(lang: Language, brain: Any, rows: list[Row]) -> list[float]:
    """Next-word accuracy with every previous word given (component evidence)."""
    brain.reset()
    for t in range(len(rows[0].pre)):
        brain.act(lang.dense([r.pre[t] for r in rows]), greedy=True)
    hits = []
    for j in range(len(rows[0].target)):
        x = (lang.dense([(lang.go,)] * len(rows)) if j == 0
             else lang.dense([lang.spoken(r.target[j - 1]) for r in rows]))
        action = np.asarray(brain.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
        if j >= rows[0].force:
            hits += [float(action[i] == r.target[j]) for i, r in enumerate(rows)]
    return hits


def sources(lang: Language, row: Row) -> list[int | None]:
    """For each target word, the heard moment that holds its content, or None for a word the
    grammar alone decides (``near``, ``end``)."""
    if "sources" in row.meta:
        return list(row.meta["sources"])
    if row.family == "clause":
        m = row.meta
        L = m["L"]
        agent, mods, patient = ((2, list(range(3, 3 + L)), 1) if m["inverted"]
                                else (1, list(range(2, 2 + L)), 2 + L))
        out: list[int | None] = [agent]
        for i in mods:
            out += [None, i]
        tail: list[int | None] = [patient, 0] if m["dialect"] else [0, patient]
        return out + tail + [None]
    if row.family == "dyck":
        out, stack = [], []
        for i, x in enumerate(row.pre):
            if lang.c_close in x:
                out.append(stack.pop())
            else:
                out.append(i)
                stack.append(i)
        return out + list(reversed(stack)) + [None]
    out = [len(row.pre) - 1]  # the start, named by the meaning
    heard_words = [x[0] for x in row.pre[:-1]]
    for a in row.meta["chain"][:-1]:
        at = next(i for i in range(0, len(heard_words), 4) if heard_words[i] == a)
        out += [None, at + 2]  # the fact's second word as heard
    return out + [None]


def kind(lang: Language, r: Row, j: int) -> str:
    if "slots" in r.meta:
        return lang.kind(r, j)
    if r.family == "dyck" and "sources" in r.meta:
        return "dyck_" + ("open" if r.target[j] in lang.opens else "close")
    if r.family == "clause":
        m = r.meta
        k = next(i for i, w in enumerate(r.target) if lang.is_verb(w))
        if j == 0:
            name = "agent"
        elif j == k:
            name = "verb"
        elif r.target[j] == lang.noun(*m["patient"]) and (
                j == k + 1 or (m["dialect"] and j == k - 1)):
            name = "patient"
        else:
            name = "mod"
        return f"clause_{name}"
    if r.family == "dyck":
        return "dyck_" + ("open" if r.target[j] in lang.opens else "close")
    return "path_" + ("start" if j == 0 else "hop")


def locate(lang: Language, brain: Any, batches: list[list[Row]]) -> dict[str, Any]:
    """Where the read lands and what is said, teacher-forced, on a fork: for every word whose
    content a heard moment holds, whether the recall memory's first head attends most to that
    moment or to the moment before it (``delivered``), and whether the brain says the word. A
    word delivered and not said is lost at the readout, one not delivered at the read."""
    fork = copy.deepcopy(brain)
    groups: dict[str, list[int]] = {}
    for rows in batches:
        fork.reset()
        written = 0
        for t in range(len(rows[0].pre)):
            fork.act(lang.dense([r.pre[t] for r in rows]), greedy=True)
            written += 1
        expected = [sources(lang, r) for r in rows]
        for j in range(len(rows[0].target)):
            x = (lang.dense([(lang.go,)] * len(rows)) if j == 0
                 else lang.dense([lang.spoken(r.target[j - 1]) for r in rows]))
            ages = None if fork.recall is None else np.asarray(fork.recall.ages())
            action = np.asarray(fork.act(x, greedy=True)).reshape(len(x), -1)[:, 0]
            heads = fork.recall.attended_heads if fork.recall is not None else None
            p = heads[0] if heads else None
            for i, r in enumerate(rows):
                source = expected[i][j]
                if source is None:
                    continue
                landed = -2
                if p is not None and ages is not None:
                    landed = written - 1 - int(ages[int(np.argmax(p[i]))])
                g = groups.setdefault(kind(lang, r, j), [0, 0, 0, 0])
                found = landed in (source, source - 1)
                g[0] += int(found)
                g[1] += int(action[i] == r.target[j])
                g[2] += int(found and action[i] == r.target[j])
                g[3] += 1
            written += 1
    return {k: {"n": v[3], "delivered": round(v[0] / v[3], 4), "said": round(v[1] / v[3], 4),
                "said_when_delivered": round(v[2] / v[0], 4) if v[0] else None}
            for k, v in sorted(groups.items())}


TERMS = ("key", "event", "cortex", "bias", "position_event", "position_cortex", "position_bias")


def logit_terms(brain: Any, x: np.ndarray) -> dict[str, np.ndarray]:
    """Head 0's logits over each stream's slots at the moment ``x``, by what they come from,
    read before the act: the event against a slot's key (``key``), the address map's event rows,
    cortex columns and bias against it (``event``, ``cortex``, ``bias``), and the same three of
    every position map at the slot's buckets (``position_*``). They sum to the logits of the read
    (``total``), whose softmax is the attention the act reads with."""
    recall = brain.recall
    moment = recall._moment(brain.observation_code(x))
    inputs, c = moment.inputs, moment.cortex
    w_x, w_c, b = recall.head_map(0)
    keys, temperature = recall.keys, recall.temperature
    out = {
        "key": np.einsum("bk,bmk->bm", moment.key, keys) / temperature,
        "event": np.einsum("bk,bmk->bm", inputs @ w_x, keys) / temperature,
        "cortex": np.einsum("bk,bmk->bm", c @ w_c, keys) / temperature,
        "bias": np.einsum("k,bmk->bm", b, keys) / temperature,
    }
    zero_x, zero_c = np.zeros_like(inputs), np.zeros_like(c)
    whole = recall.position_term(0, inputs, c)
    if whole is None:
        whole = np.zeros_like(out["key"])
        event_only = bias_only = whole
    else:
        event_only = recall.position_term(0, inputs, zero_c)
        bias_only = recall.position_term(0, zero_x, zero_c)
    out["position_cortex"] = whole - event_only
    out["position_event"] = event_only - bias_only
    out["position_bias"] = bias_only
    total = sum(out[name] for name in TERMS)
    if recall.recency:
        total = total - recall.recency * recall.ages()[None, :]
    out["total"] = np.where(recall.held, total, -np.inf)
    return out


def terms(lang: Language, brain: Any, batches: list[list[Row]]) -> dict[str, Any]:
    """Where the read's selection is decided, teacher-forced on a fork: at every word whose
    content a heard moment holds, each term's logit at the right slot (the source moment or the
    one before it, whichever the read scores higher) less its logit at the best other slot, per
    kind of word, and over the rows the read misses. ``check`` is the largest gap between the
    softmax of the summed terms and the attention the act read with."""
    fork = copy.deepcopy(brain)
    sums: dict[str, dict[str, Any]] = {}
    check = 0.0
    for rows in batches:
        fork.reset()
        written = 0
        for t in range(len(rows[0].pre)):
            fork.act(lang.dense([r.pre[t] for r in rows]), greedy=True)
            written += 1
        expected = [sources(lang, r) for r in rows]
        for j in range(len(rows[0].target)):
            x = (lang.dense([(lang.go,)] * len(rows)) if j == 0
                 else lang.dense([lang.spoken(r.target[j - 1]) for r in rows]))
            parts = logit_terms(fork, x)
            ages = np.asarray(fork.recall.ages()).astype(int)
            fork.act(x, greedy=True)
            heads = fork.recall.attended_heads
            if heads:
                total = parts["total"]
                p = np.exp(total - total.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                check = max(check, float(np.abs(p - heads[0]).max()))
            for i, r in enumerate(rows):
                source = expected[i][j]
                if source is None:
                    continue
                places = [int(s) for s in np.flatnonzero(np.isin(
                    written - 1 - ages, (source, source - 1)) & fork_held(parts, i))]
                if not places:
                    continue
                right = max(places, key=lambda s: parts["total"][i, s])
                others = np.array([s for s in range(len(ages)) if s not in places
                                   and np.isfinite(parts["total"][i, s])])
                if not len(others):
                    continue
                rival = int(others[np.argmax(parts["total"][i, others])])
                missed = parts["total"][i, rival] > parts["total"][i, right]
                for group in (kind(lang, r, j), "all"):
                    g = sums.setdefault(group, {"n": 0, "missed": 0, "terms": dict.fromkeys(
                        (*TERMS, "total"), 0.0), "missed_terms": dict.fromkeys(
                        (*TERMS, "total"), 0.0)})
                    g["n"] += 1
                    g["missed"] += int(missed)
                    for name in (*TERMS, "total"):
                        gap = float(parts[name][i, right] - parts[name][i, rival])
                        g["terms"][name] += gap
                        if missed:
                            g["missed_terms"][name] += gap
            written += 1
    out: dict[str, Any] = {"check": round(check, 6)}
    for group, g in sorted(sums.items()):
        n, m = g["n"], g["missed"]
        out[group] = {
            "n": n, "missed": round(m / n, 4),
            "gap": {k: round(v / n, 3) for k, v in g["terms"].items()},
            "gap_missed": {k: round(v / m, 3) for k, v in g["missed_terms"].items()} if m else None,
        }
    return out


def fork_held(parts: dict[str, np.ndarray], row: int) -> np.ndarray:
    """The slots a stream holds, from the terms' total."""
    return np.isfinite(parts["total"][row])


def read_panel(lang: Language, brain: Any, batches: list[list[Row]], *, twins: bool = False,
               components: bool = False, intervene: Any = None) -> dict[str, Any]:
    fork = copy.deepcopy(brain)
    return lang.read_speaker(
        lambda rows: speak(lang, fork, rows, teach=False, intervene=intervene),
        (lambda rows: forced(lang, fork, rows)) if components else None,
        batches, twins=twins)


ERASED = ("ctx_agent", "ctx_verb", "ctx_dyck")


def reading(lang: Language, brain: Any, tests: dict[str, list[list[Row]]],
            names: tuple[str, ...], components: bool = False) -> dict[str, Any]:
    """Each named panel; with the context panels, the same twins with the context erased from
    the recall memory or from the working trace at the continuation's start."""
    out = {name: read_panel(lang, brain, tests[name], twins=name in TWINS, components=components)
           for name in names}
    for name in ERASED:
        if name in names:
            for where in ("recall", "trace"):
                out[f"{name}_erased_{where}"] = read_panel(
                    lang, brain, tests[name], twins=True, intervene=erase(where))
    return out


def code_read(lang: Language, brain: Any, sample: int = 48) -> dict[str, Any]:
    """The learned code itself: how words of a class lie together and whether a word keeps its
    identity. Cosines of the keys of noun concepts (with the agent's flag), the nouns' own heard
    words and the verbs' concepts; ``same`` the share of a class's pairs at cosine .98 or above
    (two words the memories cannot tell apart); ``units`` the mean active units of a key."""

    def keys(moments: list[tuple[int, ...]]) -> np.ndarray:
        rows = [brain.observation_code(lang.dense(moments[i:i + 16]))
                for i in range(0, len(moments), 16)]
        k = np.concatenate(rows)
        return k / np.maximum(np.linalg.norm(k, axis=1, keepdims=True), 1e-12)

    def within(k: np.ndarray) -> tuple[float, float]:
        c = k @ k.T
        off = c[~np.eye(len(k), dtype=bool)]
        return float(off.mean()), float((off >= 0.98).mean())

    fork = copy.deepcopy(brain)
    brain = fork
    stems = range(min(sample, lang.trained))
    concepts = keys([lang.noun_moment(i, 0, lang.agent) for i in stems])
    words = keys([lang.heard(lang.noun(i, 0)) for i in stems])
    verbs = keys([(lang.verb_concept0 + v,) for v in range(min(sample, lang.verbs))])
    out: dict[str, Any] = {}
    for name, k in (("noun_concepts", concepts), ("noun_words", words), ("verb_concepts", verbs)):
        mean, same = within(k)
        out[name] = {"within": round(mean, 4), "same": round(same, 4),
                     "units": round(float((k > 0).sum(axis=1).mean()), 2)}
    out["concept_to_verb"] = round(float((concepts @ verbs.T).mean()), 4)
    out["word_to_verb"] = round(float((words @ verbs.T).mean()), 4)
    out["concept_to_own_word"] = round(float(np.mean(np.sum(concepts * words, axis=1))), 4)
    return out


def rss_gb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(usage / (1024**3 if sys.platform == "darwin" else 1024**2), 2)


def language_of(spec: dict[str, Any]) -> Any:
    """The spec's language: by wordpiece with ``lexicon``, else by word at ``words``."""
    if spec.get("lexicon"):
        families = spec.get("families")
        return PieceLanguage(spec["lexicon"], str(spec.get("route", "heard")),
                             bool(spec.get("swap")), families, bool(spec.get("start")),
                             float(spec.get("pseudo", 0.0)), bool(spec.get("verb_tokens")))
    nouns, verbs = (int(spec["nouns"]), int(spec["verbs"])) if "nouns" in spec else by_size(
        int(spec["words"]))
    return Language(nouns, verbs, int(spec.get("fresh", 0)), bool(spec.get("verb_flag")))


def run_life(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """One life: ``words`` (or ``nouns`` and ``verbs``, or ``lexicon``, ``route`` and ``swap``
    for the wordpiece language), ``fresh``, ``arm``, ``founder``,
    ``settings``, ``module``, ``checkpoints`` (batches at which a lean reading is taken and,
    with ``save`` (a folder), the brain saved), ``full`` (batches read on every panel), ``out``
    (JSON lines, appended at each checkpoint), ``panel_batches``, ``locate``."""
    warnings.simplefilter("ignore")
    import cadence as cd

    lang = language_of(spec)
    arm, founder = str(spec.get("arm", "Fd")), int(spec["founder"])
    module = int(spec.get("module", 64))
    started = time.monotonic()
    brain = make_brain(lang, arm, founder, spec.get("settings"), module)
    split = lang.split(founder)
    tests = lang.panels(founder, int(spec.get("panel_batches", 4)))
    rng = np.random.default_rng(np.random.SeedSequence([founder, SEEDS["train"]]))
    checkpoints = sorted({int(c) for c in spec.get("checkpoints", [2000])})
    full = {int(c) for c in spec.get("full", [])}
    folder = Path(spec["save"]) if spec.get("save") else None
    sink = Path(spec["out"]) if spec.get("out") else None
    out: list[dict[str, Any]] = []
    head = {"arm": arm, "founder": founder, "words": lang.words, "nouns": lang.nouns,
            "verbs": lang.verbs, "fresh": getattr(lang, "fresh", 0), "inputs": lang.inputs,
            "actions": lang.actions, "module": module, "settings": spec.get("settings", {}),
            "parameters": int(brain.parameters()), "cadence": cd.__version__,
            "tag": spec.get("tag", "")}
    work = new_work()
    teaching = 0.0
    batch = 0
    grow = spec.get("grow")  # {"words": at the start, "hold": batches, "ramp": batches to all}
    whole = (lang.nouns, lang.verbs)
    for checkpoint in checkpoints:
        begun = time.monotonic()
        while batch < checkpoint:
            if grow:
                start = by_size(int(grow["words"]))
                share = min(1.0, max(0.0, (batch - int(grow["hold"])) / max(1, int(grow["ramp"]))))
                lang.grown(round(start[0] + share * (whole[0] - start[0])),
                           round(start[1] + share * (whole[1] - start[1])))
            speak(lang, brain, lang.train_batch(rng, split), teach=True, work=work)
            batch += 1
        teaching += time.monotonic() - begun
        begun = time.monotonic()
        names = (tuple(tests) if isinstance(lang, PieceLanguage)
                 else EVERYTHING if checkpoint in full else LEAN)
        row = {**head, "batches": batch, "work": dict(work), "teaching_seconds": round(teaching, 1),
               "panels": reading(lang, brain, tests, tuple(names), components=False)}
        if spec.get("locate", True) and brain.recall is not None:
            row["located"] = {name: locate(lang, brain, tests[name])
                              for name in (("taught", "novel_all", "path", "dyck")
                                           if isinstance(lang, PieceLanguage)
                                           else ("clause", "path", "dyck")) if name in tests}
        if spec.get("terms") and brain.recall is not None:
            row["terms"] = {name: terms(lang, brain, tests[name][:2])
                            for name in ("clause", "path") if name in tests}
        if getattr(brain, "learned_code", None) is not None and isinstance(lang, Language):
            row["code"] = code_read(lang, brain)
        moments = work["acts"]
        row["seconds_per_batch"] = round(teaching / max(1, batch), 4)
        row["seconds_per_moment"] = round(teaching / max(1, moments), 5)
        row["sweeps_per_act"] = round(work["act_sweeps"] / max(1, moments), 1)
        row["reading_seconds"] = round(time.monotonic() - begun, 1)
        row["rss_gb"] = rss_gb()
        row["seconds"] = round(time.monotonic() - started, 1)
        if folder is not None:
            folder.mkdir(parents=True, exist_ok=True)
            row["saved"] = str(brain.save(folder / f"{arm}-w{lang.words}-f{founder}-b{batch}"))
        out.append(row)
        if sink is not None:
            sink.parent.mkdir(parents=True, exist_ok=True)
            with sink.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
    if spec.get("continuation"):
        out[-1]["continuation"] = continuation(lang, brain, rng, split, tests["clause"][0])
    return out


def continuation(lang: Language, brain: Any, rng: np.random.Generator, split: Any,
                 rows: list[Row]) -> dict[str, Any]:
    """Save and load a fork; both copies take the same three lessons and say the same panel."""
    import cadence as cd

    live = copy.deepcopy(brain)
    with tempfile.TemporaryDirectory() as folder:
        loaded = cd.Brain.load(live.save(Path(folder) / "brain"))
    stream = copy.deepcopy(rng)
    batches = [lang.train_batch(stream, split) for _ in range(3)]
    for b in batches:
        speak(lang, live, b, teach=True)
        speak(lang, loaded, b, teach=True)
    said = [speak(lang, x, rows, teach=False) for x in (live, loaded)]
    return {"identical": said[0] == said[1], "sentences": len(rows)}


COLUMNS = (("clause", "exact", "cl"), ("clause", "L2", "L2"), ("clause_held", "exact", "held"),
           ("clause_long", "exact", "long"), ("roles", "exact", "roles"),
           ("clause", "agree", "agree"), ("dyck", "exact", "dyck"), ("path", "exact", "path"),
           ("path", "hops2", "2hop"), ("path_long", "exact", "3hop"))


def line(row: dict[str, Any]) -> str:
    cells = []
    for panel, key, name in COLUMNS:
        p = row["panels"].get(panel, {})
        value = p.get(key, p.get("conditions", {}).get(key))
        cells.append(f"{name} {value:.2f}" if value is not None else f"{name}  --")
    return (f"{row['arm']:6s} w{row['words']} f{row['founder']} b{row['batches']:6d} "
            f"p{row['parameters']:9d} {row['seconds_per_moment']:.4f}s/m "
            f"{row['sweeps_per_act']:5.1f}sw " + " ".join(cells))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--words", type=int, default=200)
    parser.add_argument("--fresh", type=int, default=0)
    parser.add_argument("--verb-flag", action="store_true")
    parser.add_argument("--grow", type=json.loads, default=None,
                        help='{"words": 200, "hold": 2000, "ramp": 3000}: a vocabulary that grows')
    parser.add_argument("--lexicon", default="", help="the wordpiece language's lexicon (JSON)")
    parser.add_argument("--route", default="heard", choices=("heard", "concept"))
    parser.add_argument("--swap", action="store_true")
    parser.add_argument("--start", action="store_true", help="a flag on a word's first piece")
    parser.add_argument("--pseudo", type=float, default=0.0, help="share of made-up stems")
    parser.add_argument("--verb-tokens", action="store_true", help="a verb's own agreeing pieces")
    parser.add_argument("--arm", default="Fd", choices=list(ARMS))
    parser.add_argument("--founder", type=int, default=0)
    parser.add_argument("--module", type=int, default=64)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[2000])
    parser.add_argument("--full", nargs="*", type=int, default=[])
    parser.add_argument("--settings", type=json.loads, default={})
    parser.add_argument("--panel-batches", type=int, default=4)
    parser.add_argument("--save", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    if args.summary is not None:
        rows = [json.loads(x) for x in args.summary.read_text().splitlines() if x]
        for row in sorted(rows, key=lambda r: (r["arm"], r["words"], r["founder"], r["batches"])):
            print(line(row))
        return 0
    run_life({"words": args.words, "fresh": args.fresh, "lexicon": args.lexicon,
              "verb_flag": args.verb_flag, "grow": args.grow,
              "route": args.route, "swap": args.swap, "start": args.start,
              "pseudo": args.pseudo, "verb_tokens": args.verb_tokens, "arm": args.arm,
              "founder": args.founder,
              "module": args.module, "checkpoints": args.checkpoints, "full": args.full,
              "settings": args.settings, "panel_batches": args.panel_batches,
              "save": args.save, "tag": args.tag, "out": str(args.out) if args.out else ""})
    return 0


if __name__ == "__main__":
    sys.exit(main())
