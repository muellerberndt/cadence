"""One hearing: a word heard once in one role, used in another, against the matched networks.

A life is taught on every stem but ``fresh`` of them. On a fork of it, sixteen fresh noun stems are
heard: one batch of sixteen taught sentences (``agent verb patient end``), each with its own fresh
stem in the singular, eight as the agent and eight as the patient, the other nouns taught ones.
Then the same stems are asked in new sentences, in the role they were heard in (``same``) and in the
other (``other``), by the same scorer, closed loop, greedy. The reading repeats after 2, 4, ... 512
such batches, on new sentences of the same stems in the same roles, so it gives the exposures a
fresh word takes as well as what one hearing does; zero exposures is the baseline.

- ``said``: the fresh word said in its place, closed loop.
- ``exact``: the whole sentence.
- ``rank``: teacher-forced, the fresh word's rank among all the motor neurons or output rows at its
  place, 1 the best: what the reading moved when a word is not yet said.

``mention`` asks a fresh word in the episode it is first heard in: a path ``X near wug end`` whose
fact the world says once. The word's own motor neuron or output row has had only the hearings of
the ladder, so at zero hearings a speaker says it only by a route that needs no lesson on that word.

A network hears the same batches: one Adam step per batch at the pass's rate (equal exposure), or
``steps`` per batch (equal compute, by the seconds a brain's batch takes).

    python benchmarks/vocabulary/hearing.py --speaker brain --words 200 --fresh 16 --founder 0 \\
        --checkpoints 2000 8000 --out DIR/hearing.jsonl
    python benchmarks/vocabulary/hearing.py --speaker transformer --parameters 224311 \\
        --words 200 --fresh 16 --founder 0 --checkpoints 2000 8000 --out DIR/hearing.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import controls
import life
import numpy as np
from language import SEEDS, Language, Row, by_size

LADDER = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
ROLES = ("agent", "patient")


def sentences(lang: Language, rng: np.random.Generator, plan: list[tuple[int, str]]) -> list[Row]:
    """A batch of ``agent verb patient end``, each row with its fresh stem in its role."""
    rows = []
    for stem, role in plan:
        other = (int(rng.integers(lang.trained)), int(rng.integers(2)))
        agent = (stem, 0) if role == "agent" else other
        patient = (stem, 0) if role == "patient" else other
        rows.append(lang.clause(rng, 0, agent=agent, patient=patient,
                                v=int(rng.integers(lang.verbs)), mods=[]))
    return rows


def mentioned(lang: Language, rng: np.random.Generator, plan: list[tuple[int, str]]) -> list[Row]:
    """A batch of one-hop paths whose chain ends in the row's fresh word, heard once in a fact."""
    rows = []
    entities = 2 * lang.trained
    for stem, _ in plan:
        draws = [lang._entity(int(e)) for e in rng.permutation(entities)[:5]]
        start, firsts, seconds = draws[0], draws[1:3], draws[3:5]
        word = lang.noun(stem, 0)
        facts = [(start, word), *zip(firsts, seconds, strict=True)]
        pre: list[tuple[int, ...]] = []
        for i in rng.permutation(len(facts)):
            x, y = facts[int(i)]
            pre += [lang.heard(x), lang.heard(lang.near), lang.heard(y), lang.heard(lang.end)]
        pre.append(lang.noun_moment(start % lang.nouns, start // lang.nouns, lang.c_from))
        rows.append(Row("path", pre, [start, lang.near, word, lang.end],
                        {"hops": 1, "chain": [start, word], "facts": len(facts)}))
    return rows


def place(role: str) -> int:
    """Where a fresh noun stands in ``agent verb patient end``, or in ``X near wug end``."""
    return 0 if role == "agent" else 2


def plans(lang: Language) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """The heard stems with the roles they are heard in, and the same stems in the other role."""
    stems = list(range(lang.trained, lang.nouns))[:16]
    heard = [(stem, ROLES[i % 2]) for i, stem in enumerate(stems)]
    return heard, [(stem, ROLES[1 - i % 2]) for i, stem in enumerate(stems)]


def score(lang: Language, rows: list[Row], said: list[list[int]],
          plan: list[tuple[int, str]]) -> dict[str, float]:
    right, exact = [], []
    for r, s, (stem, role) in zip(rows, said, plan, strict=True):
        at = place(role)
        right.append(len(s) > at and s[at] == lang.noun(stem, 0))
        exact.append(s == r.target)
    return {"said": float(np.mean(right)), "exact": float(np.mean(exact))}


class BrainSpeaker:
    def __init__(self, lang: Language, brain: Any) -> None:
        self.lang, self.brain = lang, copy.deepcopy(brain)

    def fork(self) -> BrainSpeaker:
        return BrainSpeaker(self.lang, self.brain)

    def hear(self, rows: list[Row]) -> None:
        life.speak(self.lang, self.brain, rows, teach=True)

    def say(self, rows: list[Row]) -> list[list[int]]:
        return life.speak(self.lang, copy.deepcopy(self.brain), rows, teach=False)

    def ranks(self, rows: list[Row], plan: list[tuple[int, str]]) -> list[int]:
        lang, brain = self.lang, copy.deepcopy(self.brain)
        brain.reset()
        for t in range(len(rows[0].pre)):
            brain.act(lang.dense([r.pre[t] for r in rows]), greedy=True)
        out = [0] * len(rows)
        motor = np.asarray(brain.connectome.populations["motor"])
        for j in range(len(rows[0].target)):
            x = (lang.dense([(lang.go,)] * len(rows)) if j == 0
                 else lang.dense([lang.spoken(r.target[j - 1]) for r in rows]))
            brain.act(x, greedy=True)
            activity = np.asarray(brain.basal_ganglia.state.activation)[:, motor]
            for i, (stem, role) in enumerate(plan):
                if place(role) == j:
                    word = lang.noun(stem, 0)
                    out[i] = int((activity[i] > activity[i, word]).sum()) + 1
        return out


class NetworkSpeaker:
    def __init__(self, lang: Language, learner: Any, steps: int = 1) -> None:
        self.lang, self.learner, self.steps = lang, copy.deepcopy(learner), steps

    def fork(self) -> NetworkSpeaker:
        return NetworkSpeaker(self.lang, self.learner, self.steps)

    def hear(self, rows: list[Row]) -> None:
        for _ in range(self.steps):
            self.learner.update(rows)

    def say(self, rows: list[Row]) -> list[list[int]]:
        self.learner.net.eval()
        said = self.learner.speak(rows)
        self.learner.net.train()
        return said

    def ranks(self, rows: list[Row], plan: list[tuple[int, str]]) -> list[int]:
        torch = self.learner.torch
        x, _ = controls.encode(self.lang, rows)
        self.learner.net.eval()
        with torch.no_grad():
            logits = self.learner.net(torch.from_numpy(x)).numpy()
        self.learner.net.train()
        out = []
        for i, (stem, role) in enumerate(plan):
            at = len(rows[i].pre) + place(role)
            word = self.lang.noun(stem, 0)
            out.append(int((logits[i, at] > logits[i, at, word]).sum()) + 1)
        return out


def ladder(lang: Language, speaker: Any, founder: int, tag: dict[str, Any],
           rounds: tuple[int, ...] = LADDER, test_batches: int = 2) -> dict[str, Any]:
    """The reading on a fork of ``speaker``, at no hearing and after each count of hearings."""
    heard, other = plans(lang)
    fork = speaker.fork()
    rng = np.random.default_rng(np.random.SeedSequence([founder, 266, 1]))
    test_rng = np.random.default_rng(np.random.SeedSequence([founder, 266, 2]))
    tests = {name: [sentences(lang, test_rng, plan) for _ in range(test_batches)]
             for name, plan in (("same", heard), ("other", other))}
    tests["mention"] = [mentioned(lang, test_rng, heard) for _ in range(test_batches)]
    told = [(stem, "mention") for stem, _ in heard]
    readings = []
    given, seconds = 0, 0.0

    def read() -> None:
        row: dict[str, Any] = {"hearings": given, "seconds": round(seconds, 3)}
        for name, plan in (("same", heard), ("other", other), ("mention", told)):
            said = [fork.say(b) for b in tests[name]]
            scores = [score(lang, b, s, plan) for b, s in zip(tests[name], said, strict=True)]
            ranks = [x for b in tests[name] for x in fork.ranks(b, plan)]
            row[name] = {"said": float(np.mean([s["said"] for s in scores])),
                         "exact": float(np.mean([s["exact"] for s in scores])),
                         "rank": float(np.median(ranks)), "rank_mean": float(np.mean(ranks))}
        readings.append(row)

    read()
    for count in rounds:
        begun = time.monotonic()
        while given < count:
            fork.hear(sentences(lang, rng, heard))
            given += 1
        seconds += time.monotonic() - begun
        read()
    return {**tag, "actions": lang.actions, "readings": readings}


def run_life(spec: dict[str, Any]) -> None:
    warnings.simplefilter("ignore")
    nouns, verbs = by_size(int(spec["words"]))
    lang = Language(nouns, verbs, int(spec.get("fresh", 16)))
    founder = int(spec["founder"])
    split = lang.split(founder)
    tests = lang.panels(founder)
    rng = np.random.default_rng(np.random.SeedSequence([founder, SEEDS["train"]]))
    sink = Path(spec["out"])
    sink.parent.mkdir(parents=True, exist_ok=True)
    kind = spec["speaker"]
    checkpoints = sorted(int(c) for c in spec["checkpoints"])
    panels = ("clause", "roles", "path")
    batch = 0
    if kind == "brain":
        brain = life.make_brain(lang, "Fd", founder, spec.get("settings"),
                                int(spec.get("module", 64)))
        parameters = int(brain.parameters())
        for checkpoint in checkpoints:
            while batch < checkpoint:
                life.speak(lang, brain, lang.train_batch(rng, split), teach=True)
                batch += 1
            tag = {"speaker": kind, "settings": spec.get("settings", {}), "founder": founder,
                   "words": lang.words, "fresh": lang.fresh, "batches": batch,
                   "parameters": parameters, "steps": 1, "tag": spec.get("tag", "")}
            base = {n: life.read_panel(lang, brain, tests[n], twins=n == "roles")["exact"]
                    for n in panels}
            row = ladder(lang, BrainSpeaker(lang, brain), founder, {**tag, "base": base})
            with sink.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        return
    import torch

    torch.set_num_threads(1)
    tied = kind in ("tied", "pretrained")
    model = "lstm" if kind == "lstm" else "transformer"
    width = controls.matched_width(model, lang.inputs, lang.actions, int(spec["parameters"]),
                                   tied)
    learner = controls.Learner(lang, model, width, founder, controls.RATES["online"], tied,
                               kind == "pretrained")
    for checkpoint in checkpoints:
        while batch < checkpoint:
            learner.update(lang.train_batch(rng, split))
            batch += 1
        base = {n: controls.reading(learner, tests, (n,))[n]["exact"] for n in panels}
        for steps in spec.get("steps", [1]):
            tag = {"speaker": kind, "width": width, "founder": founder, "words": lang.words,
                   "fresh": lang.fresh, "batches": batch, "parameters": controls.count(learner.net),
                   "steps": int(steps), "tag": spec.get("tag", ""), "base": base}
            row = ladder(lang, NetworkSpeaker(lang, learner, int(steps)), founder, tag)
            with sink.open("a") as handle:
                handle.write(json.dumps(row) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--speaker", default="brain",
                        choices=("brain", "transformer", "tied", "pretrained", "lstm"))
    parser.add_argument("--words", type=int, default=200)
    parser.add_argument("--fresh", type=int, default=16)
    parser.add_argument("--founder", type=int, default=0)
    parser.add_argument("--parameters", type=int, default=0)
    parser.add_argument("--checkpoints", nargs="*", type=int, default=[2000])
    parser.add_argument("--settings", type=json.loads, default={})
    parser.add_argument("--steps", nargs="*", type=int, default=[1])
    parser.add_argument("--module", type=int, default=64)
    parser.add_argument("--tag", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run_life({"speaker": args.speaker, "words": args.words, "fresh": args.fresh,
              "founder": args.founder, "parameters": args.parameters,
              "checkpoints": args.checkpoints, "settings": args.settings, "steps": args.steps,
              "module": args.module, "tag": args.tag, "out": str(args.out)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
