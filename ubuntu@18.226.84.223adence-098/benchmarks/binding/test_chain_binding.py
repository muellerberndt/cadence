"""Contracts of the chained-recall instrument."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from cadence import brain as cadence_brain

HERE = Path(__file__).resolve().parent


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cb = _load("chain_binding")


def test_holdout_is_one_target_per_source_without_self_links() -> None:
    for seed in range(6):
        held = cb.holdout(seed)
        assert (held.sum(axis=1) == 1).all() and (held.sum(axis=0) == 1).all()
        assert not np.diag(held).any()


def test_training_avoids_held_links_and_held_panels_use_them() -> None:
    held = cb.holdout(2)
    rng = np.random.default_rng(0)
    trials = cb.draw(rng, 300, held)
    t = trials["tokens"]
    assert all(len(set(row)) == 5 for row in t)
    assert not (held[t[:, 0], t[:, 1]] | held[t[:, 1], t[:, 2]] | held[t[:, 3], t[:, 4]]).any()
    assert set(np.unique(trials["hops"])) == {1, 2}
    two = cb.draw(rng, 200, held, hops=2, use_held=True)
    t = two["tokens"]
    assert (held[t[:, 0], t[:, 1]] | held[t[:, 1], t[:, 2]]).all()
    assert not held[t[:, 3], t[:, 4]].any()
    assert (cb.answers(two) == t[:, 2]).all()


def test_the_question_shares_a_token_with_the_first_link_only() -> None:
    held = cb.holdout(1)
    trials = cb.draw(np.random.default_rng(3), 32, held)
    seq = cb.events(trials)
    question = seq[3 + cb.NEUTRAL]
    t = trials["tokens"]
    rows = np.arange(32)
    first = cb.link_event(t[:, 0], t[:, 1])
    second = cb.link_event(t[:, 1], t[:, 2])
    assert ((question > 0) & (first > 0)).sum(axis=1).min() == 1
    assert not ((question > 0) & (second > 0)).any()
    assert (seq[-1][:, cb.coordinate("think")] == 1).all()
    links = np.stack(seq[:3])  # (3, n, inputs): every link appears once
    for link in (first, second, cb.link_event(t[:, 3], t[:, 4])):
        assert (np.all(links == link[None], axis=2).sum(axis=0) == 1).all()
    asked = np.where(trials["hops"] == 1, cb.coordinate("ask1"), cb.coordinate("ask2"))
    assert (question[rows, asked] == 1).all()


def test_the_fixed_read_recalls_the_first_link_and_never_the_second() -> None:
    brain = cb.build("content", 0, "act")
    held = cb.holdout(0)
    trials = cb.draw(np.random.default_rng(4), 16, held, hops=2)
    seq = cb.events(trials)
    brain.reset()
    for x in seq[: 3 + cb.NEUTRAL]:
        brain.act(x, greedy=True)
    recalled = brain.content.recall(seq[3 + cb.NEUTRAL])
    t = trials["tokens"]
    rows = np.arange(16)
    assert (recalled[rows, cb.coordinate("to") + t[:, 1]] > 0).all()
    assert not recalled[rows, cb.coordinate("to") + t[:, 2]].any()


def test_learners_build_and_a_round_trains() -> None:
    held = cb.holdout(5)
    rng = np.random.default_rng(5)
    # the learned read enters the settle through the nudge's drive, which the compiled dense
    # kernel never calls: its coupling refuses the kernel (run with CADENCE_FUSED=0)
    learners = ("graph", "content") if cadence_brain._FUSED else ("graph", "content", "learned")
    for learner in learners:
        brain = cb.build(learner, 5, "act")
        work = cb.tb.new_work()
        assert cb.train_round(brain, cb.draw(rng, cb.BATCH, held), work)
        assert work["lessons"] == 1 and work["lesson_refusals"] == 0
    model = cb.Supervised(5)
    trials = cb.draw(rng, cb.BATCH, held)
    model.lesson(cb.events(trials), cb.answers(trials))
    assert model.answer(cb.events(trials)).shape == (cb.BATCH,)
