"""A minimal contextual choice for the public single-head attention recipe.

Every decision sees the same literal ``ASK`` observation. Two immutable Recall episodes
contain ``context-0, ASK, value-0`` and ``context-1, ASK, value-1``. The learned-code window
therefore has to select the ASK record belonging to the earlier context; its successor is the
action-relevant evidence. The assay cold-clears ordinary neural and trace state before ASK,
and it compares the intact reader with present-only, memory-off and slot-erasure controls.

This is bounded component evidence. It does not establish continuing-world benefit, a
within-settle query-revision advantage, multiple heads or a reason to change the default.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INPUTS = 12
ACTIONS = 2
CODE = 8
CONTEXTS = (0, 1)
RANDOM_BASELINE = 1.0 / ACTIONS


def sources() -> list[tuple[str, Path]]:
    """The public implementation bytes this result depends on."""
    names = (
        "benchmarks/attention/contextual_choice.py",
        "src/cadence/generic.py",
        "src/cadence/stream.py",
        "src/cadence/code.py",
    )
    return [(name, ROOT / name) for name in names]


def observation(kind: str, value: int | None = None) -> np.ndarray:
    """One literal event; ASK is byte-identical in every trial."""
    x = np.zeros((1, INPUTS), np.float64)
    x[0, 11] = 1.0
    if kind == "context":
        if value not in CONTEXTS:
            raise ValueError("context value must be zero or one")
        x[0, int(value)] = 1.0
    elif kind == "ask":
        if value is not None:
            raise ValueError("ASK has no value")
        x[0, 2] = 1.0
    elif kind == "value":
        if value not in CONTEXTS:
            raise ValueError("value must be zero or one")
        x[0, 3 + int(value)] = 1.0
    elif kind == "neutral":
        if value is not None:
            raise ValueError("neutral has no value")
        x[0, 5] = 1.0
    else:
        raise ValueError("unknown observation kind")
    return x


class ContextChoiceBrain(cd.Brain):
    """The public brain with controlled read interventions for this assay only."""

    attention_memory_enabled = True
    attention_context_enabled = True

    def stimulus(self, observations: Any, *, memory: bool = True) -> np.ndarray:
        recall = self.recall
        enabled = memory and self.attention_memory_enabled
        if recall is None or self.attention_context_enabled or not enabled:
            return super().stimulus(observations, memory=enabled)
        past = recall.past
        recall.past = np.zeros_like(past)
        try:
            return super().stimulus(observations, memory=True)
        finally:
            recall.past = past


def compose(seed: int, *, solve: int = 1024, module: int = 24) -> ContextChoiceBrain:
    """One opt-in attention brain with no hippocampus or episodic action vote; the protocol
    predates the recall position term (0.89.0), which it leaves out."""
    brain = ContextChoiceBrain.compose(
        INPUTS,
        ACTIONS,
        learned_code=CODE,
        attention=True,
        recall_position=0,
        modules=(module,),
        episodic=False,
        memory_choice=False,
        arousal=None,
        learning_surprise=False,
        learning_qualified=True,
        learning_damping=0,
        learning_free_steps=solve,
        learning_nudged_steps=solve,
        learning_tolerance=0.003,
        seed=seed,
    )
    if brain.hippocampus is not None or brain._episodic is not None:  # noqa: SLF001
        raise AssertionError("the contextual assay must have no episodic memory")
    return brain


def cold_policy(brain: ContextChoiceBrain) -> None:
    """Clear every contextual route except Recall's ordered learned-code window."""
    recall = brain.recall
    if recall is None or recall.window != 2 or not len(recall.past):
        raise ValueError("live Recall streams with a two-event window are required")
    kept_past = recall.past.copy()
    batch = len(kept_past)
    brain.basal_ganglia.reset()
    if brain.working_memory is not None:
        brain.working_memory.reset(batch)
    if brain.history is not None:
        brain.history.reset(batch)
    if brain.efference is not None:
        brain.efference.reset(batch)
    if brain.learned_code is not None:
        brain.learned_code.last = None
    recall.previous = np.zeros_like(recall.previous)
    recall.past = kept_past
    recall.last = None
    brain._coded_moment = None  # noqa: SLF001 - explicit component boundary
    brain._code_settlement = None  # noqa: SLF001
    brain._prepared = None  # noqa: SLF001
    brain._awaiting = None  # noqa: SLF001


def advance_context(brain: ContextChoiceBrain, context: int) -> None:
    """Present a real cue, then remove every route except its learned code in Recall."""
    brain.act(observation("context", context), greedy=True)
    cold_policy(brain)


def warm_codes(brain: ContextChoiceBrain, rounds: int = 16) -> None:
    """Let the local code learner see every literal event, then freeze its meaning map."""
    if rounds < 1:
        raise ValueError("at least one code-warmup round is required")
    events = (
        observation("context", 0),
        observation("context", 1),
        observation("ask"),
        observation("value", 0),
        observation("value", 1),
        observation("neutral"),
    )
    for _ in range(rounds):
        for event in events:
            brain.reset(keep=())
            brain.act(event, greedy=True)
    assert brain.learned_code is not None
    brain.learned_code.learning = False
    brain.reset(keep=())


def build_bank(brain: ContextChoiceBrain) -> dict[str, Any]:
    """Write exactly two three-event episodes with stable learned-code keys."""
    recall = brain.recall
    if recall is None or brain.learned_code is None or brain.learned_code.learning:
        raise ValueError("a frozen learned code and Recall are required")
    recall._private = False  # noqa: SLF001 - explicit immutable-bank construction
    brain.reset(keep=())
    for context in CONTEXTS:
        if recall.slots:
            brain.reset(keep=("slots",))
        brain.act(observation("context", context), greedy=True)
        cold_policy(brain)
        brain.act(observation("ask"), greedy=True)
        brain.act(observation("value", context), greedy=True)
    brain.reset(keep=("slots",))
    recall._private = True  # noqa: SLF001
    if recall.slots != 6 or not recall.held.all():
        raise AssertionError("the immutable bank must contain exactly six witnessed slots")
    evidence = {0: np.array([1], np.int64), 1: np.array([4], np.int64)}
    if (
        not np.array_equal(recall.values[0, 1], recall.values[0, 4])
        or not recall.nexts[0, 1].any()
        or not recall.nexts[0, 4].any()
        or np.array_equal(recall.nexts[0, 1], recall.nexts[0, 4])
    ):
        raise AssertionError("the two ASK records need identical values and distinct successors")
    return {
        "evidence_slots": evidence,
        "keys": recall.keys.copy(),
        "values": recall.values.copy(),
        "nexts": recall.nexts.copy(),
        "held": recall.held.copy(),
    }


def train_policy(brain: ContextChoiceBrain, rounds: int = 128) -> dict[str, int]:
    """Teach both opposing contextual answers in every local update."""
    if rounds < 1:
        raise ValueError("at least one policy round is required")
    recall = brain.recall
    if recall is None or not recall._private:  # noqa: SLF001
        raise ValueError("policy acquisition requires the immutable private bank")
    if recall.streams != 1:
        raise ValueError("policy acquisition starts from the one-stream canonical bank")
    recall.keys = np.repeat(recall.keys, 2, axis=0)
    recall.values = np.repeat(recall.values, 2, axis=0)
    recall.nexts = np.repeat(recall.nexts, 2, axis=0)
    recall.held = np.repeat(recall.held, 2, axis=0)
    recall.previous = np.zeros((2, recall.previous.shape[1]), np.float64)
    recall.past = np.zeros((2, recall.window - 1, recall.key_width), np.float64)
    recall.ended = np.zeros(2, bool)
    contexts = np.concatenate([observation("context", context) for context in CONTEXTS], axis=0)
    asks = np.repeat(observation("ask"), 2, axis=0)
    labels = np.asarray(CONTEXTS, np.int64)
    accepted = refused = sweeps = 0
    for _ in range(rounds):
        brain.reset(keep=("slots",))
        recall._private = True  # noqa: SLF001
        brain.act(contexts, greedy=True)
        cold_policy(brain)
        try:
            report = brain.lesson(asks, labels)
        except RuntimeError:
            refused += 1
            continue
        accepted += 1
        sweeps += sum(
            int(report.get(name, 0.0)) for name in ("free_steps", "nudged_steps", "opposite_steps")
        )
    brain.reset(keep=("slots",))
    recall.keep(np.array([0], np.int64))
    recall._private = True  # noqa: SLF001
    return {"rounds": rounds, "accepted": accepted, "refused": refused, "sweeps": sweeps}


def _masks(brain: ContextChoiceBrain, bank: dict[str, Any]) -> dict[int, np.ndarray]:
    assert brain.recall is not None
    out = {}
    for context in CONTEXTS:
        mask = np.zeros(brain.recall.slots, bool)
        mask[np.asarray(bank["evidence_slots"][context], dtype=np.int64)] = True
        out[context] = mask
    return out


def evaluate(
    acquired: ContextChoiceBrain,
    bank: dict[str, Any],
    *,
    trials: int,
    mode: str = "contextual",
    erase: str | None = None,
) -> dict[str, Any]:
    """Read one acquired brain under a declared causal intervention."""
    if trials < 2 or trials % 2:
        raise ValueError("evaluation trials must be even and at least two")
    if mode not in ("contextual", "present_only", "memory_off"):
        raise ValueError("unknown evaluation mode")
    if erase not in (None, "relevant", "other", "all"):
        raise ValueError("unknown erasure")
    brain = deepcopy(acquired)
    recall = brain.recall
    assert recall is not None
    recall._private = True  # noqa: SLF001
    brain.attention_context_enabled = mode != "present_only"
    brain.attention_memory_enabled = mode != "memory_off"
    masks = _masks(brain, bank)
    original_held = recall.held.copy()
    answers: list[int] = []
    masses: list[tuple[float, float]] = []
    codes: list[np.ndarray] = []
    refusals = 0
    for index in range(trials):
        context = index % 2
        recall.held = original_held.copy()
        if erase == "all":
            recall.held[:] = False
        elif erase == "relevant":
            recall.held[:, masks[context]] = False
        elif erase == "other":
            recall.held[:, masks[1 - context]] = False
        brain.reset(keep=("slots",))
        recall._private = True  # noqa: SLF001
        advance_context(brain, context)
        codes.append(brain.observation_code(observation("ask"))[0].copy())
        try:
            answers.append(int(brain.act(observation("ask"), greedy=True)[0]))
        except RuntimeError:
            refusals += 1
            answers.append(-1)
        weights = recall.attended
        if weights is None:
            masses.append((0.0, 0.0))
        else:
            masses.append(
                (
                    float(weights[0, masks[context]].sum()),
                    float(weights[0, masks[1 - context]].sum()),
                )
            )
    correct = [int(answer == index % 2) for index, answer in enumerate(answers)]
    encoded = np.asarray(codes)
    return {
        "accuracy": float(np.mean(correct)),
        "per_context_accuracy": {
            str(context): float(np.mean(correct[context::2])) for context in CONTEXTS
        },
        "refusals": refusals,
        "mean_correct_evidence_mass": float(np.mean([mass[0] for mass in masses])),
        "mean_wrong_evidence_mass": float(np.mean([mass[1] for mass in masses])),
        "identical_final_code": bool(np.array_equal(encoded[0::2], encoded[1::2])),
    }


def run_founder(seed: int, *, trials: int = 32) -> dict[str, Any]:
    """Acquire and evaluate the frozen single-head gate for one founder."""
    brain = compose(seed)
    warm_codes(brain)
    bank = build_bank(brain)
    training = train_policy(brain)
    before = tuple(
        value.copy() for value in (brain.recall.keys, brain.recall.values, brain.recall.nexts)
    )
    panels = {
        "contextual": evaluate(brain, bank, trials=trials),
        "present_only": evaluate(brain, bank, trials=trials, mode="present_only"),
        "memory_off": evaluate(brain, bank, trials=trials, mode="memory_off"),
        "relevant_erased": evaluate(brain, bank, trials=trials, erase="relevant"),
        "other_erased": evaluate(brain, bank, trials=trials, erase="other"),
        "all_erased": evaluate(brain, bank, trials=trials, erase="all"),
    }
    unchanged = all(
        np.array_equal(left, right)
        for left, right in zip(
            before, (brain.recall.keys, brain.recall.values, brain.recall.nexts), strict=True
        )
    )
    intact = panels["contextual"]
    gates = {
        "training_completed": training["accepted"] == training["rounds"]
        and training["refused"] == 0,
        "accuracy": intact["accuracy"] >= 0.95,
        "each_context": min(intact["per_context_accuracy"].values()) >= 0.95,
        "no_refusals": intact["refusals"] == 0,
        "identical_final_code": intact["identical_final_code"],
        "selected_relevant_evidence": intact["mean_correct_evidence_mass"]
        >= intact["mean_wrong_evidence_mass"] + 0.25,
        "beats_present_only": intact["accuracy"] - panels["present_only"]["accuracy"] >= 0.2,
        "beats_memory_off": intact["accuracy"] - panels["memory_off"]["accuracy"] >= 0.2,
        "beats_uniform_random": intact["accuracy"] - RANDOM_BASELINE >= 0.2,
        "relevant_erasure_matters": intact["accuracy"] - panels["relevant_erased"]["accuracy"]
        >= 0.2,
        "all_erasure_matters": intact["accuracy"] - panels["all_erased"]["accuracy"] >= 0.2,
        "irrelevant_erasure_stable": abs(intact["accuracy"] - panels["other_erased"]["accuracy"])
        <= 0.05,
        "bank_unchanged": unchanged,
        "no_episodic_memory": brain.hippocampus is None and brain._episodic is None,
    }
    return {
        "seed": seed,
        "training": training,
        "panels": panels,
        "uniform_random_baseline_accuracy": RANDOM_BASELINE,
        "gates": gates,
        "passed": all(gates.values()),
    }


def check(body: dict[str, Any]) -> str | None:
    """Recompute the retained pass decision from its recorded behavioral readings."""
    try:
        founders = body["founders"]
        if body["schema"] != "cadence-contextual-attention/1" or len(founders) < 2:
            return "wrong schema or too few founders"
        for founder in founders:
            panels = founder["panels"]
            intact = panels["contextual"]
            expected = {
                "training_completed": founder["training"]["accepted"]
                == founder["training"]["rounds"]
                and founder["training"]["refused"] == 0,
                "accuracy": intact["accuracy"] >= 0.95,
                "each_context": min(intact["per_context_accuracy"].values()) >= 0.95,
                "no_refusals": intact["refusals"] == 0,
                "identical_final_code": intact["identical_final_code"],
                "selected_relevant_evidence": intact["mean_correct_evidence_mass"]
                >= intact["mean_wrong_evidence_mass"] + 0.25,
                "beats_present_only": intact["accuracy"] - panels["present_only"]["accuracy"]
                >= 0.2,
                "beats_memory_off": intact["accuracy"] - panels["memory_off"]["accuracy"] >= 0.2,
                "beats_uniform_random": intact["accuracy"]
                - founder["uniform_random_baseline_accuracy"]
                >= 0.2,
                "relevant_erasure_matters": intact["accuracy"]
                - panels["relevant_erased"]["accuracy"]
                >= 0.2,
                "all_erasure_matters": intact["accuracy"] - panels["all_erased"]["accuracy"] >= 0.2,
                "irrelevant_erasure_stable": abs(
                    intact["accuracy"] - panels["other_erased"]["accuracy"]
                )
                <= 0.05,
                "bank_unchanged": founder["gates"]["bank_unchanged"],
                "no_episodic_memory": founder["gates"]["no_episodic_memory"],
            }
            if founder["gates"] != expected or founder["passed"] != all(expected.values()):
                return f"founder {founder['seed']} decisions do not match its readings"
        if body["passed"] != all(founder["passed"] for founder in founders):
            return "aggregate decision does not match founder decisions"
    except (KeyError, TypeError, ValueError):
        return "incomplete or invalid contextual-attention readings"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    founders = [run_founder(seed, trials=args.trials) for seed in args.seeds]
    result = {
        "schema": "cadence-contextual-attention/1",
        "scope": "single-head learned-code contextual choice; bounded component evidence",
        "configuration": {
            "inputs": INPUTS,
            "actions": ACTIONS,
            "learned_code": CODE,
            "modules": [24],
            "code_rounds": 16,
            "policy_rounds": 128,
            "recall_window": 2,
            "recall_temperature": 0.1,
            "recall_recency": 0.0,
            "recall_address": 0.0,
            "recall_position": 0,
        },
        "founders": founders,
        "passed": len(founders) >= 2 and all(founder["passed"] for founder in founders),
    }
    problem = check(result)
    if problem is not None:
        raise AssertionError(problem)
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        cd.Receipt.build("cadence-contextual-attention/1", result, sources()).write(args.out)
    print(encoded, end="")


if __name__ == "__main__":
    main()
