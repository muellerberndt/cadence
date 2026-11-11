"""Retrieved content becomes the answer: the copy pathway (``recall_copy``, ``recall_tokens``).

Two tasks on eight tokens, each observation a token one-hot with three flags (HEAR, ASK,
NOISE), eight actions, one per token; the brain declares that observation coordinate ``i`` and
action ``i`` are the same token where the arm says so (``recall_tokens=range(8)``).

- **Held-out tokens.** A token is heard (HEAR + token), two other tokens follow (NOISE + token),
  then a cue (HEAR + ASK) asks for the heard token back. The cue's key names the heard slot, so
  the read is right before any lesson; the question is whether the answer follows it. Tokens 0
  to 4 are taught as answers (a lesson at the cue, then the act); tokens 5 to 7 are heard and
  follow as distractors but are never taught as answers. Tests ask for every token.
- **Succession.** Five distinct tokens are heard (HEAR + token), then a cue (ASK + token) names
  one of them other than the last; the answer is the token that followed it, which the cue's
  read holds on its ``next`` port while its ``recall`` port holds the cue's own token. Only
  successors among tokens 0 to 4 are taught; tests ask for those (``succession``) and for
  successors 5 to 7 (``succession_held``).

The founder memory is compared with the copy pathway: the recipe (``recall_copy=2.0`` with the
token identities), a weaker identity (``1.0``), and the control without shared identities
(``recall_copy=0.5``, and ``1.2``, the recall projection's own scale, a plain direct projection),
drawn at random as every projection is. Each founder lives
once per arm, at ``teaching="normalized"`` with lessons where the answer differs
(``learning_surprise=True``), ``episodic=False``, on a brain of one module of 24 neurons; the
recipe and the founder also on a deep apical brain (``modules=(32, 24, 24)``, ``apical=0.25``
with the depth benchmark's learning settings). Every reading reports the read's own prediction
beside the answer: the token most driven on the port that holds the answer (``read_*``).

Component evidence for the pathway: a token the read delivers drives its own action, untaught
tokens included, and in-context succession is kept. It is not evidence for a composer, a
story or a language task; those keep their own measurements.
"""

from __future__ import annotations

import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
T = 8
HEAR, ASK, NOISE = T, T + 1, T + 2
INPUTS = T + 3
TAUGHT = 5  # tokens 0..4 are taught as answers; 5..7 never
LENGTH = 5  # tokens in a succession trial
LESSONS = {"heldout": 300, "succession": 600}
TESTS = 160
UNIFORM = 1 / T
FLAT = {"modules": (24,)}
DEEP = {
    "modules": (32, 24, 24), "apical": 0.25, "learning_normalize_relative": 0.1,
    "learning_nudged_steps": 32, "learning_nudged_fixed": True,
}
# arm -> the copy options
ARMS: dict[str, dict[str, Any]] = {
    "founder": {},
    "identity": {"recall_copy": 2.0, "recall_tokens": list(range(T))},
    "identity_weak": {"recall_copy": 1.0, "recall_tokens": list(range(T))},
    "random": {"recall_copy": 0.5},
    "random_strong": {"recall_copy": 1.2},
}
FLAT_ARMS = tuple(ARMS)
DEEP_ARMS = ("founder", "identity", "random")
FLAT_SEEDS, DEEP_SEEDS = (0, 1, 2, 3, 4, 5), (0, 1)


def sources() -> list[tuple[str, Path]]:
    """The public implementation bytes this result depends on."""
    names = (
        "benchmarks/attention/copy_tokens.py",
        "src/cadence/generic.py",
        "src/cadence/stream.py",
    )
    return [(name, ROOT / name) for name in names]


def obs(*on: int) -> np.ndarray:
    x = np.zeros((1, INPUTS))
    x[0, list(on)] = 1.0
    return x


def compose(seed: int, arm: str, depth: str = "flat") -> cd.Brain:
    options = dict(FLAT if depth == "flat" else DEEP)
    options.update(ARMS[arm])
    return cd.Brain.compose(
        INPUTS, T, seed=seed, episodic=False, recall=True, teaching="normalized",
        learning_surprise=True, **options,
    )


def answer(brain: cd.Brain, x: np.ndarray, port: str) -> tuple[int, int]:
    """The brain's greedy answer to ``x`` and the read's own prediction on ``port``."""
    drive = brain.stimulus(x)
    read = int(np.argmax(drive[0, brain.connectome.populations[port]][:T]))
    return int(np.asarray(brain.act(x, greedy=True)).reshape(-1)[0]), read


def heldout_episode(rng: np.random.Generator, token: int) -> list[np.ndarray]:
    others = rng.choice([u for u in range(T) if u != token], size=2, replace=False)
    return [obs(HEAR, token), obs(NOISE, int(others[0])), obs(NOISE, int(others[1]))]


def heldout(seed: int, arm: str, depth: str) -> dict[str, float]:
    brain = compose(seed, arm, depth)
    rng = np.random.default_rng([seed, 1])
    cue = obs(HEAR, ASK)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a lesson after an act on the same moment is refused
        for _ in range(LESSONS["heldout"]):
            token = int(rng.integers(TAUGHT))
            brain.reset()
            for x in heldout_episode(rng, token):
                brain.act(x, greedy=True)
            brain.lesson(cue, np.array([token]))
            brain.act(cue, greedy=True)
    test = np.random.default_rng([seed, 2])
    hits = {"taught": 0, "held": 0}
    reads = {"taught": 0, "held": 0}
    count = {"taught": 0, "held": 0}
    for i in range(TESTS):
        token = i % T
        brain.reset()
        for x in heldout_episode(test, token):
            brain.act(x, greedy=True)
        given, read = answer(brain, cue, "recall")
        group = "taught" if token < TAUGHT else "held"
        hits[group] += given == token
        reads[group] += read == token
        count[group] += 1
    return {
        "taught": hits["taught"] / count["taught"], "held": hits["held"] / count["held"],
        "read_taught": reads["taught"] / count["taught"],
        "read_held": reads["held"] / count["held"],
    }


def succession_trial(rng: np.random.Generator, answers: set[int]) -> tuple[np.ndarray, int, int]:
    while True:
        sequence = rng.permutation(T)[:LENGTH]
        places = [k for k in range(LENGTH - 1) if int(sequence[k + 1]) in answers]
        if places:
            k = int(rng.choice(places))
            return sequence, k, int(sequence[k + 1])


def succession(seed: int, arm: str, depth: str) -> dict[str, float]:
    brain = compose(seed, arm, depth)
    rng = np.random.default_rng([seed, 3])
    taught = set(range(TAUGHT))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(LESSONS["succession"]):
            sequence, k, label = succession_trial(rng, taught)
            brain.reset()
            for s in sequence:
                brain.act(obs(HEAR, int(s)), greedy=True)
            cue = obs(ASK, int(sequence[k]))
            brain.lesson(cue, np.array([label]))
            brain.act(cue, greedy=True)
    test = np.random.default_rng([seed, 4])
    out: dict[str, float] = {}
    for name, answers in (("succession", taught), ("succession_held", set(range(TAUGHT, T)))):
        hits = reads = 0
        for _ in range(TESTS):
            sequence, k, label = succession_trial(test, answers)
            brain.reset()
            for s in sequence:
                brain.act(obs(HEAR, int(s)), greedy=True)
            given, read = answer(brain, obs(ASK, int(sequence[k])), "next")
            hits += given == label
            reads += read == label
        out[name] = hits / TESTS
        out["read_" + name] = reads / TESTS
    recall = brain.recall
    if recall.copy and recall.tokens is not None:  # each port's identity gain after the life
        synapses, value, motor = recall.copy_synapses()
        pre = np.asarray(brain.connectome.pre)[synapses]
        weight = brain.brain.efficacy[synapses]
        same = value == motor
        following = np.isin(pre, recall.next_neurons)
        out["gain_recall"] = float(weight[same & ~following].mean())
        out["gain_next"] = float(weight[same & following].mean())
    return out


def run_one(job: tuple[str, str, int]) -> dict[str, Any]:
    depth, arm, seed = job
    return {"depth": depth, "arm": arm, "seed": seed,
            **heldout(seed, arm, depth), **succession(seed, arm, depth)}


def jobs() -> list[tuple[str, str, int]]:
    return [("flat", arm, seed) for arm in FLAT_ARMS for seed in FLAT_SEEDS] + [
        ("deep", arm, seed) for arm in DEEP_ARMS for seed in DEEP_SEEDS
    ]


def summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Means over founders per operating point and arm."""
    keys = ("taught", "held", "read_taught", "read_held", "succession", "succession_held",
            "read_succession", "read_succession_held")
    out: dict[str, Any] = {}
    for depth in ("flat", "deep"):
        for arm in ARMS:
            rows = [r for r in runs if r["depth"] == depth and r["arm"] == arm]
            if rows:
                out[f"{depth}/{arm}"] = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return out


def gates_of(runs: list[dict[str, Any]]) -> dict[str, bool]:
    """The declared gates, from the readings."""
    def rows(depth: str, arm: str) -> list[dict[str, Any]]:
        return sorted((r for r in runs if r["depth"] == depth and r["arm"] == arm),
                      key=lambda r: r["seed"])

    out = {}
    for depth in ("flat", "deep"):
        recipe, founder, control = rows(depth, "identity"), rows(depth, "founder"), rows(
            depth, "random")
        out[f"{depth}_untaught_tokens_produced"] = all(r["held"] >= 0.8 for r in recipe)
        out[f"{depth}_taught_answers_kept"] = all(
            r["taught"] >= f["taught"] - 0.05 for r, f in zip(recipe, founder, strict=True))
        out[f"{depth}_succession_kept"] = all(
            r["succession"] >= f["succession"] - 0.05 for r, f in zip(recipe, founder, strict=True))
        out[f"{depth}_control_without_identities_produces_none"] = all(
            r["held"] <= 0.1 for r in control)
        out[f"{depth}_founder_produces_no_untaught_token"] = all(r["held"] <= 0.1 for r in founder)
        out[f"{depth}_beats_uniform_random"] = all(r["held"] >= UNIFORM + 0.5 for r in recipe)
    return out


def check(body: dict[str, Any]) -> str | None:
    """The decision the readings support, or the reason they do not."""
    runs = body.get("runs", [])
    if sorted((r["depth"], r["arm"], r["seed"]) for r in runs) != sorted(jobs()):
        return "the runs are not the declared arms and founders"
    if body.get("summary") != summary(runs):
        return "the summary does not follow from the runs"
    gates = gates_of(runs)
    if body.get("gates") != gates or body.get("passed") != all(gates.values()):
        return "the gates do not follow from the readings"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.workers) as pool:
        runs = list(pool.map(run_one, jobs()))
    gates = gates_of(runs)
    result = {
        "schema": "cadence-copy-tokens/1",
        "scope": "a token the read delivers drives its own action; component evidence",
        "configuration": {
            "tokens": T, "taught": TAUGHT, "inputs": INPUTS, "length": LENGTH,
            "lessons": LESSONS, "tests": TESTS, "teaching": "normalized",
            "learning_surprise": True, "episodic": False, "arms": ARMS,
            "flat": {"modules": list(FLAT["modules"]), "seeds": list(FLAT_SEEDS)},
            "deep": {**{k: list(v) if isinstance(v, tuple) else v for k, v in DEEP.items()},
                     "seeds": list(DEEP_SEEDS), "arms": list(DEEP_ARMS)},
            "uniform_random": UNIFORM,
        },
        "runs": runs,
        "summary": summary(runs),
        "gates": gates,
        "passed": all(gates.values()),
    }
    problem = check(result)
    if problem is not None:
        raise AssertionError(problem)
    if args.out is not None:
        cd.Receipt.build("cadence-copy-tokens/1", result, sources()).write(args.out)
    print(json.dumps({"summary": result["summary"], "gates": gates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
