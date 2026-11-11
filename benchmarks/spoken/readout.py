"""Where a saved brain loses the word between its association and what it says (cadence#303).

At the word steps of held-out phrases, each word read four ways:

- ``said``: what the brain says in the stream, as a reading hears it (``chamber.read``);
- ``said, pools to motor cut`` and ``said, association to motor cut``: the same with the efficacies
  of one pathway into the motor region set to zero (both directions of the association's pair);
- ``association``: a ridge readout of the association's settled state from rest, without
  memories, fitted on training word steps (``chamber.association``);
- ``drive``: from the same settled state, the motor region's input through each pathway alone,
  and through both: the action whose input is largest.

Per word: how often each reading names it, and the confusions (which word is said for which).

    PYTHONPATH=src python benchmarks/spoken/readout.py --data /data/phrases \\
        --brain runs/learners/fields_wide-0.npz --soft 0.5 --out results/readout-0.json
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    key = f"spoken_{name}"  # a name of this chamber's own: other benchmarks have their modules
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


ch = _load("chamber")
ph = _load("phrases")

WORDS = ph.WORDS


def pathway(brain: Any, pre: str, post: str) -> np.ndarray:
    """The synapses from population ``pre`` to population ``post``."""
    c = brain.connectome
    return np.asarray(
        np.isin(c.pre, np.asarray(c.populations[pre]))
        & np.isin(c.post, np.asarray(c.populations[post]))
    )


def cut(brain: Any, *pairs: tuple[str, str]) -> Any:
    """A copy with the efficacies of the named pathways at zero."""
    out = copy.deepcopy(brain)
    efficacy = np.array(out.brain.efficacy, copy=True)
    for pre, post in pairs:
        efficacy[pathway(out, pre, post)] = 0.0
    out.brain.efficacy = efficacy
    return out


def said_at_words(
    brain: Any, phrases: Any, rows: np.ndarray, hearing: Any
) -> tuple[np.ndarray, np.ndarray]:
    """What the brain says at each word step, and the word due there."""
    learner = ch.BrainLearner(brain, "fields").reader()
    said, due = [], []
    for chunk in ch.groups(phrases, rows, ch.BATCH):
        actions = learner.hear(ch.stream(phrases, chunk, hearing))
        targets = phrases.target[chunk, : actions.shape[1]]
        mask = targets < ph.SILENCE
        said.append(actions[mask])
        due.append(targets[mask])
    return np.concatenate(said), np.concatenate(due)


def drives(brain: Any, x: np.ndarray) -> dict[str, np.ndarray]:
    """The motor region's input through each pathway from the settled state from rest."""
    c = brain.connectome
    motor = np.asarray(c.populations["motor"])
    index = np.full(c.n, -1)
    index[motor] = np.arange(len(motor))
    efficacy = np.asarray(brain.brain.efficacy)
    out: dict[str, list[np.ndarray]] = {"association": [], "pools": [], "both": []}
    paths = {
        "association": pathway(brain, "association", "motor"),
        "pools": pathway(brain, "sensory", "motor"),
    }
    for start in range(0, len(x), 64):
        state = brain._qualified(
            brain.stimulus(x[start : start + 64], memory=False), operation="predict"
        )
        a = np.atleast_2d(np.asarray(state.activation))
        total = np.zeros((len(a), len(motor)))
        for name, mask in paths.items():
            drive = np.zeros((len(a), len(motor)))
            pre, post = c.pre[mask], index[c.post[mask]]
            np.add.at(drive.T, post, (a[:, pre] * efficacy[mask]).T)
            out[name].append(drive)
            total += drive
        out["both"].append(total)
    return {name: np.concatenate(parts) for name, parts in out.items()}


def per_word(said: np.ndarray, due: np.ndarray) -> dict[str, Any]:
    right = {WORDS[w]: float(np.mean(said[due == w] == w)) for w in np.unique(due)}
    confusions = Counter(
        (WORDS[int(d)], ph.LABELS[int(s)] if s >= 0 else "refused")
        for s, d in zip(said, due, strict=True)
        if s != d
    )
    return {
        "right": float(np.mean(said == due)),
        "silence": float(np.mean(said == ph.SILENCE)),
        "per_word": right,
        "confusions": [[d, s, n] for (d, s), n in confusions.most_common(25)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--brain", required=True)
    parser.add_argument("--phrases", type=int, default=600)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from cadence import Brain

    sp = _load("spoken")
    train = sp.phrases(args.data, "base_train")
    test = sp.phrases(args.data, "base_test")
    hearing = ch.Hearing.measured(train)
    brain = Brain.load(args.brain)
    rows = sp.sample(len(test), args.phrases, "readout")
    report: dict[str, Any] = {"brain": args.brain, "parameters": int(brain.parameters())}
    views = {
        "said": brain,
        "said, pools to motor cut": cut(brain, ("sensory", "motor")),
        "said, association to motor cut": cut(
            brain, ("association", "motor"), ("motor", "association")
        ),
    }
    for name, view in views.items():
        report[name] = per_word(*said_at_words(view, test, rows, hearing))
    fit_x, fit_y = ch.word_steps(train, sp.sample(len(train), 3000, "probe"), hearing, 3000)
    x, y = ch.word_steps(test, rows, hearing, 10**6)
    assoc_fit, assoc = ch.association(brain, fit_x), ch.association(brain, x)
    probe = ch.ridge(assoc_fit, fit_y, {"test": (assoc, y)})
    mean = assoc_fit.mean(axis=0)
    scale = float(np.sqrt(np.mean((assoc_fit - mean) ** 2))) or 1.0
    z = (assoc_fit - mean) / scale
    target = np.eye(ph.SILENCE)[fit_y] - 1.0 / ph.SILENCE
    w = np.linalg.solve(z.T @ z + probe["penalty"] * len(z) * np.eye(z.shape[1]), z.T @ target)
    report["association"] = per_word(np.argmax(((assoc - mean) / scale) @ w, axis=1), y)
    report["association"]["probe"] = probe
    for name, drive in drives(brain, x).items():
        report[f"drive, {name}"] = per_word(np.argmax(drive, axis=1), y)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(
        json.dumps(
            {
                k: (v["right"], v["silence"])
                for k, v in report.items()
                if isinstance(v, dict) and "right" in v
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
