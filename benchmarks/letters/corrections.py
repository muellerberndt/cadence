"""Corrections of read speech, as on issue 303: a held-out speaker corrected, before and after.

For each saved life and each held-out speaker with at least ``least`` utterances, from the life
as saved: the speaker's utterances in the cache's order, the first half corrections and the
second half read; every other held-out utterance read as retention. A correction is the
utterance's forced alignment (``speech.align.forced_path``) through the brain as it stands,
taught at ``rate`` of the life's lesson rates:

- ``plain``: the correction alone (``Rehearsal(rows=0)``);
- ``rehearsal``: beside three training utterances at weight .25 (``Rehearsal()``), its bank
  holding ``bank`` training utterances aligned through the saved life.

Readings are characters (one less the character edits per reference character) of the letter
decoder's text. The lives learned labels from the forced alignment, so corrections are labels.

    PYTHONPATH=src python benchmarks/letters/corrections.py --cache CACHE \\
        --brains runs/speech-304500-b198.npz ... --out results/corrections.jsonl --workers 12
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

MODES = ("plain", "rehearsal")


def edits(reference: str, hypothesis: str) -> int:
    row = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        previous, row[0] = row[0], i
        for j, h in enumerate(hypothesis, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (r != h))
    return row[-1]


def characters(brain: Any, held: list[tuple[np.ndarray, np.ndarray]], batch: int,
               folder: Path) -> float:
    """The letter decoder's characters on ``held`` from a frozen copy of ``brain``."""
    from cadence import Brain
    from cadence.speech.align import log_beliefs
    from cadence.speech.decode import Alphabet, collapse

    alphabet = Alphabet()
    brain.reset()
    copy = Brain.load(brain.save(folder / "copy.npz"))
    errors = total = 0
    for k in range(0, len(held), batch):
        part = held[k : k + batch]
        for (_, text), logp in zip(part, log_beliefs(copy, [x for x, _ in part]), strict=True):
            reference = alphabet.decode(text)
            errors += edits(reference, alphabet.decode(collapse(np.argmax(logp, axis=1))))
            total += len(reference)
    return 1.0 - errors / max(total, 1)


def at_rate(brain: Any, rate: float) -> None:
    """Every rate of the brain's lesson learner at ``rate`` of its own."""
    config = brain.learner.config
    eta_bias = config.eta_bias if config.eta_bias is not None else config.eta / 10
    brain.retune(learning=dataclasses.replace(config, eta=config.eta * rate,
                                              eta_bias=eta_bias * rate))


def episode(brain_path: str, own: list[tuple[np.ndarray, np.ndarray]],
            others: list[tuple[np.ndarray, np.ndarray]],
            bank_rows: list[tuple[np.ndarray, np.ndarray]], mode: str, rate: float,
            batch: int, seed: int) -> dict[str, Any]:
    """One speaker corrected in one mode from the saved life: readings before and after."""
    from cadence import Brain
    from cadence.speech.align import forced_path, log_beliefs
    from cadence.speech.correct import Rehearsal

    folder = Path(tempfile.mkdtemp(prefix="corrections-"))
    started = time.process_time()
    brain = Brain.load(brain_path)
    corrections, readings = own[: len(own) // 2], own[len(own) // 2 :]
    before = {"speaker": characters(brain, readings, batch, folder),
              "others": characters(brain, others, batch, folder)}
    if mode == "rehearsal":
        bank = Rehearsal(rows=3, weight=0.25, seed=seed)
        brain.reset()
        copy = Brain.load(brain.save(folder / "copy.npz"))
        for k in range(0, len(bank_rows), batch):
            part = bank_rows[k : k + batch]
            for (x, text), logp in zip(part, log_beliefs(copy, [x for x, _ in part]),
                                       strict=True):
                bank.remember(x, forced_path(logp, text))
    else:
        bank = Rehearsal(rows=0)
    at_rate(brain, rate)
    taught = time.process_time()
    for x, text in corrections:
        brain.reset()
        copy = Brain.load(brain.save(folder / "copy.npz"))
        path = forced_path(log_beliefs(copy, [x])[0], text)
        bank.correct(brain, x, path)
    taught = time.process_time() - taught
    after = {"speaker": characters(brain, readings, batch, folder),
             "others": characters(brain, others, batch, folder)}
    return {"brain": brain_path, "mode": mode, "rate": rate, "corrections": len(corrections),
            "read": len(readings), "others": len(others), "before": before, "after": after,
            "seconds_per_correction": taught / max(len(corrections), 1),
            "process_seconds": time.process_time() - started}


def speakers(identities: list[str], least: int) -> dict[str, list[int]]:
    """Held-out utterances by speaker, the speakers with at least ``least`` of them."""
    by: dict[str, list[int]] = {}
    for k, identity in enumerate(identities):
        by.setdefault(identity.split("-")[0], []).append(k)
    return {s: rows for s, rows in sorted(by.items()) if len(rows) >= least}


_CACHE: dict[str, Any] = {}


def _job(spec: dict[str, Any]) -> dict[str, Any]:
    import importlib.util
    import sys

    here = Path(__file__).resolve().parent
    if "letters_speech" not in sys.modules:
        found = importlib.util.spec_from_file_location("letters_speech", here / "speech.py")
        assert found is not None and found.loader is not None
        module = importlib.util.module_from_spec(found)
        sys.modules["letters_speech"] = module
        found.loader.exec_module(module)
    sp = sys.modules["letters_speech"]
    cache = Path(spec["cache"])
    if "held" not in _CACHE:
        _CACHE["held"] = sp.load(cache, "held")
        train = sp.load(cache, "train")
        rng = np.random.default_rng(spec["seed"])
        _CACHE["bank"] = [train[int(i)] for i in rng.choice(len(train), spec["bank"],
                                                             replace=False)]
    held, rows = _CACHE["held"], spec["rows"]
    own = [held[k] for k in rows]
    others = [held[k] for k in range(len(held)) if k not in set(rows)]
    try:
        out = episode(spec["brain"], own, others, _CACHE["bank"], spec["mode"], spec["rate"],
                      spec["batch"], spec["seed"])
        return {"speaker": spec["speaker"], **out}
    except Exception as exc:  # a failed episode is a reading too
        return {"spec": {k: v for k, v in spec.items() if k != "rows"}, "error": repr(exc)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, help="a prepare-listed cache (speech.py)")
    parser.add_argument("--brains", nargs="+", required=True)
    parser.add_argument("--least", type=int, default=8, help="a speaker's utterances, at least")
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--bank", type=int, default=256, help="training utterances in the bank")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    with np.load(Path(args.cache) / "held.npz") as data:
        identities = [str(i) for i in data["identities"]]
    chosen = speakers(identities, args.least)
    specs = [{"cache": args.cache, "brain": brain, "speaker": s, "rows": rows, "mode": mode,
              "rate": args.rate, "bank": args.bank, "batch": args.batch, "seed": args.seed}
             for brain in args.brains for s, rows in chosen.items() for mode in MODES]
    print(json.dumps({"speakers": {s: len(r) for s, r in chosen.items()},
                      "episodes": len(specs)}), flush=True)
    with open(args.out, "a") as handle:
        if args.workers > 1:
            with get_context("spawn").Pool(args.workers) as pool:
                for result in pool.imap_unordered(_job, specs):
                    handle.write(json.dumps(result) + "\n")
                    handle.flush()
        else:
            for spec in specs:
                handle.write(json.dumps(_job(spec)) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()
