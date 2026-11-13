"""What the read-speech brain is given, read by gradient learners: where a plateau sits.

A brain of the read-speech chamber (``speech.py``) hears each 40 ms step through a fixed chain:
the window of log-mel frames, the learned fields' pooled code of it, one association over that
code and the code of the step before, and a teacher's forced alignment as per-step lessons. This
instrument trains gradient learners on single links of that chain and reads them as every
learner of the chamber is read. Each is an independently scored control, never part of a
brain's answer path. The reading tells which link holds the letters a brain does not say:

- ``--hear window``: the standardized window the brain's fields read (``--frames`` 10 ms frames,
  24 as cached, or wider, rebuilt from the cached frames);
- ``--hear fields``: the pooled code of a saved brain's learned fields (``--brain``), frozen;
- ``--hear state``: a saved brain's settled activity at each step, read greedily from rest
  (``--populations``, any of the brain's populations side by side);
- ``--past k``: the heard vectors of the ``k`` steps before, side by side with the step's own;
- ``--model linear | mlp | lstm | transformer`` at ``--width`` (``mlp``: one hidden layer;
  ``lstm`` and ``transformer``: the chamber's controls);
- ``--loss ctc``: CTC on the utterance's text; ``--loss path``: cross-entropy on the forced
  alignment of the text through a frozen copy's beliefs, an even split before the first copy
  and a new copy every ``--align-every`` utterances: the brain's teacher on a gradient learner.

Training is the offline control's: shuffled epochs, one Adam step per utterance (rate 0.001),
read on the held-out utterances after every epoch.

    python benchmarks/letters/ceiling.py --cache CACHE --hear fields --brain BRAIN.npz \
        --model mlp --width 256 --loss ctc --seed 0 --epochs 12 --out DIR
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("speech", _here / "speech.py")
assert _spec is not None and _spec.loader is not None
sp = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("speech", sp)
_spec.loader.exec_module(sp)
lt = sp.lt

Pairs = list[tuple[np.ndarray, np.ndarray]]


# ------------------------------------------------------------------------------- what is heard


def wide_windows(cache: Path, name: str, frames: int) -> Pairs:
    """The cache's utterances heard through a window of ``frames`` 10 ms frames, standardized
    with the cache's per-band statistics as ``speech.load_windows`` does its own window."""
    meta = json.loads((cache / "meta.json").read_text())
    return list(sp.load_windows(cache, name, {**meta, "window": int(frames)}))


def fields_code(brain: Any, pairs: Pairs, chunk: int = 2048) -> Pairs:
    """Each step's pooled code of the brain's learned fields, the tables held still."""
    fields = brain.fields
    was, fields.learning = fields.learning, False
    out = []
    for x, text in pairs:
        code = np.concatenate([fields.code(x[k : k + chunk]) for k in range(0, len(x), chunk)])
        out.append((code.astype(np.float32), text))
    fields.learning = was
    return out


def settled_states(brain: Any, pairs: Pairs, batch: int = 32) -> Pairs:
    """Every neuron's settled activity at each step, the brain answering greedily from rest,
    ``batch`` utterances side by side; nothing is learned."""
    if brain.fields is not None:
        brain.fields.learning = False
    out: Pairs = []
    for k in range(0, len(pairs), batch):
        rows = [x for x, _ in pairs[k : k + batch]]
        x, mask = sp.padded(rows)
        brain.reset()
        held = np.zeros((*mask.shape, brain.connectome.n), dtype=np.float32)
        for t in range(len(x)):
            brain.act(x[t], greedy=True)
            held[t] = np.asarray(brain.basal_ganglia.state.activation)
        out += [(held[mask[:, r], r], pairs[k + r][1]) for r in range(len(rows))]
    return out


def with_past(pairs: Pairs, past: int) -> Pairs:
    """Each step's vector beside the vectors of the ``past`` steps before it (the first step's
    repeated before the utterance began)."""
    if past <= 0:
        return pairs
    out = []
    for x, text in pairs:
        padded = np.concatenate([np.repeat(x[:1], past, axis=0), x])
        out.append((np.concatenate([padded[past - k : len(padded) - k]
                                    for k in range(past + 1)], axis=1), text))
    return out


def standardized(train: Pairs, held: Pairs) -> tuple[Pairs, Pairs]:
    """Both sets less the training steps' mean, over their standard deviation, per value."""
    stacked = np.concatenate([x for x, _ in train])
    mean, scale = stacked.mean(0), stacked.std(0) + 1e-6
    return ([((x - mean) / scale, t) for x, t in train],
            [((x - mean) / scale, t) for x, t in held])


# ---------------------------------------------------------------------------------- learners


def build(model: str, width: int, inputs: int, seed: int) -> Any:
    import controls as ct
    import torch
    from torch import nn

    torch.manual_seed(int(seed))
    if model == "linear":
        return nn.Linear(inputs, sp.SYMBOLS)
    if model == "mlp":
        return nn.Sequential(nn.Linear(inputs, width), nn.ReLU(), nn.Linear(width, sp.SYMBOLS))
    return ct.build(model, width, inputs, sp.SYMBOLS, seed)


def run(train: Pairs, held: Pairs, *, model: str, width: int, loss: str, seed: int,
        epochs: int, align_every: int = 256, rate: float = 0.001) -> dict[str, Any]:
    """One learner, trained offline and read after every epoch."""
    import controls as ct
    import torch

    torch.set_num_threads(1)
    inputs = train[0][0].shape[1]
    net = build(model, width, inputs, seed)
    opt = torch.optim.Adam(net.parameters(), lr=rate)
    record: dict[str, Any] = {"model": model, "width": width, "loss": loss, "seed": seed,
                              "inputs": inputs, "parameters": ct.count(net), "probes": []}

    def forward(x: np.ndarray) -> Any:
        return net(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None])[0]

    frozen: Any = None
    heard, started = 0, time.monotonic()
    for epoch in range(epochs):
        order = np.random.default_rng([seed, epoch]).permutation(len(train))
        for i in order:
            x, text = train[int(i)]
            if loss == "path" and heard and heard % align_every == 0:
                frozen = build(model, width, inputs, seed)
                frozen.load_state_dict(net.state_dict())
            logits = forward(x)
            if loss == "ctc":
                logp = torch.log_softmax(logits, -1)[:, None]
                value = torch.nn.functional.ctc_loss(
                    logp, torch.from_numpy(text)[None], torch.tensor([len(x)]),
                    torch.tensor([len(text)]), blank=0, zero_infinity=True)
            else:
                if frozen is None:
                    path = sp.even_path(len(x), text)
                else:
                    with torch.no_grad():
                        belief = torch.log_softmax(frozen(torch.from_numpy(
                            np.ascontiguousarray(x, dtype=np.float32))[None])[0], -1)
                    path = lt.ctc_path(belief.numpy().astype(np.float64), text)
                value = torch.nn.functional.cross_entropy(logits, torch.from_numpy(path))
            opt.zero_grad()
            value.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), ct.CLIP)
            opt.step()
            heard += 1
        actions, logps = [], []
        with torch.no_grad():
            for x, _ in held:
                logits = forward(x)
                actions.append(np.argmax(logits.numpy(), -1))
                logps.append(torch.log_softmax(logits, -1).numpy().astype(np.float64))
        score = {**sp.read(actions, held), **sp.locate(actions, logps, held)}
        record["probes"].append({"epoch": epoch + 1, "seconds": time.monotonic() - started,
                                 "score": score,
                                 "sample": [sp.decode(lt.collapse(a)) for a in actions[:2]]})
        print(json.dumps({"epoch": epoch + 1, "characters": round(score["characters"], 3),
                          "letter": round(score["letter_from_sound"], 3),
                          "emission": round(score["emission_given_letter"], 3),
                          "seconds": round(time.monotonic() - started)}), flush=True)
    best = max(record["probes"], key=lambda q: float(q["score"]["characters"]))
    record["best"] = {"epoch": best["epoch"], **{k: best["score"][k] for k in (
        "characters", "letter_from_sound", "emission_given_letter", "deletions_per_character")}}
    return record


def split(stacked: np.ndarray, like: Pairs) -> Pairs:
    """Stacked steps cut back into the utterances of ``like``, with their texts."""
    ends = np.cumsum([len(x) for x, _ in like])
    return [(stacked[end - len(x) : end], text) for end, (x, text) in zip(ends, like, strict=True)]


def heard(args: argparse.Namespace) -> tuple[Pairs, Pairs, dict[str, Any]]:
    """The training and held-out utterances as the learner hears them."""
    note: dict[str, Any] = {"hear": args.hear, "past": args.past}
    if args.hear == "window":
        frames = args.frames or int(json.loads((args.cache / "meta.json").read_text())["window"])
        train = wide_windows(args.cache, "train", frames)
        held = wide_windows(args.cache, "held", frames)
        note["frames"] = frames
    else:
        import cadence as cd

        train, held = sp.load(args.cache, "train"), sp.load(args.cache, "held")
        note["brain"] = args.brain.name
        place = ({"backend": "cpu"} if args.device is None else
                 {"backend": "torch", "device": args.device, "precision": "float32"})
        brain = cd.Brain.load(args.brain, **place)
        kept = None if args.keep is None else args.keep / f"{args.brain.stem}-{args.hear}.npz"
        if kept is not None and kept.exists():
            with np.load(kept) as data:  # read once, heard by every learner
                train, held = split(data["train"], train), split(data["held"], held)
        else:
            if args.hear == "fields":
                train, held = fields_code(brain, train), fields_code(brain, held)
            else:
                train = settled_states(brain, train, args.batch)
                held = settled_states(brain, held, args.batch)
            if kept is not None:
                kept.parent.mkdir(parents=True, exist_ok=True)
                beside = kept.with_suffix(".tmp.npz")
                np.savez(beside, train=np.concatenate([x for x, _ in train]),
                         held=np.concatenate([x for x, _ in held]))
                os.replace(beside, kept)
        if args.hear == "state":  # the named populations' neurons, side by side
            note["populations"] = args.populations
            index = np.concatenate([np.asarray(brain.connectome.populations[p], dtype=np.int64)
                                    for p in args.populations])
            train = [(x[:, index], text) for x, text in train]
            held = [(x[:, index], text) for x, text in held]
    if args.limit:
        train = train[: args.limit]
    train, held = with_past(train, args.past), with_past(held, args.past)
    if args.hear != "window":
        train, held = standardized(train, held)
    return train, held, note


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--hear", choices=("window", "fields", "state"), default="window")
    parser.add_argument("--frames", type=int, help="window: 10 ms frames a step hears")
    parser.add_argument("--brain", type=Path, help="fields, state: a saved brain")
    parser.add_argument("--populations", nargs="+", default=["association"])
    parser.add_argument("--device", help="fields, state: a torch device the saved brain runs on")
    parser.add_argument("--batch", type=int, default=32,
                        help="state: utterances the brain hears side by side")
    parser.add_argument("--past", type=int, default=0)
    parser.add_argument("--model", choices=("linear", "mlp", "lstm", "transformer"),
                        default="mlp")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--loss", choices=("ctc", "path"), default="ctc")
    parser.add_argument("--align-every", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, help="the first training utterances only")
    parser.add_argument("--keep", type=Path,
                        help="fields, state: a folder that keeps what was heard for the next run")
    parser.add_argument("--name", help="the record's name; made from the settings without it")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    train, held, note = heard(args)
    record = run(train, held, model=args.model, width=args.width, loss=args.loss, seed=args.seed,
                 epochs=args.epochs, align_every=args.align_every)
    record.update(note)
    name = args.name or "-".join(str(v) for v in (
        args.hear, *(args.populations if args.hear == "state" else []),
        f"f{args.frames}" if args.frames else "", f"past{args.past}", args.model, args.width,
        args.loss) if v != "")
    args.out.mkdir(parents=True, exist_ok=True)
    sp.write_atomic(args.out / f"ceiling-{name}-{args.seed}.json.gz", record)
    print(json.dumps({"name": name, "seed": args.seed, "parameters": record["parameters"],
                      **{k: round(v, 3) if isinstance(v, float) else v
                         for k, v in record["best"].items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
