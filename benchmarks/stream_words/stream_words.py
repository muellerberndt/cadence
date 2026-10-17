"""The streaming spoken-word chamber: words as frame sequences, heard one frame per moment.

A synthetic speaker says words from a lexicon of ``words`` (12), each a short sequence of
phones from an inventory of ``phones`` (16); a phone is a fixed vector of ``features`` (16)
values in [0, 1], and every frame adds Gaussian noise. Words come in pairs that share all but
their final phone, so a word stays ambiguous until its last frame; every word's penultimate
phone is the same joint phone, and each final phone ends two words of different pairs, so the
last two phones leave two candidates and the phone before them decides. Every phone before
the final one lasts one or two frames (duration jitter); the final phone lasts one frame,
which makes the end of a word a frame the history can identify. One to three silent frames
separate the words. The label of a frame is blank (0) except at a word's last frame, where it
is the word (1 to ``words``): one output slot of ``words + 1`` choices.

Arms. ``surprise``: one continuing ``Brain.compose`` life (the founder composition with its
working trace) hears every frame through ``Brain.step(frame, teacher=label)`` with the
learner's surprise rule, a lesson only where its act's answer differs from the label.
``every_frame``: the same with a lesson on every frame, the paired control. ``window``: the
same composed brain and rule given the concatenation of the last ``window_frames`` frames
(the longest word) as one observation, the pressed-in window as a control. ``gru``: a
supervised online gated recurrent unit (the cell of ``benchmarks/competitors.py``) with a
softmax readout, the cross-entropy of every frame taken through truncated backpropagation
over the last ``truncation`` frames, one update per frame. The world pays nothing: the
reward learner's rates are zero in every brain arm, so the zero-reward feedback of ``step``
changes nothing. Controls: ``majority`` (blank on every frame), ``uniform`` (a uniform
choice per frame) and ``phone`` (the most frequent label of each noise-free phone on the
evaluation stream, in sample: a bound for any answer read from the current frame alone).

Readings come from saved copies every ``probe_every`` training words: the copy is reset and
answers a held-out stream greedily without learning. Word accuracy at word ends, emissions
elsewhere (a word answered on a frame that ends no word) per word, precision and recall of
the emitted words, and the confusions at word ends (the pair partner, the rhyme partner, any
other word). One life per process: ``--arm``, ``--seed``, ``--out``; ``--report`` tabulates a
directory of receipts. ``protocol.json`` declares the settings and the gate.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import gzip  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import signal  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

import cadence as cd  # noqa: E402
from cadence.receipts import Receipt, canonical_json  # noqa: E402

SCHEMA = "stream-words/1"
PROTOCOL = Path(__file__).with_name("protocol.json")
BLANK = 0
BRAIN_ARMS = ("surprise", "every_frame", "window", "history")
LEARNER_ARMS = ("gru",)
CONTROLS = ("majority", "uniform", "phone")
COMPETITORS_PATH = Path(__file__).resolve().parents[1] / "competitors.py"


def _load_competitors() -> Any:
    spec = importlib.util.spec_from_file_location("competitors", COMPETITORS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


competitors = _load_competitors()


def load_protocol(path: Path = PROTOCOL) -> dict[str, Any]:
    protocol: dict[str, Any] = json.loads(path.read_bytes())
    for key in ("world", "training_words", "probe_every", "evaluation_words", "seeds", "brain"):
        if key not in protocol:
            raise ValueError(f"protocol lacks {key}")
    return protocol


# ----------------------------------------------------------------------------------- world


def rng_for(seed: int, purpose: str) -> np.random.Generator:
    """The independent random stream of one seed for one purpose."""
    purposes = ("lexicon", "training", "evaluation", "uniform")
    return np.random.default_rng([int(seed), purposes.index(purpose)])


def make_lexicon(
    seed: int,
    *,
    phones: int = 16,
    features: int = 16,
    pairs: int = 6,
    prefix_phones: tuple[int, int] = (2, 3),
) -> dict[str, Any]:
    """Phones and words of one seed.

    Phone 0..phones-1 are fixed vectors in [0, 1]. One phone is the joint, the penultimate
    phone of every word, which no prefix contains. ``pairs`` final phones f_0..f_{pairs-1}:
    pair k says ``prefix_k + [joint, f_k]`` (word 2k) and ``prefix_k + [joint, f_{k+1}]``
    (word 2k + 1), indices modulo ``pairs``, so final phone f_k ends word 2k and word 2k - 1,
    of the pairs k and k - 1. A prefix has ``prefix_phones`` phones without the joint and
    without a phone repeated back to back, and the last prefix phones of neighbouring pairs
    differ, so the phone before the joint tells the two words of one final phone apart.
    """
    rng = rng_for(seed, "lexicon")
    if pairs < 2 or phones < pairs + 3:
        raise ValueError("a lexicon needs at least two pairs and more phones than finals")
    vectors = rng.random((phones, features))
    order = rng.permutation(phones)
    joint = int(order[0])
    finals = [int(p) for p in order[1 : pairs + 1]]
    usable = [p for p in range(phones) if p != joint]
    while True:
        prefixes = []
        for _ in range(pairs):
            length = int(rng.integers(prefix_phones[0], prefix_phones[1] + 1))
            prefix = [int(rng.choice(usable))]
            while len(prefix) < length:
                phone = int(rng.choice(usable))
                if phone != prefix[-1]:
                    prefix.append(phone)
            prefixes.append(prefix)
        distinct = len({tuple(p) for p in prefixes}) == pairs
        neighbours = all(prefixes[k][-1] != prefixes[k - 1][-1] for k in range(pairs))
        if distinct and neighbours:
            break
    words = []
    for k in range(pairs):
        words.append(prefixes[k] + [joint, finals[k]])
        words.append(prefixes[k] + [joint, finals[(k + 1) % pairs]])
    return {
        "seed": int(seed),
        "phones": vectors,
        "joint": joint,
        "finals": finals,
        "prefixes": prefixes,
        "words": words,
        "pair": [w ^ 1 for w in range(2 * pairs)],
        "rhyme": [rhyme_partner(w, pairs) for w in range(2 * pairs)],
    }


def rhyme_partner(word: int, pairs: int) -> int:
    """The other word that ends in the same final phone (word 2k with word 2k - 1)."""
    return (word - 1) % (2 * pairs) if word % 2 == 0 else (word + 1) % (2 * pairs)


def word_frames(word: list[int], rng: np.random.Generator, repeat: float) -> list[int]:
    """The phone of every frame of one spoken word: each phone before the final one lasts one
    frame, or two with probability ``repeat``; the final phone lasts one frame."""
    frames: list[int] = []
    for phone in word[:-1]:
        frames.extend([phone] * (2 if rng.random() < repeat else 1))
    frames.append(word[-1])
    return frames


def make_stream(
    lexicon: dict[str, Any],
    words: int,
    rng: np.random.Generator,
    *,
    noise: float = 0.1,
    repeat: float = 0.5,
    silence: tuple[int, int] = (1, 3),
) -> dict[str, Any]:
    """A stream of ``words`` words drawn uniformly, each after one to three silent frames.

    ``frames`` (T, features) are the phone vectors (zero for silence) plus Gaussian noise;
    ``phones`` (T,) the noise-free phone of every frame, -1 for silence; ``labels`` (T,)
    blank except the last frame of every word, which carries the word plus one; ``ends`` (T,)
    marks those frames; ``said`` lists the words in order and ``boundaries`` the index after
    each word's last frame."""
    vectors = lexicon["phones"]
    phone_track: list[int] = []
    labels: list[int] = []
    said = rng.integers(len(lexicon["words"]), size=words)
    boundaries = []
    for word in said:
        gap = int(rng.integers(silence[0], silence[1] + 1))
        phone_track.extend([-1] * gap)
        labels.extend([BLANK] * gap)
        spoken = word_frames(lexicon["words"][int(word)], rng, repeat)
        phone_track.extend(spoken)
        labels.extend([BLANK] * (len(spoken) - 1) + [int(word) + 1])
        boundaries.append(len(labels))
    track = np.asarray(phone_track, dtype=np.int64)
    clean = np.where(track[:, None] >= 0, vectors[np.maximum(track, 0)], 0.0)
    frames = clean + rng.normal(0.0, noise, clean.shape)
    label_array = np.asarray(labels, dtype=np.int64)
    return {
        "frames": frames,
        "phones": track,
        "labels": label_array,
        "ends": label_array != BLANK,
        "said": np.asarray(said, dtype=np.int64),
        "boundaries": np.asarray(boundaries, dtype=np.int64),
    }


def window(frames: np.ndarray, n: int) -> np.ndarray:
    """Each frame's observation for the window control: the last ``n`` frames concatenated,
    oldest first and the current frame last, zeros before the stream began."""
    frames = np.asarray(frames, dtype=float)
    padded = np.concatenate([np.zeros((n - 1, frames.shape[1])), frames])
    return np.stack([padded[t : t + n].reshape(-1) for t in range(len(frames))])


# --------------------------------------------------------------------------------- scoring


def score(answers: np.ndarray, stream: dict[str, Any], lexicon: dict[str, Any]) -> dict[str, Any]:
    """Word accuracy at word ends and the emissions elsewhere.

    An emission is a frame answered with a word (any choice but blank). A word is recalled
    when its last frame is answered with that word; precision is the share of emissions that
    recall a word. Emissions elsewhere are emissions on frames that end no word, counted per
    word of the stream. A wrong word at a word's end is a confusion with its pair partner
    (same prefix), its rhyme partner (same final phone) or another word."""
    a = np.asarray(answers, dtype=np.int64).reshape(-1)
    labels, ends = stream["labels"], stream["ends"]
    if a.shape != labels.shape:
        raise ValueError("one answer per frame")
    words = int(ends.sum())
    emitted = a != BLANK
    hits = int(np.sum(a[ends] == labels[ends]))
    elsewhere = int(np.sum(emitted & ~ends))
    confusions = {"pair": 0, "rhyme": 0, "other": 0, "blank": 0}
    for answer, label in zip(a[ends], labels[ends], strict=True):
        if answer == label:
            continue
        if answer == BLANK:
            confusions["blank"] += 1
        elif answer - 1 == lexicon["pair"][label - 1]:
            confusions["pair"] += 1
        elif answer - 1 == lexicon["rhyme"][label - 1]:
            confusions["rhyme"] += 1
        else:
            confusions["other"] += 1
    return {
        "words": words,
        "frames": int(len(a)),
        "word_accuracy": hits / words,
        "recall": hits / words,
        "precision": hits / int(emitted.sum()) if emitted.any() else None,
        "end_emission": float(np.mean(emitted[ends])),
        "emissions_elsewhere": elsewhere,
        "emissions_elsewhere_per_word": elsewhere / words,
        "emission_rate_elsewhere": elsewhere / int(np.sum(~ends)),
        "confusions": confusions,
    }


def controls(stream: dict[str, Any], lexicon: dict[str, Any], seed: int) -> dict[str, Any]:
    """The controls' answers scored: blank everywhere, a uniform choice per frame, and the
    most frequent label of each noise-free phone (silence included) on this stream."""
    classes = len(lexicon["words"]) + 1
    labels, phones = stream["labels"], stream["phones"]
    uniform = rng_for(seed, "uniform").integers(classes, size=len(labels))
    table: dict[int, np.ndarray] = {}
    for phone, label in zip(phones, labels, strict=True):
        table.setdefault(int(phone), np.zeros(classes, dtype=np.int64))[label] += 1
    by_phone = np.array([int(np.argmax(table[int(p)])) for p in phones], dtype=np.int64)
    return {
        "majority": score(np.zeros_like(labels), stream, lexicon),
        "uniform": score(uniform, stream, lexicon),
        "phone": score(by_phone, stream, lexicon),
    }


# ------------------------------------------------------------------------------ brain arms


def compose(arm: str, inputs: int, seed: int, protocol: dict[str, Any]) -> cd.Brain:
    """The composed founder for an arm: the protocol's shared genes and the arm's own."""
    genes = {**protocol["brain"]["genes"], **protocol["arms"][arm].get("genes", {})}
    return cd.Brain.compose(inputs, len(protocol_words(protocol)) + 1, seed=seed, **genes)


def protocol_words(protocol: dict[str, Any]) -> range:
    return range(2 * int(protocol["world"]["pairs"]))


def observations(arm: str, stream: dict[str, Any], protocol: dict[str, Any]) -> np.ndarray:
    if arm == "window":
        return window(stream["frames"], int(protocol["window_frames"]))
    return np.asarray(stream["frames"], dtype=float)


def brain_answers(path: Path, x: np.ndarray) -> np.ndarray:
    """A saved brain's greedy answers to a fresh stream: loaded, reset, no learning."""
    brain = cd.Brain.load(path)
    brain.reset()
    return np.array([int(brain.act(row[None], greedy=True)[0]) for row in x], dtype=np.int64)


def brain_life(
    arm: str,
    seed: int,
    protocol: dict[str, Any],
    out: Path,
    sizes: dict[str, int],
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One continuing life: every training frame through ``Brain.step(frame, teacher=label)``;
    a saved copy answers the evaluation stream every ``probe_every`` words. ``record`` fills
    as the life goes, so a life the wall limit stops keeps its probes."""
    record = {} if record is None else record
    world = protocol["world"]
    lexicon = make_lexicon(seed, **lexicon_settings(world))
    train = make_stream(
        lexicon, sizes["training_words"], rng_for(seed, "training"), **stream_settings(world)
    )
    test = make_stream(
        lexicon, sizes["evaluation_words"], rng_for(seed, "evaluation"), **stream_settings(world)
    )
    x, test_x = observations(arm, train, protocol), observations(arm, test, protocol)
    brain = compose(arm, x.shape[1], seed, protocol)
    work = {
        "frames": 0,
        "lessons": 0,
        "lesson_sweeps": 0,
        "act_sweeps": 0,
        "feedback_sweeps": 0,
        "refused": 0,
    }
    probes: list[dict[str, Any]] = []
    record.update({"lexicon": plain_lexicon(lexicon), "work": work, "probes": probes})
    word = 0
    started = time.monotonic()
    next_end = iter(train["boundaries"])
    boundary = next(next_end)
    since = {"frames": 0, "lessons": 0, "end_hits": 0, "ends": 0}
    for t in range(len(x)):
        try:
            issued = brain.step(x[t][None], teacher=[int(train["labels"][t])])
        except RuntimeError as exc:  # a refused act; custody: the life ends here
            work["refused"] += 1
            record["refusal"] = {"frame": t, "error": f"{type(exc).__name__}: {exc}"}
            record["describe"] = brain.describe()
            return record
        learned = brain.last_learning
        taught = int(learned.get("demonstration_accepted_presentations", 1))
        work["frames"] += 1
        work["lessons"] += taught
        work["lesson_sweeps"] += int(learned.get("demonstration_total_steps", 0))
        work["feedback_sweeps"] += int(learned.get("free_steps", 0))
        settled = brain.last_settlement
        work["act_sweeps"] += int(settled["steps"]) if settled is not None else 0
        since["frames"] += 1
        since["lessons"] += taught
        if train["ends"][t]:
            since["ends"] += 1
            since["end_hits"] += int(issued[0] == train["labels"][t])
        if t + 1 == boundary:
            word += 1
            boundary = next(next_end, -1)
            if word % sizes["probe_every"] == 0 or word == sizes["training_words"]:
                path = brain.save(out / f"{arm}-{seed}-w{word}.npz")
                answers = brain_answers(path, test_x)
                probes.append(
                    {
                        "words": word,
                        "frames": t + 1,
                        "seconds": time.monotonic() - started,
                        "training": dict(since),
                        "score": score(answers, test, lexicon),
                        "answers": answers.tolist() if word == sizes["training_words"] else None,
                    }
                )
                since = {"frames": 0, "lessons": 0, "end_hits": 0, "ends": 0}
                if word < sizes["training_words"]:
                    path.unlink()
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "seed": seed,
                            "words": word,
                            "accuracy": round(probes[-1]["score"]["word_accuracy"], 3),
                            "elsewhere": round(
                                probes[-1]["score"]["emissions_elsewhere_per_word"], 3
                            ),
                            "elapsed": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    record.update(
        {
            "describe": brain.describe(),
            "parameters": brain.parameters(),
            "neurons": brain.connectome.n,
            "synapses": brain.connectome.synapses,
            "controls": controls(test, lexicon, seed),
        }
    )
    return record


def lexicon_settings(world: dict[str, Any]) -> dict[str, Any]:
    return {
        "phones": int(world["phones"]),
        "features": int(world["features"]),
        "pairs": int(world["pairs"]),
        "prefix_phones": tuple(world["prefix_phones"]),
    }


def stream_settings(world: dict[str, Any]) -> dict[str, Any]:
    return {
        "noise": float(world["noise"]),
        "repeat": float(world["repeat"]),
        "silence": tuple(world["silence"]),
    }


def plain_lexicon(lexicon: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in lexicon.items()}


# ------------------------------------------------------------------------------ online GRU


def gru_input_gradient(cell: Any, g: np.ndarray, cache: dict[str, Any]) -> np.ndarray:
    """The gradient of ``g . h`` with respect to the previous hidden state ``h_prev`` of one
    GRU step (``competitors.GRUCell``), the weights held fixed: what truncated
    backpropagation carries one frame back."""
    w = cell.weights
    z, r, n, h_prev = cache["z"], cache["r"], cache["n"], cache["h_prev"]
    gn = g * (1.0 - z) * (1.0 - n * n)
    gz = g * (h_prev - n) * z * (1.0 - z)
    through_n = w["Un"].T @ gn
    gr = through_n * h_prev * r * (1.0 - r)
    return np.asarray(g * z + w["Uz"].T @ gz + w["Ur"].T @ gr + through_n * r)


class OnlineGRU:
    """A gated recurrent unit with a softmax readout, learning online from labelled frames.

    ``learn(x, label)`` answers the frame, then takes the gradient of that frame's
    cross-entropy through the last ``truncation`` frames (their cached forward passes, the
    current weights) and makes one Adam step on its norm-clipped value; the hidden state
    carries on. ``answer(x)`` answers greedily and carries the state without learning.
    Work: updates, forward and backward cell passes, and an analytic count of
    multiply-accumulates."""

    def __init__(
        self,
        inputs: int,
        classes: int,
        seed: int,
        *,
        hidden: int = 32,
        truncation: int = 16,
        rate: float = 0.003,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        clip: float = 5.0,
        readout_init: float = 0.1,
    ) -> None:
        if hidden <= 0 or truncation <= 0 or rate <= 0 or clip <= 0:
            raise ValueError("hidden, truncation, rate and clip must be positive")
        rng = np.random.default_rng(seed)
        self.inputs, self.classes, self.hidden, self.truncation = (
            inputs,
            classes,
            hidden,
            truncation,
        )
        self.rate, self.betas, self.eps, self.clip = rate, betas, eps, clip
        self.cell = competitors.GRUCell(inputs, hidden, rng)
        self.W = rng.normal(0.0, readout_init, (classes, hidden))
        self.b = np.zeros(classes)
        self.h = np.zeros(hidden)
        self.history: deque[dict[str, Any]] = deque(maxlen=truncation)
        self.moments = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in self.params().items()}
        self.work = {
            "updates": 0,
            "forward": 0,
            "backward": 0,
            "clipped": 0,
            "multiply_accumulates": 0,
        }

    def params(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b, **self.cell.parameters()}

    @property
    def parameter_count(self) -> int:
        return int(sum(v.size for v in self.params().values()))

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
        h, cache = self.cell.forward(np.asarray(x, dtype=float).reshape(-1), self.h)
        logits = self.W @ h + self.b
        p = np.exp(logits - logits.max())
        self.work["forward"] += 1
        self.work["multiply_accumulates"] += self.cell.forward_macs + self.classes * self.hidden
        return h, cache, p / p.sum()

    def answer(self, x: np.ndarray) -> int:
        h, _, p = self._forward(x)
        self.h = h
        return int(np.argmax(p))

    def gradients(self, label: int, p: np.ndarray, h: np.ndarray) -> dict[str, np.ndarray]:
        """The cross-entropy gradient of the current frame through the cached history."""
        dlogits = p.copy()
        dlogits[label] -= 1.0
        grads = {"W": np.outer(dlogits, h), "b": dlogits}
        for name, value in self.cell.parameters().items():
            grads[name] = np.zeros_like(value)
        g = self.W.T @ dlogits
        for cache in reversed(self.history):
            for name, value in self.cell.backward(g, cache).items():
                grads[name] += value
            g = gru_input_gradient(self.cell, g, cache)
            self.work["backward"] += 1
            self.work["multiply_accumulates"] += self.cell.backward_macs + 3 * self.hidden**2
        return grads

    def learn(self, x: np.ndarray, label: int) -> int:
        h, cache, p = self._forward(x)
        self.history.append(cache)
        grads = self.gradients(int(label), p, h)
        norm = float(np.sqrt(sum(float(np.sum(v * v)) for v in grads.values())))
        scale = min(1.0, self.clip / max(norm, 1e-300))
        self.work["clipped"] += int(scale < 1.0)
        self.work["updates"] += 1
        step = self.work["updates"]
        b1, b2 = self.betas
        for name, value in self.params().items():
            m, v = self.moments[name]
            grad = scale * grads[name]
            m *= b1
            m += (1 - b1) * grad
            v *= b2
            v += (1 - b2) * grad * grad
            value -= self.rate * (m / (1 - b1**step)) / (np.sqrt(v / (1 - b2**step)) + self.eps)
        self.work["multiply_accumulates"] += 6 * self.parameter_count
        self.h = h
        return int(np.argmax(p))

    def frozen_copy(self) -> OnlineGRU:
        twin = copy.deepcopy(self)
        twin.h = np.zeros(self.hidden)
        twin.history.clear()
        return twin


def gru_life(
    seed: int,
    protocol: dict[str, Any],
    out: Path,
    sizes: dict[str, int],
    record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One online GRU life on the training stream, probed like a brain life."""
    record = {} if record is None else record
    world = protocol["world"]
    lexicon = make_lexicon(seed, **lexicon_settings(world))
    train = make_stream(
        lexicon, sizes["training_words"], rng_for(seed, "training"), **stream_settings(world)
    )
    test = make_stream(
        lexicon, sizes["evaluation_words"], rng_for(seed, "evaluation"), **stream_settings(world)
    )
    settings = {k: v for k, v in protocol["gru"].items() if k != "note"}
    settings["betas"] = tuple(settings["betas"])
    learner = OnlineGRU(train["frames"].shape[1], len(lexicon["words"]) + 1, seed, **settings)
    probes: list[dict[str, Any]] = []
    record.update(
        {
            "lexicon": plain_lexicon(lexicon),
            "work": learner.work,
            "probes": probes,
            "parameters": learner.parameter_count,
            "settings": protocol["gru"],
        }
    )
    word = 0
    started = time.monotonic()
    boundaries = set(int(b) for b in train["boundaries"])
    for t, (x, label) in enumerate(zip(train["frames"], train["labels"], strict=True)):
        learner.learn(x, int(label))
        if t + 1 in boundaries:
            word += 1
            if word % sizes["probe_every"] == 0 or word == sizes["training_words"]:
                twin = learner.frozen_copy()
                answers = np.array([twin.answer(f) for f in test["frames"]], dtype=np.int64)
                probes.append(
                    {
                        "words": word,
                        "frames": t + 1,
                        "seconds": time.monotonic() - started,
                        "score": score(answers, test, lexicon),
                        "answers": answers.tolist() if word == sizes["training_words"] else None,
                    }
                )
                print(
                    json.dumps(
                        {
                            "arm": "gru",
                            "seed": seed,
                            "words": word,
                            "accuracy": round(probes[-1]["score"]["word_accuracy"], 3),
                            "elapsed": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
    record["controls"] = controls(test, lexicon, seed)
    return record


# -------------------------------------------------------------------------------- receipts


def sources() -> list[tuple[str, Path]]:
    package = Path(cd.__file__).resolve().parent
    return [
        ("stream_words.py", Path(__file__).resolve()),
        ("competitors.py", COMPETITORS_PATH),
        ("protocol.json", PROTOCOL),
        *(
            ("cadence/" + path.relative_to(package).as_posix(), path)
            for path in sorted(package.rglob("*.py"))
        ),
    ]


def write_receipt(path: Path, body: dict[str, Any]) -> None:
    receipt = Receipt.build(SCHEMA, body, sources())
    data = (canonical_json(receipt.to_dict()) + "\n").encode()
    path.write_bytes(gzip.compress(data, mtime=0))


def read_receipt(path: Path) -> dict[str, Any]:
    stored: dict[str, Any] = json.loads(gzip.decompress(path.read_bytes()))
    return stored


def verify(path: Path, *, current: bool = False) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as directory:
        plain = Path(directory) / "receipt.json"
        plain.write_bytes(gzip.decompress(path.read_bytes()))
        return Receipt.verify(plain, sources=sources() if current else None)


# ---------------------------------------------------------------------------------- report


def final(body: dict[str, Any]) -> dict[str, Any] | None:
    """The final probe's score of a complete life, or None for one that refused or timed out."""
    probes = body["result"].get("probes") or []
    return probes[-1]["score"] if probes and body["status"] == "complete" else None


def table_row(name: str, scores: list[dict[str, Any]], lessons: str) -> str:
    accuracy = [s["word_accuracy"] for s in scores]
    precision = [s["precision"] for s in scores if s["precision"] is not None]
    confusions = {k: sum(s["confusions"][k] for s in scores) for k in scores[0]["confusions"]}
    cells = [
        name,
        str(len(scores)),
        f"{np.mean(accuracy):.3f}",
        " ".join(f"{a:.2f}" for a in accuracy),
        f"{np.mean([s['emissions_elsewhere_per_word'] for s in scores]):.3f}",
        f"{np.mean(precision):.3f}" if precision else "-",
        f"{np.mean([s['end_emission'] for s in scores]):.3f}",
        " / ".join(str(confusions[k]) for k in ("pair", "rhyme", "other", "blank")),
        lessons,
    ]
    return "| " + " | ".join(cells) + " |"


def report(directory: Path) -> str:
    """The receipts of a directory as Markdown: per arm the final probe over seeds, the
    controls, then the declared gate on the means over seeds."""
    rows: dict[str, list[dict[str, Any]]] = {}
    control_scores: dict[str, dict[int, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json.gz")):
        body = read_receipt(path)["body"]
        rows.setdefault(body["arm"], []).append(body)
        for name, value in (body["result"].get("controls") or {}).items():
            control_scores.setdefault(name, {})[int(body["seed"])] = value
    lines = [
        "| arm | seeds | word accuracy | per seed | emissions elsewhere per word | precision"
        " | end emission | confusions pair / rhyme / other / blank | lessons per frame |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    means: dict[str, tuple[float, float]] = {}
    for arm in (*BRAIN_ARMS, *LEARNER_ARMS):
        bodies = sorted(rows.get(arm, []), key=lambda b: b["seed"])
        if not bodies:
            continue
        scores = [final(b) for b in bodies]
        complete = [s for s in scores if s is not None]
        if len(complete) < len(scores):
            lines.append(
                f"| {arm} | {len(bodies)} | {len(scores) - len(complete)} refused or"
                " incomplete | | | | | | |"
            )
            continue
        work = [b["result"]["work"] for b in bodies]
        lessons = (
            f"{np.mean([w['lessons'] / w['frames'] for w in work]):.3f}"
            if "lessons" in work[0]
            else "-"
        )
        lines.append(table_row(arm, complete, lessons))
        means[arm] = (
            float(np.mean([s["word_accuracy"] for s in complete])),
            float(np.mean([s["emissions_elsewhere_per_word"] for s in complete])),
        )
    for name in CONTROLS:
        if name in control_scores:
            by_seed = control_scores[name]
            lines.append(table_row(name, [by_seed[s] for s in sorted(by_seed)], "-"))
    lines.append("")
    lines.append(f"Gate: {gate(means)}")
    return "\n".join(lines)


def gate(means: dict[str, tuple[float, float]]) -> dict[str, Any] | None:
    """The declared gate on the means over seeds: the surprise arm's word accuracy at or
    above the window control's and above 0.5, with at most 0.2 emissions elsewhere per word."""
    if "surprise" not in means or "window" not in means:
        return None
    accuracy, elsewhere = means["surprise"]
    parts = {
        "at_or_above_window": accuracy >= means["window"][0],
        "above_final_phone_bound": accuracy > 0.5,
        "elsewhere_within_bound": elsewhere <= 0.2,
    }
    return {**parts, "passes": all(parts.values())}


# ------------------------------------------------------------------------------------ main


def timeout(*_: Any) -> None:
    raise TimeoutError("declared wall limit per life")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", choices=BRAIN_ARMS + LEARNER_ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, help="the directory of this life's receipt")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--smoke", action="store_true", help="a tiny life; not a declared run")
    parser.add_argument("--report", type=Path, help="tabulate a directory of receipts")
    parser.add_argument("--verify", type=Path, help="check a receipt's digest")
    parser.add_argument("--current", action="store_true", help="--verify against these sources")
    args = parser.parse_args(argv)
    if args.report is not None:
        print(report(args.report))
        return 0
    if args.verify is not None:
        ok, message = verify(args.verify, current=args.current)
        print(message)
        return 0 if ok else 1
    if args.arm is None or args.seed is None or args.out is None:
        parser.error("--arm, --seed and --out are required for a life")
    protocol = load_protocol(args.protocol)
    if args.arm in BRAIN_ARMS and args.arm not in protocol["arms"]:
        parser.error(f"the protocol declares no arm {args.arm!r}")
    sizes = {k: int(protocol[k]) for k in ("training_words", "probe_every", "evaluation_words")}
    if args.smoke:
        sizes = {"training_words": 40, "probe_every": 20, "evaluation_words": 20}
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.arm}-{args.seed}.json.gz"
    if target.exists():
        parser.error(f"{target} exists; a life is run once")
    started = time.monotonic()
    result: dict[str, Any] = {}
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, float(protocol["max_wall_seconds_per_life"]))
    try:
        if args.arm in BRAIN_ARMS:
            brain_life(args.arm, args.seed, protocol, args.out, sizes, result)
        else:
            gru_life(args.seed, protocol, args.out, sizes, result)
        status = "refused" if result.get("refusal") else "complete"
    except TimeoutError as exc:
        result["error"], status = str(exc), "timeout"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    body = {
        "arm": args.arm,
        "seed": args.seed,
        "status": status,
        "smoke": args.smoke,
        "sizes": sizes,
        "protocol": protocol,
        "result": result,
        "seconds": time.monotonic() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cadence": cd.__version__,
    }
    write_receipt(target, body)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "seed": args.seed,
                "status": status,
                "seconds": round(body["seconds"], 1),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
