"""The spoken-phrase chamber's learners and readings (cadence#303).

A learner hears a batch of phrases of equal steps in lockstep: at every step the window as the
brain hears it (``Hearing``), with a lesson on the step's targets before the act when taught.
The transcript is read from the actions: the words up to the first ``end``.

Brain arms (``ARMS``) name what the brain sees and how it is composed; the baselines are
``controls.Sequence``. Readings never change a life: a brain is read on a copy whose fields
take no step, a baseline without a step.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import sys
import time
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


ph = _load("phrases")
ct = _load("controls")
audio = _load("audio")

ACTIONS, SILENCE, END = ph.ACTIONS, ph.SILENCE, ph.END
BANDS, FRAMES = audio.BANDS, ph.WINDOW
SCALE = 0.3  # the standardized window's scale, as the second run of cadence#183
CLIP_SIGMA = 3.0
STATS_PHRASES = 400
BATCH = 32

# What each brain arm sees and how it is composed: ``image`` reads the window as a
# time-frequency image ``(bands, frames)``; ``flat`` as one vector (the keyword front end).
# Without the recall memory a brain composes no lessons of its own: ``TAUGHT`` names the
# default's teaching and heard moment, so an arm without the memory differs by the memory alone.
TAUGHT: dict[str, Any] = {
    "recall": False,
    "teaching": "deep",
    "learning_surprise": True,
    "learning_fast_settle": True,
    "carried_gain": 0.01,
    "history_decays": (0.0,),
}
ARMS: dict[str, dict[str, Any]] = {
    "fields": {"sees": "image", "compose": {"learned_fields": 64}},
    "fields_no_history": {"sees": "image", "compose": {"learned_fields": 64, "history_decays": ()}},
    "fields_no_recall": {"sees": "image", "compose": {"learned_fields": 64, **TAUGHT}},
    "fields_founder": {"sees": "image", "compose": {"learned_fields": 64, "recall": False}},
    "fields_drawn": {"sees": "image", "hold": True, "compose": {"learned_fields": 64}},
    "keyword": {"sees": "flat", "compose": {"recall": False}},
    "keyword_taught": {"sees": "flat", "compose": dict(TAUGHT)},
    # development arms of Step 0: a wider association, and the pixel chamber's teaching point
    "fields_wide": {"sees": "image", "compose": {"learned_fields": 64, "modules": (256,)}},
    "fields_normalized": {
        "sees": "image",
        "compose": {"learned_fields": 64, "recall": False, "teaching": "normalized"},
    },
    "fields_wide_normalized": {
        "sees": "image",
        "compose": {
            "learned_fields": 64,
            "modules": (256,),
            "recall": False,
            "teaching": "normalized",
        },
    },
}
CONTROLS: dict[str, tuple[str, str]] = {
    "transformer_ctc": ("transformer", "ctc"),
    "transformer_step": ("transformer", "step"),
    "lstm_ctc": ("lstm", "ctc"),
    "lstm_step": ("lstm", "step"),
}


@dataclasses.dataclass
class Hearing:
    """Per-band statistics of training windows: the window as the brain hears it is centred on
    its own mean (one gain per window), standardized per band, clipped and scaled, and laid out
    as an image of ``BANDS`` rows by ``FRAMES`` columns."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def measured(cls, phrases: Any, seed: int = 7) -> Hearing:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(len(phrases), min(STATS_PHRASES, len(phrases)), replace=False))
        total = np.zeros(BANDS)
        square = np.zeros(BANDS)
        count = 0
        for r in rows:
            mel = phrases.logmel[r].astype(np.float64)
            for k in range(1, int(phrases.steps[r]) + 1):
                w = ph.window(mel, k)
                w = w - w.mean()
                total += w.sum(axis=0)
                square += (w**2).sum(axis=0)
                count += len(w)
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean**2, 1e-6))
        return cls(mean, np.maximum(std, 1e-3))

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        """(rows, FRAMES, BANDS) log-mel windows to (rows, BANDS * FRAMES)."""
        x = np.asarray(windows, dtype=np.float64)
        x = x - x.mean(axis=(1, 2), keepdims=True)
        z = np.clip((x - self.mean) / self.std, -CLIP_SIGMA, CLIP_SIGMA) * SCALE
        return np.ascontiguousarray(z.transpose(0, 2, 1)).reshape(len(z), BANDS * FRAMES)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def stream(phrases: Any, rows: np.ndarray, hearing: Hearing) -> np.ndarray:
    """The heard windows of phrases of equal steps: (rows, steps, BANDS * FRAMES)."""
    steps = int(phrases.steps[rows[0]])
    mel = phrases.logmel[rows].astype(np.float32)
    return np.stack([hearing(ph.window(mel, k)) for k in range(1, steps + 1)], axis=1)


# ------------------------------------------------------------------------------------ learners


def compose(arm: str, founder: int, extra: dict[str, Any] | None = None) -> Any:
    from cadence import Brain

    spec = ARMS[arm]
    inputs = (BANDS, FRAMES) if spec["sees"] == "image" else BANDS * FRAMES
    settings: dict[str, Any] = {"backend": "cpu", **spec["compose"], **(extra or {})}
    direct = float(settings.pop("direct", 0.0))
    if direct:  # round 2: the plastic pathway from the pools to the motor region
        settings["fields_motor"] = direct
    brain = Brain.compose(inputs, ACTIONS, seed=founder, **settings)
    if spec.get("hold"):
        brain.fields.learning = False
    return brain


def soft_targets(target: np.ndarray, share: float) -> np.ndarray:
    """One-hot targets per step, except that the silent step after a word step is taught the
    word at ``share`` and silence at the rest: its window holds the word's last frame as well."""
    rows, steps = target.shape
    out = np.zeros((rows, steps, ACTIONS))
    safe = np.where(target < 0, SILENCE, target)
    out[np.arange(rows)[:, None], np.arange(steps)[None, :], safe] = 1.0
    for r in range(rows):
        for k in np.flatnonzero(target[r] < SILENCE):
            if k + 1 < steps and target[r, k + 1] == SILENCE:
                out[r, k + 1] = 0.0
                out[r, k + 1, SILENCE] = 1.0 - share
                out[r, k + 1, target[r, k]] = share
    return out


class BrainLearner:
    """A brain hearing phrases: a lesson on the step's targets before each act when taught.
    ``silence_weight`` weighs the rows taught silence in each lesson (``Brain.lesson(weights=)``),
    and ``soft`` teaches the step after a word step the word at that share (``targets=``)."""

    silence_weight = 1.0
    soft = 0.0

    kind = "brain"
    hold = 1  # steps a word stays due (``held``)

    def __init__(self, brain: Any, arm: str) -> None:
        self.brain, self.arm = brain, arm
        self.work = {
            "phrases_taught": 0,
            "lessons": 0,
            "refused_lessons": 0,
            "exhausted_acts": 0,
            "train_seconds": 0.0,
            "read_seconds": 0.0,
        }

    def parameters(self) -> int:
        return int(self.brain.parameters())

    def hear(
        self,
        x: np.ndarray,
        target: np.ndarray | None = None,
        refs: list[tuple[int, ...]] | None = None,
        row_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Actions (rows, steps); with ``target`` each step is taught before the act, each row
        at its ``row_weights`` (one where not given)."""
        from cadence import LearningPhaseError

        started = time.process_time()
        rows, steps = x.shape[:2]
        actions = np.full((rows, steps), -1, np.int64)
        soft = None
        if target is not None:
            target = held(target, self.hold)
            if self.soft:
                soft = soft_targets(np.asarray(target), self.soft)
        self.brain.reset()
        for k in range(steps):
            if target is not None:
                self.work["lessons"] += 1
                labels = np.asarray(target[:, k], dtype=np.int64)
                taught: dict[str, Any] = {}
                if self.silence_weight != 1.0 or row_weights is not None:
                    weights = np.where(labels == SILENCE, self.silence_weight, 1.0)
                    if row_weights is not None:
                        weights = weights * row_weights
                    taught["weights"] = weights
                if soft is not None:
                    taught["targets"] = soft[:, k]
                else:
                    taught["labels"] = labels
                try:
                    self.brain.lesson(x[:, k], **taught)
                except (LearningPhaseError, RuntimeError):
                    self.work["refused_lessons"] += 1
            try:
                actions[:, k] = np.asarray(self.brain.act(x[:, k], greedy=True)).reshape(rows)
            except RuntimeError:
                self.work["exhausted_acts"] += 1
        spent = time.process_time() - started
        if target is not None:
            self.work["phrases_taught"] += rows
            self.work["train_seconds"] += spent
        else:
            self.work["read_seconds"] += spent
        return actions

    def reader(self) -> BrainLearner:
        """A copy to read on: its fields take no step and its lessons never reach this life."""
        clone = copy.deepcopy(self.brain)
        if clone.fields is not None:
            clone.fields.learning = False
        out = BrainLearner(clone, self.arm)
        out.hold, out.silence_weight, out.soft = self.hold, self.silence_weight, self.soft
        return out

    def set_rate(self, scale: float) -> None:
        """Every rate of the lesson's learner at ``scale`` of the life's."""
        config = self.brain.learner.config
        base = getattr(self, "_base_config", None) or config
        self._base_config = base
        eta_bias = base.eta_bias if base.eta_bias is not None else base.eta / 10
        self.brain.retune(
            learning=dataclasses.replace(base, eta=base.eta * scale, eta_bias=eta_bias * scale)
        )

    def copy(self) -> BrainLearner:
        out = BrainLearner(copy.deepcopy(self.brain), self.arm)
        out.work = dict(self.work)
        out.hold, out.silence_weight, out.soft = self.hold, self.silence_weight, self.soft
        return out


class ControlLearner:
    """A baseline with the brain's interface."""

    kind = "control"
    hold = 1

    def __init__(self, model: Any) -> None:
        self.model = model
        self.base_rate = ct.RATE

    @property
    def work(self) -> dict[str, Any]:
        return dict(self.model.work)

    def parameters(self) -> int:
        return int(self.model.parameters())

    def hear(
        self,
        x: np.ndarray,
        target: np.ndarray | None = None,
        refs: list[tuple[int, ...]] | None = None,
        row_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        if target is not None:
            assert refs is not None
            if row_weights is not None:  # a row at weight zero adds no gradient: leave it out
                keep = np.flatnonzero(row_weights > 0)
                x, target, refs = x[keep], target[keep], [refs[int(i)] for i in keep]
            self.model.teach(x, held(target, self.hold), refs, END)
            return np.empty((len(x), 0), np.int64)
        return np.asarray(self.model.actions(x, SILENCE))

    def reader(self) -> ControlLearner:
        return self

    def set_rate(self, scale: float) -> None:
        self.model.set_rate(self.base_rate * scale)

    def copy(self) -> ControlLearner:
        out = ControlLearner(self.model.copy())
        out.hold = self.hold
        return out


def control(name: str, founder: int, parameters: int) -> ControlLearner:
    kind, loss = CONTROLS[name]
    width = ct.width_for(kind, ACTIONS + (1 if loss == "ctc" else 0), parameters)
    return ControlLearner(ct.Sequence(kind, loss, width, ACTIONS, seed=founder))


# ------------------------------------------------------------------------------------ readings


def held(target: np.ndarray, hold: int) -> np.ndarray:
    """Per-step targets with each word due at ``hold`` steps: the step whose window first holds
    the word's last frame and the silent steps after it, up to ``hold`` in all. A phrase's words
    differ, so a word said at two steps in a row reads as one (``decode(collapse=True)``)."""
    if hold <= 1:
        return target
    out = np.array(target, copy=True)
    for r, row in enumerate(np.asarray(target)):
        for k in np.flatnonzero((row >= 0) & (row < SILENCE)):
            for j in range(int(k) + 1, min(int(k) + hold, len(row))):
                if row[j] != SILENCE:
                    break
                out[r, j] = row[k]
    return out


def decode(actions: np.ndarray, collapse: bool = False) -> list[dict[str, Any]]:
    """Each row's words up to the first ``end``, and whether it ended; with ``collapse`` a word
    said at consecutive steps reads once."""
    out = []
    for row in np.asarray(actions):
        words: list[int] = []
        ended = False
        previous = -1
        for a in row:
            a = int(a)
            if a == END:
                ended = True
                break
            if 0 <= a < SILENCE and not (collapse and a == previous):
                words.append(a)
            previous = a
        out.append({"words": tuple(words), "ended": ended})
    return out


def edits(hyp: tuple[int, ...], ref: tuple[int, ...]) -> tuple[int, int, int]:
    """Substitutions, insertions and deletions of the cheapest alignment."""
    n, m = len(ref), len(hyp)
    cost = np.zeros((n + 1, m + 1), np.int64)
    kind = np.zeros((n + 1, m + 1, 3), np.int64)
    cost[:, 0], cost[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        kind[i, 0] = (0, 0, i)
    for j in range(1, m + 1):
        kind[0, j] = (0, j, 0)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = ref[i - 1] == hyp[j - 1]
            options = (
                (
                    cost[i - 1, j - 1] + (0 if same else 1),
                    kind[i - 1, j - 1] + (0 if same else np.array((1, 0, 0))),
                ),
                (cost[i, j - 1] + 1, kind[i, j - 1] + np.array((0, 1, 0))),
                (cost[i - 1, j] + 1, kind[i - 1, j] + np.array((0, 0, 1))),
            )
            best = min(options, key=lambda o: o[0])
            cost[i, j], kind[i, j] = best[0], best[1]
    s, ins, d = kind[n, m]
    return int(s), int(ins), int(d)


def groups(
    phrases: Any, rows: np.ndarray, batch: int, rng: np.random.Generator | None = None
) -> list[np.ndarray]:
    """Batches of rows of equal steps; shuffled within and across groups with ``rng``."""
    out: list[np.ndarray] = []
    for steps in sorted(np.unique(phrases.steps[rows])):
        members = rows[phrases.steps[rows] == steps]
        if rng is not None:
            members = rng.permutation(members)
        out.extend(members[i : i + batch] for i in range(0, len(members), batch))
    if rng is not None:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def read(
    learner: Any, phrases: Any, rows: np.ndarray, hearing: Hearing, batch: int = BATCH
) -> dict[str, Any]:
    """Word accuracy (1 minus the word error rate), exact phrases, and at the word steps the
    share said right, another word, silence or ``end`` (the per-step learners only)."""
    reader = learner.reader()
    started = time.process_time()
    s = ins = d = words = exact = ended = 0
    at_word = np.zeros(4)
    for chunk in groups(phrases, np.asarray(rows), batch):
        actions = reader.hear(stream(phrases, chunk, hearing))
        targets = phrases.target[chunk, : actions.shape[1]]
        for j, r in enumerate(chunk):
            ref = phrases.reference(int(r))
            collapse = learner.hold > 1 or getattr(learner, "soft", 0.0) > 0
            got = decode(actions[j : j + 1], collapse=collapse)[0]
            a, b, c = edits(got["words"], ref)
            s, ins, d = s + a, ins + b, d + c
            words += len(ref)
            exact += int(got["words"] == ref and got["ended"])
            ended += int(got["ended"])
        is_word = targets < SILENCE
        said = actions[is_word]
        right = said == targets[is_word]
        at_word += (
            int(right.sum()),
            int(((said < SILENCE) & ~right).sum()),
            int((said == SILENCE).sum()),
            int(((said == END) | (said < 0)).sum()),
        )
    n = len(rows)
    out = {
        "phrases": n,
        "words": words,
        "word_accuracy": 1.0 - (s + ins + d) / max(words, 1),
        "exact": exact / max(n, 1),
        "substitutions": s,
        "insertions": ins,
        "deletions": d,
        "ended": ended / max(n, 1),
        "seconds": time.process_time() - started,
    }
    if learner.kind == "brain" or getattr(learner.model, "loss", "") == "step":
        total = max(at_word.sum(), 1)
        out["at_word"] = {
            "right": at_word[0] / total,
            "other": at_word[1] / total,
            "silence": at_word[2] / total,
            "end": at_word[3] / total,
        }
    return out


def train(learner: Any, phrases: Any, batches: list[np.ndarray], hearing: Hearing) -> None:
    for chunk in batches:
        x = stream(phrases, chunk, hearing)
        target = phrases.target[chunk, : x.shape[1]]
        learner.hear(x, target, [phrases.reference(int(r)) for r in chunk])


def adapt(
    learner: Any,
    own: Any,
    evaluate: Any,
    retention: Any,
    retention_rows: np.ndarray,
    hearing: Hearing,
    minutes: list[float],
    *,
    replay: Any = None,
    replays: int = 0,
    rate: float = 1.0,
    seed: int = 0,
    held: float | None = None,
    rehearsal: Any = None,
) -> list[dict[str, Any]]:
    """A new speaker's labelled phrases one at a time, in order, read at each of ``minutes``
    (cumulative seconds of the speaker's phrases heard) on the speaker's evaluation phrases and
    on ``retention``. With ``replay``, each phrase is taught beside ``replays`` phrases of the base
    training set of equal steps; with ``held`` they are heard beside it and taught at that weight
    (``Brain.lesson(weights=)``; at zero the lesson's update is the corrected phrase's alone, while
    the brain acts on every row). With ``rehearsal`` (a ``cadence.speech.correct.Rehearsal``),
    each phrase is taught by the library's correction instead: its labels on every row
    (``learning_surprise=False``), or the soft targets of a life taught them, beside the bank's
    phrases at the rehearsal's weight."""
    life = learner.copy()
    life.set_rate(rate)
    if rehearsal is not None:  # the recipe's lessons: every row taught
        life.brain.retune(learning_surprise=False)
    rng = np.random.default_rng(seed)
    every = np.arange(len(evaluate))
    heard, taught, seconds_taught = 0.0, 0, 0.0
    out: list[dict[str, Any]] = []
    pending = sorted(minutes)
    row = 0
    while pending:
        if heard >= 60.0 * pending[0] - 1e-9 or row >= len(own):
            reading = {
                "minutes": pending.pop(0),
                "heard_seconds": heard,
                "phrases": taught,
                "own": read(life, evaluate, every, hearing),
                "retention": read(life, retention, retention_rows, hearing),
                "seconds_per_correction": seconds_taught / max(taught, 1),
            }
            out.append(reading)
            continue
        chunk = np.array([row])
        x = stream(own, chunk, hearing)
        target = own.target[chunk, : x.shape[1]]
        refs = [own.reference(row)]
        if replay is not None and replays:
            same = np.flatnonzero(replay.steps == own.steps[row])
            if len(same):
                extra = rng.choice(same, min(replays, len(same)), replace=False)
                x = np.concatenate([x, stream(replay, extra, hearing)])
                target = np.concatenate([target, replay.target[extra, : x.shape[1]]])
                refs += [replay.reference(int(r)) for r in extra]
        started = time.process_time()
        if rehearsal is not None:
            if getattr(life, "soft", 0.0):  # a life taught soft targets is corrected by them
                rehearsal.correct(life.brain, x[0], targets=soft_targets(target, life.soft)[0])
            else:
                rehearsal.correct(life.brain, x[0], target[0])
        elif held is not None and len(x) > 1:
            weights = np.full(len(x), float(held))
            weights[0] = 1.0
            life.hear(x, target, refs, row_weights=weights)
        else:
            life.hear(x, target, refs)
        seconds_taught += time.process_time() - started
        heard += float(own.seconds[row])
        taught += 1
        row += 1
    return out


# ------------------------------------------------------------------------------------ probes


def word_steps(
    phrases: Any, rows: np.ndarray, hearing: Hearing, limit: int
) -> tuple[np.ndarray, np.ndarray]:
    """Heard windows at word-target steps and their words, at most ``limit``."""
    xs, ys = [], []
    for r in rows:
        steps = int(phrases.steps[r])
        mel = phrases.logmel[r : r + 1].astype(np.float32)
        for k in range(1, steps + 1):
            t = int(phrases.target[r, k - 1])
            if t < SILENCE:
                xs.append(hearing(ph.window(mel, k))[0])
                ys.append(t)
        if len(ys) >= limit:
            break
    return np.asarray(xs[:limit]), np.asarray(ys[:limit])


def ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    panels: dict[str, tuple[np.ndarray, np.ndarray]],
    classes: int = SILENCE,
) -> dict[str, Any]:
    """A linear readout by ridge regression on one-hot words, its penalty chosen on the last
    quarter of the fitting rows, read on each panel."""
    mean = train_x.mean(axis=0)
    scale = float(np.sqrt(np.mean((train_x - mean) ** 2))) or 1.0
    x = (train_x - mean) / scale
    y = np.eye(classes)[train_y] - 1.0 / classes
    cut = (3 * len(x)) // 4

    def fit(rows: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
        if rows.shape[1] <= rows.shape[0]:
            gram = rows.T @ rows
            return np.asarray(
                np.linalg.solve(gram + penalty * len(rows) * np.eye(len(gram)), rows.T @ targets)
            )
        gram = rows @ rows.T
        alpha = np.linalg.solve(gram + penalty * len(rows) * np.eye(len(rows)), targets)
        return np.asarray(rows.T @ alpha)

    scores = {}
    for penalty in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        w = fit(x[:cut], y[:cut], penalty)
        scores[penalty] = float(np.mean(np.argmax(x[cut:] @ w, axis=1) == train_y[cut:]))
    best = max(scores, key=lambda p: (scores[p], p))
    w = fit(x, y, best)
    return {
        "penalty": best,
        "validation": scores[best],
        "width": int(train_x.shape[1]),
        "panels": {
            name: float(np.mean(np.argmax((px - mean) / scale @ w, axis=1) == py))
            for name, (px, py) in panels.items()
        },
    }


def pools(brain: Any, x: np.ndarray, chunk: int = 64) -> np.ndarray:
    return np.concatenate(
        [np.asarray(brain.observation_code(x[i : i + chunk])) for i in range(0, len(x), chunk)]
    )


def association(brain: Any, x: np.ndarray, chunk: int = 64) -> np.ndarray:
    """The association's settled state of each window from rest, without memories."""
    index = np.asarray(brain.connectome.populations["association"])
    out = []
    for i in range(0, len(x), chunk):
        state = brain._qualified(
            brain.stimulus(x[i : i + chunk], memory=False), operation="predict"
        )
        out.append(np.atleast_2d(np.asarray(state.activation))[:, index])
    return np.concatenate(out)
