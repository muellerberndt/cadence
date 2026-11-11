"""Lives in the pixel chamber (cadence#284): a brain or the convolutional baseline learns the
screen-dependent policy from rendered frames, and its panels are read at each budget.

An arm names what the brain sees and how it is composed (``ARMS``). A life is one arm, one
founder, one condition and one frame shape:

- ``lesson``: every batch of ``rows`` frames is one ``lesson(frames, labels)`` (the
  baseline: one cross-entropy step).
- ``reward``: ``rows`` streams through ``step``; the reward of each action (1 right, 0 wrong)
  arrives with the next frame, each frame an episode of its own (``done``).

At each budget (frames seen) the panels are read on the life without changing it, by
``predict`` (each frame on its own, from rest, without memories). The readings at the last
budget add the probes (``probe``): a linear readout fitted by ridge regression on a fixed set
of training-range frames, read on every panel, from the representation the arm's cortex reads
(``code``) and from the association region's settled state (``association``). The probes are
readings of what a representation holds and never enter a life.

Every life reports its parameters and its analytic multiply-accumulates (``macs``): a brain's
lesson as its phases' sweeps of every synapse and the fields' reading and learning of the
frame, a read as the settle's sweeps and the fields' reading; the baseline's forward pass, and
a training step as three of them (a backward pass at twice the forward).

    PYTHONPATH=src python benchmarks/pixels/pixels.py --arms visual pooled conv \\
        --founders 0 1 --budgets 1024 4096 --out results/step0.jsonl --workers 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    key = f"pixels_{name}"  # a name of this chamber's own: other benchmarks have a chamber.py
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


ch = _load("chamber")
cn = _load("convnet")

POOL = 4  # the coarse fixed code's block
PANEL = 256  # frames per panel
PROBE = 2048  # frames the probes are fitted on
PROBE_RIDGE = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
CHUNK = 64  # frames per predict call
ARMS: dict[str, dict[str, Any]] = {
    # the visual cortex as composed: 8 maps of 3 by 3 fields at stride one
    "visual": {"sees": "image", "compose": {}},
    # the raw pixels as a vector
    "pixels": {"sees": "flat", "compose": {}},
    # the coarse fixed code: block means
    "pooled": {"sees": "pooled", "compose": {}},
    # the fixed random sparse code of 1,024 units over the pixels
    "random_code": {"sees": "flat", "compose": {"code": 1024}},
    # the learned code of 128 units over the pixels at unit length
    "learned_code": {"sees": "unit", "compose": {"learned_code": 128}},
    # learned receptive fields (``learned_fields``), at the normalized teaching point
    "fields": {"sees": "image", "compose": {"learned_fields": 64, "teaching": "normalized"}},
    # the same brain with its fields held at their draw from birth
    "fields_drawn": {"sees": "image", "hold": True,
                     "compose": {"learned_fields": 64, "teaching": "normalized"}},
    # colour controls of the fields: every field reads all three channels, or one channel each
    "fields_joint": {"sees": "image", "colour": "joint",
                     "compose": {"learned_fields": 64, "teaching": "normalized"}},
    "fields_channels": {"sees": "image", "colour": "channels",
                        "compose": {"learned_fields": 64, "teaching": "normalized"}},
    # the fields' three pool rows read through one set of synapses (``tied``, cadence#308):
    # a field's response in a pool column drives the association alike in whichever row it lies
    "fields_tied": {"sees": "image",
                    "compose": {"learned_fields": 64, "teaching": "normalized",
                                "tied": {("sensory", "association"): (3, 1)}}},
    # the baseline at the hidden width of the Atari network
    "conv": {"sees": "image", "conv": {"hidden": 256}},
    # the baseline at the parameter count of the ``fields`` arm's brain
    "conv_matched": {"sees": "image", "conv": {"hidden": "fields"}},
    # the baseline at the parameter count of the ``fields_tied`` arm's brain
    "conv_matched_tied": {"sees": "image", "conv": {"hidden": "fields_tied", "least": 1}},
}


def seen(arm: str, batch: np.ndarray) -> np.ndarray:
    """What the arm's brain receives of a batch of frames."""
    kind = ARMS[arm]["sees"]
    if kind in ("image", "flat"):
        return ch.flat(batch)
    if kind == "unit":
        return ch.unit(batch)
    if kind == "pooled":
        return ch.pooled(batch, POOL)
    raise ValueError(kind)


def inputs_of(arm: str, shape: tuple[int, int, int]) -> Any:
    kind = ARMS[arm]["sees"]
    if kind == "image":
        return shape
    if kind == "pooled":
        return (shape[0] // POOL) * (shape[1] // POOL) * shape[2]
    return int(np.prod(shape))


def compose(arm: str, shape: tuple[int, int, int], founder: int, condition: str,
            extra: dict[str, Any] | None = None) -> Any:
    from cadence import Brain

    settings: dict[str, Any] = {"recall": False, "backend": "cpu"}
    settings.update(ARMS[arm]["compose"])
    if condition == "reward":
        settings["arousal"] = True
    settings.update(extra or {})
    return Brain.compose(inputs_of(arm, shape), ch.ACTIONS, seed=founder, **settings)


class Life:
    """One arm's learner: a brain, or the baseline, with one interface for the life."""

    def __init__(self, arm: str, shape: tuple[int, int, int], founder: int, condition: str,
                 extra: dict[str, Any] | None = None) -> None:
        self.arm, self.shape, self.condition = arm, shape, condition
        self.conv = "conv" in ARMS[arm]
        if self.conv:
            hidden = ARMS[arm]["conv"]["hidden"]
            if isinstance(hidden, str):  # matched to that arm's brain
                count = compose(hidden, shape, founder, condition).parameters()
                hidden = cn.hidden_for(
                    shape, ch.ACTIONS, count, ARMS[arm]["conv"].get("least", 16)
                )
            self.learner: Any = cn.ConvActorCritic(
                shape, ch.ACTIONS, int(hidden), founder, condition
            )
        else:
            self.learner = compose(arm, shape, founder, condition, extra)
            if ARMS[arm].get("hold"):
                self.learner.fields.learning = False
            if ARMS[arm].get("colour"):  # the same stage reading colour another way, from birth
                held = self.learner.fields
                self.learner.fields = type(held)(
                    held.height, held.width, held.channels, held.features, seed=held.seed,
                    colour=ARMS[arm]["colour"],
                )
        self.pending: np.ndarray | None = None  # the last actions awaiting their reward
        self.labels: np.ndarray | None = None
        self.paid: list[float] = []
        # the work of the lessons and of the reads: frames, and a brain's rows times sweeps
        self.work = {"lesson_frames": 0, "lesson_row_sweeps": 0.0, "read_frames": 0,
                     "read_row_sweeps": 0.0}

    def parameters(self) -> int:
        return int(self.learner.parameters())

    def predict(self, batch: np.ndarray) -> np.ndarray:
        out = []
        for start in range(0, len(batch), CHUNK):
            part = batch[start : start + CHUNK]
            out.append(np.asarray(self.learner.predict(
                part if self.conv else seen(self.arm, part))).reshape(-1))
            self.work["read_frames"] += len(part)
            if not self.conv:
                settled = self.learner.last_settlement or {}
                self.work["read_row_sweeps"] += len(part) * float(settled.get("steps", 0))
        return np.concatenate(out)

    def macs(self) -> dict[str, float] | None:
        """Analytic multiply-accumulates per frame of the life so far, ``lesson`` (a frame
        taught) and ``read`` (a frame answered), and the lessons' total; None for a life under
        reward, whose steps report no sweeps."""
        if self.condition != "lesson":
            return None
        taught, read = self.work["lesson_frames"], self.work["read_frames"]
        if self.conv:
            forward = float(cn.macs(self.shape, ch.ACTIONS, self.learner.hidden))
            lesson, answer = 3.0 * forward, forward
        else:
            brain = self.learner
            synapses = float(brain.connectome.synapses)
            fields = {"code": 0, "learn": 0} if brain.fields is None else brain.fields.macs()
            learns = brain.fields is not None and brain.fields.learning
            lesson = fields["code"] + (fields["learn"] if learns else 0) + (
                synapses * self.work["lesson_row_sweeps"] / taught if taught else 0.0
            )
            answer = fields["code"] + (
                synapses * self.work["read_row_sweeps"] / read if read else 0.0
            )
        return {"lesson": float(lesson), "read": float(answer), "lessons": float(lesson * taught)}

    def experience(self, batch: np.ndarray, labels: np.ndarray) -> None:
        if self.condition == "lesson":
            self.work["lesson_frames"] += len(batch)
            if self.conv:
                self.learner.lesson(batch, labels)
            else:
                report = self.learner.lesson(seen(self.arm, batch), labels)
                self.work["lesson_row_sweeps"] += float(report.get("total_row_sweeps", 0.0))
            return
        if self.conv:
            actions = self.learner.act(batch)
            rewards = (actions == labels).astype(float)
            self.learner.reward(batch, actions, rewards)
            self.paid.extend(rewards.tolist())
            return
        x = seen(self.arm, batch)
        if self.pending is None:
            actions = self.learner.step(x)
        else:
            assert self.labels is not None
            rewards = (self.pending == self.labels).astype(float)
            self.paid.extend(rewards.tolist())
            actions = self.learner.step(x, reward=rewards, done=np.ones(len(x), bool))
        self.pending, self.labels = np.asarray(actions).reshape(-1), labels

    def states(self, batch: np.ndarray, populations: tuple[str, ...]) -> dict[str, np.ndarray]:
        """The settled activity of named populations, each frame from rest without memories."""
        brain = self.learner
        index = {name: np.asarray(brain.connectome.populations[name]) for name in populations}
        out: dict[str, list[np.ndarray]] = {name: [] for name in populations}
        for start in range(0, len(batch), CHUNK):
            x = seen(self.arm, batch[start : start + CHUNK])
            state = brain._qualified(brain.stimulus(x, memory=False), operation="predict")
            activation = np.atleast_2d(np.asarray(state.activation))
            for name in populations:
                out[name].append(activation[:, index[name]])
        return {name: np.concatenate(parts) for name, parts in out.items()}

    def code(self, batch: np.ndarray) -> np.ndarray:
        """The representation the arm's cortex reads: the visual maps, the code, or the input."""
        if self.conv:
            torch = cn._torch()
            with torch.no_grad():
                feats = self.learner.body[:5](self.learner._input(batch))
            return np.asarray(feats.numpy())
        if self.arm == "visual":
            return self.states(batch, ("visual/output",))["visual/output"]
        brain = self.learner
        if (brain.learned_code is not None or brain.sparse_code is not None
                or brain.fields is not None):
            return np.concatenate([
                np.asarray(brain.observation_code(seen(self.arm, batch[s : s + CHUNK])))
                for s in range(0, len(batch), CHUNK)
            ])
        return seen(self.arm, batch)

    def association(self, batch: np.ndarray) -> np.ndarray:
        if self.conv:
            torch = cn._torch()
            with torch.no_grad():
                return np.asarray(self.learner.body(self.learner._input(batch)).numpy())
        return self.states(batch, ("association",))["association"]


def ridge_probe(train: np.ndarray, labels: np.ndarray, panels: dict[str, np.ndarray],
                panel_labels: dict[str, np.ndarray]) -> dict[str, Any]:
    """A linear readout by ridge regression on one-hot targets, in the dual form; the penalty
    is chosen on the last quarter of the fitting frames, then refitted on all of them."""
    mean = train.mean(axis=0)
    scale = float(np.sqrt(np.mean((train - mean) ** 2))) or 1.0

    def standard(x: np.ndarray) -> np.ndarray:
        return (x - mean) / scale

    x = standard(train)
    y = np.eye(ch.ACTIONS)[labels] - 1.0 / ch.ACTIONS
    cut = (3 * len(x)) // 4

    def fit(rows: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
        gram = rows @ rows.T
        alpha = np.linalg.solve(gram + penalty * len(rows) * np.eye(len(rows)), targets)
        return np.asarray(rows.T @ alpha)

    scores = {}
    for penalty in PROBE_RIDGE:
        w = fit(x[:cut], y[:cut], penalty)
        scores[penalty] = ch.accuracy(np.argmax(x[cut:] @ w, axis=1), labels[cut:])
    best = max(PROBE_RIDGE, key=lambda p: (scores[p], p))
    w = fit(x, y, best)
    readings = {
        name: ch.accuracy(np.argmax(standard(feats) @ w, axis=1), panel_labels[name])
        for name, feats in panels.items()
    }
    return {"penalty": best, "validation": scores[best], "width": int(train.shape[1]),
            "active": float(np.mean(train > 0)), "silent_rows": int(np.sum(~(train > 0).any(1))),
            "panels": readings}


def read_panels(life: Life, panels: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    return {name: ch.accuracy(life.predict(frames), labels)
            for name, (frames, labels) in panels.items()}


def probe_set(founder: int, shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(ch.founder_seed("panel", founder, f"probe:{shape}"))
    aspect = shape[1] / shape[0]
    scenes = [ch.draw_scene(rng, ch.TRAINED, aspect) for _ in range(PROBE)]
    return ch.frames(scenes, shape, rng), np.array([s.label for s in scenes])


def life(spec: dict[str, Any]) -> dict[str, Any]:
    warnings.simplefilter("ignore", RuntimeWarning)
    arm, founder, condition = spec["arm"], int(spec["founder"]), spec["condition"]
    shape = tuple(ch.SHAPES[spec.get("shape", "grey84")])
    rows, budgets = int(spec.get("rows", 8)), [int(b) for b in spec["budgets"]]
    control = spec.get("control")
    started = time.process_time()
    built = Life(arm, shape, founder, condition, spec.get("extra"))
    schedule = ch.Schedule(founder, shape, rows=rows,
                           control=control if control in ("blank", "shuffled") else None)
    size = int(spec.get("panel", PANEL))
    panels = {name: ch.panel(founder, name, shape, size) for name in ch.PANELS}
    # beside panels of another size, the panels of the first declaration (256 frames)
    declared = None if size == PANEL else {
        name: ch.panel(founder, name, shape, PANEL) for name in ch.PANELS
    }
    readings: list[dict[str, Any]] = []
    seen_frames, learn_seconds = 0, 0.0
    frozen = control == "frozen"
    batches = schedule.batches(max(budgets) // rows)
    for budget in budgets:
        while seen_frames < budget:
            batch, labels = next(batches)
            if not frozen:
                t = time.process_time()
                built.experience(batch, labels)
                learn_seconds += time.process_time() - t
            seen_frames += len(batch)
        t = time.process_time()
        reading = {"frames": seen_frames, "panels": read_panels(built, panels)}
        reading["read_seconds_per_frame"] = (time.process_time() - t) / (size * len(ch.PANELS))
        if declared is not None:
            reading["panels_256"] = read_panels(built, declared)
        reading["learn_seconds_per_frame"] = learn_seconds / max(1, seen_frames)
        if built.paid:
            reading["paid_last_1024"] = float(np.mean(built.paid[-1024:]))
        readings.append(reading)
    out: dict[str, Any] = {
        "arm": arm, "founder": founder, "condition": condition, "shape": list(shape),
        "rows": rows, "control": control, "extra": spec.get("extra"), "panel": size,
        "parameters": built.parameters(), "readings": readings, "macs": built.macs(),
    }
    if built.conv:
        hidden = built.learner.hidden
        out["macs_per_frame"] = cn.macs(shape, ch.ACTIONS, hidden)
    if spec.get("probe"):
        fit_frames, fit_labels = probe_set(founder, shape)
        probes = {}
        for kind in ("code", "association"):
            read = built.code if kind == "code" else built.association
            probes[kind] = ridge_probe(
                read(fit_frames), fit_labels,
                {name: read(frames) for name, (frames, _) in panels.items()},
                {name: labels for name, (_, labels) in panels.items()},
            )
        out["probes"] = probes
    out["process_seconds"] = time.process_time() - started
    return out


def _run(spec: dict[str, Any]) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
    try:
        return {"spec": spec, **life(spec)}
    except Exception as exc:  # a failed life is a reading too
        return {"spec": spec, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["visual"])
    parser.add_argument("--founders", nargs="+", type=int, default=[0])
    parser.add_argument("--condition", default="lesson", choices=("lesson", "reward"))
    parser.add_argument("--shape", default="grey84", choices=tuple(ch.SHAPES))
    parser.add_argument("--budgets", nargs="+", type=int, default=[1024, 4096])
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--control", default=None,
                        choices=(None, "blank", "shuffled", "frozen"))
    parser.add_argument("--panel", type=int, default=PANEL)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--extra", default=None, help="JSON of compose settings")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    specs = [
        {"arm": arm, "founder": f, "condition": args.condition, "shape": args.shape,
         "budgets": args.budgets, "rows": args.rows, "control": args.control,
         "panel": args.panel, "probe": args.probe,
         "extra": json.loads(args.extra) if args.extra else None}
        for arm in args.arms for f in args.founders
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as handle:
        if args.workers > 1:
            with get_context("spawn").Pool(args.workers) as pool:
                for result in pool.imap_unordered(_run, specs):
                    handle.write(json.dumps(result) + "\n")
                    handle.flush()
        else:
            for spec in specs:
                handle.write(json.dumps(_run(spec)) + "\n")
                handle.flush()


if __name__ == "__main__":
    main()
