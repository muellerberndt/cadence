"""Where the recall chamber's lesson lands, and what a founder carries before any lesson.

Two development diagnostics for [issue 84](https://github.com/muellerberndt/cadence/issues/84),
on the finite recall chamber's declared recipe (``vanished_cue.make_brain`` at trace amplitude
0.3 and decay 0.8, the ``every`` teaching rule). They measure; they freeze no gate and change
no runtime.

``--untrained`` asks what a founder carries before any lesson. For every founder, a fresh brain
meets START, then WRITE with token 0 in one row and token 1 in its paired row, then neutral
events, then QUERY, with the chamber's own frozen episodes. It records the paired distance of
the association state and of the working trace at every event. The lessons of the chamber
arrive at QUERY, where the rows' current inputs are identical: whatever separates the pair
there came through the trace, and the trace is a decaying copy of earlier association states
that no lesson differentiates through time. If the founders that learned recall in the
historical freezes are the founders whose untrained association layer separated the two
tokens at WRITE, acquisition is bounded by a fixed random representation, not by the lessons.

``--credit`` trains founders with the chamber's rule for ``--repeats`` repeats and records,
per lesson, the step every projection received (sensory to association, association to
motor and back, prefrontal to association), the sign consistency of those steps over the
run, the paired association and trace distances at QUERY, and every ``--every`` repeats the
recall of a saved and reloaded copy on the clean, replacement and order conditions of the
frozen test episodes. The living brain is never read for a measurement. This shows where the
credit of a lesson lands, whether later lessons keep pushing an already learned mapping, and
when recall comes and goes.

Run ``python benchmarks/recall/credit_landing.py --help``.
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
import gzip  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

import cadence as cd  # noqa: E402
from cadence.learning import LearningPhaseError  # noqa: E402
from cadence.receipts import Receipt, canonical_json  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = _load("finite_horizon_inputs")
chamber = _load("finite_horizon")

SCHEMA = "recall-credit-landing/1"
PROTOCOL = HERE / "protocol-finite-2.json"
# founders of the historical freezes, and whether they acquired a nonnegative recall horizon
HISTORICAL = {
    301: False,
    302: False,
    303: True,  # recall/1, amplitude 1.0: 303 learned
    304: False,
    305: False,
    306: True,  # recall/2, amplitude 0.3, every rule
    307: False,
    308: False,
    309: True,  # recall/3, amplitude 0.3, surprise rule
}
REGIONS = ("sensory", "association", "prefrontal", "motor")
EVALUATED = ("clean-0", "clean-1", "clean-2", "distractor-1", "replacement-1", "order-latest-1")


def load_protocol() -> dict[str, Any]:
    return json.loads(PROTOCOL.read_text())


def make_brain(protocol: dict[str, Any], seed: int, amplitude: float, decay: float) -> cd.Brain:
    return chamber.make_brain(protocol, seed, decay=decay, amplitude=amplitude)


def groups_of(brain: cd.Brain) -> dict[str, np.ndarray]:
    w = brain.connectome
    region = np.full(w.n, -1)
    for index, name in enumerate(REGIONS):
        region[np.asarray(w.populations[name])] = index
    assert (region >= 0).all()
    pre, post = region[w.pre], region[w.post]
    groups = {}
    for a, source in enumerate(REGIONS):
        for b, target in enumerate(REGIONS):
            mask = (pre == a) & (post == b)
            if mask.any():
                groups[f"{source}->{target}"] = mask
    return groups


def paired(x: np.ndarray) -> float:
    """Mean distance between the two rows of each pair (rows 2k and 2k+1 carry opposite cues)."""
    return float(np.linalg.norm(x[::2] - x[1::2], axis=1).mean())


def association_and_trace(brain: cd.Brain) -> tuple[np.ndarray, np.ndarray]:
    state = brain.basal_ganglia.state
    assert state is not None and brain.working_memory is not None
    activation = np.atleast_2d(np.asarray(state.activation))
    return activation[:, brain.association_index], brain.working_memory.trace.copy()


# ----------------------------------------------------------------------------- untrained


def untrained(seed: int, protocol: dict[str, Any], amplitude: float, decay: float) -> dict:
    """A fresh founder meets the frozen clean-0/1/2 and distractor-1 test episodes once each,
    with no lesson; the paired association and trace distances at every event are recorded,
    and the motor margin at QUERY."""
    frozen = chamber.episode_fixture(seed, {"train": 1, "test": 1})
    brain = make_brain(protocol, seed, amplitude, decay)
    readings: dict[str, Any] = {"seed": seed, "historical": HISTORICAL.get(seed), "episodes": {}}
    sensory = brain.connectome.populations["sensory"]
    for index in range(len(frozen["test/condition"])):
        name = chamber.condition_name(frozen, "test", index)
        if name not in ("clean-0", "clean-1", "clean-2", "distractor-1", "replacement-1"):
            continue
        arrays = chamber.episode_arrays(frozen, "test", index)
        events = []
        for x in arrays["observations"]:
            brain.act(x, greedy=True)
            association, trace = association_and_trace(brain)
            state = brain.basal_ganglia.state
            assert state is not None
            activation = np.atleast_2d(np.asarray(state.activation))
            motor = activation[:, brain.motor_index]
            labels = arrays["labels"]
            margin = (
                motor[np.arange(len(labels)), labels] - motor[np.arange(len(labels)), 1 - labels]
            )
            events.append(
                {
                    "paired_input": paired(x[:, : len(sensory)]),
                    "paired_association": paired(association),
                    "association_norm": float(np.linalg.norm(association, axis=1).mean()),
                    "paired_trace": paired(trace),
                    "motor_margin": float(margin.mean()),
                    "motor_level": float(np.abs(motor).mean()),
                }
            )
        readings["episodes"][name] = events
    return readings


# -------------------------------------------------------------------------------- credit


class Tally:
    def __init__(self, synapses: int) -> None:
        self.total = np.zeros(synapses)
        self.square = np.zeros(synapses)
        self.count = 0

    def add(self, step: np.ndarray) -> None:
        self.total += step
        self.square += step * step
        self.count += 1

    def by_group(self, groups: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        out = {}
        if not self.count:
            return out
        rms = np.sqrt(self.square / self.count)
        moved = rms > 0
        consistency = np.zeros(len(rms))
        consistency[moved] = np.abs(self.total[moved]) / np.sqrt(self.count * self.square[moved])
        for name, mask in groups.items():
            part = mask & moved
            out[name] = {
                "consistency": float(consistency[part].mean()) if part.any() else 0.0,
                "rms_step": float(rms[mask].mean()),
                "net": float(np.abs(self.total[mask]).mean()),
                "moved": int(part.sum()),
                "synapses": int(mask.sum()),
            }
        return out


def evaluate(brain: cd.Brain, frozen: dict[str, np.ndarray]) -> dict[str, float]:
    """Recall of a saved and reloaded copy on the frozen test episodes of the evaluated
    conditions: intact only, no lesson, the copy's own continuing stream."""
    with tempfile.TemporaryDirectory() as directory:
        path = brain.save(os.path.join(directory, "brain.npz"))
        copy = cd.Brain.load(path)
    correct: dict[str, list[int]] = {name: [] for name in EVALUATED}
    for index in range(len(frozen["test/condition"])):
        name = chamber.condition_name(frozen, "test", index)
        if name not in correct:
            continue
        arrays = chamber.episode_arrays(frozen, "test", index)
        answer = None
        for x in arrays["observations"]:
            answer = copy.act(x, greedy=True)
        assert answer is not None
        correct[name] += (np.asarray(answer) == arrays["labels"]).astype(int).tolist()
    return {name: float(np.mean(values)) for name, values in correct.items() if values}


def set_motor_bias(brain: cd.Brain, value: np.ndarray | float) -> None:
    """Replace the motor neurons' biases on the live graph; derived weights are remade."""
    bias = brain.brain.bias.copy()
    bias[brain.motor_index] = value
    brain.learner.brain = brain.learner.brain.with_parameters(bias=bias)


def homeostasis(brain: cd.Brain, target: float, rate: float) -> None:
    """A chamber-level probe of intrinsic plasticity: after an act, each motor neuron's bias
    moves toward the level that would keep its activation at ``target``, by ``rate`` times
    the shortfall, averaged over the rows. Local to the neuron, counter-free, off at rate 0.
    Diagnostic only: this is not a library mechanism."""
    state = brain.basal_ganglia.state
    if state is None or rate <= 0.0:
        return
    motor = np.atleast_2d(np.asarray(state.activation))[:, brain.motor_index].mean(axis=0)
    bias = brain.brain.bias.copy()
    bias[brain.motor_index] += rate * (target - motor)
    brain.learner.brain = brain.learner.brain.with_parameters(bias=bias)


def credit(
    seed: int,
    protocol: dict[str, Any],
    amplitude: float,
    decay: float,
    repeats: int,
    every: int,
    rule: str,
    motor_bias: float | None = None,
    homeostatic: tuple[float, float] | None = None,
) -> dict:
    """Train one founder with the chamber's rule and record where each lesson's credit lands."""
    started = time.perf_counter()
    frozen = chamber.episode_fixture(seed, {"train": repeats, "test": 24})
    brain = make_brain(protocol, seed, amplitude, decay)
    if motor_bias is not None:
        set_motor_bias(brain, motor_bias)
    groups = groups_of(brain)
    tally = Tally(brain.connectome.synapses)
    reference = brain.brain.efficacy.copy()
    conditions = len(inputs.TRAIN_CONDITIONS)
    curve = [{"repeat": 0, "lessons": 0, "recall": evaluate(brain, frozen)}]
    lessons: list[dict[str, Any]] = []
    refused = 0
    accepted = 0
    count = len(frozen["train/condition"])

    def act(x: np.ndarray) -> np.ndarray:
        answer = brain.act(x, greedy=True)
        if homeostatic is not None:
            homeostasis(brain, *homeostatic)
        return np.asarray(answer)

    for index in range(count):
        arrays = chamber.episode_arrays(frozen, "train", index)
        observations = arrays["observations"]
        labels = np.asarray(arrays["labels"])
        condition = chamber.condition_name(frozen, "train", index)
        for event, x in enumerate(observations):
            query = event == len(observations) - 1
            if not query:
                act(x)
                continue
            drive = brain.stimulus(x)
            prefrontal = drive[:, brain.connectome.populations["prefrontal"]]
            if rule == "surprise":
                answer = act(x)
                wrong = np.asarray(answer) != labels
                rows = np.flatnonzero(wrong)
            else:
                rows = np.arange(len(labels))
            record: dict[str, Any] = {
                "episode": index,
                "condition": condition,
                "rows": int(len(rows)),
                "paired_prefrontal_drive": paired(prefrontal),
            }
            if len(rows):
                before = brain.brain.efficacy.copy()
                try:
                    brain.learner.step(drive[rows], labels[rows])
                    accepted += 1
                    record["accepted"] = True
                except LearningPhaseError:
                    refused += 1
                    record["accepted"] = False
                step = brain.brain.efficacy - before
                if record["accepted"]:
                    tally.add(step)
                record["step"] = {
                    name: float(np.sqrt(np.mean(step[mask] ** 2))) for name, mask in groups.items()
                }
            if rule != "surprise":
                answer = act(x)
            association, trace = association_and_trace(brain)
            state = brain.basal_ganglia.state
            assert state is not None
            motor = np.atleast_2d(np.asarray(state.activation))[:, brain.motor_index]
            record["paired_association"] = paired(association)
            record["paired_trace"] = paired(trace)
            record["association_level"] = float(np.abs(association).mean())
            record["motor_level"] = float(np.abs(motor).mean())
            record["motor_max"] = float(motor.max())
            record["motor_bias"] = float(brain.brain.bias[brain.motor_index].mean())
            record["correct"] = float(np.mean(np.asarray(answer) == labels))
            lessons.append(record)
        repeat = (index + 1) // conditions
        if (index + 1) % conditions == 0 and repeat % every == 0:
            efficacy = brain.brain.efficacy
            curve.append(
                {
                    "repeat": repeat,
                    "lessons": accepted,
                    "recall": evaluate(brain, frozen),
                    "drift": {
                        name: float(
                            np.sqrt(np.mean((efficacy[mask] - reference[mask]) ** 2))
                            / (np.sqrt(np.mean(reference[mask] ** 2)) + 1e-12)
                        )
                        for name, mask in groups.items()
                    },
                    "weight_rms": {
                        name: float(np.sqrt(np.mean(efficacy[mask] ** 2)))
                        for name, mask in groups.items()
                    },
                }
            )
    return {
        "seed": seed,
        "historical": HISTORICAL.get(seed),
        "rule": rule,
        "amplitude": amplitude,
        "decay": decay,
        "repeats": repeats,
        "motor_bias": motor_bias,
        "homeostasis": None if homeostatic is None else list(homeostatic),
        "accepted": accepted,
        "refused": refused,
        "curve": curve,
        "lessons": lessons,
        "consistency": tally.by_group(groups),
        "seconds": time.perf_counter() - started,
    }


# ----------------------------------------------------------------------------------- run


def _untrained_job(args: tuple[int, dict, float, float]) -> dict:
    return untrained(*args)


def _credit_job(args: tuple) -> dict:
    return credit(*args)


def run(function: Any, jobs: list[Any], workers: int) -> list[dict]:
    if workers <= 1 or len(jobs) <= 1:
        return [function(job) for job in jobs]
    with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        return list(pool.imap_unordered(function, jobs))


def sources() -> list[tuple[str, Path]]:
    package = Path(cd.__file__).resolve().parent
    return [
        ("credit_landing.py", Path(__file__).resolve()),
        ("finite_horizon.py", HERE / "finite_horizon.py"),
        ("finite_horizon_inputs.py", HERE / "finite_horizon_inputs.py"),
        *(
            ("cadence/" + path.relative_to(package).as_posix(), path)
            for path in sorted(package.rglob("*.py"))
        ),
    ]


def write(path: Path, body: dict[str, Any], manifest: dict[str, Any]) -> None:
    receipt = Receipt.build(SCHEMA, body, sources())
    if receipt.source != manifest:
        raise RuntimeError("a source file changed during the run; discard it and run again")
    data = (canonical_json(receipt.to_dict()) + "\n").encode()
    if path.suffix == ".gz":
        data = gzip.compress(data, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def read(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    stored = json.loads(data)
    return stored["body"] if "body" in stored else stored


def verify(path: Path, *, current: bool = False) -> tuple[bool, str]:
    """Canonical form and digest of a receipt; with ``current`` also the sources' identity."""
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    with tempfile.TemporaryDirectory() as directory:
        plain = Path(directory) / "receipt.json"
        plain.write_bytes(data)
        stored = json.loads(data)
        if stored.get("kind") != SCHEMA:
            return False, f"not a {SCHEMA} receipt"
        return Receipt.verify(plain, sources=sources() if current else None)


# -------------------------------------------------------------------------------- report


def report_untrained(body: dict[str, Any]) -> str:
    lines = [
        "# Untrained founders: what the trace carries before any lesson",
        "",
        "Paired distances between the two rows of a pair (opposite tokens, identical other "
        "events) at WRITE and at QUERY; the motor margin at QUERY. `learned` marks the "
        "founders that acquired a nonnegative recall horizon in the historical freezes.",
        "",
        "| Founder | Learned | Assoc. at WRITE | Trace at QUERY clean-0 | clean-1 | clean-2 | "
        "distractor-1 | Assoc. at QUERY clean-0 | clean-1 | Motor margin clean-0 | Motor level |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for founder in sorted(body["founders"], key=lambda f: f["seed"]):
        episodes = founder["episodes"]

        def at(condition: str, event: int, key: str, episodes: dict = episodes) -> str:
            events = episodes.get(condition)
            if events is None or len(events) <= event:
                return "-"
            return f"{events[event][key]:.3f}"

        learned = founder.get("historical")
        mark = "-" if learned is None else ("yes" if learned else "no")
        cells = [
            str(founder["seed"]),
            mark,
            at("clean-0", 1, "paired_association"),
            at("clean-0", -1, "paired_trace"),
            at("clean-1", -1, "paired_trace"),
            at("clean-2", -1, "paired_trace"),
            at("distractor-1", -1, "paired_trace"),
            at("clean-0", -1, "paired_association"),
            at("clean-1", -1, "paired_association"),
            at("clean-0", -1, "motor_margin"),
            at("clean-0", -1, "motor_level"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


STILL = 1e-5  # an RMS efficacy step below this leaves the greedy answers where they are
RUN = 12  # one repeat of the twelve conditions without a moving lesson: the readout is dead


def death(lessons: list[dict]) -> dict[str, Any] | None:
    """The first lesson from which no later lesson moved any weight by more than ``STILL``,
    with the motor level before and after; None when the weights kept moving. A saturated
    readout leaves steps of 1e-6 to 1e-4: not exactly zero, but no longer learning."""
    moving = [
        (index, max(record["step"].values()) > STILL)
        for index, record in enumerate(lessons)
        if "step" in record and record.get("accepted", True)
    ]
    if not moving or moving[-1][1]:
        return None
    first = len(moving)
    for position in range(len(moving) - 1, -1, -1):
        if moving[position][1]:
            break
        first = position
    if len(moving) - first < RUN:
        return None  # a few still lessons at the end are not a dead readout
    index = moving[first][0]
    before = [r.get("motor_level") for r in lessons[:index] if r.get("motor_level") is not None]
    after = [r.get("motor_level") for r in lessons[index:] if r.get("motor_level") is not None]
    return {
        "lesson": int(index),
        "of": len(lessons),
        "motor_level_before": float(np.mean(before[-12:])) if before else float("nan"),
        "motor_level_after": float(np.mean(after)) if after else float("nan"),
        "motor_bias_after": float(np.mean([r["motor_bias"] for r in lessons[index:]]))
        if lessons[index:] and "motor_bias" in lessons[index]
        else float("nan"),
    }


def report_credit(body: dict[str, Any]) -> str:
    lines = ["# Where the lesson's credit lands", ""]
    for founder in sorted(body["founders"], key=lambda f: f["seed"]):
        learned = founder.get("historical")
        mark = "" if learned is None else (" (learned historically)" if learned else " (did not)")
        lines.append(
            f"## Founder {founder['seed']}{mark}: {founder['accepted']} lessons, "
            f"{founder['refused']} refused, rule `{founder['rule']}`"
        )
        lines.append("")
        notes = []
        if body.get("learning_overrides"):
            notes.append(f"learning overrides {body['learning_overrides']}")
        if founder.get("motor_bias") is not None:
            notes.append(f"motor bias {founder['motor_bias']:g}")
        if founder.get("homeostasis"):
            target, rate = founder["homeostasis"]
            notes.append(f"motor homeostasis toward {target:g} at rate {rate:g}")
        lines.append(
            "Recall of a saved copy along the run (intact, 24 episodes per condition)"
            + ("; " + "; ".join(notes) if notes else "")
            + ":"
        )
        lines.append("")
        header = (
            "| Repeat | Lessons | " + " | ".join(EVALUATED) + " | Drift pf->assoc | |W| pf->assoc |"
        )
        lines.append(header)
        lines.append("|" + " --- |" * (3 + len(EVALUATED) + 1))
        for point in founder["curve"]:
            recall = point["recall"]
            drift = point.get("drift", {}).get("prefrontal->association")
            weight = point.get("weight_rms", {}).get("prefrontal->association")
            lines.append(
                f"| {point['repeat']} | {point['lessons']} | "
                + " | ".join(f"{recall.get(name, float('nan')):.2f}" for name in EVALUATED)
                + f" | {'-' if drift is None else f'{drift:.3f}'} | "
                + f"{'-' if weight is None else f'{weight:.3f}'} |"
            )
        lines.append("")
        lines.append(
            "Steps per projection (RMS per lesson, mean over lessons) and their consistency:"
        )
        lines.append("")
        lines.append("| Projection | RMS step | Net | Consistency | Moved / synapses |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, value in founder["consistency"].items():
            lines.append(
                f"| {name} | {value['rms_step']:.4f} | {value['net']:.4f} | "
                f"{value['consistency']:.2f} | {value['moved']} / {value['synapses']} |"
            )
        lessons = founder["lessons"]
        early = [r for r in lessons[: len(lessons) // 4]]
        late = [r for r in lessons[-len(lessons) // 4 :]]

        def mean(records: list[dict], key: str) -> float:
            values = [r[key] for r in records if key in r]
            return float(np.mean(values)) if values else float("nan")

        lines.append("")
        lines.append(
            f"Paired association distance at QUERY: {mean(early, 'paired_association'):.3f} "
            f"in the first quarter of lessons, {mean(late, 'paired_association'):.3f} in the "
            f"last; paired trace {mean(early, 'paired_trace'):.3f} then "
            f"{mean(late, 'paired_trace'):.3f}; paired prefrontal drive "
            f"{mean(early, 'paired_prefrontal_drive'):.4f} then "
            f"{mean(late, 'paired_prefrontal_drive'):.4f}; motor level "
            f"{mean(early, 'motor_level'):.3f} then {mean(late, 'motor_level'):.3f}; motor bias "
            f"{mean(early, 'motor_bias'):.3f} then {mean(late, 'motor_bias'):.3f}; training "
            f"answers right {mean(early, 'correct'):.2f} then {mean(late, 'correct'):.2f}."
        )
        dead = death(lessons)
        if dead is not None:
            lines.append("")
            lines.append(
                f"**The weights stopped moving at lesson {dead['lesson']} of {dead['of']}**: "
                f"motor level {dead['motor_level_before']:.3f} in the twelve lessons before, "
                f"{dead['motor_level_after']:.3f} after; motor bias after "
                f"{dead['motor_bias_after']:.3f}."
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--untrained", action="store_true")
    mode.add_argument("--credit", action="store_true")
    mode.add_argument("--report", type=Path)
    mode.add_argument("--verify", type=Path, help="check a receipt's digest and canonical form")
    parser.add_argument(
        "--current", action="store_true", help="with --verify: also require the current sources"
    )
    parser.add_argument("--founders", nargs="*", type=int, default=None)
    parser.add_argument("--amplitude", type=float, default=0.3)
    parser.add_argument("--decay", type=float, default=0.8)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--every", type=int, default=4)
    parser.add_argument("--rule", choices=("every", "surprise"), default="every")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--motor-bias",
        type=float,
        default=None,
        help="set the motor neurons' biases at construction (a fixed operating-point probe)",
    )
    parser.add_argument(
        "--homeostasis",
        nargs=2,
        type=float,
        default=None,
        metavar=("TARGET", "RATE"),
        help="after every act, move each motor bias by RATE times (TARGET - its activation)",
    )
    parser.add_argument(
        "--learning",
        nargs="*",
        default=[],
        metavar="FIELD=VALUE",
        help="override fields of the chamber's LearnerConfig, e.g. normalize=0 momentum=0 eta=0.5",
    )
    args = parser.parse_args(argv)
    if args.verify is not None:
        ok, message = verify(args.verify, current=args.current)
        print(message or ("verified" if ok else "failed"))
        return 0 if ok else 1
    if args.report is not None:
        body = read(args.report)
        print(report_untrained(body) if body["mode"] == "untrained" else report_credit(body))
        return 0
    protocol = load_protocol()
    overrides: dict[str, Any] = {}
    for item in args.learning:
        field, _, raw = item.partition("=")
        if not field or not raw:
            parser.error("--learning takes FIELD=VALUE pairs")
        value: Any
        if raw in ("True", "False", "true", "false"):
            value = raw.lower() == "true"
        elif raw.lstrip("-").isdigit():
            value = int(raw)
        else:
            value = float(raw)
        protocol["learning"][field] = overrides[field] = value
    founders = args.founders
    if founders is None:
        founders = [0, 1, *sorted(HISTORICAL)] if args.untrained else [0, 1, 304, 305, 306]
    manifest = Receipt.build(SCHEMA, {}, sources()).source
    started = time.perf_counter()
    if args.untrained:
        results = run(
            _untrained_job,
            [(seed, protocol, args.amplitude, args.decay) for seed in founders],
            args.workers,
        )
        mode_name = "untrained"
    else:
        results = run(
            _credit_job,
            [
                (
                    seed,
                    protocol,
                    args.amplitude,
                    args.decay,
                    args.repeats,
                    args.every,
                    args.rule,
                    args.motor_bias,
                    None if args.homeostasis is None else tuple(args.homeostasis),
                )
                for seed in founders
            ],
            args.workers,
        )
        mode_name = "credit"
    body = {
        "schema": SCHEMA,
        "mode": mode_name,
        "protocol": protocol,
        "amplitude": args.amplitude,
        "decay": args.decay,
        "repeats": args.repeats,
        "every": args.every,
        "rule": args.rule,
        "learning_overrides": overrides,
        "motor_bias": args.motor_bias,
        "homeostasis": args.homeostasis,
        "founders": sorted(results, key=lambda r: r["seed"]),
        "cadence": cd.__version__,
        "seconds": time.perf_counter() - started,
    }
    out = args.out or HERE / "results" / f"{mode_name}-{time.strftime('%Y-%m-%d-%H%M%S')}.json.gz"
    write(out, body, manifest)
    print(f"wrote {out}", file=sys.stderr)
    print(report_untrained(body) if args.untrained else report_credit(body))
    return 0


if __name__ == "__main__":
    sys.exit(main())
