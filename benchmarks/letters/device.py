"""The read-speech recipe's composition on a device (cadence#314, R3).

The composition is the chamber's ``fields256_p`` arm: ``Brain.compose((40, 24), 29,
learned_fields=64, modules=(256,), fields_motor=1.0, learning_surprise=False)`` with the actor's
rates at zero (``letters``), or the same reading the spoken chamber's 40 by 98 window over 37
symbols (``spoken``). A training step is the forced-alignment lesson's form,
``step(x, teacher=labels)``: one target per stream, the utterances heard side by side as
streams. Frames are random windows at the chamber's scale (clipped standard normals times 0.3);
the measured quantities read the arithmetic, not what is learned.

- ``match``: a short life from one CPU brain loaded on each place. After it, each place's
  efficacies, biases and fields against the first place's, the efficacy differences relative to
  the first place's own lesson steps, the motor's settled activity and greedy choices on one more
  act, and whether a save on the place loads on the CPU and on the place to the same parameters.
- ``carry``: a brain saved on some backend (``--saved``) loads on each place to the saved
  parameters, and a further short life from it on each place against the first place's.
- ``time``: seconds per lesson step and per greedy act at each batch of streams, after warm-up,
  and seconds per hour of audio (a step hears 40 ms per stream). A training step is a lesson and,
  for the teacher's re-alignment, one greedy act of the frozen copy.

A place is ``cpu``, ``mlx`` or ``torch:<device>:<precision>``; one JSON line per reading.

    PYTHONPATH=src python benchmarks/letters/device.py match --rows 32 \
        --places cpu torch:cuda:float32 mlx
    PYTHONPATH=src python benchmarks/letters/device.py time --places torch:cuda:float32 \
        --batches 32 128
"""
import argparse
import json
import os
import tempfile
import time
import warnings

import numpy as np

import cadence as cd

SHAPES = {"letters": ((40, 24), 29), "spoken": ((40, 98), 37)}
GENES = dict(learned_fields=64, modules=(256,), fields_motor=1.0, learning_surprise=False,
             actor_eta=0.0, actor_eta_bias=0.0, actor_eta_critic=0.0)
STEP_SECONDS = 0.04


def data(shape, actions, rows, steps, seed=0):
    rng = np.random.default_rng(seed)
    width = int(np.prod(shape))
    x = np.clip(rng.standard_normal((steps, rows, width)), -3, 3) * 0.3
    labels = rng.integers(actions, size=(steps, rows))
    return x.astype(np.float32).astype(np.float64), labels


def place_kw(place):
    if place.startswith("torch"):
        _, device, precision = place.split(":")
        return {"backend": "torch", "device": device, "precision": precision}
    if place == "mlx":
        return {"backend": "mlx"}
    return {"backend": "cpu"}


def sync(brain):
    kernel = brain.brain._torch
    if kernel is not None and str(kernel.device).startswith("cuda"):
        kernel.torch.cuda.synchronize()


def snapshot(brain):
    return {"efficacy": np.asarray(brain.brain.efficacy, dtype=np.float64).copy(),
            "bias": np.asarray(brain.brain.bias, dtype=np.float64).copy(),
            "dictionary": brain.fields.dictionary.copy(),
            "thresholds": brain.fields.thresholds.copy()}


def match(args, shape, actions):
    x, labels = data(shape, actions, args.rows, args.steps)
    founder = cd.Brain.compose(shape, actions, seed=args.seed, backend="cpu", **GENES)
    folder = tempfile.mkdtemp()
    start = founder.save(os.path.join(folder, "start.npz"))
    first = np.asarray(founder.brain.efficacy, dtype=np.float64)
    motor = np.asarray(founder.motor_index)
    ref = None
    for place in args.places:
        kw = place_kw(place)
        row = {"mode": "match", "shape": args.shape, "rows": args.rows, "steps": args.steps,
               "place": place, "parameters": founder.parameters()}
        try:
            brain = cd.Brain.load(start, **kw)
            row["backend"] = brain.brain.backend
            brain.reset()
            for t in range(len(x)):
                brain.step(x[t], teacher=labels[t])
            sync(brain)
            snap = snapshot(brain)
            saved = brain.save(os.path.join(args.keep or folder,
                                            f"{args.shape}-{place.replace(':', '_')}.npz"))
            back, again = cd.Brain.load(saved, backend="cpu"), cd.Brain.load(saved, **kw)
            row["saved_loads_on_cpu_to_the_bit"] = bool(
                np.array_equal(np.asarray(back.brain.efficacy), snap["efficacy"])
                and np.array_equal(np.asarray(back.brain.bias), snap["bias"])
                and np.array_equal(back.fields.dictionary, snap["dictionary"]))
            row["saved_loads_on_backend_to_the_bit"] = bool(
                np.array_equal(np.asarray(again.brain.efficacy, dtype=np.float64),
                               snap["efficacy"]))
            brain.reset()
            brain.act(x[-1], greedy=True)
            activity = np.asarray(brain.basal_ganglia.state.activation, dtype=np.float64)[:, motor]
            choice = np.argmax(activity, axis=1)
            if ref is None:
                ref = (snap, activity, choice)
                row["largest_lesson_step"] = float(np.max(np.abs(snap["efficacy"] - first)))
            else:
                step = np.abs(ref[0]["efficacy"] - first)
                for name in snap:
                    row[f"max_diff_{name}"] = float(np.max(np.abs(snap[name] - ref[0][name])))
                diff = np.abs(snap["efficacy"] - ref[0]["efficacy"])
                row["max_diff_efficacy_over_largest_step"] = float(diff.max() / step.max())
                moved = step > 0.1 * step.max()
                row["steps_relative_error_median"] = float(np.median(diff[moved] / step[moved]))
                row["max_diff_motor_activity"] = float(np.max(np.abs(activity - ref[1])))
                row["greedy_choices_agree"] = float(np.mean(choice == ref[2]))
        except Exception as error:  # a refused backend is a reading
            import traceback
            row["error"] = f"{type(error).__name__}: {error}"
            row["trace"] = traceback.format_exc()[-2000:]
        print(json.dumps(row), flush=True)


def carry(args, shape, actions):
    """A brain saved on one backend loads on each place to the saved parameters, and a further
    short life from it on each place follows the first place's within the solve's tolerance."""
    x, labels = data(shape, actions, args.rows, args.steps, seed=2)
    with np.load(args.saved) as archive:
        saved = {"efficacy": archive["efficacy"].astype(np.float64), "bias": archive["bias"]}
    ref = None
    for place in args.places:
        row = {"mode": "carry", "saved": os.path.basename(args.saved), "place": place,
               "rows": args.rows, "steps": args.steps}
        try:
            brain = cd.Brain.load(args.saved, **place_kw(place))
            row["backend"] = brain.brain.backend
            row["loads_to_the_saved_parameters"] = bool(
                np.array_equal(np.asarray(brain.brain.efficacy, dtype=np.float64),
                               saved["efficacy"])
                and np.array_equal(np.asarray(brain.brain.bias, dtype=np.float64), saved["bias"]))
            first = np.asarray(brain.brain.efficacy, dtype=np.float64).copy()
            brain.reset()
            for t in range(len(x)):
                brain.step(x[t], teacher=labels[t])
            sync(brain)
            snap = snapshot(brain)
            if ref is None:
                ref = snap
                row["largest_lesson_step"] = float(np.max(np.abs(snap["efficacy"] - first)))
            else:
                step = np.abs(ref["efficacy"] - first)
                diff = np.abs(snap["efficacy"] - ref["efficacy"])
                row["max_diff_efficacy_over_largest_step"] = float(diff.max() / step.max())
                moved = step > 0.1 * step.max()
                row["steps_relative_error_median"] = float(np.median(diff[moved] / step[moved]))
                row["max_diff_dictionary"] = float(np.max(np.abs(snap["dictionary"]
                                                                 - ref["dictionary"])))
        except Exception as error:
            import traceback
            row["error"] = f"{type(error).__name__}: {error}"
            row["trace"] = traceback.format_exc()[-2000:]
        print(json.dumps(row), flush=True)


def timing(args, shape, actions):
    for place in args.places:
        for rows in args.batches:
            row = {"mode": "time", "shape": args.shape, "place": place, "rows": rows}
            try:
                x, labels = data(shape, actions, rows, args.warm + args.steps, seed=1)
                brain = cd.Brain.compose(shape, actions, seed=args.seed, **place_kw(place),
                                         **GENES)
                row["backend"] = brain.brain.backend
                brain.reset()
                lessons, acts = [], []
                for t in range(len(x)):
                    tick = time.perf_counter()
                    brain.step(x[t], teacher=labels[t])
                    sync(brain)
                    lessons.append(time.perf_counter() - tick)
                brain.reset()
                for t in range(len(x)):
                    tick = time.perf_counter()
                    brain.act(x[t], greedy=True)
                    sync(brain)
                    acts.append(time.perf_counter() - tick)
                lesson = float(np.median(lessons[args.warm:]))
                act = float(np.median(acts[args.warm:]))
                audio = rows * STEP_SECONDS  # seconds of audio a step hears
                row.update({
                    "lesson_seconds_per_step": lesson, "act_seconds_per_step": act,
                    "lesson_seconds_per_audio_hour": lesson / audio * 3600,
                    "act_seconds_per_audio_hour": act / audio * 3600,
                    "training_seconds_per_audio_hour": (lesson + act) / audio * 3600,
                    "first_step_seconds": lessons[0],
                })
                kernel = brain.brain._torch
                if kernel is not None and str(kernel.device).startswith("cuda"):
                    torch = brain.brain._torch.torch
                    row["device_name"] = torch.cuda.get_device_name(0)
                    row["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            except Exception as error:
                import traceback
                row["error"] = f"{type(error).__name__}: {error}"
                row["trace"] = traceback.format_exc()[-2000:]
            print(json.dumps(row), flush=True)


def main() -> None:
    warnings.simplefilter("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("match", "time", "carry"))
    p.add_argument("--keep", default=None, help="match: keep each place's saved brain here")
    p.add_argument("--saved", default=None, help="carry: a brain saved on some backend")
    p.add_argument("--shape", default="letters", choices=tuple(SHAPES))
    p.add_argument("--rows", type=int, default=32)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--warm", type=int, default=3)
    p.add_argument("--batches", nargs="+", type=int, default=[32, 128])
    p.add_argument("--places", nargs="+", default=["cpu"],
                   help="cpu, torch:<device>:<precision>, mlx")
    p.add_argument("--seed", type=int, default=304400)
    args = p.parse_args()
    shape, actions = SHAPES[args.shape]
    {"match": match, "time": timing, "carry": carry}[args.mode](args, shape, actions)


if __name__ == "__main__":
    main()
