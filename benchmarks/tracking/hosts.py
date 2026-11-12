"""The map of things in a host brain (issue #321, M1 to M4): the built-in place module held by
brains that took no lesson, a task the host learns from the map, a second task, and the module's
file on another backend.

- **M1, the skill arrives.** A host composed for the map's observation holds the built-in module
  (`brain.space = cadence.modules.load("place")`); on a founder's scenes of a person's hand and
  the two recorded scenes its map's error is read beside the module alone (its donor).
- **M2, the host reads it.** A task that needs the hand's place: which quarter of the plane the
  hand is in, at every moment, out of view as in view. A host that reads the map's observation
  and a host of the same composition that reads the camera's report (its place and whether it
  was seen) learn it by the library's lessons; each is read on held-out scenes after a number of
  lessons.
- **M3, the skill stays.** The host learns a second task (whether the hand is in view), and the
  map's error is read again. The map holds no taught synapse, so lessons of the host do not reach
  it; the reading states that.
- **M4, the file.** The module's file read on the torch backend answers as on the CPU.

    python benchmarks/tracking/hosts.py --host 3300
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mapped  # noqa: E402
import tracking as tr  # noqa: E402
import world as w  # noqa: E402

HAND = 0


def quarter(points: np.ndarray, plane: tuple[float, ...]) -> np.ndarray:
    """Which quarter of the plane each point is in: 0 to 3."""
    x_mid, y_mid = (plane[0] + plane[1]) / 2, (plane[2] + plane[3]) / 2
    pts = np.asarray(points).reshape(-1, 2)
    return (2 * (pts[:, 0] > x_mid) + (pts[:, 1] > y_mid)).astype(np.int64)


def moments(scenes: list[w.Scene], held: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                       np.ndarray]:
    """Every moment of ``scenes`` after the first: the map's observation, the camera's report
    (its place, and one where it saw the hand), the hand's true quarter, and whether it was
    seen."""
    looks, raw, labels, seen = [], [], [], []
    for scene in scenes:
        feed = tr.given(scene)
        held.reset()
        held.moment(seen=[(HAND, feed["report"][0])])
        for t in range(1, scene.steps):
            sight = [(HAND, feed["report"][t])] if feed["seen"][t] else []
            held.moment(seen=sight, lit=mapped.seeing(scene, t))
            looks.append(held.observation())
            raw.append(np.r_[feed["report"][t] if feed["seen"][t] else np.zeros(2),
                             float(feed["seen"][t])])
            labels.append(int(quarter(scene.truth[t, 0], scene.geometry.plane)[0]))
            seen.append(bool(feed["seen"][t]))
    return np.array(looks), np.array(raw), np.array(labels), np.array(seen)


def learn(inputs: np.ndarray, labels: np.ndarray, test: np.ndarray, test_labels: np.ndarray,
          hidden: np.ndarray, seed: int, marks: tuple[int, ...], rows: int = 16
          ) -> list[dict[str, float]]:
    """A host of the default composition learns the task from ``inputs``, ``rows`` moments a
    lesson in a shuffled order, and is read after each number of lessons in ``marks``: the
    share right on the held-out moments, and on those out of view."""
    import cadence as cd

    brain = cd.Brain.compose(inputs=inputs.shape[1], actions=4, seed=seed, backend="cpu")
    order = np.random.default_rng(seed).permutation(len(inputs))
    out, done = [], 0
    for mark in marks:
        while done < mark:
            start = (done * rows) % max(len(order) - rows, 1)
            pick = order[start : start + rows]
            brain.reset()
            brain.lesson(inputs[pick], labels[pick])
            done += 1
        guess = []
        for k in range(0, len(test), 256):
            brain.reset()  # each batch of held-out moments as streams of their own
            guess.append(np.asarray(brain.act(test[k : k + 256], greedy=True)).reshape(-1))
        brain.reset()
        guess = np.concatenate(guess)
        right = guess == test_labels
        out.append({"lessons": mark, "right": float(np.mean(right)),
                    "right out of view": float(right[hidden].mean())})
    return out


def main(argv: list[str] | None = None) -> int:
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--host", type=int, default=3300)
    parser.add_argument("--scenes", type=int, default=40)
    parser.add_argument("--marks", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    import cadence as cd
    from cadence.space.map import Map

    out: dict[str, Any] = {"host": args.host}
    held_out = tr.test_scenes("other", args.host, args.scenes, 0)
    files = [w.load(Path(__file__).resolve().parent / "scenes" / f"{n}.json")
             for n in ("promise", "promise-folded")]
    donor = cd.modules.load("place")
    host = cd.Brain.compose(inputs=donor.count + donor.units, actions=4, seed=args.host,
                            backend="cpu")
    host.space = cd.modules.load("place")

    def error(held: Any) -> float:
        table = mapped.read(held_out + files, {"map": held}, None, controls=False)["map"]
        return float(table["out_of_view"])

    out["M1"] = {"donor": error(donor), "host": error(host.space)}
    # M2: the task from the map's observation, and from the camera's report
    training = w.scenes("other", tr.seed(args.host, "train"), args.scenes)
    looks, raw, labels, _ = moments(training, host.space)
    t_looks, t_raw, t_labels, t_seen = moments(held_out, host.space)
    out["M2"] = {"map": learn(looks, labels, t_looks, t_labels, ~t_seen, args.host,
                              tuple(args.marks)),
                 "report": learn(raw, labels, t_raw, t_labels, ~t_seen, args.host,
                                 tuple(args.marks))}
    # M3: a second task (whether the hand is in view) learnt by the host, the map read again
    before = np.array(host.space._plain.efficacy)
    second = cd.Brain.compose(inputs=looks.shape[1], actions=2, seed=args.host, backend="cpu")
    for k in range(0, min(len(looks), 4000) - 16, 16):
        second.reset()
        second.lesson(looks[k : k + 16], (raw[k : k + 16, 2] > 0).astype(np.int64))
    out["M3"] = {"after the second task": error(host.space),
                 "map's synapses unchanged by the host's lessons": bool(
                     np.array_equal(before, host.space._plain.efficacy))}
    # M4: the module's file on the torch backend
    try:
        import torch  # noqa: F401

        path = Path(f"/tmp/place-{args.host}.npz")
        donor.save(path)
        on_torch = Map.load(path, backend="torch")
        out["M4"] = {"torch": error(on_torch), "cpu": error(Map.load(path))}
    except ImportError:
        out["M4"] = {"torch": None}
    print(json.dumps(out, default=float), flush=True)
    if args.out:
        args.out.write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
