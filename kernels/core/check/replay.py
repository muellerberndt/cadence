"""A part's calls in real lives, replayed in the core: Wild's founders live in Python (the
reference); every call of the part's methods is kept with the part's state before it, its
arguments, what it returned and the state after it (``cadence.core.parts``); each is then made
in the core on the state before, and what it returned and the state it leaves are compared,
entry by entry: to the bit, within a tolerance (shown), or apart (the first few shown). A brain
error in the life (Wild logs it as ``brain_error`` and lets the animal act at random) or a part
with no call recorded fails the check.

    python kernels/core/check/replay.py arousal [steps] [--wild PATH] [--stream]

A part module may define ``calling(obj, method, args, kwargs) -> (args, kwargs, after)``: the
call's arguments as the Python method takes them (a callback wrapped so its questions and
answers are kept) and ``after()``, the named arguments known once the call has run. ``--stream``
replays each call as it is made and keeps none, so a long life is checked call by call.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from cadence.core import Core
from cadence.core.parts import part as part_module

p = argparse.ArgumentParser()
p.add_argument("part")
p.add_argument("steps", type=int, nargs="?", default=60)
p.add_argument("--wild", default=os.environ.get("WILD", ""),
               help="a checkout of cadence-wild (its wild package is imported from there)")
p.add_argument("--tolerance", type=float, default=1e-9)
p.add_argument("--calls", type=int, default=4000, help="the most calls kept")
p.add_argument("--stream", action="store_true",
               help="replay each call as it is made, keeping none (every call of the life)")
a = p.parse_args()
if a.wild:
    sys.path.insert(0, a.wild)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from wild.sim import Host, Sim  # noqa: E402
from wild.world import Rules, World  # noqa: E402

module = part_module(a.part)
calling = getattr(module, "calling", None)
core = Core() if a.stream else None
kept: list = []
tally = {"calls": 0, "bitwise": 0, "close": 0, "apart": 0, "worst": 0.0, "shown": 0}
off: dict[str, list] = {}  # each entry not to the bit: the calls where it is not, its largest gap
methods: dict[str, int] = {}  # the calls checked of each method


def check(k, name, before, args, theirs_out, theirs_after):
    """One call made in the core on the state before, against what the Python did."""
    tally["calls"] += 1
    methods[name] = methods.get(name, 0) + 1
    try:
        after, out = core.call(a.part, name, before, args)
    except (RuntimeError, ValueError) as error:  # the core refused the call, or faulted
        after, out = {}, {"refused": str(error)}
    gaps = []
    for label, ours, theirs in (("out", out, theirs_out), ("state", after, theirs_after)):
        if set(ours) != set(theirs):
            gaps.append((f"{label} names", set(ours) ^ set(theirs), np.inf))
            continue
        for key, value in theirs.items():
            x, y = np.asarray(ours[key]), np.asarray(value)
            if x.shape != y.shape:
                gaps.append((f"{label}.{key} shape", (x.shape, y.shape), np.inf))
            elif x.tobytes() != y.astype(x.dtype).tobytes():
                gap = float(np.max(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0
                gaps.append((f"{label}.{key}", gap, gap))
    for label, _, gap in gaps:
        held = off.setdefault(f"{name}: {label}", [0, 0.0])
        held[0] += 1
        held[1] = max(held[1], gap)
    if not gaps:
        tally["bitwise"] += 1
        return
    largest = max(g[2] for g in gaps)
    tally["worst"] = max(tally["worst"], largest)
    if largest <= a.tolerance:
        tally["close"] += 1
    else:
        tally["apart"] += 1
        if tally["shown"] < 5:
            tally["shown"] += 1
            print(f"  call {k} {name}: apart {gaps[:4]}")


for name, (arguments, result) in module.METHODS.items():
    method = getattr(module.CLASS, name)
    read = isinstance(method, property)  # a property is recorded as a call of its getter

    def recorded(self, *args, _m=method.fget if read else method, _n=name, _a=arguments,
                 _r=result, **kwargs):
        before = module.state_of(self) if a.stream or len(kept) < a.calls else None
        after = dict
        if calling is not None:
            args, kwargs, after = calling(self, _n, args, kwargs)
        value = _m(self, *args, **kwargs)
        if before is not None:
            entry = (_n, before, {**_a(*args, **kwargs), **after()}, _r(value),
                     module.state_of(self))
            if a.stream:
                check(tally["calls"], *entry)
            else:
                kept.append(entry)
        return value
    setattr(module.CLASS, name, property(recorded) if read else recorded)

world = World(Rules(), seed=1)
sim = Sim(world, minds=Host(world.rules.width, world.rules.height, world.seed, None, []))
began = time.perf_counter()
seen = 0
lived = {"birth": 0, "death": 0}
for step in range(a.steps):
    sim.step()
    # Wild catches a mind's error and the animal acts at random: a broken recording or state
    # shows only there, so the first one ends the check
    events = getattr(world, "events", [])
    seen = 0 if len(events) < seen else seen  # a world that clears its events each step
    errors = [e for e in events[seen:] if e.get("what") == "brain_error"]
    for e in events[seen:]:
        if e.get("what") in lived:
            lived[e["what"]] += 1
    seen = len(events)
    if errors:
        print(f"FAIL: a brain error at step {step}: {errors[0]}")
        sys.exit(2)
recorded_calls = tally["calls"] if a.stream else len(kept)
print(f"{a.part}: {recorded_calls} calls over {a.steps} steps, {lived['birth']} births and "
      f"{lived['death']} deaths ({time.perf_counter() - began:.0f} s)")
if not recorded_calls:
    print(f"FAIL: no call of {a.part} was recorded: its methods ({', '.join(module.METHODS)}) "
          "never ran, or their recording broke")
    sys.exit(2)

core = core or Core()
for k, entry in enumerate(kept):
    check(k, *entry)
print("  calls: " + ", ".join(f"{n} {k}" for k, n in sorted(methods.items())))
for key, (calls, gap) in sorted(off.items()):
    print(f"  not to the bit: {key} in {calls} calls, largest {gap:.1e}")
print(f"REPLAY {a.part}: {tally['calls']} calls, {tally['bitwise']} bitwise, {tally['close']} "
      f"within {a.tolerance:g} (largest {tally['worst']:.1e}), {tally['apart']} apart")
sys.exit(1 if tally["apart"] or not tally["calls"] else 0)
