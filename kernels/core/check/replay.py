"""A part's calls in real lives, replayed in the core: Wild's founders live in Python (the
reference); every call of the part's methods is kept with the part's state before it, its
arguments, what it returned and the state after it (``cadence.core.parts``); each is then made
in the core on the state before, and what it returned and the state it leaves are compared,
entry by entry: to the bit, within a tolerance (shown), or apart (the first few shown). A brain
error in the life (Wild logs it as ``brain_error`` and lets the animal act at random) or a part
with no call recorded fails the check.

    python kernels/core/check/replay.py arousal [steps] [--wild PATH]
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
a = p.parse_args()
if a.wild:
    sys.path.insert(0, a.wild)
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
from wild.sim import Host, Sim  # noqa: E402
from wild.world import Rules, World  # noqa: E402

module = part_module(a.part)
kept: list = []
for name, (arguments, result) in module.METHODS.items():
    method = getattr(module.CLASS, name)
    read = isinstance(method, property)  # a property is recorded as a call of its getter

    def recorded(self, *args, _m=method.fget if read else method, _n=name, _a=arguments,
                 _r=result, **kwargs):
        before = module.state_of(self) if len(kept) < a.calls else None
        value = _m(self, *args, **kwargs)
        if before is not None:
            kept.append((_n, before, _a(*args, **kwargs), _r(value), module.state_of(self)))
        return value
    setattr(module.CLASS, name, property(recorded) if read else recorded)

world = World(Rules(), seed=1)
sim = Sim(world, minds=Host(world.rules.width, world.rules.height, world.seed, None, []))
began = time.perf_counter()
seen = 0
for step in range(a.steps):
    sim.step()
    # Wild catches a mind's error and the animal acts at random: a broken recording or state
    # shows only there, so the first one ends the check
    events = getattr(world, "events", [])
    seen = 0 if len(events) < seen else seen  # a world that clears its events each step
    errors = [e for e in events[seen:] if e.get("what") == "brain_error"]
    seen = len(events)
    if errors:
        print(f"FAIL: a brain error at step {step}: {errors[0]}")
        sys.exit(2)
print(f"{a.part}: {len(kept)} calls over {a.steps} steps ({time.perf_counter() - began:.0f} s)")
if not kept:
    print(f"FAIL: no call of {a.part} was recorded: its methods ({', '.join(module.METHODS)}) "
          "never ran, or their recording broke")
    sys.exit(2)

core = Core()
bitwise = close = apart = 0
worst = 0.0
shown = 0
for k, (name, before, args, theirs_out, theirs_after) in enumerate(kept):
    after, out = core.call(a.part, name, before, args)
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
    if not gaps:
        bitwise += 1
        continue
    largest = max(g[2] for g in gaps)
    worst = max(worst, largest)
    if largest <= a.tolerance:
        close += 1
    else:
        apart += 1
        if shown < 5:
            shown += 1
            print(f"  call {k} {name}: apart {gaps[:4]}")
print(f"REPLAY {a.part}: {len(kept)} calls, {bitwise} bitwise, {close} within "
      f"{a.tolerance:g} (largest {worst:.1e}), {apart} apart")
sys.exit(1 if apart or not kept else 0)
