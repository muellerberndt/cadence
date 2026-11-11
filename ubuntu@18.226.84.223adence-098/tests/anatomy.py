"""The single-region layout of earlier releases (``Brain.build`` up to 0.84.3), wired by hand
for the pins and fixtures recorded on it: a sensory region (an image's visual cortex), one
association region and the motor region, and with ``working_memory`` a prefrontal region
read at ``memory_scale``. ``Brain.compose(inputs, actions, modules=(hidden,))`` is the
composed brain of that size."""

from __future__ import annotations

from typing import Any

import cadence as cd
from cadence.genome import Genome, Projection, develop
from cadence.regions import motor_cortex, prefrontal_cortex, visual_cortex


def single_region(
    inputs: int | tuple[int, int],
    actions: int,
    *,
    hidden: int = 64,
    working_memory: bool = False,
    memory_scale: float = 12.0,
    seed: int = 0,
    **options: Any,
) -> cd.Brain:
    if isinstance(inputs, tuple):
        sensory = visual_cortex(inputs[0], inputs[1], seed=seed)
        forward = Projection("visual", "association")
    else:
        sensory = cd.Region("sensory", inputs)
        forward = Projection("sensory", "association", reciprocal=False)
    regions = [
        sensory,
        cd.Region("association", hidden),
        motor_cortex(actions, lateral=-0.5 if actions <= 8 else 0.0),
    ]
    projections = [forward, Projection("association", "motor")]
    if working_memory:
        regions.insert(1, prefrontal_cortex(hidden))
        projections.append(
            Projection("prefrontal", "association", scale=memory_scale, reciprocal=False)
        )
    genome = Genome(tuple(regions), tuple(projections), label="generic-brain")
    return cd.Brain(develop(genome, seed=seed), seed=seed, **options)
