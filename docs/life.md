# Plasticity for life

A synapse has a fast strength that moves with every experience and a slow one that
consolidates what was repeated and reinforced; unused synapses are pruned and co-active
neurons sprout new ones. `Seams` gives a learner's seams that life.

```python
seams = cd.Seams(learner, cd.SleepConfig(consolidate=0.5, downscale=0.5, prune_below=0.01,
                                         sprout_above=0.5, budget=40, strength=3.0))
for day in days:
    for x, y in stream:                      # the day: the learner learns as it always does
        learner.step(x, y)
        seams.observe(activation)            # fast strength moved; tags accumulate; silent overlaps count co-activation
    report = seams.sleep()                   # the night: consolidate, downscale, prune, sprout
```

- **Fast and slow.** The learner writes the fast strength. `sleep` moves a fraction of every
  tagged fast strength into the slow one (`consolidate`) and downscales the rest. The slow
  strength is long-term memory; the fast one is plasticity that never stops.
- **Tags.** A seam is tagged by how much it moved since the last sleep, so sleep can tell
  what was repeated and reinforced from what merely happened.
- **Prune and sprout.** A seam whose total falls below `prune_below` goes silent: zero
  strength, frozen. A silent overlap of the wiring sprouts when its two endpoints kept firing
  together (`sprout_above`), within each owner's `budget` of live incoming seams. The wiring
  holds every overlap that could ever carry a seam; `alive` marks the ones that do, and its
  digest is the wiring as a state variable.
- **Conserved strength.** Each owner holds at most `strength` of incoming strength; the
  strengths a learner writes are relative. This is what made learned seams selective in the
  composer, where per-owner rate homeostasis had made rare inputs hyperexcitable.

The learner's `trainable_overlaps` is kept equal to `alive`, so nothing it does touches a
silent overlap, and the conformance reference sees exactly the live seams. The traps this
was built against, in cadence-examples rung 10, are the tests.
