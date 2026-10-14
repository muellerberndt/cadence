# Context: what a brain carries from one moment to the next

A Cadence brain does not keep a list of what it has seen. At every moment it settles on an
interpretation of what it senses, and it carries that interpretation into the next moment.
The next interpretation is formed from two things together: the new observation and the
interpretation the brain brought with it. That carried interpretation is the brain's context.

## How it works

```text
moment t:    observation ──► settle ──► interpretation (association region) ──► action
                                ▲                      │
                                │                      ▼ copied by the working trace
moment t+1:  observation ──► settle ◄── carried interpretation (prefrontal region)
```

- The **association region** is the last module of the brain. Its settled activity is the
  interpretation of the moment.
- The **working trace** copies that activity after each moment into the **prefrontal**
  region, one neuron per association neuron. The prefrontal region holds the copy clearly.
- The prefrontal region feeds the copy back into the next settle through **plastic synapses
  that start quiet**. A newborn brain answers mostly from what it sees now.
- **Learning decides how much the past counts.** Where the carried interpretation predicts
  the outcome, learning strengthens those synapses; where it predicts nothing, they stay
  quiet and the present keeps the decision.
- **New evidence revises it.** An observation that says something new moves the settle away
  from what was carried; an observation that says nothing new leaves it where it was.

Nothing here is a separate memory module: the context lives in the same settled activity that
answers the present moment.

## The defaults

```python
from cadence import Brain

brain = Brain.compose(inputs=3, actions=2, arousal=True)  # context is on by default
action = brain.live([[1.0, 0.0, 0.0]])                    # a cue
action = brain.live([[0.0, 0.0, 1.0]], reward=[0.0])      # the cue is gone; its context is not
```

| Setting | Default | What it does |
| --- | --- | --- |
| `working_memory_amplitude` | `3.0` | How clearly the prefrontal region holds the carried copy |
| `working_memory_scale` | `1.2` | How strongly a newborn brain lets the copy shape the next settle (learning changes it) |
| `working_memory_decay` | `0.0` | How much of older moments the copy keeps; 0 carries exactly the last interpretation |
| `modules` | `(64,)` | The last number is the width of the association region: how much context fits |

These defaults are new in 0.82.0. Earlier releases read the copy back ten times more
strongly from birth (scale 12) and averaged it with older moments (retention 0.2). A carry
that loud drowns what the brain sees right now: an untrained brain completed a fully observed
chain of four actions less often than uniform random, and a brain whose history did not matter
could not learn to ignore it. A carry that starts quiet but is held clearly lets the present
decide at birth and lets learning raise the past where it pays. Older moments are not lost at
retention 0; they reach the present through the interpretations they shaped.

## Wire the brain for better retention

1. **Make the last module wide enough.** Context capacity is the width of the association
   region. A task that holds one fact across a short gap works at the default 64; holding more
   items or ignoring distractors needs more width, for example `modules=(128,)` or
   `modules=(48, 128)`. A narrow last module such as `modules=(48, 24)` carries little.
2. **Live a continuing life.** Use `live`: a calm brain answers from routine and learns nothing,
   and a surprised one learns. Learning only from surprise protects context that already works.
   If you teach with `step(..., teacher=...)`, teach where the answer was wrong, not at every
   moment.
3. **Show the brain what matters.** Context is learned from experience: the outcome or lesson
   has to depend on something the brain saw earlier. Distinct cues help, such as a marker that
   says "this is new information" and another that says "this is noise".
4. **Mark episode boundaries only where the world really restarts.** `done=True` clears the
   carried context for the next episode; a continuing life keeps it.
5. **Let the brain feel its own actions when the next step depends on them.** For sequences of
   its own actions, add the efference copy with `efference_amplitude=1.0`: the issued action
   becomes part of the next moment's input.

## Expert settings

Pass these to `Brain.compose(...)`. The amplitude, retention and focus can also be changed
later with `brain.retune(...)`, which keeps the acquired brain; the scale sets the synapses a
brain is born with, so it is a construction setting.

| Setting | Raise it when | Lower it when |
| --- | --- | --- |
| `working_memory_scale` | context must matter from the first moment, before learning (the steady-rhythm chambers run the earlier 12) | the brain repeats itself and ignores what it sees now |
| `working_memory_amplitude` | the copy should weigh more in learning and in the settle | the carried context should fade out (0 switches it off) |
| `working_memory_decay` | the context changes slowly and should be averaged over moments | (default 0 carries only the last interpretation) |
| `working_memory_focus` | only the neurons that just changed should be carried | (default 0 carries everything) |
| `efference_amplitude` | the next action depends on what the brain just did | (default 0 has no copy) |

Saved brains keep their own settings and wiring: a brain saved with an older release loads
with the trace and the synapses it was built with. To rebuild the composition of releases
before 0.82.0, pass `working_memory_scale=12.0, working_memory_decay=0.2`.

## Check that the context does the work

Compare the same acquired brain with and without its context, on copies:

```python
import os
import tempfile

path = os.path.join(tempfile.mkdtemp(), "brain.npz")
brain.save(path)
intact, erased = Brain.load(path), Brain.load(path)
erased.working_memory.reset(1)  # forget the carried interpretation only
go = [[0.0, 0.0, 1.0]]
print(intact.act(go, greedy=True), erased.act(go, greedy=True))
```

If the answers differ where the history should matter, the brain is using its context. If they
never differ, it answers from the present observation alone.

## What context is not

- It is not a recording of past inputs. Two different histories that lead to the same
  interpretation are the same context. The [episodic memory](episodic.md) keys its records by
  a fixed fading trace of what was seen and done, which does tell such histories apart.
- There is no separate gate deciding what to store. A distractor that looks like real evidence
  can overwrite the context of a small brain; a wider association region resists it better.
- `act` settles the present moment with the carried context; it does not plan a whole future
  sequence at once.
- `predict` and `fit` answer independent samples and ignore the context; use `live`, `step`
  or `act` on a continuing stream.

See [episodic memory](episodic.md) for what happened in a situation before, [memory](memory.md)
for the longer-term associative stores, [sequential tasks](sequential-tasks.md)
for chains of actions, and the [composition guide](brain.md) for every default.
