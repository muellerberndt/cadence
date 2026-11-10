# Pixel chamber: a screen-dependent policy at 84 by 84

This chamber asks whether a composed brain learns a policy that depends on what a rendered
screen shows, at the size of an Atari or VizDoom frame, and whether what it learns holds where
and how an object appears. It is the instrument of
[issue 284](https://github.com/muellerberndt/cadence/issues/284), prerequisite 3 of the
feasibility check ([issue 287](https://github.com/muellerberndt/cadence/issues/287)).

## The task

Every frame is rendered by the chamber (`chamber.py`), so the tests need no game data: 84 by 84
in grey (the gated shape) and in colour, and 160 by 120 in grey. A frame holds one task object
on a lit background with noise, and up to two distractors. The object's kind is told by its
shape alone:

| Object | Shape | Rewarded action |
| --- | --- | --- |
| food | a filled disc | approach: the action toward its side |
| hazard | a diagonal cross | avoid: the action away from its side |
| target | a ring | aim: toward it, `FIRE` once it is in the centre |

Three actions, `LEFT`, `RIGHT` and `FIRE`; eight situations drawn with equal chance (food and
the hazard left or right, the target left, right, or centre twice). Food and the hazard on the
same side take opposite actions, so the answer needs the object's kind and its place.

Each factor of a frame has a trained range and a held-out range that no training frame shows:

| Factor | Trained | Held out (its panel) |
| --- | --- | --- |
| vertical place, share of the height | 0.20 to 0.38 and 0.62 to 0.80 | 0.45 to 0.55 (`positions`) |
| scale, of a radius of 0.11 of the shorter side | 0.75 to 0.90 and 1.10 to 1.25 | 0.95 to 1.05 (`scales`) |
| lighting, background and object | 0.05 to 0.25, 0.75 to 1.00 | 0.35 to 0.45, 0.55 to 0.65 (`lighting`) |
| distractors | a bar, a square, in half the frames | a triangle in every frame (`distractor`) |

`held_out` is fresh frames at the trained ranges. Each panel is 256 frames, a fixed draw per
founder. Invariance on a factor is its panel within .05 of `held_out`.

## Arms and controls

| Arm | What it is |
| --- | --- |
| `fields` | `Brain.compose(inputs=(84, 84), actions=3, learned_fields=64, recall=False, teaching="normalized")`: [learned receptive fields](../../docs/senses.md#learned-receptive-fields) |
| `fields_drawn` | the same brain with its fields held at their draw from birth |
| `visual`, `pixels`, `pooled`, `random_code`, `learned_code` | the visual cortex, the raw pixels, 4 by 4 block means, `code=1024`, `learned_code=128` (Step 0) |
| `conv_matched`, `conv` | two convolutions as in the Atari network, a hidden layer at the `fields` brain's parameter count or at 256 units, policy and value heads, trained online by Adam on the same batches (`convnet.py`) |

Controls, on any brain arm (`--control`): `blank` (the background alone, labels as drawn),
`shuffled` (each batch's labels permuted among its frames), `frozen` (no learning); uniform
random actions read 1/3.

Two conditions: `lesson` (each batch of 8 frames one `lesson`, the baseline one cross-entropy
step) and `reward` (8 streams through `step`, reward 1 for the right action and 0 otherwise,
each frame an episode; the baseline a one-step actor-critic). Panels are read by `predict` at
each budget, on the life without changing it. `--probe` adds ridge readouts of the representation
the cortex reads and of the association, fitted on 2,048 training-range frames: readings that
locate a failure, never part of a life.

```sh
PYTHONPATH=src python benchmarks/pixels/pixels.py --arms fields conv_matched \
    --founders 0 1 --budgets 4096 16384 65536 --out results/lesson.jsonl --workers 2
python benchmarks/pixels/summary.py results/*.jsonl
PYTHONPATH=src python benchmarks/pixels/timing.py --backend cpu --out results/timing.jsonl
```

## Readings

READINGS
