# Forager chamber: a body taught by imitation, its cortex reading the code or the senses too

A small animal's brain with senses by name and a fixed code where they meet
(`Brain.compose(senses=..., code=128, modules=(32, 32))`) is taught to forage by a teacher that
turns toward the nearest plant in sight, walks to it and bites it. With `code=units` the cortex
reads the code alone; a code of 128 units over 104 inputs holds which senses are on together,
and a body's action here follows a bearing and a distance. The chamber reads, for
[`cortex_reads`](../../docs/senses.md#what-the-cortex-reads), whether the cortex holds that
geometry and whether the brain imitates the teacher, when the cortex reads the code alone and
when it reads the code and the senses.

`world.py` holds the worlds and the teacher: each body on a walled ground of 12 m with plants and
rocks, its senses in the layout of a larger simulation's animals (eyes in eight sectors, the
ears and a heard call, the thing it attends, the nearest food as what, bearing and distance, and
its body; 104 inputs). `chamber.py` holds the life and the readings; its docstring gives each
step, `summary.py` the gates.

```sh
PYTHONPATH=src python benchmarks/forager/chamber.py --founders 0 1 --out dev.json
python benchmarks/forager/summary.py dev.json
python -m pytest -q benchmarks/forager
```

## The arms

| Arm | The brain |
| --- | --- |
| `code` | `code=128`: the cortex reads the code (the founder) |
| `both` | `code=128, cortex_reads="both"`: the cortex reads the code and the senses |
| `observation` | `code=128, cortex_reads="observation"`: the cortex reads the senses, the memories key on the code |
| `none` | no code: the cortex reads the senses and the memories key on them |

## The life and the readings

Sixteen bodies, one stream each, live 1,200 moments of lessons: at every moment the brain is
shown the teacher's action (`Brain.lesson`, each action's rows weighed so that every action
pulls alike, at most 8) and acts greedily. Half of the bodies move as the brain answers and half
as the teacher shows; every 100 moments the halves change places and the worlds are drawn anew.
Every 300 moments a frozen copy is read on sixteen worlds no lesson used (150 moments each):

- **On its own path:** every body moves as the copy answers; its agreement with the teacher at
  the states it reaches, and the plants it eats (against the teacher's and a random body's on the
  same worlds).
- **On the teacher's path:** the teacher moves every body, and the copy and the founder's newborn
  brain settle on each moment, so every arm is read on the same states: the copy's agreement
  with the teacher, and **the turn in the cortex**, a ridge readout of the teacher's turn (left,
  walk on, right, at the moments a plant is in sight and no bite) from the association's settled
  activity, fitted on the even moments and read on the odd ones, balanced accuracy, chance one
  third. The same readout from the observation and from its code is the ceiling of what the
  senses hold.

A life's reading is the mean over its four readings.

## Declared protocol

Declared before the first life of its founders. Founders **1011100 to 1011111** (twelve, fresh).
Development used founders 0 to 3. One life a founder and arm on the CPU backend
(`backend="cpu"`), one thread.

| Gate | Reading | Bar |
| --- | --- | --- |
| F1 | the turn in the cortex: the state's readout with `both` above the same founder's with `code` | on 10 of 12 |
| F2 | imitation on the teacher's path: the agreement with `both` above the same founder's with `code` | on 10 of 12 |

Reported without a bar: the agreement and the plants eaten on the brain's own path, the agreement
while taught, the newborn's readout, and the `observation` and `none` arms.

**Expectations, from development (founders 0 to 3).** F1 on every founder (`both` .823 to .855
against `code` .637 to .750); F2 on every founder, by .009 to .147 (`both` .636 to .763 against
`code` .616 to .665), and F2 may miss on two or three founders where the code imitates well. On
the brain's own path `both` ate more on 3 of 4 (8.8 plants a body in 150 moments against 6.3,
the teacher 11.8 to 14.0, a random body under 0.2). `observation` read about as `both` (.825 and .711), `none`
below both (.799 and .678).

## Readings

READINGS-PLACEHOLDER
