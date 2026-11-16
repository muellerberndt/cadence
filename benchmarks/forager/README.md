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
  walk on, right, at the moments a plant is in sight and no bite, a walk toward the plant or a
  turn from a rock or a wall in the way) from the association's settled activity, fitted on every
  other sample, in the order the moments and the bodies give them, and read on the rest;
  balanced accuracy, chance one third. The same readout from the observation and from its code
  reads what the senses hold.
- **While taught:** the agreement of the brain's greedy answers with the teacher's, each right
  after the moment's lesson, over the last 100 moments of lessons before the reading.

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

### The declared run: F1 holds, F2 fails

Founders 1011100 to 1011111 at `44f2fed2`, as declared above (Cadence 0.101.1 before its
release; [receipt](results/declared.json)). Founders 1011110 and 1011111 were read in two parts
on a shared machine, the `code` arm and then the other three, with the same source and seeds.
Each cell is a life's mean over its four readings.

| founder | turn in the cortex, `code` | `both` | teacher's path, `code` | `both` | own path, `code` | `both` | plants, `code` | `both` | while taught, `code` | `both` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1011100 | .767 | .806 | .702 | .708 | .386 | .705 | 5.25 | 9.56 | .654 | .704 |
| 1011101 | .726 | .806 | .729 | .659 | .591 | .662 | 8.47 | 8.77 | .642 | .710 |
| 1011102 | .813 | .881 | .693 | .696 | .590 | .661 | 7.03 | 8.30 | .653 | .733 |
| 1011103 | .747 | .852 | .605 | .698 | .513 | .528 | 5.05 | 8.09 | .618 | .675 |
| 1011104 | .791 | .855 | .609 | .713 | .497 | .609 | 5.97 | 8.50 | .650 | .727 |
| 1011105 | .781 | .850 | .716 | .685 | .473 | .577 | 6.88 | 7.86 | .629 | .712 |
| 1011106 | .795 | .869 | .655 | .676 | .510 | .678 | 6.45 | 9.45 | .603 | .713 |
| 1011107 | .696 | .845 | .661 | .658 | .500 | .540 | 4.27 | 6.38 | .588 | .739 |
| 1011108 | .724 | .823 | .690 | .644 | .569 | .654 | 7.95 | 8.78 | .638 | .684 |
| 1011109 | .726 | .823 | .684 | .677 | .410 | .515 | 5.16 | 6.59 | .660 | .717 |
| 1011110 | .782 | .829 | .666 | .697 | .416 | .500 | 5.22 | 6.19 | .591 | .689 |
| 1011111 | .724 | .804 | .650 | .632 | .545 | .584 | 7.30 | 8.05 | .628 | .725 |

| Gate | Founders that hold it | |
| --- | --- | --- |
| F1, the turn in the cortex | 12 of 12 (`both` .804 to .881, `code` .696 to .813; by .039 to .149) | holds |
| F2, agreement on the teacher's path | 6 of 12 (`both` .632 to .713, `code` .605 to .729; mean .679 against .672) | fails |

The declared readout is fitted on every other sample, so moments of one body fall on both
sides of it. Fitted on the even bodies and read on the odd ones (`<source>_bodies`, added after
the run; [receipt](results/bodies.json), founders 1011100 to 1011103 read again, every declared
mean to the last digit), the state reads .806, .810, .884 and .872 with `both` against .737,
.722, .826 and .756 with `code`, and the observation .848 to .903, the code .829 to .888.

Without a bar, `both` against `code` on the same founders: on its own path the brain agrees
with the teacher more on 12 of 12 (.601 against .500 on average) and eats more on 12 of 12 (8.04
plants a body in 150 moments against 6.25; the teacher 11.5 to 14.3, a random body under 0.4);
while taught it agrees more on 12 of 12 (.711 against .630). The newborn's readout is .518
against .498. The observation and its code hold the turn at .875 and .850 for a linear readout.

| Arm | turn in the cortex | teacher's path | own path | plants | while taught |
| --- | --- | --- | --- | --- | --- |
| `code` | .756 | .672 | .500 | 6.25 | .630 |
| `both` | .837 | .679 | .601 | 8.04 | .711 |
| `observation` | .824 | .688 | .571 | 7.53 | .694 |
| `none` | .788 | .671 | .471 | 6.01 | .677 |

**Located (F2).** On the teacher's path the walks make up about three in five moments, and with
`both` the brain turns or bites at more of them. Read by action after the run (the reading's
`by_action`, added for this; [receipt](results/by_action.json), the declared lives read again
with their means to the last digit) on the two founders where `both` falls furthest below `code`
(1011101 and 1011108), `both` agrees with the teacher's walks at .539 and .513 against .739 and
.701, and with its turns and bites at .747 to .939 against .553 to .824. The lessons weigh every
action alike, so a turn's row weighs about three times a walk's; the brain that reads the
bearing learns the turns and moves the border between a turn and a walk off the teacher's. On
its own path that border costs it no plants. Agreement over the actions, each weighed alike,
reads .774 and .764 for `both` against .733 and .674 for `code` on those two founders.

