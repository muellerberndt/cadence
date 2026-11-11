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
| `fields_tied` | the `fields` brain with `tied={("sensory", "association"): (3, 1)}`: its three pool rows read through one set of synapses ([tied projections](../../docs/tune.md#tied-projections-opt-in)) |
| `visual`, `pixels`, `pooled`, `random_code`, `learned_code` | the visual cortex, the raw pixels, 4 by 4 block means, `code=1024`, `learned_code=128` (Step 0) |
| `conv_matched`, `conv` | two convolutions as in the Atari network, a hidden layer at the `fields` brain's parameter count or at 256 units, policy and value heads, trained online by Adam on the same batches (`convnet.py`) |
| `conv_matched_tied` | the same baseline at the `fields_tied` brain's parameter count (5 hidden units) |

Controls, on any brain arm (`--control`): `blank` (the background alone, labels as drawn),
`shuffled` (each batch's labels permuted among its frames), `frozen` (no learning); uniform
random actions read 1/3.

Two conditions: `lesson` (each batch of 8 frames one `lesson`, the baseline one cross-entropy
step) and `reward` (8 streams through `step`, reward 1 for the right action and 0 otherwise,
each frame an episode; the baseline a one-step actor-critic). Panels are read by `predict` at
each budget, on the life without changing it. `--probe` adds ridge readouts of the representation
the cortex reads and of the association, and for a brain with learned fields of its pools summed
over the pool rows, fitted on 2,048 training-range frames: readings that locate a failure, never
part of a life. Every life under lessons reports its parameters and its analytic
multiply-accumulates per lesson frame and per read frame: a brain's phases' row sweeps of its
synapses and the fields' reading and learning of the frame, the baseline's forward pass and a
backward pass at twice the forward.

```sh
PYTHONPATH=src python benchmarks/pixels/pixels.py --arms fields conv_matched \
    --founders 0 1 --budgets 4096 16384 65536 --out results/lesson.jsonl --workers 2
python benchmarks/pixels/summary.py results/*.jsonl
python benchmarks/pixels/summary.py results/*.jsonl --panel 2048 --founders 10 \
    --brain fields_tied --baseline conv_matched_tied --against fields
PYTHONPATH=src python benchmarks/pixels/timing.py --backend cpu --out results/timing.jsonl
```

## Readings

### Step 0: what the brain gets from the frame (development founders 0 and 1)

Grey 84 by 84, lessons of 8 frames, 4,096 frames, every brain `recall=False`. The probe is the
ridge readout of the representation the cortex reads (`--probe`).

| What the cortex reads | Parameters | Brain, fresh frames | Brain, held-out band | Probe, fresh frames and band | Probe of the association |
| --- | --- | --- | --- | --- | --- |
| the visual cortex (8 maps of 3 by 3 fields, 53,792 neurons) | 4,013,316 | .734, .777 | .723, .664 | .742 .699, .805 .617 | .766, .789 |
| raw pixels, 7,056 inputs | 484,292 | .750, .781 | .734, .727 | .746 .676, .801 .598 | .754, .781 |
| 4 by 4 block means, 441 inputs | 34,472 | .723, .746 | .707, .508 | .758 .676, .801 .586 | .754, .773 |
| fixed random code, `code=1024` | 74,116 | .711, .746 | .555, .547 | .746 .652, .770 .586 | .746, .762 |
| learned code, `learned_code=128` | 931,604 | .391, .355 | .406, .344 | no unit active | .391, .320 |
| convolutional baseline, 256 hidden | 674,100 | .820, .824 | .672, .629 | .957 .297, .957 .281 | .859, .867 |

The learned code is silent at 7,056 inputs; the untaught fields and the fixed codes hold the
answer up to about .75 to .80 for a linear readout, and the brain reaches that level; a
representation that keeps every place apart leaves the held-out band unread.

### The declared run (cadence#284, fresh founders 2840 to 2845, 65,536 frames)

| Condition | Frame | Arm | Parameters | Fresh frames (range) | Band | Scale | Lighting | Distractor | At .90 | Within .05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lesson | 84 by 84 | `fields` | 47,812 | .974 (.930 to .992) | .948 | .975 | .832 | .947 | 6 of 6 | 4 of 6 |
| lesson | 84 by 84 | `fields_drawn` | 47,812 | .832 (.770 to .910) | .857 | .860 | .801 | .837 | 1 of 6 | 6 of 6 |
| lesson | 84 by 84 | `conv_matched` | 50,820 | 1.000 | .145 | 1.000 | .934 | .998 | 6 of 6 | 0 of 6 |
| lesson | 84 by 84 | `conv` | 674,100 | .995 (.973 to 1.000) | .270 | 1.000 | .913 | .995 | 6 of 6 | 0 of 6 |
| lesson | 84 by 84 | `fields`, blank screen | 47,812 | .386 | .352 | .380 | .404 | .379 | | |
| lesson | 84 by 84 | `fields`, labels of independent scenes | 47,812 | .357 | .337 | .363 | .374 | .349 | | |
| lesson | 84 by 84 | `fields`, frozen | 47,812 | .352 | .381 | .371 | .359 | .362 | | |
| reward | 84 by 84 | `fields` | 47,812 | .673 (.574 to .758) | .690 | .676 | .648 | .647 | | |
| reward | 84 by 84 | `fields`, blank screen | 47,812 | .372 | .352 | .377 | .400 | .376 | | |
| reward | 84 by 84 | `conv_matched` | 50,820 | .894 (.758 to .988) | .506 | .915 | .634 | .888 | | |
| lesson | 84 by 84 by 3 | `fields` | 56,004 | .767 (.734 to .809) | .772 | .756 | .657 | .755 | 0 of 6 | 4 of 6 |
| lesson | 84 by 84 by 3 | `conv_matched` | 55,465 | 1.000 | .083 | .999 | .786 | .998 | 6 of 6 | 0 of 6 |
| lesson | 160 by 120 | `fields` | 47,812 | .965 (.934 to .992) | .827 | .968 | .904 | .925 | 6 of 6 | 0 of 6 |
| lesson | 160 by 120 | `conv_matched` | 129,156 | 1.000 | .141 | 1.000 | .738 | .998 | 6 of 6 | 0 of 6 |

Where the readings fall short (ridge probes): on the two grey founders outside .05 (2841 on the
distractor, 2843 on the band) the pools read those panels at .988 to 1.000 and the association
at .914 and .949: the brain's readout trails what its pools hold. At 160 by 120 the pools of
founder 2843 after 16,384 frames read the band at .996: the same. In colour the pools of founder
2840 read .914 on fresh frames and .715 on the band: the 64 fields divide among the six hues.

Uniform random actions read 1/3. The shuffled control as first declared permuted each batch's
labels among its frames, which keeps a frame's own label in one draw of eight; it read .582 and
was replaced by labels of independent scenes before its re-read.

### Where the first run's invariance fell short, located

The first run's six founders, their lives replayed with the pools cached and the brain's read of
them varied one term at a time (panels of 2,048 frames, 65,536 frames; "within .05": founders
whose band, scale and distractor panels all read within .05 of fresh frames):

| What was varied | Fresh frames | Band | Scale | Distractor | Within .05 |
| --- | --- | --- | --- | --- | --- |
| nothing | .965 | .949 | .976 | .944 | 4 of 6 |
| the association at 128 | .965 | .941 | .980 | .954 | 6 of 6 (5 of 6 at 131,072 frames) |
| the association at 256 | .965 | .936 | .976 | .948 | 4 of 6 |
| the pools' scale into the cortex, 4 times | .978 | .942 | .991 | .961 | 3 of 6 |
| the lesson's step at a third | .963 | .963 | .974 | .937 | 4 of 6 |
| the lesson's step at three times | .966 | .905 | .974 | .948 | 2 of 6 |
| 32 rows per lesson | .967 | .969 | .981 | .944 | 4 of 6 |
| twice the exposure | .973 | .934 | .982 | .957 | 4 of 6 |
| every synapse's step at a tenth from 49,152 frames on | .956 | .892 | .969 | .924 | 1 of 6 |
| the fields held still from 16,384 frames on | .969 | .967 | .977 | .955 | 6 of 6 |
| fields that settle with use (`MATURITY` 2,048) | .973 | .973 | .982 | .970 | 6 of 6 |

On these six founders the fields' motion looked like the failing term: held still, or settling
with use, every founder read within .05. The second run tested that on fresh founders.

### The second run (cadence#284, amended: fresh founders 2846 to 2857, 65,536 frames)

Bars on 10 of 12 founders; panels of 2,048 frames. `fields` is the mechanism the amendment put
under test (fields that settle with use, `MATURITY` 2,048); `fields_restless` the first run's
fields, which keep their whole step and which the library keeps.

| Condition | Arm | Parameters | Fresh frames (range) | Band | Scale | Lighting | Distractor | At .90 | Within .05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lesson | `fields` (settling) | 47,876 | .967 (.847 to .991) | .939 | .977 | .811 | .959 | 11 of 12 | 9 of 12 |
| lesson | `fields_restless` | 47,876 | .963 (.880 to .988) | .941 | .975 | .808 | .945 | 11 of 12 | 10 of 12 |
| lesson | `fields_drawn` | 47,876 | .859 (.758 to .916) | .852 | .886 | .795 | .849 | 2 of 12 | 12 of 12 |
| lesson | `conv_matched` | 50,820 | .999 (.996 to 1.000) | .162 | 1.000 | .852 | .999 | 12 of 12 | 0 of 12 |
| lesson | `conv` | 674,100 | .999 (.997 to 1.000) | .200 | 1.000 | .806 | .998 | 12 of 12 | 0 of 12 |
| reward | `fields` (settling) | 47,876 | .633 (.474 to .722) | .665 | .634 | .599 | .605 | | |
| reward | `fields_restless` | 47,876 | .635 (.459 to .733) | .665 | .632 | .610 | .614 | | |
| reward | `fields`, blank screen | 47,876 | .374 (.326 to .391) | .366 | .372 | .374 | .375 | | |
| reward | `conv_matched` | 50,820 | .857 (.610 to .984) | .563 | .865 | .629 | .854 | 7 of 12 | 3 of 12 |

(The `fields` parameter counts include the settling fields' `use` table, 64 values, which the
library does not keep.) On the 256-frame panels of the first declaration `fields` reads 8 of 12
within .05 and `fields_restless` 10 of 12. Settling with use, selected on the six opened
founders where it read 6 of 6, does not carry to fresh founders, and the library keeps the
first run's fields.

Where the readings fall short: on every founder outside .05 the pools read the band at .973 to
.995 for a linear readout, the association at .894 to .978, and the brain at .830 to .925. An
object in the band spreads its pools' energy over the three pool rows (.22, .56, .23 of it on
one founder) where an object at a trained place puts .66 to .69 into its outer row and .31 to
.34 into the middle one: the middle row is taught by the edges of trained objects alone, and
the brain's read of it is the term that trails.

### The third run: the pool rows tied

(cadence#284, protocol 3: fresh founders 2858 to 2869, 65,536 frames of lessons, bars on 10 of
12 founders, panels of 2,048 frames.) `fields_tied` is the `fields` brain with
`tied={("sensory", "association"): (3, 1)}`: the association reads the three pool rows through
one set of 192 by 64 synapses, each stepped by the mean of its three copies' steps. No constant
was selected.

| Arm | Parameters | Multiply-accumulates per lesson frame | Per read frame | Fresh frames (range) | Band | Scale | Lighting | Distractor | At .90 | Band within .05 | Band, scale and distractor within .05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fields_tied` | 23,236 | 2.73e7 | 1.26e7 | .965 (.914 to .990) | .977 | .976 | .781 | .946 | 12 of 12 | 12 of 12 | 10 of 12 |
| `fields` | 47,812 | 2.71e7 | 1.22e7 | .965 (.913 to .989) | .947 | .977 | .818 | .955 | 12 of 12 | 9 of 12 | 8 of 12 |
| `conv_matched_tied` | 22,253 | 3.26e6 | 1.09e6 | .893 (.356 to 1.000) | .186 | .895 | .665 | .895 | 10 of 12 | 2 of 12 | 2 of 12 |
| `conv_matched` | 50,820 | 3.34e6 | 1.11e6 | 1.000 (.999 to 1.000) | .163 | 1.000 | .926 | .998 | 12 of 12 | 0 of 12 | 0 of 12 |
| `conv` | 674,100 | 5.21e6 | 1.74e6 | 1.000 (.999 to 1.000) | .192 | 1.000 | .887 | .999 | 12 of 12 | 0 of 12 | 0 of 12 |

(Of a brain's lesson frame the fields' reading and learning are 2.49e7; the rest is the phases'
row sweeps of its 41,344 synapses, which the tie leaves in place.)

| Gate | Reading | |
| --- | --- | --- |
| T1, the band within .05 of fresh frames with fresh frames at .90, on 10 of 12 | 12 of 12; every founder reads the band at or above its fresh frames | met |
| T2, band, scale and distractor each within .05, on 10 of 12 | 10 of 12; founders 2862 and 2864 read the distractor panel .062 and .086 under fresh frames | met |
| T3, the band gap of `fields` minus that of `fields_tied`, paired, above zero by two standard errors | +.029, standard error .011 | met |
| T4, fresh frames at .90 on 10 of 12; `conv_matched_tied` minus `fields_tied` at most .05; `fields_tied` at most .02 under `fields` | 12 of 12; -.072; +.001 (standard error .009) | met |

On the 256-frame panels: T1 12 of 12, T2 10 of 12, T3 +.023 (standard error .011).

The tie moves the band, from .947 to .977 at the same fresh frames. The association's probe
reads the band at .992 with the tie and .964 without; the pools read it at .989 and their sum
over the rows at .999 in both arms. It costs elsewhere: the lighting panel, outside the gates,
reads .781 against .818 (its gap .038 wider, standard error .010), and the distractor panel .946
against .955 (.010 wider, standard error .005). On the two founders outside .05 the pools read
the distractor panel at .986 and .993, their sum over the rows at .975 and .991, the association
at .931 and .890 and the brain at .928 and .863: the term is the brain's read of its code beside
an untrained distractor. The untied brain reads those two founders' distractor panels at .959 and
.930. The baseline at the tied brain's parameter count reads fresh frames at .998 or above on
ten founders and at .356 and .370 on two.

Development on opened founders 2846 to 2849, 2853 and 2855, which selected nothing:
`fields_tied` read the band at .969 against .920, and all three panels within .05 on 6 of 6
founders against 4 of 6.

### Seconds per frame (`timing.py`, the `fields` brain, median of 32 moments)

| Where | Frame | `act`, 1 stream | `lesson`, 1 stream | `act`, 16 streams | `lesson`, 16 streams | One stream within 28.6 ms |
| --- | --- | --- | --- | --- | --- | --- |
| one Intel Xeon 8488C core (c7i) | 84 by 84 grey | 2.6 ms | 3.3 ms | 3.1 ms | 3.2 ms | yes |
| one Intel Xeon 8488C core | 84 by 84 colour | 9.0 ms | 10.0 ms | 10.9 ms | 11.1 ms | yes |
| one Intel Xeon 8488C core | 160 by 120 grey | 8.5 ms | 9.5 ms | 12.1 ms | 12.2 ms | yes |
| NVIDIA L4, float32 (`CADENCE_TRITON=0`) | 84 by 84 grey | 3.6 ms | 6.5 ms | 3.3 ms | 3.5 ms | yes |
| NVIDIA L4, float32 (`CADENCE_TRITON=0`) | 84 by 84 colour | 15.3 ms | 18.1 ms | 10.3 ms | 10.5 ms | yes |
| MLX, Apple M4 (shared, load 17 to 33) | 84 by 84 grey | 6.7 ms | 13.5 ms | 5.6 ms | 6.0 ms | yes |
| MLX, Apple M4 (shared) | 84 by 84 colour | 23.0 ms | 29.6 ms | 25.9 ms | 26.5 ms | yes |
| one Apple M4 core (shared) | 84 by 84 grey | 5.9 ms | 7.6 ms | 6.2 ms | 6.0 ms | yes |
| one Apple M4 core (shared) | 84 by 84 colour | 20.4 ms | 22.5 ms | 24.5 ms | 24.8 ms | yes |

The default visual cortex on the same Xeon core: `act` 163 ms and `lesson` 185 ms per frame at
84 by 84 grey (4,013,316 parameters), 220 ms and 251 ms in colour (5,038,020). The fields read
every frame on the host, so 16 streams cost about 16 frames; on the M4 the fields alone take 2.6
ms per grey frame to read and 1.5 ms to learn. On the L4 the small-group Triton kernel of the
torch backend does not compile under Triton 3.3 at any row count read
([issue 299](https://github.com/muellerberndt/cadence/issues/299)); the readings above take the
torch path without it.
