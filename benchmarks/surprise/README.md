# One surprise signal, two measures of it

Issue: [#337](https://github.com/muellerberndt/cadence/issues/337).
`Brain.compose(surprise=True)` reads each neuron's surprise from the residual its settle starts
from; the guide is
[surprise, and the focus it draws](../../docs/start.md#surprise-and-the-focus-it-draws-opt-in).
This chamber reads a second measure beside it, on the same brains and the same streams, at every
moment, so the two can be compared:

- **residual** (the library's, `brain.surprise`): the size of the potential equation's error at
  the state the stream carried into the moment, under the moment's drive, before the settle
  repairs it. For a carried state that settled within the tolerance it is the change of each
  neuron's drive since that settle.
- **move** (in this chamber alone): how far the moment moved each neuron's activity, the state
  after the moment less the state carried into it.

Both sizes pass through one law with one set of genes (`cadence.surprise.Surprise` at its
founders, `tolerance=2.0`, `floor=0.05`, `adapt=0.5`, `recover=0.01`): a size against the size
the neuron is used to. A reading is the moment's largest surprise over the neurons: over the
whole brain (`Surprise.moment`), over the sensory neurons alone (what the focus reads), and
over the association and motor regions.

## The chamber

`chamber.py` composes brains of 24 inputs and 6 actions
(`Brain.compose(24, 6, seed=seed, backend="cpu", surprise=True)`, 230 neurons). A pattern holds
4 inputs at one. A stream's patterns are drawn from the first 16 inputs; the other 8 are spare,
for a thing the brain has not had.

```bash
PYTHONPATH=src python benchmarks/surprise/chamber.py --seeds 0 1 2
```

`--main <src>` names the `src` folder of another source tree of the library for section (e), and
`--json <file>` writes every reading. `test_surprise_chamber.py` runs each section in short;
`tests/test_surprise.py` holds the signal's contract.

## Readings (seeds 0, 1 and 2)

### (a) A repeating stream with rare deviants

A stream repeats (`repeat`: one pattern; `cycle`: six patterns in order) and one moment in ten
on average brings a deviant in place of the standard: one pattern on spare inputs that comes
back (`units`), one new combination of the inputs the cycle's own patterns use that comes back
(`combination`), or a new such combination each time (`fresh`). The area is the area under the
curve of deviant against standard moments (0.5: the two cannot be told apart), the mean with the
range over the seeds; the means are the reading at the deviants / at the moment after one / at
the standards, with the ratio of the first to the last. Lived by greedy acts (the weights stay)
or by `live` with reward only.

| Stream | Deviant, one in | Lived by, read over | residual: area | residual: means | move: area | move: means |
| --- | --- | --- | --- | --- | --- | --- |
| repeat | units, 10 | act, whole brain | 0.54 (0.54 to 0.54) | 0.30 / 0.27 / 0.01 (26) | 0.55 (0.54 to 0.57) | 0.22 / 0.21 / 0.01 (19) |
| repeat | units, 10 | act, senses | 0.52 (0.52 to 0.53) | 0.13 / 0.00 / 0.00 | 0.52 (0.52 to 0.53) | 0.09 / 0.00 / 0.00 |
| repeat | units, 60 | act, whole brain | 0.70 (0.67 to 0.75) | 0.43 / 0.46 / 0.01 (40) | 0.71 (0.67 to 0.78) | 0.36 / 0.37 / 0.01 (44) |
| repeat | units, 60 | act, senses | 0.66 (0.60 to 0.71) | 0.20 / 0.00 / 0.00 | 0.64 (0.60 to 0.68) | 0.16 / 0.00 / 0.00 |
| cycle | units, 10 | act, whole brain | 0.54 (0.53 to 0.56) | 0.31 / 0.28 / 0.02 (20) | 0.64 (0.57 to 0.74) | 0.26 / 0.28 / 0.01 (19) |
| cycle | units, 10 | act, senses | 0.52 (0.52 to 0.53) | 0.13 / 0.00 / 0.00 | 0.52 (0.52 to 0.53) | 0.09 / 0.00 / 0.00 |
| cycle | combination, 10 | act, whole brain | 0.50 (0.49 to 0.50) | 0.00 / 0.09 / 0.00 (0) | 0.58 (0.55 to 0.62) | 0.05 / 0.13 / 0.01 (8) |
| cycle | combination, 10 | act, senses | 0.50 (0.50 to 0.50) | 0.00 / 0.00 / 0.00 | 0.50 (0.50 to 0.50) | 0.00 / 0.00 / 0.00 |
| cycle | fresh, 10 | act, whole brain | 0.50 (0.49 to 0.52) | 0.01 / 0.06 / 0.01 (1) | 0.57 (0.50 to 0.65) | 0.05 / 0.13 / 0.01 (4) |
| cycle | fresh, 10 | act, senses | 0.50 (0.50 to 0.50) | 0.00 / 0.00 / 0.00 | 0.50 (0.50 to 0.50) | 0.00 / 0.00 / 0.00 |
| repeat | units, 10 | live, whole brain | 0.55 (0.54 to 0.55) | 0.30 / 0.24 / 0.01 (45) | 0.54 (0.53 to 0.55) | 0.22 / 0.18 / 0.01 (35) |
| repeat | units, 10 | live, senses | 0.52 (0.52 to 0.53) | 0.13 / 0.00 / 0.00 | 0.52 (0.52 to 0.53) | 0.09 / 0.00 / 0.00 |
| cycle | fresh, 10 | live, whole brain | 0.53 (0.49 to 0.55) | 0.02 / 0.04 / 0.01 (4) | 0.55 (0.49 to 0.59) | 0.04 / 0.07 / 0.01 (6) |
| cycle | fresh, 10 | live, senses | 0.50 (0.50 to 0.50) | 0.00 / 0.00 / 0.00 | 0.50 (0.50 to 0.50) | 0.00 / 0.00 / 0.00 |

At the senses no standard moment reads in either measure, so those rows carry no ratio. Over
the association and motor regions the residual's largest reading at any moment is 0.00 on every
stream lived by greedy acts and 1.72 on the streams lived by `live`; the move's is 2.40 and
2.40.

### (b) Onset and habituation

One input is held for 20 moments; a change comes for one moment and the held input returns for
four, eight times over; then a new change comes on spare inputs. Each cell: the whole brain's
reading, and in brackets the senses' and the association and motor regions' (the mean of the
seeds; the whole brain's and the senses' are the same on every seed to the second decimal).

| Moment | residual | move |
| --- | --- | --- |
| the input held, its last moment | 0.00 (0.00; 0.00) | 0.00 (0.00; 0.00) |
| the change, its first coming | 3.00 (3.00; 0.00) | 2.22 (2.22; 0.17) |
| the change, its second coming | 4.09 (0.00; 0.00) | 2.90 (0.00; 1.13) |
| the change, its third, fourth and eighth coming | 0.00 (0.00; 0.00) | 0.00 (0.00; 0.00) |
| the held input back after the first change | 4.09 (0.00; 0.00) | 2.90 (0.00; 1.56) |
| a new change on spare inputs | 3.00 (3.00; 0.00) | 2.22 (2.22; 0.00) |

### (c) Learned, not built

Two brains of one seed live a regular sequence of six patterns for 40 cycles: `exposed` by greedy
acts (the weights stay; each neuron's usual size learns) and `taught` by `step` with the next
element as the teacher (the weights learn the order: its greedy answer names the next element
at 1.00 of a regular cycle's moments, the exposed brain's at 0.22). Copies of each brain are
then shown one more cycle, regular or broken. Each cell: the mean reading with, in brackets, the
share of the moments that read a surprise, over the whole brain; then the same at the senses.
A break's moments are those whose step from the moment before is none of the sequence's steps.

| Brain | Moments | residual | move |
| --- | --- | --- | --- |
| exposed | the sequence's first cycle, a fresh brain | 3.75 (1.00); 2.16 (0.72) | 2.67 (1.00); 1.61 (0.72) |
| exposed | the regular cycle, after 40 | 0.00 (0.00); 0.00 (0.00) | 0.00 (0.00); 0.00 (0.00) |
| exposed | two neighbours change places | 0.06 (0.22); 0.00 (0.00) | 0.24 (0.67); 0.00 (0.00) |
| exposed | an element twice, the next left out | 0.04 (0.33); 0.00 (0.00) | 0.12 (0.33); 0.00 (0.00) |
| exposed | an element on spare inputs | 3.55 (1.00); 1.50 (0.50) | 2.56 (1.00); 1.11 (0.50) |
| exposed | an element a new combination of the sequence's inputs | 0.16 (0.33); 0.00 (0.00) | 0.23 (0.83); 0.00 (0.00) |
| taught | the sequence's first cycle, a fresh brain | 3.80 (1.00); 2.16 (0.72) | 2.71 (1.00); 1.61 (0.72) |
| taught | the regular cycle, after 40 | 0.00 (0.00); 0.00 (0.00) | 0.00 (0.00); 0.00 (0.00) |
| taught | two neighbours change places | 0.19 (0.44); 0.00 (0.00) | 0.38 (0.67); 0.00 (0.00) |
| taught | an element twice, the next left out | 0.09 (0.17); 0.00 (0.00) | 0.26 (0.67); 0.00 (0.00) |
| taught | an element on spare inputs | 3.54 (1.00); 1.50 (0.50) | 2.56 (1.00); 1.11 (0.50) |
| taught | an element a new combination of the sequence's inputs | 0.12 (0.33); 0.00 (0.00) | 0.27 (0.67); 0.00 (0.00) |

### (d) Cost

Milliseconds a greedy act with the signal off and on, and microseconds of the parts: the
residual's one extra evaluation (`NeuralGraph.residual(per_unit=True)`), the law
(`Surprise.observe`) and the move (a subtraction and the law). Medians of 400 moments, the
median of five runs; one thread of an Apple-silicon laptop shared with other work (a load
average near 9), so the ratios carry further than the absolute times.

| Brain | neurons | synapses | off, ms | on, ms | residual, us | law, us | move, us |
| --- | --- | --- | --- | --- | --- | --- | --- |
| composed | 230 | 11,008 | 0.29 | 0.40 | 53 | 12 | 12 |
| wide (`modules=(256,)`) | 614 | 93,184 | 0.34 | 0.48 | 78 | 15 | 15 |
| composed, compiled moment (`learning_compiled_moment=True`) | 230 | 11,008 | 0.08 | 0.21 | 53 | 12 | 12 |
| wide, compiled moment | 614 | 93,184 | 0.13 | 0.29 | 87 | 17 | 17 |

- **The residual costs one more evaluation a moment**: one product of the weights with the
  carried activity and one activation, in NumPy on the host, beside the law's arithmetic. With
  the signal a greedy act takes 0.11 to 0.16 ms more, 1.4 times the act without it. The move
  needs no evaluation: both of its states are held by the stream.
- **With the compiled moment the signal costs more than its parts** (2.2 to 2.6 times the act
  without it): a greedy act of such a brain runs as one compiled call that keeps no carried
  state apart from its solve, and a brain with surprise takes the same calls in turn (the same
  action and state, to the bit).
- The settle's own first check computes the same error and reads its largest size alone; a
  solve that handed back that check's error a neuron would spare the extra evaluation.

### (e) To the bit

One short life (30 moments of `live` with reward only, then 15 of `step` with a teacher at
every third on two fresh streams) is hashed over its actions, its weights and every array of
its saved file, without the surprise's own arrays and record.

- The option unnamed and `surprise=False` give one digest.
- `surprise=True` gives that digest: the signal changes no action, weight or saved array beside
  its own.
- The unnamed life under the source of release 0.99.1 (`--main`, a checkout of `v0.99.1`) gives
  that digest: with the option unset a brain is that release's, to the bit.

## Where the two measures agree, and where they differ

**They agree** on which moments surprise wherever a thing is new to the senses:

- An input held reads nothing; a change reads at once; the same change reads nothing from its
  third coming; a new change reads as the first did (b).
- A lived sequence reads nothing after 40 cycles, and an element on spare inputs reads at every
  one of its moments, in both brains (c).
- At the senses the two read the same moments on every row of (a) and (c): their areas lie
  within 0.02 of each other and their shares are equal.

**They differ:**

- **Where it reads.** The residual reads the neurons a drive enters: the senses and the regions
  the memories drive (the working trace's, the heard history's, the recall memory's ports, and
  under `live` the neurons the place store's records read into). A neuron that no drive enters
  carries no residual at the start of a settle: under greedy acts the association and motor
  regions read 0.00 at every moment of (a) and (b). The move reads those regions as well (1.13
  and 1.56 at the second coming and at the return in (b), up to 2.40 in (a)).
- **Scale.** The residual is in units of drive (a first change of one unit of sensory drive
  reads `log(1 / 0.05)`, 3.00; a neuron a memory drives at amplitude 3 reads 4.09). The move is
  in units of activity (2.22 and 2.90 for the same moments).
- **A new arrangement of known inputs.** The residual reads none of it at the moment it comes:
  in (a) a new combination of the cycle's own inputs reads 0.00 against 0.00 at the standards
  (area 0.50), since each neuron's drive changes by a size the neuron knows. The move reads a
  little there (0.05 against 0.01; area 0.58, and 0.57 with a range of 0.50 to 0.65 for a new
  combination each time). In (c) the move reads a surprise at more of an order break's moments
  (0.67 of them against 0.22 and 0.44). These readings are small beside a new thing's (means
  of 0.23 to 0.38 against 2.56) and rest on few moments: a broken cycle holds two or three
  break moments a seed, and the seeds' means run from 0.00 to 1.06.
- **A deviant on spare inputs within a cycle.** The move's area over the whole brain is 0.64
  (0.57 to 0.74), the residual's 0.54 (0.53 to 0.56); at the senses both read 0.52.
- **Cost and time.** The residual costs one more evaluation a moment (d) and is known before
  the settle runs; the move costs a subtraction and is known once the settle has finished.

## What both measures read alike of the signal itself

- **A deviant that returns is usual by its third coming.** The patterns are zeros and ones, so
  an input changes by one size, and the founder genes take half of an excess at once and give
  back a hundredth a moment. One in ten moments a deviant on spare inputs reads at its first
  two comings and at none after (on the repeating stream areas of 0.52 to 0.55, with a ratio of
  means of 19 to 45); one in sixty it reads again (0.70 and 0.71).
- **The moment after a change reads as high as the change.** The neurons the memories drive
  (the heard history, the recall memory's ports, the working trace) take the change in one
  moment later, at amplitude 3, so `moment` over the whole brain reads the echo (0.27 after a
  deviant against 0.30 at it); the senses read none of it.
- **Neither reads order.** The brain forecasts no next input: it carries its last equilibrium
  into a moment. Two neighbours changing places and an element left out read 0.00 at the senses
  in both measures, in the brain that was taught the order as in the brain that was not.

## What is designed, by name

The law and its genes; the potential equation as the residual read; the largest over the
neurons as `moment`; this chamber's streams. Learned: each neuron's usual size, and whatever the
brain's weights and memories bring into the drive. The readings are seeds 0, 1 and 2 of one
brain size on patterns of zeros and ones, lived by greedy acts but for the rows of (a) lived by
`live` and the taught brain of (c); the times are one shared machine's.
