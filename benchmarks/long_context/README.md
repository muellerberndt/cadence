# Long context: how far the read reaches

These instruments read [#268](https://github.com/muellerberndt/cadence/issues/268): a transformer's
attention reaches every token in its window, in one step, at the same quality. How far does the
recall memory's read reach under the read's rule, what fails first as context grows, and what does
it cost? They measure settings of the library against each other and against independently scored
networks; they establish no default.

## The instruments

`reach.py` lives one brain per spec and reads it frozen. The answer moments are counted, so a life's
work stays comparable across distances: a spec names `answers` (answer batches taught, 16 streams
each), and the life runs whole episodes until it has them. A taught answer moment has its lesson
before the act, as in the [sequence battery](../sequence/README.md), and only rows answered wrong are
taught.

| Task | A stream, then | The answer | What it asks of the read |
| --- | --- | --- | --- |
| `nback_dense` | tokens from 8; after enough moments every 4th moment is a cue for one of four odd distances up to `n` | the token that many moments back | the plain reach test: a position the cue names, no content to find it by |
| `periodic` | a motif of period `P` repeated four times, a tenth of the symbols varied | from the second period on, the motif's next symbol | a position nothing cues, at every moment |
| `lookback` | tokens from 4, then a cue, `first`, `second` or `last` | the token at that place | taught at 5 to 12, read at 40 to 1,000: a place in the episode, at lengths no lesson reached |
| `lookback_dense` | one story with a cue every 4th moment | the token at that place | the same taught inside the length |

The dense forms hold a life's work near constant: 1,500 answers cost 6,000 to 12,000 moments at any
distance, where one lesson per story costs a story per lesson.

**Controls that need no distant moment.** `lookback` with `same` holds one token through the whole
story. `nback_dense` at distances up to 7 among 1,280 slots and `periodic` at a period of 8 on a
stream of 1,000 moments read whether many slots or a long stream hurt a read that needs neither.

**Slots.** `recall_capacity` scales with the distance, at 1.25 or 2.56 times it ([below](#slots)).
The table is exact to the capacity (`recall_position` at the capacity), or the founder's 32 with or
without the grid.

**What a frozen read reports**, per condition, for head 0 and read-only before the act: the answer,
each port's named token (the held read), mass and first choice on the right slots, the distance of the
first choice from them, the effective slots (the exponential of the attention's entropy), and each
term's share of the margin between the better right slot and the best other slot and the slots beside
it (content, table, anchor, relative, slope, grid). The right slots are the answer's own moment, on the
recall port, and the slot before it, on the next port; on a periodic stream every period back as well.

`cost.py` reads the read's cost against its slots: seconds per moment split into read, write and the
rest, sweeps, and memory. `nets.py` lives the matched transformer and LSTM on the same episodes with
the window to match, and reads both comparisons in one life: equal exposure at the brain's answers,
equal compute at the brain's seconds on the same machine and one thread.

```bash
PYTHONPATH=src python benchmarks/long_context/reach.py --plan plan.json --out DIR/rows.jsonl --workers 6
PYTHONPATH=src python benchmarks/long_context/nets.py --plan nets.json --out DIR/nets.jsonl --workers 4
PYTHONPATH=src python benchmarks/long_context/cost.py --out DIR/cost.jsonl --workers 3
python -m pytest -q benchmarks/long_context
```

## Readings

Every number below is from a frozen snapshot on one c7i.16xlarge shared with other work, one thread per
process; seconds are shared-load seconds. **Development** readings are on founders 0 to 2: three lives are a
screen, reported life by life where they split. **Declared** readings are on fresh founders under a protocol saved
before the first life, with nothing tuned after it. On a six-life mean a difference under about .07 is noise.

`Fd` is `Brain.compose(..., episodic=False, teaching="deep", recall=True, recall_recency=0.0,
learning_surprise=True, recall_boundary=True, recall_anchor=True, recall_slope=0.1, recall_position_rate=0.01,
recall_relative=True)`; `FdT` adds `working_memory_amplitude=0.0`, the trace silent. The lives were read with the
contrast as the option it then was, `recall_address_contrast=True`, which is now the memory's only address lesson.
"The grid" is `recall_grid=(7, 8, 9, 11)` on the founder's table of 32, in the library's form; "without the peak" is
the form that steps the residues on the plain gradient. A period's reading is the mean over the three answered
periods. Chance is .17 on periods and .125 on n-back.

**The finding.** A place the episode marks reaches as far as the slots hold it: look back taught at 5 to 12 reads
`first` and `second` at 1.00 at 1,000 moments, and nothing but the slots limits it. A position nothing marks needs a
code for the distance. An exact table reads n-back to 100 by 6,000 answers and stays at chance at 400, and its size
grows with the slots. The grid reads n-back at 1.00 at 400 by 6,000 answers and at .96 to 1.00 at 1,000 by 12,000, at a fixed size.
At 100 moments `Fd` with the grid holds its four bars on six fresh founders, where the exact table fails the period's
bar. Periods of 400 and 1,000 do not hold. Beside the matched networks the read wins n-back at 100, 400 and 1,000 at
equal exposure and at equal compute, draws level on a period of 100 at equal compute, loses seconds on dense look back,
and on long periods trades reliability for reach: the transformer holds 400 and stays at chance at 1,000. In the sequence composition
the grid with 128 slots reads a period of 40 at .999 where table 64 reads .821 to .925, and every other gate is the same to the digit.

### Slots

A distance past `recall_capacity` has no slot to land on, so the slots scale with the context. The first runs took
the smallest of 64, 128, 512 and 1,280 holding 1.25 times the distance. On a stream that repeats that is too few: the
slots hold the last period and a noisy copy of it, a read one period back meets a tenth of its symbols varied, and the
three answered periods have a ceiling of r1 1.00, r2 and r3 .917, mean .944. At 2.56 times the distance (40 to 128, 100
to 256, 400 to 1,024, 1,000 to 2,560) the slots still hold the first period and the read lands on it: effective slots 2
at a period of 40, 4 to 19 at 100.

### Which term fails first

Each term was read by its margin: its share of the difference between the better right slot and the best other slot,
and the slot beside it, at the frozen read.

| Order | Term | Where | Reading (development) |
| --- | --- | --- | --- |
| 1 | Slots | a distance past `recall_capacity` | look back `first` and `second` .10 to .39 with mass .00 and `last` .91 to 1.00: 16 slots at length 40, 64 at 100, 256 at 400 |
| 2 | Position code | a distance past `recall_position` | table 64, n-back at 100: distances 25 and 51 read .81 to .97, distances 75 and 101 read .13 to .18, neighbour margin .00 |
| 3 | Slope's step | exact table at 400 and up | n-back at 400, 6,000 answers: .11 to .15, slope margin -7.4; at 1,000, 1,500 answers: .09 to .14, slope margin -17.5 |
| 4 | Lesson's size | every distance | the right slot's share of a lesson's gradient is about `(tokens - 1) / slots`; right share .5 at about 1,000, 1,700 and 3,400 answers at n of 16, 40 and 100 on the exact table |
| 5 | Relative term | periods of 400 and up with the grid | at fallen reads its margin is -1.5 to -3.1 with grid and table near 0: the read stays near its own last focus, away from the answer |
| not reached | The read's spread | n-back to 1,000 | first choice 1.00, mass .29 to .48, effective 30 to 91 at the reads that hold |

The slope also drifts when its step is bounded: lean, grid and table move together along a direction the read does
not see (slope margin -345, grid +200, table +109, the lean's largest entry from 1 to 72) while training's right share
stays .90, and a frozen read then falls. With the grid the slope takes no step once a slot passes the table.

### Look back

Development, 1,500 answers, taught at 5 to 12, read at lengths no lesson reached, 1,280 slots. The grid changes no digit:
lessons on short stories never pass the table. Mass stays flat with length (`first` .50 to .77, `second` .58 to .65, `last`
.40 to .48), effective slots 5 to 16, first choice 1.00. One token held through the story, the control that needs no
distant moment, reads 1.00 at every length.

| Read at | `Fd` `first` | `second` | `last` | `FdT` `first` | `second` | `last` |
| --- | --- | --- | --- | --- | --- | --- |
| 40 | .99-1.00 | 1.00 | .98-1.00 | 1.00 | .96-1.00 | .99-1.00 |
| 100 | .98-1.00 | 1.00 | 1.00 | .98-1.00 | .98-1.00 | .98-1.00 |
| 400 | 1.00 | 1.00 | .98-1.00 | 1.00 | 1.00 | .95-.97 |
| 1,000 | 1.00 | 1.00 | 1.00 | 1.00 | .92-1.00 | .97-1.00 |

**Taught inside the length** (`lookback_dense`, each life's mean by cue). The slope carries the place when it is
taught, and the grid form takes it from the slope: the form taught still reads, and a story with one cue no longer does.

| Length, slots | Form, answers | Dense `first`, `second`, `last` | One cue, at the length: `first`, `second` |
| --- | --- | --- | --- |
| 100, 128 | table 64, 1,500 | 1.00, 1.00, 1.00 | .99-1.00, 1.00 |
| 100, 128 | grid, 3,000 | 1.00, 1.00, 1.00 | .17-.36, .91-1.00 |
| 400, 512 | table 64, 1,500 | .93-1.00, .99-1.00, .90-1.00 | .95-1.00, .91-1.00 |
| 400, 512 | grid, 3,000 | 1.00, .98-1.00, .94-1.00 | .31-.33, .68-.74 |
| 1,000, 1,280 | table 64, 1,500 | .89-1.00, .88-.98, .86-.96 | .88-1.00, .68-.98 |
| 1,000, 1,280 | grid, 3,000 | .87-.88, .94-.95, .84-.88 | .27-.39, .65-.76 |

### N-back

Development, dense n-back, each cell the range over the four distances and the lives. The first choice is 1.00 at
every read that holds. Controls: distances up to 7 among 1,280 slots read 1.00 at 6,000 answers, and on a stream of
1,600 moments (one life): many slots and a long stream cost a near read nothing.

| n | Slots | Form | 1,500 | 3,000 | 6,000 | 12,000 | Mass, effective at the last | Seconds at the last |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 64 | exact | .42-.97 | | 1.00 | | .46-.54, 6-12 | |
| 40 | 64 | exact | .25-.49 | | 1.00 | | .37-.42, 12-17 | |
| 100 | 128 | exact | .13-.22 | | .99-1.00 | | .38-.46, 19-26 | |
| 100 | 256 | grid | .12-.78 | .33-1.00 | 1.00 | | .30-.38, 35-54 | 138-209 |
| 100 | 256 | grid, `FdT`, declared | | | .996-.999 at the lowest distance | | .24-.46, 26-71 | 221-307 |
| 400 | 512 | exact | .10-.14 | | .11-.15 | | .00-.01, 47-234 | |
| 400 | 1,024 | grid | .12-.20 | .28-1.00 | 1.00 | | .33-.48, 30-59 | 202-389 |
| 400 | 1,024 | grid, `FdT` | .64-1.00 | .99-1.00 | 1.00 | | .35-.69, 8-58 | 245-333 |
| 1,000 | 1,280 | exact | .09-.14 | | | | .00, 51-560 | |
| 1,000 | 1,280 | grid | | .12-.15 | .11-.15 | .96-1.00 | .29-.95, 2-91 | 727-1,401 |
| 1,000 | 1,280 | grid, `FdT`, one life | | .88-.96 | 1.00 | 1.00 | .30-.35, 57-81 | 694 |

The trace silent reads like `Fd` with the grid: at the exact table it costs n-back at 100 (.18 to .86 at 6,000 answers),
and with the grid the read at 400 comes at 1,500 to 3,000 answers.

### Periods

Each life's mean over the three answered periods. Without the peak the read holds 400 and reads n-back only to 400, by
12,000 answers at 512 slots, and at 1,000 stays at chance by 24,000; the library's form reads n-back to 1,000 and holds
periods to 100.

| Period | Slots | Form | Answers | Reading |
| --- | --- | --- | --- | --- |
| 40 | 64 | exact | 180,000 | .931 .939 .944 |
| 40 | 64 | grid without the peak | 180,000 | .833 .934 .931 |
| 40 | 128 | exact | 45,000 | .988 .999 .872 |
| 40 | 128 | grid | 45,000 | .998 1.000 |
| 100 | 128 | exact, declared | 180,000 | .896 .833 .909 .936 .929 .585 |
| 100 | 256 | exact | 45,000 | .831 .932 .927, with .316 on the way |
| 100 | 256 | grid without the peak, declared | 180,000 | .962 .961 .963 .964 .964 .965 |
| 100 | 256 | grid, declared | 180,000 | .956 .968 .965 .958 .959 .937 |
| 400 | 1,024 | grid without the peak | 90,000 | .964 .966 .965 |
| 400 | 1,024 | grid | 90,000 | .247 .961 .180 |
| 1,000 | 1,280 | grid without the peak | 90,000 | .860 .759 .888 |
| 1,000 | 2,560 | grid | 90,000 | .947 .200 .911; all three at chance at 22,500 |

A period of 8 on a stream of 1,000 moments among 1,280 slots, the control that needs no distant moment, reads .98 to 1.00
by 15,000 answers (one life). With the trace silent a period of 400 on the grid reads .171 .303 .344 .186 on one life, at 3,750 to 45,000 answers, and a
period of 100 without the peak reads .965 to .967 at 90,000 on three.

### The declarations at 100 moments

Each gate passes at its bar on 5 of 6 founders: C1 look back `first` at 100 (.90), C2 `second` (.90), C3 the lowest of n-back's
four distances at 6,000 answers (.90), C4 a period of 100 at 180,000 answers (.85). Each declaration has its own six founders.

| Declaration | Arm | Slots | C1 | C2 | C3 | C4 | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| First | `Fd`, exact table | 128 | 6 of 6 | 5 of 6 | 6 of 6 | .585-.936, 4 of 6 | **fails C4** |
| First | `Fd`, first grid form | 128 | 6 of 6 | 5 of 6 | 6 of 6 | .808-.910, 3 of 6 | **fails C4** |
| First | `FdT`, first grid form | 128 | 6 of 6 | 6 of 6 | 6 of 6 | not read | |
| Second | `Fd`, grid without the peak | 256 | 5 of 6 (.873) | 5 of 6 (.746) | 6 of 6 | .961-.965, 6 of 6 | holds |
| Second | `FdT`, the same | 256 | 6 of 6 | 6 of 6 | 5 of 6 (.404) | stopped at three lives, .965-.967 at 90,000 | |
| Third | `Fd`, grid | 256 | 1.000 on all | .984-1.000 | .998-1.000 | .937-.968, 6 of 6 | **holds, the acceptance on #268** |
| Third | `FdT`, grid | 256 | 1.000 on all | .951-1.000 | .996-.999 | not read | C1 to C3 hold |

The third `Fd` arm reads 15,418 parameters on look back and 17,698 on n-back, 40 to 56 seconds, 141 to 222 and 1,889 to 2,397 for
C1 and C2, C3 and C4; a period reads .958 to .967 by 15,000 answers, in 150 to 200 seconds.

### In the sequence composition

`recall_position=64` earned S6 alone in the composition of #256. The grid replaces it only with the slots beside it: at 64
slots a read one period back meets the noisy copy, and one life of three falls. At 128 the first period is still there, and
the read lands on it.

Development, the gate table of #256 on its founders 0 to 2, without `recall_when`, with tokens and a bound on each lesson's contrast that 0.95 leaves out:

| Row (bar) | Table 64, 64 slots | Table 32, 64 slots | Grid, 64 slots | Grid, 128 slots |
| --- | --- | --- | --- | --- |
| S6, a period of 40 (.85) | .897 .940 .940 | .615 .603 .597 | .238 .940 .863 | **.998 .999 .999** |
| S1 to S5, S8, S10, T1, U1, U2, period 8 | pass | the same digits | the same digits | the same digits |
| S9, 12 founders (10 at .90) | 11 | | 11, the same values | 11, the same values |
| Chamber, three lives: every panel field, lessons, act sweeps | | | | the same digits |

Declared (six fresh founders, S9 on twelve), on the composition without that bound and tokens, with the store on
and `working_memory_amplitude=0.3`. The arms read alike to the digit on every row but S6:

| Gate | Bar | Table 64, 64 slots | Grid, 128 slots |
| --- | --- | --- | --- |
| S1 `first` | .90 | .993-1.000, 6 of 6 | the same digits |
| S2 `second` | .90 | .981-1.000, 6 of 6 | the same digits |
| S3 anchors erased, fall of `second` | .30 | .436-.531, 6 of 6 | the same digits |
| S4 copying unique tokens | .95 | .999-1.000, 6 of 6 | the same digits |
| S5 copying repeated tokens | .90 | .899 .901 .924 .893 .913 .891, **3 of 6** | the same digits |
| S6 a period of 40 | .85 | .862 .882 .865 .867 .821 .925, 5 of 6, mean .870 | .999 .999 1.000 1.000 .999 .999, 6 of 6, mean .999 |
| T1 two hops | .90 | .993-.998, 6 of 6 | the same digits |
| U1 induction | .90 | .976 .894 .941 .998 .996 .986, 5 of 6 | the same digits |
| U2 n-back | .90 | .996-.999, 6 of 6 | the same digits |
| S9 ordered recall, 12 founders | .90 on 10 | 12 of 12 | 12 of 12 |

The protocol's rule asked the grid's arm to pass every gate, so by its letter table 64 stays: S5 fails on both arms, to the
same digits. S5's fall is the amplitude's and not the grid's: the same founders at the amplitude's default read .940 to .964.
The row that separates the arms is S6. The grid at 128 slots has fewer parameters (15,044 against 16,819 at 6 inputs, 28,084
against 30,584 on the chamber's brain) and an act costs .13 ms more at 12 inputs on one stream (.67 against .54), with twice the
slots' memory. What it costs is the place in a long story: lessons past the table stop the slope, as the look back table shows.

### Beside the matched networks

Independently scored, each at the brain's parameter count with the window to match, one thread on the same machine.

| Task | Brain with the grid | Transformer | LSTM |
| --- | --- | --- | --- |
| Look back at unseen lengths | 1.00, 1,500 answers, 40-56 s | lowest cue .00-.34 at any exposure | .92-1.00, 1,500 answers, 3-4 s: a latch does it |
| N-back at 100 | 1.00, 6,000 answers, 141-222 s | .11-.13 at 24,000, 170-261 s; lowest .13-.26, mean .15-.79 at 48,000, 349-481 s | chance at 48,000, 655-729 s |
| N-back at 400 | 1.00, 6,000 answers, 202-389 s | .12-.13 at 48,000, 415-430 s | chance at 48,000, 313 s |
| N-back at 1,000 | .96-1.00, 12,000 answers, 727-1,401 s | chance at 48,000, 775-969 s | not read |
| Period of 100 | .967 .969 at 3,750, 35 s; .958-.967 at 15,000, 150-200 s | .19-.94 at 15,000, 15-22 s; .94-.97 at 45,000, 45-67 s | chance at 180,000, 338-382 s |
| Period of 400 | .906 .941 .204 at 7,500, 104-213 s; .247 .961 .180 at 90,000 | chance at 45,000; .19 .94 at 90,000, 165 s; .94 .94 at 180,000, 303-314 s | chance at 180,000, 203 s |
| Period of 1,000 | .947 .200 .911 at 90,000, 2,827-3,350 s | chance at 180,000, 405-407 s | not read |
| Dense look back at 100 | 1.00, 3,000 answers, 30 s | 1.00, 1,500 answers, 2-5 s | lowest .24, mean .54 at 12,000, 12 s |
| Dense look back at 400 | .94-1.00, 3,000 answers, 41 s | lowest .67-.92 at 6,000, 17-33 s; 1.00 at 12,000, 33 s | lowest .11, mean .51 at 12,000, 52 s |
| Dense look back at 1,000 | .84-.95, 3,000 answers, 73 s | .26 lowest, mean .85 at 12,000, 57 s | |

### Cost

Newborn brains, one thread, 12 inputs and 64 association units, 32 sweeps per act at every slot count; trained n-back lives
read 49 to 59. Seconds grow with the slots through the read: a write adds one slot in place, so at 4,096 slots and 16 streams the
read takes 4.2 ms, the write .2 and the settle 1.3. The grid adds to the read and nothing to the write.

| Slots | Parameters: table 32, exact, grid | One stream, ms per act: table 32, exact, grid | 16 streams, ms per act: table 32, exact, grid | Lesson moment, 16 streams, ms | Memory per stream |
| --- | --- | --- | --- | --- | --- |
| 64 | 14,695, 19,469, 17,390 | .53, .54, .64 | 1.49, 1.49, 1.57 | 9.8 | 18 kB |
| 128 | 14,849, 29,325, 17,544 | .55, .56, .67 | 1.57, 1.60, 1.69 | 9.9 | 37 kB |
| 512 | 15,157, 88,461, 17,852 | .59, .61, .71 | 1.91, 2.21, 2.07 | 12.6 | 148 kB |
| 1,280 | 15,465, 206,733, 18,160 | .66, .78, .80 | 2.58, 3.11, 2.93 | 18.1 | 370 kB |
| 4,096 | 15,619, 640,397, 18,314 | .84, 1.00, 1.07 | 5.68, 7.14, 6.70 | 35.8 | 1.18 MB |

At 64 inputs and 16 streams an act takes 1.9 ms at 64 slots, 2.0 at 128, 2.8 at 512, 6.4 at 1,280 and 13 at 4,096.

### What these readings do not establish

- That a composition with the grid reads like this on fresh founders past 100 moments: everything at 400 and 1,000 is development,
  on three lives or fewer.
- Why the read falls at long periods beyond the terms' margins. The peak's step grows with a spread read, since it divides a gradient
  summed over every residue by a small peak; that it causes the fall has not been tested.
- A form that holds both n-back at 1,000 and periods at 400. The two forms read on three lives each sit on either side.
- Periods of 400 and 1,000 on the exact table, and the LSTM at 1,000.
- Anything about language: these tasks ask where, not what.
