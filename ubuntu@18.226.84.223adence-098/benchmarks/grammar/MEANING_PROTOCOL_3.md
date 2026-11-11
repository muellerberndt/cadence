# The quiet message beside a free read, and roles from the brain's own message: declared protocol 3 (cadence#286)

Declared on 2026-11-10, committed to the lane branch before the first declared life. The founders,
arms, readings and gates below are fixed; a failed gate is reported as failed. It follows
[the second protocol](MEANING_PROTOCOL_2.md), whose brain writes its own message
(`Brain.compose(message=units)`).

## The questions

Two, in this order; the second part runs only where the first part's gate holds for at least one
arm.

1. **The quiet message beside a free read, at no cost.** In the second run `message_quiet` (70
   units a block at amplitude .15, the recall read free) lifted clauses (+.03), held-out
   combinations (+.03), roles (+.08), brackets (+.15) and paths (+.04) and cost three-hop paths
   .22, against the default composition, paired. The cost was located in the three-hop sentence
   holding more moments than any lesson's message did (21 heard and 8 said, against at most 17 and
   6 in a lesson): the message wraps around its four blocks, or crowds them. The arms below test
   that cause directly.
2. **Roles from the brain's own message.** With `message=280` at 256 association neurons and the
   read barred, role twins read .90 (4 of 6 founders at .90), where the first protocol's
   instrument-written message read .95 (5 of 6) on other founders. The code is the same (a test
   holds the brain's write equal to `cadence.Message`'s), so the gap, if it is one, lies in the
   association's read of its own region or in the founders. The instrument's message is driven at
   the same amplitude through the same projection scale and differs in one thing: it sits on
   heard-history blocks of 35 neurons, one learned gain each (32 gains), where the brain's own
   region has one gain per block of 280 (4 gains), and the second run's gains settled at 1.2 to
   7.9 per block. The arms below locate the gap and test that one mechanism.

## The opt-in's additions (off by default)

- `message_gains=n`: `n` learned gains per block under `carried_gain`, one per `units / n` neurons,
  in place of one per block.
- `message_boundary=channel`: a heard moment with that input channel on is a boundary the brain
  can see. The message is emptied and that moment begins it again; with
  `message_boundary_holds=True` the message keeps what it heard before the boundary and nothing
  from that moment on joins, until the episode ends. In the chamber the boundary is `GO`
  (channel 33).

Without these a brain is the founder's to the bit (the second protocol's gate B).

## What development found (design input)

- Six founders, the read free, the default lessons, 64 association neurons, 70 units a block at
  amplitude .15 with no learned gain: a message held from `GO` (the brain's own words not joining)
  read paths .81 against .96 for the default composition with no learned gain, and three hops .08
  against .27; with the gain learned, paths .69 at amplitude .15 and .36 at .45.
- The second run's `message_quiet`, read with the message silent: clauses .12, paths .61, three
  hops .16 (.05 with it).
- Episode lengths in the chamber: clauses 7 to 13 moments, brackets 4 to 15, paths 17 and 23 in
  lessons; three-hop paths 29 (21 heard, 8 said). No lesson's episode has more than 23 moments.

## Instrument

`benchmarks/grammar/meaning.py` (`plan --third`, `lives`, `read --third free`, `read --third
roles`). A brain life is `Brain.compose(35, 18, modules=(width,), seed=founder, backend="cpu",
recall=True)` with the arm's settings; lessons, panels, scores and the bar are the first
protocol's. The instrument's message (part 2) is the first protocol's `cadence.Message` written
onto heard-history neurons after every act, 280 units a block at scale .5.

## Part 1: founders and arms

Founders 9828600 to 9828605 (no life in any lane). Every arm has the recall read free and the
default lessons (`learning_surprise=True`), 64 association neurons.

| Arm | The brain | Parameters |
| --- | --- | --- |
| `default` | the default composition: the reference | 28,716 |
| `heard` | `default` with `history_decays=(0.0,)`: the second reference | 30,992 |
| `quiet` | `default` with `message=70`, `message_amplitude=0.15` (the second run's `message_quiet`) | 46,920 |
| `quiet_24` | `quiet` with `message_phases=24`: every lesson's episode fits without a wrap (23 moments at most); a three-hop path wraps by five moments | 137,940 |
| `quiet_32` | `quiet` with `message_phases=32`: past the longest episode read; blocks 23 to 31 are written by no lesson | 174,348 |
| `quiet_hold` | `quiet` with `message_boundary=33`, `message_boundary_holds=True`: the message is held from `GO`, the brain's own words do not join | 46,920 |
| `quiet_clear` | `quiet` with `message_boundary=33`: the message is emptied at `GO` and holds `GO` and the words said | 46,920 |
| `quiet_10` | `quiet` with `message_amplitude=0.10` | 46,920 |

## Part 1: readings and gate

Whole sentences on the six panels `clause`, `clause_held`, `roles` (both twins), `dyck`, `path`
and `path_long` (three hops); each message arm less `default` and less `heard`, paired by
founder, per panel: the mean difference, its standard error and the number of founders whose
difference is at least -.03; each message arm read again with the message silent and with the
read barred; the learned gains; parameters and seconds.

**Gate F3.** An arm holds where, on each of the six panels, its difference from `default` is at
least -.03 on 5 of the 6 founders; that covers "every panel within .03 of the default or above it,
and three hops within .03 of the default's". The differences from `heard` are reported beside it.

**Expectations, stated before the run.** `quiet` reads as in the second run: paths kept, three
hops .2 under. `quiet_24` and `quiet_32` do not hold three hops: a three-hop sentence's message
differs from every lesson's in the number of moments it holds, whatever the phase count, and
`quiet_32` drives the association from blocks no lesson wrote. `quiet_hold` costs paths and holds
three hops no better. `quiet_clear` loses the lifts on clauses and roles and costs three hops
more, the words said so far being a count the brain learns to end on. `quiet_10` lifts and costs
less than `quiet`. F3 fails on every arm on three hops.

**What a failure names.** Where `quiet_24` and `quiet_32` cost three hops as `quiet` does, the
cause is not the wrap: it is the length, the message of a longer sentence being one no lesson
settled under. Where `quiet_hold` costs three hops as `quiet` does, the brain's own words are not
the cause either. Where `quiet_clear` costs more, the count of words said is part of the cost.

## Part 2: founders and arms

Run only where F3 holds for at least one arm. Three arms on the second run's founders 9728600 to
9728605 (paired with that run's `state_256`), then two on the fresh founders 9828600 to 9828605.
Every arm has 256 association neurons, every row taught, the read barred.

| Arm | The brain | Parameters |
| --- | --- | --- |
| `own_256` | `message=280`: the brain's own message, one gain per block (the second run's `state_256`) | 432,528 |
| `instrument_256` | the first protocol's `state_256`: the instrument's message of 280 units a block at scale .5 on 32 heard-history blocks, one gain each | 432,556 |
| `fine_256` | `own_256` with `message_gains=8`: one gain per 35 neurons, the instrument's read of the brain's own message | 432,556 |

**Readings.** Whole sentences on `clause`, `clause_held`, `roles`, `dyck`, `path`; each word kind
said right with the previous words given; the learned gains; `instrument_256` and `fine_256`
less `own_256`, paired by founder.

**Location.** Where `instrument_256` reads roles over `own_256` by .05 and beyond two standard
errors on the second run's founders, the gap is in the association's read of its own region;
where it does not, the gap is the founders' spread and no mechanism is at fault. Where
`fine_256` recovers what `instrument_256` reads, the mechanism is the gain per group.

**Gate R3.** `fine_256` on the fresh founders: `roles` .90 on 5 of 6, with `clause` and
`clause_held` at .90 on 5 of 6.

**Expectations.** `instrument_256` reads roles .93 to .97 on the second run's founders and
`fine_256` follows it within .03; R3 holds on 5 of 6.

## Left to a later protocol

The step term, the learned map from words to concept codes, lengths, depths, persistence, and
`history_decays=(0.0,)` as a default candidate.

## Compute

One AWS c7i.8xlarge, one process per life, one BLAS thread per process.

## Result

Part 1 read on one c7i.8xlarge, founders 9828600 to 9828605, 48 lives on commit 79bb2c17. F3
fails on every arm, so part 2 did not run. Every table is on the chamber's page
([the quiet message's cost](README.md#the-quiet-messages-cost)).

| Arm | Parameters | clause | clause_held | roles | dyck | path | three hops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `default` | 28,716 | .93 | .84 | .81 | .77 | .98 | .35 |
| `heard` | 30,992 | 1.00 | .97 | .99 | .86 | .98 | .71 |
| `quiet` | 46,920 | 1.00 | .84 | .92 | .92 | .90 | .13 |
| `quiet_24` | 137,940 | .99 | .89 | .89 | .91 | .90 | .26 |
| `quiet_32` | 174,348 | .99 | .86 | .94 | .83 | .89 | .23 |
| `quiet_hold` | 46,920 | .99 | .89 | .91 | .89 | .95 | .35 |
| `quiet_clear` | 46,920 | 1.00 | .88 | .92 | .89 | .98 | .24 |
| `quiet_10` | 46,920 | .99 | .82 | .93 | .84 | .96 | .18 |

Each arm less `default`, paired by founder, mean (standard error) and founders at or above -.03
of 6 (a difference of two sentences in 64 is -.031 and counts as under):

| Arm | clause | clause_held | roles | dyck | path | three hops | F3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `quiet` | +.07 (.04) 6 | +.00 (.08) 4 | +.11 (.06) 5 | +.15 (.11) 5 | -.07 (.08) 5 | -.22 (.07) 1 | fails |
| `quiet_24` | +.06 (.05) 5 | +.05 (.06) 5 | +.07 (.06) 4 | +.15 (.14) 5 | -.08 (.02) 1 | -.09 (.11) 2 | fails |
| `quiet_32` | +.06 (.04) 5 | +.02 (.06) 3 | +.12 (.05) 5 | +.06 (.11) 4 | -.09 (.05) 1 | -.12 (.08) 1 | fails |
| `quiet_hold` | +.06 (.04) 6 | +.05 (.07) 5 | +.10 (.07) 4 | +.12 (.08) 4 | -.03 (.02) 3 | +.00 (.06) 3 | fails |
| `quiet_clear` | +.07 (.04) 6 | +.04 (.07) 5 | +.10 (.06) 5 | +.12 (.10) 4 | +.01 (.01) 5 | -.11 (.08) 2 | fails |
| `quiet_10` | +.06 (.04) 5 | -.02 (.03) 4 | +.12 (.07) 5 | +.07 (.11) 4 | -.02 (.03) 5 | -.17 (.08) 1 | fails |

Against `heard`: every message arm is under it on held-out combinations (-.08 to -.15), roles
(-.06 to -.11) and three hops (-.36 to -.58), level on clauses and brackets.

**Against the expectations.** `quiet` read as stated (paths -.07 with one founder at .53, three
hops -.22). `quiet_24` and `quiet_32` cost three hops (-.09, -.12) and paths (-.08, -.09) as
stated: the phase count is not the cause. `quiet_hold` was expected to cost paths and hold three
hops no better; it held three hops at the default's mean (+.00: .56, .28, .41, .44, .22, .22
against .28, .27, .42, .47, .31, .36) and cost paths .03 at the mean, with the lifts kept (clauses
+.06, held-out +.05, roles +.10, brackets +.12). `quiet_clear` kept paths (+.01) and cost three
hops -.11 where a larger cost was expected. `quiet_10` cost three hops -.17 where less was
expected.

**What the failure names.** The cause of the three-hop cost is the brain's own words joining the
message while it speaks: with them kept out (`quiet_hold`) three hops read the default's mean, and
with the message holding them alone (`quiet_clear`) three hops fall .11. The phase count is not
the cause (`quiet_24`, `quiet_32`), and the amplitude a notch lower leaves it (`quiet_10`). F3
fails on `quiet_hold` by the founders' spread: three hops +.28, +.02, -.02, -.03, -.09, -.14, and
paths +.02, -.06, -.03, -.02, -.11, +.05, three of six at or above -.03 on each. Read with the
message silent, `quiet_hold` says clauses at .86 (the other arms .10 to .16): with its own words
kept out of the message the brain says its clauses through the read and lifts them with the
message, and three hops fall to .01, so the held meaning is what it says a long path from.

Part 2 did not run; its arms and gate stand declared for a later run.

## Amendment 1: the held message on twelve founders

Written after part 1 was read and before any life of these arms.

**Why.** F3 asked every panel to sit within .03 of the default on 5 of 6 founders. A panel is 64
sentences, so a per-founder bar of -.03 is two sentences, inside the panel's noise, and it read
noise as failure: `quiet_hold` held three hops and every other panel at the mean and failed F3 on
the founders' spread. The confirmatory run reads the same arm on twelve fresh founders with a gate
on the paired mean and its standard error.

**Founders.** 9928600 to 9928611 (no life in any lane), one online pass each (4,000 batches of
the first grammar), the same panels.

| Arm | The brain | Parameters |
| --- | --- | --- |
| `default` | the default composition: the reference | 28,716 |
| `heard` | `default` with `history_decays=(0.0,)` | 30,992 |
| `quiet_hold` | `default` with `message=70`, `message_amplitude=0.15`, `message_boundary=33`, `message_boundary_holds=True`: the message held from `GO` | 46,920 |
| `quiet_hold_heard` | `quiet_hold` with `history_decays=(0.0,)`: the held meaning and the previous word together, two independent mechanisms | 49,196 |

**Readings.** The six panels; `quiet_hold`, `quiet_hold_heard` and `heard` less `default`, and
`quiet_hold_heard` less `heard`, paired by founder: the mean difference and its standard error;
the learned gains; seconds.

**Gate F3c.** An arm holds against `default` where, on each of the six panels, the paired mean
difference is at or above -.03 and the default is not above the arm by two standard errors of
the paired difference. A lift counts only where it clears two standard errors.

**Expectations.** `quiet_hold` holds F3c: three hops and paths within .03 at the mean, clauses,
roles and brackets lifted. `quiet_hold_heard` holds F3c and reads three hops at `heard`'s level
(.7); against `heard` it is level or lifts brackets. `heard` lifts paths and three hops against
`default` as in the second run.

## Result of Amendment 1 and part 2

Read on one c7i.8xlarge on commit 6fbb5f31: 48 lives of Amendment 1 on founders 9928600 to
9928611, and part 2's 18 lives on 9728600 to 9728605 and 12 on 9828600 to 9828605.

**Amendment 1.** Whole sentences, mean over twelve founders:

| Arm | Parameters | clause | clause_held | roles | dyck | path | three hops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `default` | 28,716 | .97 | .88 | .89 | .80 | .92 | .41 |
| `heard` | 30,992 | .99 | .91 | .94 | .96 | .98 | .64 |
| `quiet_hold` | 46,920 | .98 | .91 | .90 | .78 | .71 | .17 |
| `quiet_hold_heard` | 49,196 | 1.00 | .92 | .95 | .88 | .97 | .42 |

Less `default`, paired by founder, mean (standard error); `*` a lift and `-` a cost beyond two
standard errors:

| Arm | clause | clause_held | roles | dyck | path | three hops | F3c |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `quiet_hold` | +.01 (.01) | +.03 (.03) | +.01 (.03) | -.02 (.06) | -.20 (.09)- | -.24 (.06)- | fails |
| `quiet_hold_heard` | +.03 (.01)* | +.04 (.03) | +.06 (.04) | +.07 (.04) | +.05 (.05) | +.01 (.10) | holds |
| `heard` | +.02 (.02) | +.03 (.03) | +.05 (.03) | +.15 (.04)* | +.06 (.05) | +.23 (.08)* | holds |

`quiet_hold_heard` less `heard`: clause +.01 (.01), held-out +.01 (.04), roles +.01 (.03),
brackets -.08 (.03)-, paths -.01 (.01), three hops -.21 (.07)-.

**Against the expectations.** `quiet_hold` was expected to hold F3c and failed on paths (-.20)
and three hops (-.24): part 1's reading on six founders does not replicate on twelve.
`quiet_hold_heard` held F3c as stated, with one lift beyond two standard errors (clauses +.03), and
read three hops at the default's level (.42), under `heard` (.64). `heard` held as stated.

**Where `quiet_hold` fails.** On four founders of twelve (9928601, 9928606, 9928608, 9928611)
paths read .20 to .41, and with every previous word given the first hop is said at .41 to .81 and
the second at .22 to .34; the default composition says them at .94 to 1.00 on the same founders.
A message held from `GO` holds the path's start and its facts and none of the hops said, so the
brain that leans on it loses its place in the path on those founders. With the previous word
beside it (`quiet_hold_heard`) every founder reads paths .94 to 1.00.

**Part 2: roles.**

| Arm | Founders | Parameters | clause | clause_held | roles | at .90 of 6 |
| --- | --- | --- | --- | --- | --- | --- |
| `own_256` | second run's | 432,528 | .99 | .95 | .90 | 6, 5, 4 |
| `instrument_256` | second run's | 432,556 | 1.00 | .92 | .94 | 6, 4, 5 |
| `fine_256` | second run's | 432,556 | .99 | .94 | .88 | 6, 4, 3 |
| `own_256` | fresh | 432,528 | .99 | .87 | .85 | 6, 4, 4 |
| `fine_256` | fresh | 432,556 | 1.00 | .89 | .95 | 6, 4, 6 |

`own_256` on the second run's founders reads that run's `state_256` to the sentence. Paired,
`instrument_256` less `own_256`: clause +.01 (.01), held-out -.03 (.02), roles +.05 (.03);
`fine_256` less `own_256`: roles -.02 (.06) on those founders and +.10 (.09) on the fresh ones.

**Location.** The instrument's message reads roles .05 over the brain's own on the same founders,
inside two standard errors (.06), and loses held-out combinations by .03: by the declared rule the
gap is the founders' spread, and the read of the brain's own region is not at fault. One gain per
35 neurons does not recover anything on the same founders (-.02).

**Gate R3** fails: `fine_256` on the fresh founders reads roles .95 (6 of 6 at .90) with clauses
kept (6 of 6) and held-out combinations at .89 with 4 of 6 at .90 (.83 and .61 on two founders,
where `own_256` reads .81 and .53). The failing reading is held-out combinations on the same two
founders in both arms.

**Against the expectations.** `instrument_256` was expected at .93 to .97 on roles and read .94;
`fine_256` was expected to follow it and read .88 on the same founders. R3 was expected to hold
and failed on held-out combinations.
