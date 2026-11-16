# A slot's strength

Issue: [#336](https://github.com/muellerberndt/cadence/issues/336). A slot of the recall memory
every new brain holds grows strong when its moment comes up again, when it came with a large
surprise and when the reads keep fetching it, and a full bank gives up its weakest slot outside
its protected newest half in place of its oldest. The setting is
`Brain.compose(recall_strength=True)` (`Recall(strength=True)`, with `recall_again` and
`recall_kept`); the guide is [docs/memory.md](../../docs/memory.md#a-slots-strength-opt-in).

## The chamber

`strength.py` lives one small world. A fact is a name stated with a value: a name is 6 of 32
binary inputs, a value one of 8 inputs, and one more input is a touch no ordinary moment drives.
A cue is the name alone. The other moments of a life are facts of fresh names, each told once,
each acted on greedily. A slot is held where the bank holds the fact's key; a read is right where
the recall region's drive at the cue (`brain.stimulus`, before any settle) is largest on the
fact's value. Two brains of the composed founder (128 slots, without the associative store) live
the same moments:

- **plain**: the brain `Brain.compose` makes.
- **strength**: the same brain with `recall_strength=True`.

```bash
PYTHONPATH=src python benchmarks/recall_strength/strength.py --seeds 0 1 2
PYTHONPATH=src python benchmarks/recall_strength/strength.py --paid --seeds 0 1 2
```

The first runs in about nine minutes on two threads, the second (the paid life) in about five
minutes a seed. `test_recall_strength_chamber.py` runs the readings in short on a bank of 32 slots;
`tests/test_recall_strength.py` holds the law's contract.

## Readings (seeds 0, 1 and 2; a bank of 128 slots)

Each cell is the share held / the share read right; the last column is the capacities of other
moments a fact stayed held, the median over the facts (the range).

**The dose.** A fact told n times in a row, four facts a seed, then capacities of other moments:

| Arm, tellings | 0.5 | 2 | 4 | 8 | 16 | 24 | Held, capacities |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plain, n = 1 | 1.00 / 1.00 | 0.00 / 0.08 | | | | | 0.9 (0.9 to 1.0) |
| plain, n = 2 | 1.00 / 1.00 | 0.00 / 0.08 | | | | | 0.9 (0.9 to 1.0) |
| plain, n = 4 | 1.00 / 1.00 | 0.00 / 0.00 | | | | | 0.9 (0.8 to 1.0) |
| plain, n = 8 | 1.00 / 1.00 | 0.00 / 0.17 | | | | | 0.9 (0.8 to 1.0) |
| plain, n = 16 | 1.00 / 1.00 | 0.00 / 0.08 | | | | | 0.8 (0.6 to 1.0) |
| strength, n = 1 | 1.00 / 1.00 | 0.00 / 0.17 | 0.00 / 0.08 | 0.00 / 0.33 | 0.00 / 0.08 | 0.00 / 0.25 | 0.8 (0.6 to 1.5) |
| strength, n = 2 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.00 / 0.08 | 0.00 / 0.17 | 0.00 / 0.00 | 6.3 (6.1 to 7.1) |
| strength, n = 4 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 12.1 (11.7 to 13.1) |
| strength, n = 8 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.00 / 0.17 | 17.3 (17.1 to 18.5) |
| strength, n = 16 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | past the longest span |

The plain bank holds no fact past one capacity, however often it was told (its spans past two
capacities are not lived). With the law a fact told n times stays about `8 ln(n / 0.9)`
capacities (6.4, 11.9 and 17.5 for 2, 4 and 8 tellings), and 16 tellings keep it. A held fact is
read right at every span: the read is the plain memory's. A read at chance is right one time in
eight.

**The one shot.** In brains with `surprise=True`, after three capacities of ordinary moments, one
fact told once with the touch struck at 80 beside three facts told once without it:

| Arm, fact | 0.5 | 2 | 8 | 24 | Held, capacities | Tellings the surprise counted |
| --- | --- | --- | --- | --- | --- | --- |
| plain, struck | 1.00 / 1.00 | 0.00 / 0.33 | 0.00 / 0.00 | 0.00 / 0.33 | 0.9 (0.9 to 0.9) | 1599 |
| plain, unstruck | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.11 | 0.00 / 0.00 | 1.0 (0.9 to 1.0) | 0.00 to 1.27 |
| strength, struck | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | past the longest span | 1599 |
| strength, unstruck | 0.78 / 0.89 | 0.11 / 0.33 | 0.11 / 0.33 | 0.00 / 0.11 | 0.6 (0.5 to 13.7) | 0.00 to 1.29 |

The struck moment is kept at once. Of the ordinary moments of a life (24 capacities), 2, 0 and 2
were written past the kept depth by the brain's own surprise with the law (1, 0 and 1 counted so
in the plain brain, where nothing reads the count).

**The use.** Three facts told once; one is asked for by its name every eighth moment, one by a
part of its name (4 of its 6 inputs, another part each time) every eighth moment, one never:

| Arm, asked | 0.5 | 2 | 8 | 24 | Held, capacities |
| --- | --- | --- | --- | --- | --- |
| plain, by its name | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.9 (0.9 to 0.9) |
| plain, by a part | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.0 (1.0 to 1.0) |
| plain, never | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.33 | 0.00 / 0.00 | 1.0 (1.0 to 1.0) |
| strength, by its name | 1.00 / 1.00 | 0.33 / 0.33 | 0.00 / 0.33 | 0.00 / 0.00 | 1.4 (1.4 to 2.1) |
| strength, by a part | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | past the longest span |
| strength, never | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.33 | 0.00 / 0.00 | 0.7 (0.7 to 0.7) |

A fact asked for by parts of its name is the one slot each question reads, and it stays. A fact
asked for by its name alone shares the reads with the slots of the earlier questions, which hold
the name exactly, and those questions are one moment told again: the questions stay, the fact
leaves.

**What it costs.** The share of the last 128 moments the bank holds after 16 capacities of a
life, the age of its oldest slot and the slots past the kept depth; 0% or 25% of the moments
tell one of 12 known facts again:

| Arm, life | Recent moments held | Oldest slot, capacities | Kept slots |
| --- | --- | --- | --- |
| plain, 0% told again | 1.00 | 0.99 | 0 |
| strength, 0% told again | 0.84 to 0.89 | 1.74 to 2.71 | 0 |
| plain, 25% told again | 1.00 | 0.98 to 0.99 | 0 |
| strength, 25% told again | 0.77 to 0.84 | 1.56 to 2.05 | 12 |

A task the plain bank does: 2,400 lessons of a fact, 0 to 5 other moments and its name alone
taught to answer the value, then 24 free answers a gap (shares of the capacity; two gaps inside
the protected half, two outside it), the share right:

| Arm, life | Gap 0.0625 | Gap 0.375 | Gap 0.625 | Gap 0.875 |
| --- | --- | --- | --- | --- |
| plain, 0% told again | 0.96 | 0.92 | 0.96 | 0.93 |
| strength, 0% told again | 0.92 | 0.94 | 0.90 | 0.33 |
| plain, 25% told again | 0.97 | 0.94 | 0.97 | 0.96 |
| strength, 25% told again | 0.93 | 0.92 | 0.94 | 0.25 |

A moment once the bank is full: 0.67 ms plain, 0.72 ms with the law (two threads).

**Where the cost comes from.** In a life where no moment comes up again the law gives up 11 to
16 percent of the last 128 moments, and a taught answer about a fact told once and never read is
right one time in three at 0.875 of the capacity. The cause is the read's credit: every read
renews the slots it attends toward one telling, so an older slot any read touched outlives a
younger one no read touched. Development readings with that one line of `Recall._deepened`
replaced, everything else as built (seeds 0, 1 and 2):

| Read's credit | Recent held, 0% / 25% told again | Taught, gap 0.875, 0% / 25% | Asked by a part, held at 2 and 8 capacities |
| --- | --- | --- | --- |
| as built, `p * max(1 - depth, 0)` | 0.84 to 0.89 / 0.77 to 0.84 | 0.33 / 0.25 | 1.00, 1.00 |
| above an even share, `max(p - 1 / held, 0) * max(1 - depth, 0)` | 0.84 to 0.91 / 0.79 to 0.83 | 0.40 / 0.26 | 1.00, 1.00 |
| none | 1.00 / 0.91 to 0.93 | 0.97 / 0.94 | 0.00, 0.00 |

Without the read's credit the law costs nothing where no moment comes up again (the bank of the
25% life gives its 12 kept facts their slots), and a fact asked for by its parts leaves as an
unread one does. A fact told twice is held past 4 capacities without the credit (the longest span
read there), and a struck moment is kept by its depth alone.

**The paid life** (`--paid`). One continuing `Brain.live` life, paid only, of the composed
founder with `arousal`, the associative store and a bank of 16 slots, on the reward recall
chamber's recipe (a need of 0.2 reward a moment over a youth of 300 moments, the delayed-credit
actor point, pay +1 or -1 delivered with the next episode's first moment and `done`). An
episode: a fact of a name (5 of 24 inputs) and one of two values, lived once and sensed three
times more through `wait`, then `gap` other facts through `wait`, then the name alone through
`live`; 3,000 episodes. The share right (chance .5):

| Arm, gap | Episodes 1 to 200 | 2,001 to 3,000 | Per seed, 2,001 to 3,000 |
| --- | --- | --- | --- |
| plain, 4 | 0.54 | 0.61 | 0.56, 0.77, 0.50 |
| strength, 4 | 0.54 | 0.61 | 0.56, 0.77, 0.50 |
| plain, 32 | 0.50 | 0.47 | 0.48, 0.45, 0.49 |
| strength, 32 | 0.51 | 0.75 | 0.95, 0.79, 0.51 |

With the question within the bank (a gap of 4) the law changes no answer: the two arms answer
alike on every seed, and one seed of three learns. With the question two capacities after its
fact the bank without the law holds no fact and answers at chance; with the law the fact told
four times is held (about 12 capacities) and two seeds of three learn to answer from it. The
law's life at the long gap learns more often than either arm at the short one; the chamber does
not say why.

## What a change of the default needs

The law is opt-in. A change of the default puts `strength` in `Recall.COMPOSED` (what
`recall=True` gives a new brain), so a saved memory without the setting keeps its founder, off:
a default brain saved by 0.99.1 loads with the law off and continues as it was saved. A new
brain's life is the brain's without the law until its bank first gives up a slot (128 moments of
one stream since its reset); from that moment the bank differs, and its actions some hundreds of
moments later. Before the change, these readings on the composed founder, beside the same brain
without the law:

- **The paths the law does not run on.** `Lives` refuses it, the compiled moment
  (`learning_compiled_moment`) takes its moments in turn, and a torch or MLX device keeps its
  bank on the host. Changed as it is, the default makes `Lives` refuse every new brain and takes
  every new brain off the compiled moment and off the device memory. The law on
  `_RecallOnDevice`, in `_moment.recall_write` and in `Lives` comes first, each read to the bit
  against the host's law.
- **The options it refuses.** Several heads, a window, the command and the read of when: the
  composed setting yields to each where a caller names it (`Recall.composed`), read on the
  attention chambers that set them (`benchmarks/attention`).
- **The cost** of this chamber's last section (recent moments held, the taught answer at gaps
  outside the protected half) with the read's credit as built and without it, and the time of a
  moment (0.67 against 0.72 ms here) and on torch at a size whose memory rests on the device.
- **The chambers that read the bank past one capacity of a life**, each on the seeds of its
  recorded readings and at lives longer than 128 moments of a stream (a shorter life is
  unchanged): `benchmarks/recall/vanished_cue.py`, `benchmarks/grammar` (`grammar.py`,
  `meaning.py`, `exposure.py`, `window.py`, `gradients.py`), `benchmarks/binding`
  (`sequence_binding.py`, `chain_binding.py`, `token_binding.py`, `index_read.py`,
  `learned_read.py`, `scaling.py`), `benchmarks/memory_scale` (`retrieval.py`, `used.py`,
  `library.py`, `cost.py`), `benchmarks/keydoor/key_door.py`, and every chamber that composes a
  default brain and lives past 128 moments without a reset.
- **The founders**: the composed founder with `arousal=True`; with `surprise=True`, where the
  brain's own surprise writes ordinary moments past the kept depth (2, 0 and 2 in a life of 24
  capacities here); with the associative store and without it; with
  `learning_compiled_moment=True`; on torch. The founder memory (`recall_founder=True`) is
  unchanged by the setting.
- **The saved files.** A new brain saves `cadence-brain/5` with the feature `recall_strength`,
  which 0.99.1 refuses by the format's name. Pinned files and descriptions of default brains
  change with it and are re-recorded once every other value matches: the description and
  checkpoint digests of `tests/test_senses_by_name.py` (their lives' acts match) and the format
  and feature pins of `tests/test_longterm_records.py`, `tests/test_memory_strength.py` and
  `tests/test_record_heads.py`. No pinned life of a default brain fills a bank: the recorded
  lives of `tests/test_reset_arousal.py`, `tests/test_arousal_need_scale.py`,
  `tests/test_learned_code.py`, `tests/test_observation_code.py`, `tests/test_regions.py`,
  `tests/test_recall_memory.py`, `tests/test_recall_when.py` and
  `tests/test_generic_settling.py`, and the composition tests of the defaults
  (`test_operating_defaults.py`, `test_recording.py`, `test_history_default.py`,
  `test_store_rate_default.py`, `test_context_defaults.py`), pass with the default changed.
