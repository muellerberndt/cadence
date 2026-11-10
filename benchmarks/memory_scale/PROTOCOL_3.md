# A brain that writes and reads its own records: the third declaration

Declared 10 November 2026, before the first run on its founders. Issue:
[#288](https://github.com/muellerberndt/cadence/issues/288). It follows
[the first](PROTOCOL.md) and [the second](PROTOCOL_2.md) declarations, whose outcomes stand as
recorded there. Instrument: `brain_records.py`; plans: [`protocol-3/`](protocol-3); contracts:
`test_memory_scale.py` and `tests/test_longterm_records.py`.

## The opt-in read

`Brain.compose(longterm=records)`, off by default. It needs the brain's held message
(`message=units`, #286) held from a turn (`message_boundary=channel`,
`message_boundary_holds=True`).

- **The key** of a record is the brain's held message after the turn: its union over the blocks,
  what the episode heard before the turn. Nothing heard after the turn changes it.
- **The write.** The moment after the turn is the record's value. It is written only where it says
  more than the record the key finds: where a record holds the key (cosine `longterm_revise`, .999,
  the same key) and the same value nothing is written; another value revises that record; no
  record takes a slot. A silent moment after the turn writes nothing.
- **The read.** Before every settle after a turn the key reads the stream's records:
  `longterm_heads` heads in turn, each holding its first choice where the record's key lies at
  cosine `longterm_match` (.75) or above, at `longterm_amplitude` (3.0) times its value on a region
  of its own, which the association reads through plastic synapses, as it reads the recall
  memory's ports.
- **What stays.** Each stream's records are a `cadence.LongTermMemory` of `records` slots: reads
  deepen, idle records fade, a full memory replaces the most faded, records survive `reset`.

A brain without the option is the brain without it (a contract test).

## The chamber

The second declaration's facts with the brain in place of the instrument. `meanings.Lexicon(64,
96)`: 64 noun stems in two numbers and 96 verbs, 12,288 agent-verb pairs, and one input channel
for the turn (388 inputs). A fact is four moments: the act, the agent, the turn, the patient (one
of 8 stems). A question is the act, the agent, the turn and a silent moment at which the brain
answers with one of 8 actions. Every episode begins with `Brain.reset()`, which empties the recall
memory's slots: from a fact to its question the records are the only carrier.

**The brain.** `Brain.compose(388, 8, modules=(64,), backend="cpu", message=280,
message_fan_in=177, message_boundary=387, message_boundary_holds=True, longterm=10000, seed=f)`:
the default composition with #286's declared code (280 units a block, each unit reading the
chamber's share of the channels, 16 of 70), held from the turn, and 10,000 records a stream.

**The nursery.** 1,500 rounds over 16 streams: a fact about a pair of the nursery's pool (2,048
pairs drawn by founder) is told, then a fact told before is asked and its answer taught by one
lesson.

**The reading** at N = 1,000 and 10,000 records a stream, each on a copy of the nursery's brain
with its records emptied: N facts per stream about pairs outside the nursery's pool, which no
lesson asked, each heard once as an episode; then a tenth of them told again with another patient;
then 25 questions per stream about unrevised facts and 25 about revised ones (400 each), without
lessons.

| Score | What it is |
| --- | --- |
| found | the record the read held at the answer holds the question's key and the fact's current patient |
| stale | it holds the patient a revised fact had before |
| used | the brain's answer is the current patient (chance .125) |

**Arms.** `records`: the brain above. `plain`: the same brain without `longterm`, through the same
nursery and reading (parameters 387,862 against 413,082 at founder 0). `records_h2`: `records` with
`longterm_heads=2`, reported.

**Cost.** ms per moment of question and fact episodes (100 each, after five to warm), one process,
at 0 (the option off), 1,000 and 10,000 records a stream, over 1 and 16 streams. The records are
filled by the store's own writes of facts' keys and values, the keys the brain's message gives
(a contract test holds them equal), so the reading does not live 40,000 moments first. On one core
of a spot CPU box that runs nothing else (`protocol-3/cost.json`) and on one device, the brain on
`backend="torch", device="cuda"` (`protocol-3/cost-device.json`).

## Founders

Records: 2900 to 2905. Cost: 2906 to 2908. In no run of any lane.

## Gates

Each gate on 5 of 6 founders unless stated.

| Gate | Arm | Reading | Bar |
| --- | --- | --- | --- |
| B1 | `records` | found, unrevised, 1,000 records | .99 |
| B2 | `records` | found, unrevised, 10,000 records | .99 |
| B3 | `records` | used, unrevised, 10,000 records | .90, and within .05 of used at 1,000 |
| B4 | `records` | revised facts at 10,000 records: found current; stale; used | .95; at most .02; .90 |
| B5 | `plain` | used, unrevised, at both sizes | at most .25 |
| C3 | the option at 10,000 records, one stream | ms per moment of question and of fact episodes, one core | at most twice the same brain's without the option, on 3 of 3 |
| D1 | the same on the device | ms per moment, both kinds | at most twice the same brain's without the option on the device, on 3 of 3 |

**Reasons for the bars.** B1, B2 and B4's found are the first declaration's retrieval bar, for keys
that are exact by construction (a question hears the moments its fact did). B3 and B4's used are
#288's: a brain that uses what it finds answers at 10,000 records as at 1,000. B5 holds that the
answer is carried by the records and by nothing else. C3 and D1 are C1's budget: a moment with the
memory costs at most twice a moment without it.

**Reported without a bar.** `records_h2`; every score at 1,000; the counts of writes, revisions and
agreed statements; parameters and neurons; tell ms per moment over 16 streams.

**A measured negative** names its term: the key (a question's key differs from its fact's), the
write (a fact not written, or a sibling taken for it), the read (the record held, the answer wrong:
the association's use), or cost.

**Expectations, stated before the run** (development founder 0, 60 nursery rounds, 50 records a
stream): found 1.00 on both kinds, used .98 and .97, stale .00. B1 to B5 hold. C3 holds: at 10,000
records a read and a write score every slot (.09 ms in the second declaration) beside a moment of
about 10 ms.

## Total work

18 lives (three arms by six founders), each a nursery of 12,000 batch moments and readings of about
48,000; cost 18 cells on one core and 18 on a device.

## Not established

- A turn the brain marks by itself. The turn is an input channel the world gives (`message_boundary`).
- Several facts in one episode: one turn per episode.
- Saving a brain with records, `grow` and `graft`: refused in this version.
- Keys that differ between the fact and its question: noisy or partial keys and other wordings, the
  second declaration's limits, are not read here.

## Outcome

Read on the declared founders and recorded in [the README](README.md#the-third-declaration-a-brain-that-writes-and-reads-its-own-records),
with the receipts in [`results/`](results) and `summarize.py` (`gates3`) recomputing each gate.

| Gate | Verdict |
| --- | --- |
| B1 | passes, 6 of 6: .9975 to 1.000 |
| B2 | passes, 5 of 6: .980 to 1.000; founder 2901 reads .980, where 3.8% of its facts share their key with another |
| B3 | passes, 6 of 6: used .980 to 1.000 at 10,000 against .9975 to 1.000 at 1,000 |
| B4 | passes, 6 of 6: found current .990 to 1.000, stale .000, used .9925 to 1.000 |
| B5 | passes, 6 of 6: .1225 to .155 |
| C3 | passes, 3 of 3: 1.82 to 1.83 times the moment without the option |
| D1 | passes, 3 of 3: 1.65 to 1.69 |

**Against the expectations.** Every gate held as expected. Found at 10,000 read below the
development reading's 1.00 on four founders: the misses are keys that two facts share (2901) or that
another key contains (every founder), located in the README. Over 16 streams a fact moment at 10,000
records costs 10.5 times a moment without the option, the host's loop over the streams' stores.

**Departures from the plan of the runs.** The CPU box's package update stalled and was stopped by
hand; its Python environment was installed by the same commands. Nothing in an instrument, a plan or
a bar changed after the declaration.

## Amendment 1: the many-stream write (declared 10 November 2026, before its run)

The outcome above stands. Over 16 streams a fact moment at 10,000 records cost 10.5 times the
moment without the option: the streams' stores were read and written one after another on the
host, and a write read its store twice. The records change in three ways, none of them in what is
read or written: a stream's records are read, written and faded store by store (`BATCHED=False`)
or as one bank, and the bank holds to the store-by-store records to the bit (a contract test, on
streams that tell, revise and ask past their capacity).

- **One product over the streams' stores.** Each stream's keys are kept as an index of the slots
  that hold each unit, `(streams, units, slots)`, and a read sums the index rows of the cue's units,
  one gathered sum over every stream that reads; for keys of zeros and ones the sum is the store's
  dot product and the first choice the store's.
- **One read per write.** The read before the settle is the write's recognition, passed to the
  store (`LongTermMemory.write(..., recognised=slot)`), which reads nothing itself.
- **One fading over the streams.** The streams' depths, recent uses and fadings are rows of three
  shared arrays, faded by the store's law in one pass.

**The gate**, on founders 2909 to 2911 (in no run of any lane), one process on one spot CPU box
that runs nothing else, `protocol-3/cost-amended.json` (the records filled by the records' own
write, `"fill": "record"`):

| Gate | Reading | Bar |
| --- | --- | --- |
| C4 | ms per fact moment over 16 streams at 10,000 records a stream, over the same brain's without the option, one core | at most 2, on 3 of 3 |

C3 is read again on the same founders and reported. **Expectation:** the bank reads about ten
index rows of 10,000 slots per stream and fades 160,000 sums a moment; a fact moment over 16
streams costs about 1.5 times one without the option.

### Outcome of Amendment 1

Read on founders 2909 to 2911 and recorded in
[the README](README.md#amendment-1-the-many-stream-write), with the receipt
`results/third-cost-amended.jsonl` and `summarize.py` (`gates3`) recomputing each gate.

| Gate | Verdict |
| --- | --- |
| C4 | passes, 3 of 3: 1.661 to 1.688 times the moment without the option |
| C3, read again | passes, 3 of 3: 1.254 to 1.261 |

**Against the expectation.** C4 reads above the expected 1.5 and within its bar. **Departures from
the plan of the run.** None.

## Amendment 2: the read scored by cosine (declared 10 November 2026, before its run)

The outcome above and Amendment 1's stand. On every founder of the reading .2% to .6% of the facts
have a key that another record's key contains: the read scored each record by its key's dot
product with the message's key, so a record whose key holds the message's and more tied with the
fact's own, and the lower slot won. The records' read and the write's recognition score each
record by the cosine of its key with the message's key (`LongTermMemory(cosine=True)`; the
standalone store's default scores by the dot product). A key read with itself scores 1 and a key
that holds it and more scores below 1, so a contained key reads its own record; the bank scores
as the stores do (a contract test). What is written, the match (.75) and the revision (.999) are
as declared.

**The gates**, on founders 2912 to 2917 (in no run of any lane), the third declaration's chamber,
nursery and reading, the `records` arm alone (`protocol-3/records-amended.json`):

| Gate | Arm | Reading | Bar |
| --- | --- | --- | --- |
| B6 | `records`, read by cosine | found, unrevised, 10,000 records | .99, on 6 of 6 |
| B7 | the same | found, unrevised, 10,000 records, over the questions whose fact's key is its own | .99, on 6 of 6 |

A fact's key is its own where no other fact of the stream's reading has the same key
(`brain_records.own_keys`, the keys a `cadence.Message` of the brain's settings gives, which a
contract test holds equal to the brain's). The receipt reports, per founder, the share of facts
whose key another fact has. B1, B3 and B4 are read again on these founders and reported with
their bars; `plain` and `records_h2` are not run.

**Reasons for the bars.** B6 is B2's bar on all six founders. B7 separates the read's term from the
key's: a key that two facts share whole is one record, which the second fact revises, and no
score of the read tells the two facts apart.

**Expectation, stated before the run.** The reading's facts at 10,000 records written to and read
from a store as the records do, without the brain (`brain_records.py --store`,
`protocol-3/store-development.json`, receipt `results/third-store-development.jsonl`; four streams
each, development readings on founder 0 and on founders the third declaration read):

| Founder | Distinct keys of 10,000 facts | found by the dot product | found by cosine |
| --- | --- | --- | --- |
| 2905 | 10,000 | .9961 to .9979 | 1.000 |
| 0 | 9,932 to 9,937 | .9913 to .9928 | .9936 to .9943 |
| 2901 | 9,807 to 9,813 | .9804 to .9808 | .9816 to .9822 |

By the dot product a write can also miss its own record, tied with a key that holds it, and
write a second copy (founder 0 holds up to five records more than it has distinct keys); by cosine
the records held equal the distinct keys on every stream. B7 holds on every founder (.9975 to
1.000). B6 holds on a founder whose facts' keys are distinct or nearly so, and fails on one like
2901, where 3.8% of the facts share their key with another fact. Of the nine founders of the
third declaration and Amendment 1, 2901 and 2909 (9,873 records for 10,000 facts) hold shared keys
near that share, so on six fresh founders B6 is more likely than not to fail on one; such a
failure is the key's term, located by the receipt's share of shared keys.

### Outcome of Amendment 2

Read on founders 2912 to 2917 and recorded in
[the README](README.md#amendment-2-the-read-scored-by-cosine), with the receipt
`results/third-records-amended.jsonl` and `summarize.py` (`gates3`) recomputing each gate.

| Gate | Verdict |
| --- | --- |
| B6 | fails, 5 of 6: .9875 to 1.000; founder 2917 reads .9875, where 2.43% of its facts share their key with another |
| B7 | passes, 6 of 6: 1.000 |
| B1, read again | passes, 6 of 6: .995 to 1.000 |
| B3, read again | passes, 6 of 6: .9875 to 1.000 |
| B4, read again | passes, 6 of 6: found current .9975 to 1.000, stale .000, used .9925 to 1.000 |

**Against the expectation.** As expected: B7 holds on every founder, the founders with distinct
keys read 1.000, and B6 fails on the one founder of six whose code gives 2.43% of the facts a key
another fact has. The failure is the key's term. **Departures from the plan of the run.** None.
