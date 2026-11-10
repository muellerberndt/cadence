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
