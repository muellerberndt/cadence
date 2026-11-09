# Long-term memory at scale: the declared protocol

Declared 9 November 2026, before the first run on the declared founders. Issue:
[#288](https://github.com/muellerberndt/cadence/issues/288). Instruments: `retrieval.py`,
`used.py`, `retention.py`, `cost.py` and `nets.py` of this directory; contracts:
`test_memory_scale.py`. The plans of the declared runs are the files of [`protocol/`](protocol),
one per instrument, read by `--plan`.

## Questions

1. **Capacity and retrieval.** Does a read return the right record among N stored ones, for an
   exact, a noisy and a partial cue and among similar records, and do several heads return
   several different relevant records in one read?
2. **Found and used.** Where the read finds the record, does the brain answer with it?
3. **Retention.** Under a fixed budget of slots, do facts used often stay for the whole stream, do
   rare ones leave, and is a fact that changed revised?
4. **Cost.** What do a read and a write cost against N, and where do the 128-slot default and the
   guard at 4,096 code coordinates bind?

## Records, cues and streams

`world.py` holds them. A record is an address code with a content code, both of unit length. The
key source is a parameter. The declared one is `code`: every address drawn at random on the unit
sphere, 64 coordinates wide, standing in for a code of meaning until the message code of #286
exists. A quarter of the records come in families of 8 whose addresses share a direction at
cosine .8.

| Cue | What it is | Cosine to the stored address |
| --- | --- | --- |
| `exact` | the record's address | 1 |
| `noisy` | the address plus noise of its own energy | .71 |
| `partial` | the address with half of its coordinates silent | .71 |
| `similar` | the exact address of a family member | 1, with seven siblings at .8 |
| `similar_noisy` | a family member's address plus the noise | .71 |
| `family`, for k heads | a family's shared direction | .89 to each of its 8 members |
| `superposed`, for k heads | the unit sum of k unrelated addresses | about 1 / sqrt(k) to each |

A cue never holds a content.

## Scores

- **found.** A read returns the right record when the record holds more than half of the read's
  mass. The settle answers on the mixture a read holds, so a right first choice inside a spread
  read does not count. `first`, the share of reads whose first choice is the record, and `mass`,
  the mean mass on it, are reported beside it.
- **found, several heads.** A trial counts when every head holds more than half of its mass on
  one relevant record and the k records are different.
- **used.** The brain's act is the record's content token, on records no lesson taught; also
  among the reads whose first choice is right.
- **retrievable.** A read with a fact's address holds more than half of its mass on a record of
  that fact at its current content.

## Arms

| Arm | What it is |
| --- | --- |
| `library` | the recall memory of a new brain (`Brain.compose`, the default composition), at a capacity that holds the N records beside its boundary slot: every slot scored, temperature .25, the oldest slot dropped. Nothing is taught |
| `library@.05` | the same at `recall_temperature=0.05` |
| `bank` | the candidate, `bank.Bank`: a sparse address of 8,192 units with 64 winners picks 256 candidates, each head reads its first choice among them whole (temperature 0), a write that recognises a held key at cosine .9 rewrites that record, and a full bank replaces the most faded slot under the depth and recent-use law (depth scale 2, depth kept over 8 capacities of moments, recent use over a sixteenth of a capacity) |
| `bank-soft` | the candidate at temperature .05, the read a softmax over the candidates |
| `bank-dense` | the candidate without the coarse stage: every slot scored |
| `ring` | the candidate with the fading off: a full bank replaces the slot written longest ago, the recall memory's own rule |
| `recent` | the candidate reading recent use alone (no depth) |
| `cumulative` | the candidate with the depth kept as a whole sum, `memory_fade`'s own form |
| `append` | the candidate without revision: every write takes a slot |
| `default`, `content` (used) | the default composition with its memory at the capacity of the reading; the same with `recall_position=0` and `recall_temperature=0.05`, a read by content alone |
| `bank`, `bank+notebook` (used) | a brain without a recall memory, and the default composition at its 128 slots, each with the candidate bank beside it: the bank's read of every moment's address is held on input coordinates of its own |

## Matched references

- `exact`: an exact nearest-neighbour store over the same records, and its k nearest for k
  heads. It is the ceiling for retrieval: no read by dot product finds a record it does not.
- `transformer`: a transformer over the window (2 layers, 2 heads, width 256), taught by
  backpropagation on fresh record sets with the same families and cues, at windows of 128 and
  1,000 records, and read at 1,000 and 4,000. Its answer is scored by decoding.
- `lru`, `fifo`, `random`: least recently used, first in first out and random eviction over
  exact keys at the same budget. `lfu`: least frequently used with an exact count of every
  fact's uses, an upper reference on the plain stream.

## What the development founders fixed

Founders 0, 1 and 2, before the declaration. Nothing below was read on a declared founder.

- **The read's temperature.** At a new brain's temperature of .25 the recall memory's first
  choice is right and its mass on the right record is .28 at 128 records and .05 at 1,000
  (unit keys). At .05 the mass holds for exact and noisy cues. Under the soft law several heads
  share one first choice; at temperature 0 they read different records. The candidate reads at
  temperature 0, and the soft read is an arm.
- **The coarse stage.** On founders 0 and 1 a sparse address of 8,192 units with 64 winners and
  256 candidates reads as the exact store does at 10,000 records (noisy .999 against 1.000) and
  at 100,000 (noisy .990 against .994, partial .992 against .994). With 32 winners the noisy
  cue reads .985 and .960. Each unit's weights are at unit length, so the units win equally
  often.
- **The fading's constants.** At a budget of 1,000 on founders 0 to 2, over a grid of the depth's
  scale (1.5 to 8), the moments the depth is kept over (4 to 64 capacities, and the whole sum) and
  the recent-use window: the setting with the highest mean of the plain stream's and the drifting
  stream's `top`. That is depth scale 2 kept over 8 capacities: plain .975 to .990, drifting .950
  to .975, least recently used .82 to .83. The whole sum reads .98 on the plain stream and .58 to
  .62 on the drifting one: it does not let a fact go that fell out of use. A write deepens a slot
  and only a read marks it recently used.
- **The nursery of the used reading.** 1,500 rounds bring every arm's nursery questions to .96
  or above on founders 0 to 2.

## Declared runs

Founders 2880 to 2885. One thread per process.

| Plan | Cells | Reads |
| --- | --- | --- |
| `protocol/retrieval.json` | 6 founders by N of 128, 1,000, 10,000 and 100,000 at key width 64; arms `exact`, `library`, `library@.05`, `bank`, `bank-soft`, `bank-dense`; 1,000 cues of each kind and 200 trials per head count (2, 4, 8) and cue | found, first, mass, the share of right records among the candidates, ms per read and write, bytes |
| `protocol/retrieval-wide.json` | founders 2880 to 2882 by N of 1,000 to 100,000 at key width 256; `exact`, `bank`, `bank-dense` | the same, reported |
| `protocol/used.json` | 6 founders by four arms, a nursery of 1,500 rounds, then 400 questions at each N of 128 to 100,000 | found, used, the read's margins by source |
| `protocol/retention.json` | 6 founders by budgets of 128, 1,000 and 10,000 slots, the plain and the drifting stream, 4 budgets of facts, 100 budgets of moments, Zipf exponent 1, a change before one occurrence in twenty; the references and `fade`, `ring` at every budget; `fade-soft`, `recent`, `cumulative`, `append` to 1,000 | top, revised, stale, mixed, rare, left, hits and writes at ten checkpoints |
| `protocol/cost.json`, `cost-width.json` | founders 2880 to 2882 by N, one process at a time on a box that runs nothing else; the moment at key widths 64 to 2,048 | ms per read, write and the rest of the moment; bytes per record; the address map's bytes |
| `protocol/cost-device.json`, `nets.json` | the device box: the read as one product on the device, and the transformer | ms per read; right by cue |

## Gates

Each gate is read on the declared founders and passes on 5 of 6.

| Gate | Arm | Reading | Bar |
| --- | --- | --- | --- |
| R1 | `bank` | found, `exact` cue, N = 10,000 | .99 |
| R2 | `bank` | found, `noisy` cue, N = 10,000 | .90 |
| R3 | `bank` | found, 4 heads, `family` cue, N = 10,000 | .85 |
| U1 | `bank` (used) | used at N = 10,000 | within .05 of used at 128 |
| K1 | `fade` | top at the end of the plain stream, budget 1,000 | .95 |
| K2 | `fade` | revised at the end of the plain stream, budget 1,000 | .90 |
| K3 | `fade` | top at the end of the plain stream, budget 1,000 | above `lru` on the same founder |
| C1 | `bank` | ms per read at N = 10,000, one core | at most `moment`'s `rest_ms` at 128 records on the same box |
| C2 | `device` | ms per read at N = 10,000, on the device | at most `moment`'s `rest_ms` at 128 records on the device's box |

Every arm of the retention run passes 10 budgets of writes (statements that took a slot which
did not hold the fact); a run in which one does not is void.

**Reasons for the bars.** R1 to R3 and K1 to K3 are the issue's. U1 follows the issue: the read's
score and the brain's are kept apart, and a brain that uses what it finds answers at 10,000
records as it does at 128. C1 and C2 state the budget the issue left open: a read takes at most
as long as the rest of the moment, so a moment with a large memory costs at most twice a moment
without one.

**Reported without a bar.** N = 100,000; the partial and similar cues; 2 and 8 heads; the
superposed cue, whose ceiling the address code's width sets; key width 256; budgets of 128 and
10,000; the drifting stream; every checkpoint; the default composition's memory on every reading.

**A measured negative** names its binding term: the address code's capacity (the exact store
misses too), interference by similarity (the exact store finds what the read's mass does not
hold), the coarse stage (the right record is no candidate), the position maps or the output
(found and not used), or cost.

## Total work

Retrieval: 24 cells at width 64 and 9 at width 256, the largest about 15 minutes. Used: 24 lives
of 1,500 rounds, about one minute each, and their readings. Retention: 312 streams, the largest
one million moments. Cost: 12 cells and 4 widths, one at a time. About 20 core-hours on one
32-core box, and under two hours of one device.

## Not established by this protocol

- Any default. Every mechanism here is an instrument-level candidate or an opt-in, and a default
  moves only under the preservation panel and every gate of #256.
- A key that holds meaning. The addresses are random codes; what a learned code or the message
  code of #286 does to similarity between records is not read here.
- A brain that writes its own long-term memory in a life: which moments are written, and a read
  that enters the settle through a region of its own.

## Outcome

Read on the declared founders and recorded in [the README](README.md), with the receipts in
[`results/`](results) and `summarize.py` recomputing each gate from them.

| Gate | Verdict |
| --- | --- |
| R1, R2, R3 | pass on 6 of 6 |
| U1 | passes on 6 of 6 |
| K1, K2, K3 | pass on 6 of 6; every arm passes 14 budgets of writes or more |
| C1 | fails on 3 of 3: .245 to .251 ms against a budget of .118 to .122; the address costs more than scoring 10,000 keys |
| C2 | passes on 3 of 3: .119 to .123 ms against .199 to .207 |

**Departures from the plan of the runs.** The CPU box was a spot c7a.8xlarge and the device box a
spot g5.xlarge with one A10G. The device box was lost once while the transformer's plan ran, and
that plan was run again from its start. Nothing in an instrument, a plan or a bar changed after the
declaration. The reading in [`after/`](after) was taken after C1 failed and is marked as such.
