# Long-term memory at scale: the second declaration

Declared 10 November 2026, before the first run on its founders. Issue:
[#288](https://github.com/muellerberndt/cadence/issues/288). It follows
[the first declaration](PROTOCOL.md), whose outcome stands as recorded there, and declares two
readings: the cost gate for the store as it stands, and the memory keyed by held meanings. The
plans are the files of [`protocol-2/`](protocol-2); contracts: `test_memory_scale.py`.

## 1. The cost gate, re-declared

C1 of the first declaration failed: with the sparse address in front of 10,000 records a read
took .245 to .251 ms against a budget of .118 to .122. The library's store has since taken the
founder that reading called for: `cadence.LongTermMemory` scores every slot up to 16,384 records
(`direct`) and puts the address in front of the read past that. The reading taken after the
failure (.084 to .093 ms) does not count as a pass. This section declares the gate again for the
store as it stands.

- **Arm.** `memory`: `cadence.LongTermMemory` at its founders (`direct` at twice the address's
  units, float64), filled with N records at key width 64. A read is `read(cue, credit=False)`
  with the value it holds (`value`).
- **Budget.** As in C1: `moment`'s `rest_ms` at 128 records, the rest of the default
  composition's moment at the same key width, on the same box and founder.
- **Run.** `protocol-2/cost.json` through `cost.py`, founders 2886, 2887 and 2888 (in no run of
  any lane), one process on one spot box that runs nothing else, of the first reading's type
  (c7a.8xlarge) where capacity allows; the type is recorded with the outcome.

| Gate | Arm | Reading | Bar |
| --- | --- | --- | --- |
| C1b | `memory` | ms per read at N = 10,000, one core | at most `moment`'s `rest_ms` at 128 records on the same box and founder, on 3 of 3 |

C2 stands as read: on the device a read at 10,000 records took .119 to .123 ms against .199 to
.207, on 3 of 3 founders. The issue's cost box is ticked on C1b and C2 together.

**Reported without a bar.** N of 128, 1,000 and 100,000; the write; `memory-float32`; `exact`;
and `floor` and `floor32`, the keys' product with the cue and its first choice alone: the least
a read that scores every slot does in NumPy. If C1b fails, the floor against the budget says
whether any read that scores every slot fits it.

## 2. Keys made by the message

The first declaration read random and sparse stand-ins for a code of meaning. This section reads
keys that `cadence.Message` makes (#286), with the question: does a large memory keyed by held
meanings retrieve and keep as the stand-ins did, and which term binds first?

**Facts** (`meanings.py`). The grammar chamber's concept moments at a larger lexicon: 512 noun
stems in two numbers and 128 verbs, 1,795 input coordinates (a word and a concept per noun form
or stem and per verb, and the flags `PLURAL`, `AGENT` and `PATIENT`). A fact says of an agent and
a verb which patient. In the record format #286 gives:

- the **key** says what the fact is about: `Message.key()` after the act and the agent are
  heard, the union of their two codes;
- the **value** says it: `Message.state` after the patient is heard too, four blocks of zeros and
  ones with the patient's code in the third.

A revised fact keeps its key and takes another patient. The lexicon holds 131,072 agent-verb
pairs, and N facts are drawn about different ones. Two facts about one agent, or with one verb,
share that moment's code: every record has siblings by construction, and none are added.

**The code.** `Message` at #286's declared settings: 4 phases, 5 winners, 280 units. Its fan-in
there is 16 of the chamber's 70 channels; this lexicon's 3,590 channels take the same share, 821.
Beside it: the 70-unit code at the same share, and the 280-unit code at the class's own fan-in of
16 (which its cover rule raises to 26 here).

| Cue | What it is |
| --- | --- |
| `exact` | the fact's key |
| `noisy` | the key with three in ten of its active units exchanged for silent ones (cosine .7) |
| `partial` | the key with half of its coordinates silent |
| `reworded` | the same act and agent heard as their words in place of their concepts |
| `agent`, for k heads | the agent's moment alone, relevant to every stored fact about that agent |

**Arms.**

| Arm | What it is |
| --- | --- |
| `exact` | an exact store over the same keys: the fact whose key shares the most active units with the cue, and among facts tied for the best the one told last. The ceiling |
| `memory` | `cadence.LongTermMemory` as it stands, the key as `Message` gives it and the held message as its value |
| `memory-same` | the same with `revise=0.999`: a write is a record told again only where its key is the same key |
| `memory-direct` | the same as `memory` with every slot scored, at 100,000 records, where `memory` reads through its address |
| `lru`, `fifo`, `random`, `lfu`; `ring`, `append` | the references of the first declaration's retention reading, and the store with the fading off and without revision |

**Scores.** `found`, `first` and `held` as in the first declaration. `said`: the read's value
holds the fact's patient. `single`, on the exact store: the cue's best overlap is held by the
right fact alone; the rest are ties. For k heads a trial counts when the heads return k different
facts about the cue's agent.

**Runs.** Founders 2890 to 2895 (in no run of any lane), the code drawn from the founder.

| Plan | Cells | Reads |
| --- | --- | --- |
| `protocol-2/meaning-keys.json` through `meaning_keys.py` | 6 founders by N of 1,000, 10,000 and 100,000 by the three codes; 1,000 cues of each kind, 200 trials per head count | found, first, said, single, held, the share of distinct keys, ms |
| `protocol-2/meaning-retention.json` and `-70.json` through `retention.py` | 6 founders, a budget of 1,000 slots, 4,000 facts, 100,000 moments, the plain and the drifting stream, a change of patient before one occurrence in twenty | top, revised, stale, rare, hits, writes |

**Gates.** The first declaration's bars, on the 280-unit code and the store as it stands, each on
5 of 6 founders:

| Gate | Arm | Reading | Bar |
| --- | --- | --- | --- |
| M1 | `memory` | found, `exact` cue, 10,000 facts | .99 |
| M2 | `memory` | found, `noisy` cue, 10,000 facts | .90 |
| M3 | `memory` | 4 heads return 4 different facts about the cue's agent, 10,000 facts | .85 |
| M4 | `memory` | retention: top at the end of the plain stream | .95 |
| M5 | `memory` | retention: revised at the end of the plain stream | .90 |
| M6 | `memory` | retention: top above `lru` on the same founder | |

**The first term that binds** is named from the same receipts:

- *the key*, where the exact store misses too: facts whose keys are one key (`distinct`), or a
  cue whose best overlap several facts share (`single`);
- *the write*, where `memory` holds fewer facts apart than `memory-same` (`held`): a sibling
  taken for the fact told again;
- *the address*, where `memory` reads below `memory-direct` at 100,000 facts;
- *the form*, on the `reworded` cue.

**Expectations, stated before the run** (development founders 0 and 1, one cell each at 1,000 and
10,000 facts):

- M1 holds narrowly or fails narrowly: found .986 to .997 at 10,000, with .2% to .7% of the facts
  sharing a key with another.
- M2 fails: .79 to .87 at 10,000, and the exact store reads the same. A key holds about ten
  active units in two groups of five, each group shared with every fact about that agent or with
  that verb; a cue that loses three units ties with a sibling on about one cue in four.
- The partial cue reads .59 to .65 at 10,000.
- M3 fails or holds narrowly: .66 to .85.
- M4 to M6 hold as on the stand-ins: at 4,000 facts the keys are distinct but for a few.
- The store's recognition at cosine .9 takes a sibling for the fact told again where two keys of
  ten units differ in one: `memory` holds .983 to .994 of 10,000 facts apart, `memory-same` .993
  to .998.
- The 70-unit code reads below the 280-unit code on every cue.
- The `reworded` cue reads .00: a word's code and its concept's are unrelated draws, the limit
  #286 located and holds.

## Not established by this declaration

- A learned map from heard words to concept codes, or any key that two wordings share. That
  limit is #286's.
- A brain that writes these records itself. The instrument tells the store every fact with its
  key and value apart.
- Other lexicons and fact shapes. One lexicon and one shape of fact are read; the share of the
  agent-verb pairs a store holds grows with N, to three in four at 100,000.

## Outcome

Read on the declared founders and recorded in [the README](README.md#the-second-declaration),
with the receipts in [`results/`](results) and `summarize.py` (`gates2`) recomputing each gate.

| Gate | Verdict |
| --- | --- |
| C1b | passes on 3 of 3: .084 to .093 ms against a budget of .122; the floor reads .071 to .079 |
| M1 | passes on 6 of 6: .996 to 1.000 |
| M2 | fails, 1 of 6: .852 to .908; the exact store reads .850 to .894 |
| M3 | fails, 3 of 6: .825 to .925; the exact store reads .825 to .930, 4 of 6 |
| M4, M5, M6 | pass on 6 of 6: top .968 to .978, revised .970 to .984, least-recently-used .780 to .833 |

**The first term that binds** is the key. An exact key finds its fact as the stand-ins did to
10,000 facts. A noisy or partial key ties with the facts that share one of its two moments, on the
exact store as on the memory, from the first thousand facts. The write's recognition at cosine
.9 binds second, the address not at all on these cues, and the form wholly.

**Against the expectations.** M1 held with more room than expected (.996 to 1.000 for .986 to
.997). M2 failed as expected, and one founder passed it (.908). M3 read .825 to .925 for an
expected .66 to .85. M4 to M6 held. The store held .995 to .998 of 10,000 facts apart for an
expected .983 to .994. The reworded cue read .000 to .010.

**Departures from the plan of the runs.**

- The box for C1b was a spot m7a.8xlarge, the first reading's processor (AMD EPYC 9R14): the
  c7a.8xlarge first launched was reclaimed after five minutes, and its rows are not used.
- That box was reclaimed in turn while the meaning-key plan ran, before any of its rows were
  written; the retention plans and C1b were complete. The meaning-key plan was run again from its
  start on a spot c7i.8xlarge, in three parts with the same cells and arms: 1,000 and 10,000
  facts, 100,000 facts, and `memory-direct` at 100,000 facts.
- `memory-direct` fills 100,000 slots with every slot scored at each write, about half an hour a
  cell. The README states the cells that were read.
- Nothing in an instrument, a plan or a bar changed after the declaration. `retention.py` gained
  the `message-whole` keys of the development reading in [`after/`](after) after the declared
  retention plans had finished.
