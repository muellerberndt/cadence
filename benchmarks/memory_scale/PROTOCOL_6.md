# A brain with its own million records, beside the controls: the sixth declaration

Declared 13 November 2026, before the first run on its founders. Issue:
[#310](https://github.com/muellerberndt/cadence/issues/310), gates C6 (at 1,000,000 records) and
C7. It follows [the fifth declaration](PROTOCOL_5.md), which read a brain with its own records at
100,000. Instruments: `brain_records.py` and `window.py`; plans: [`protocol-6/`](protocol-6);
contracts: `test_memory_scale.py`, `tests/test_longterm_memory.py` and
`tests/test_longterm_records.py`. The store alone at 10,000,000 and 100,000,000 records, on disk,
is [the seventh declaration](PROTOCOL_7.md).

## The opt-ins

Each is off by default. With them off `cadence.LongTermMemory` and `Brain.compose(longterm=...)`
are what they were (contract tests), and a brain saves in the format it had.

- **`grow=True`**: the table of pairs starts at 64 cells and takes a cell at a time as records
  are written (linear hashing). A write splits the cells its own entries call for, so no write
  rebuilds the table, and the capacity is a bound that costs nothing until slots are written.
  Each record keeps the units it is filed under (`filed`, 64 bytes), which a split reads.
- **`path=directory`**: every array of the memory is a file under the directory, mapped into
  memory; `flush()`, `close()` and `LongTermMemory.open(directory)`.
- **`sure=cosine`**: a read of one head first reads the pairs of the cue's `pairs` most driven
  units (120 cells in place of 32,640), then of four times as many, and ends where a record
  found under the most pairs holds a key at that cosine to the cue. A cue no record holds so
  closely is read in full.
- **`Brain.compose(longterm_grow=, longterm_path=, longterm_sure=)`** build each stream's store
  with them; with `longterm_path` a stream's records lie under `stream-<n>` and a checkpoint
  names the directory.
- **`Brain.compose(longterm_apart=True)`**: a stream keeps the sum of the moments it heard
  before the turn and gives it to the store as the fact's source (`LongTermMemory(apart=)`, the
  fourth declaration's C2). A write whose key a record holds revises that record only where the
  heard moments agree; a read takes, among the records that hold its key, the one whose heard
  moments it shares. The heard moments are told exactly, so the stores compare a source's whole
  signature (`LongTermMemory(same=8)`, all 8 of its units), and a read or a write that carries a
  source ends early (`sure`) only at a record that holds its signature.

A brain whose records grow, lie in files or are told apart saves as `cadence-brain/5`, which
0.99.0 and earlier refuse by name.

## What is designed, by name

Nothing in this declaration is learned beside what the fifth declaration's brain learns in its
nursery. The parts an author laid out:

| Designed part | Read beside |
| --- | --- |
| the address of pairs, its table that grows, the read that ends at its record | the fifth declaration's brain at 100,000 (the table sized to the capacity, every read in full); contract tests hold the first choices together |
| the source of a fact: the sum of the heard moments, under the store's second fixed projection | the same brain without `longterm_apart` (development, below) |
| the chamber's facts and lexicon, the brain's fixed message code | the third and fifth declarations |
| the window's token: one token a fact, the sum of three embeddings and a place code | named as a form that favours the control |

## C6 at 1,000,000: the brain and the chamber

The fifth declaration's chamber and brain, with the changes that the records' number asks for:

- **A lexicon of 512 noun stems in two numbers and 1,024 verbs** (`meanings.Lexicon(512, 1024)`:
  1,048,576 agent-verb pairs, 3,588 inputs with the turn), so that 1,000,000 facts are about
  1,000,000 different pairs. The brain holds 14,828,703 parameters (the fifth declaration's
  2,403,231: the inputs' projections grow with the lexicon).
- **Each stream's store**: `longterm=2000000` (a bound), `longterm_pairs=16,
  longterm_expansion=16384, longterm_winners=256, longterm_candidates=4096, longterm_fade=False`
  as before, with `longterm_grow=True, longterm_sure=0.999, longterm_apart=True` and
  `longterm_path` a folder of the founder's own on the box's disk. A record's value is the heard
  moment in float64 (28,704 bytes), so twelve founders' records (about 400 GB) are past the
  box's memory of 256 GB; they lie in files.
- **Facts told apart.** At 280 units some agents' codes are one code (the third declaration's
  located term). Of 1,024 agent forms the development founders 0 to 5 keep 1,018, 1,020, 997,
  1,016, 887 and 1,022 different codes. At 1,000,000 facts nearly every pair of the lexicon is
  told, so on a founder like the fifth about a quarter of the facts share their key with
  another fact, and without `longterm_apart` each such pair of facts is one record.

One stream reads: the nursery keeps its 16 streams and 1,500 rounds; the reading tells 1,000,000
facts on one stream, then a tenth of them again with another patient, then asks 400 questions
about unrevised facts and 400 about revised ones, without lessons. Scores as in the fifth
declaration: **found**, **stale**, **used** (chance .125), and found over the questions whose
fact's key is its own.

## C7: the same facts held another way

Twelve founders read 1,000, 10,000 and 100,000 facts of the fifth declaration's chamber (a
lexicon of 256 stems and 256 verbs, the brain of 2,403,231 parameters) in three arms. The
controls read `brain_records.read_at`'s own draws, so every arm reads the same facts, revisions
and questions.

- **The brain** (`brain-sizes.json`): the fifth declaration's brain with the store's settings
  above (`longterm_grow`, `longterm_sure`, `longterm_apart`, `longterm_path`), one life a
  founder through its nursery and then the three sizes.
- **The window** (`window.json`, arm `context`): a causal transformer of 3 layers, 4 heads and
  the width whose parameters are nearest the brain's. A told fact is one token; the facts and
  then the revisions enter the window in told order, each token attending to all before it; a
  question is one token read against the keys and values kept. Taught by backpropagation on
  windows of 16 to 256 facts over the nursery's pairs for 3,600 steps on two threads (about 45
  minutes on a box of its own; a count of steps, so the box's load does not change what it
  learns).
  It is read at every size to 100,000 facts (110,000 tokens).
- **The index** (arm `retrieval`): every told fact listed under its act and its agent; a
  question fetches the 8 facts that hold both of its words, then its agent's, the latest first,
  and the same transformer reads them in told order.

Every row carries parameters, bytes per record (the brain's store; the window's keys and values;
the index's postings and words), the time to write a record and to read a question, and
multiply-accumulates.

## Gates

A gate holds on a founder when every reading it names holds there.

| Gate | Reading | Bar |
| --- | --- | --- |
| C6c | used, unrevised facts, 1,000,000 records | .90 (within .05 of C6a's .95), on 10 of 12 |
| C6d | revised facts at 1,000,000: found current; stale; used | .95; under .02; .90, on 10 of 12 |
| C6e | the brain of C7 at 100,000 records, facts told apart: found and used, unrevised | .99 and .95, on 10 of 12 |
| C7a | used (unrevised and revised) of the brain against right of the better control, at 1,000, 10,000 and 100,000 facts | no more than .05 under it, on 10 of 12 |
| C7, reported | every arm's bytes per record, write and read time, accuracy, parameters and multiply-accumulates at every size; each arm's read time at 100,000 over 1,000 | none |

Reported without a bar: found over facts whose key is their own, the share of facts whose key
another fact has, ms per told and per asked moment, the records' bytes in the store's arrays and
on the disk, the window's accuracy at the size it was taught and past it, and whether the index
fetched the fact.

## Founders

Development founders 0 to 5 fixed the constants; none of the following was read before this
declaration.

| Run | Plan | Founders | Sizes |
| --- | --- | --- | --- |
| the brain's million | `brain-million.json` | 3220 to 3231 | 1,000,000 |
| the brain at three sizes | `brain-sizes.json` | 3240 to 3251 | 1,000, 10,000, 100,000 |
| the window and the index | `window.json` | 3240 to 3251 | 1,000, 10,000, 100,000 |

One on-demand r7i.8xlarge (32 threads, 256 GB) in us-east-2, the records on its gp3 volume:
two threads a brain for the million, one a brain for the three sizes, two a window. The seventh
declaration's cells share the box; every time here is reported, never gated.

## Expectations, stated before the runs

EXPECTATIONS

## Cost, stated before the runs

At 9.3 ms a told moment (development, a loaded box) a million facts of four moments and their
revisions take about 11 to 12 hours a founder.
