# A million records: the fourth declaration

Declared 11 November 2026, before the first run on its founders. Issue:
[#310](https://github.com/muellerberndt/cadence/issues/310). It follows the three declarations of
[#288](https://github.com/muellerberndt/cadence/issues/288) ([first](PROTOCOL.md),
[second](PROTOCOL_2.md), [third](PROTOCOL_3.md)), whose outcomes stand as recorded there.
Instruments: `million.py` and `facts.py`; plans: [`protocol-4/`](protocol-4); contracts:
`test_memory_scale.py`, `tests/test_longterm_memory.py` and `tests/test_longterm_records.py`.

This declaration reads the store alone: gates C1 to C5 of #310. A brain that writes its own
records at 100,000 and 1,000,000 (C6) and the controls beside it (C7) are declared apart.

## The opt-ins

Each is off by default. With them off `cadence.LongTermMemory` and `Brain.compose(longterm=...)`
are what they were (contract tests).

- **An address of pairs (`pairs=units`).** The store's address of units lists every slot under
  each of its 64 winning units, so a cue's units name about half of the slots whatever their
  number, and a read's work grows with the records. With `pairs` a record is filed under every
  pair of its `pairs` most driven units, in a table of cells sized to the capacity (16 entries a
  cell, sized for 8; an entry is a slot under its pair's tag). A cue's `winners` name the pairs
  it reads; the slots found under the most pairs, `candidates` of them, go to the read's
  competition, which is unchanged. A key is centred before it drives the units, so a key of zeros
  and ones drives each unit as often as any.
- **`fill`** writes many records told once in one pass: each takes an empty slot, as a write
  with `recognised=-1` does (a contract test holds the two together).
- **`read_each`**: heads reading in turn, each with a cue of its own, for a question of several
  parts.
- **`chain`**: a read's value laid on the next cue's coordinates, one record's content the next
  read's cue.
- **`apart=width`**: a write takes the fact's source (a row of `width` values its key was coded
  from) and keeps a signature of it, its 8 most driven units under a second fixed projection of
  1,024 units. A write whose key a record holds revises that record only where the signatures
  share 5 units or more; otherwise it is another fact and takes a slot. A read that carries its
  source takes, among the records that tie at its best score, the one whose signature it shares.
- **`Brain.compose(longterm_pairs=, longterm_expansion=, longterm_winners=,
  longterm_candidates=, longterm_fade=)`** builds each stream's store with these settings.

Where Numba is installed (the library's optional `fast` extra) the address of pairs counts its
slots in one compiled loop; without it the same integers come from NumPy (a contract test), and
both are timed.

## What is designed, by name

Nothing in this declaration is learned. The parts an author laid out:

| Designed part | Read beside |
| --- | --- |
| the address: a fixed random projection (the library's `PatternSeparator`), the table of pairs, the centring | the scan (no address) and the address of units as the store stands |
| the key of parts: a subject's code beside a relation's, on coordinates of their own | the two codes summed on the same coordinates |
| one code per entity as subject and as object, so a value read is a subject | none: a brain's own key from what it reads is C6's and #311's |
| a question's parts given to the heads one each (`read_each`) | the parts summed into one cue |
| the source's signature: a second fixed projection | the same store without it |

A learned address, and keys a brain's settled state makes, are outside this declaration.

## Records, keys and cues

**Capacity (`million.py`).** N records, each a key with a value of 64 coordinates, in float32.
Key forms:

| Form | Key | Stands in for |
| --- | --- | --- |
| `code-128` | a unit code of 128 coordinates; a quarter of the records in families of 8 at cosine .8 | a dense code of meaning |
| `hashed-280` | 280 coordinates, 15 to 22 of them one and the rest zero, drawn at random | the brain's fixed message code of a sentence |
| `code-256` | as `code-128`, 256 coordinates | the settled store's key |
| `code-64` | as `code-128`, 64 coordinates | the first declaration's key |

Cues: `exact`; `noisy` (a code with noise of its own energy, cosine .71; a hashed key with three
in ten of its active units exchanged); `partial` (half of the coordinates silent);
`similar_noisy` (a family member's code with the noise). 1,000 cues of each kind per cell. For
several records: a family's shared direction read by k heads, k unrelated codes summed and read
by k heads, and the same k codes read each by a head (k = 2, 4, 8; 200 trials).

**Facts (`facts.py`).** N / 8 entities and 64 relations, each a unit code of 64 coordinates. Every
entity is the subject of 8 facts with 8 different relations; a fact's object is another entity and
its value is the object's code. The key of parts is the subject's code beside the relation's (128
coordinates); the control key is their unit sum (64 coordinates). 500 questions of each kind per
cell.

**found.** Every read is at temperature 0 and scored by cosine, so a read holds its first choice
whole. A cue is found when that choice is the record. For k heads a trial is found when the heads
hold k different relevant records; for a question of p parts when every part's fact is held; for
a chain when every hop holds its fact. With shared keys found is by value: the read holds the
fact's current object.

## The arms

| Arm | Settings | Read at |
| --- | --- | --- |
| `scan` | every record scored: the control and the ceiling | every N |
| `units` | the store as it stands: `expansion=8192, winners=64, candidates=256` | to 1,000,000 |
| `pairs` | `pairs=16, expansion=16384, winners=256, candidates=2048` | 100,000 and 1,000,000 |
| `pairs`, ten million | `pairs=16, expansion=32768, winners=512, candidates=4096` | 10,000,000 |

Every store arm: `cosine=True, fade=False, dtype="float32"`, written by `fill`. The fading is
off because it touches every slot each moment; retention under a budget was the first
declaration's.

## Gates

A gate holds on a founder when every reading it names holds there. Twelve founders each; a gate
passes on 10 of 12 unless it says otherwise.

| Gate | Reading | Bar |
| --- | --- | --- |
| C1 | `pairs` at 1,000,000 records, `code-128` and `hashed-280`: found from an exact cue; from a noisy and from a partial cue | .99; .95 |
| C1, reported | 10,000,000 records: `scan` and `pairs`, `code-128` on 3 founders and `hashed-280` on 2 | none |
| C3a | `pairs` at 1,000,000, `code-128` and `hashed-280`: found on exact, noisy and partial cues against `scan` on the same cues | within .01 |
| C3b | ms per read of `pairs` at 1,000,000 over ms at 100,000, mean of the three cue kinds, one core, `code-128` and `hashed-280`, compiled and through NumPy | under 3, on 3 of 3 |
| C2 | 1,000,000 facts of which 4 in 100 share their key with another fact, written one by one, then 1 in 100 told again with another object; `pairs` with `apart=32`: found over all facts (.04 of the sharing facts' rate plus .96 of the others') | .99 on 12 of 12 |
| C4a | `pairs` at 1,000,000 facts, key of parts: the subject alone read by k = 2, 4 and 8 heads | .95 each |
| C4b | the same store: a question of four parts, each head with a part (`read_each`) | .90 |
| C5 | `pairs` at 100,000 facts, key of parts: two hops; three hops; from an exact and from a noisy first cue | .95; .90 |

Reported beside the gates, without a bar: the `units` arm; `code-256` and `code-64`; the
family, summed and each-by-a-head readings on codes; questions of 2 and 8 parts; the parts
summed into one cue; the summed key; chains at 1,000,000; C2's store without `apart`, with the
revised facts' current and stale rates; bytes per record, the records a read scores, the table
entries it reads and its multiply-accumulates in every table.

**The timing (C3b)** is read by cells of their own: founders 3136 to 3138, three processes on a
box that runs nothing else, 300 timed reads per cue kind. The compiled read and the NumPy read
are two arms over the same records. The `scan` and `units` arms are timed beside them.

## Founders

Development founders 0 to 2 fixed the constants above; none of the following was read before
this declaration.

| Run | Plan | Founders | Sizes |
| --- | --- | --- | --- |
| capacity | `capacity.json` | 3100 to 3111 | 100,000 and 1,000,000, four key forms |
| ten million | `ten-million.json`, `ten-million-hashed.json` | 3100 to 3102; 3100 and 3101 | 10,000,000 |
| timing | `timing.json` | 3136 to 3138 | 100,000 and 1,000,000 |
| facts | `facts.json` | 3112 to 3123 | 100,000 and 1,000,000, both key forms |
| shared keys | `shared-apart.json`, `shared-plain.json` | 3124 to 3135 | 1,000,000 |

## Expectations, stated before the runs

From development founders 0 to 2 (one to three founders a cell):

- **C1 holds.** `pairs` at 1,000,000: exact 1.000; `code-128` noisy .997 to 1.000 and partial
  .980 to .997; `hashed-280` noisy .997 and partial .980, where the scan itself reads .984: a
  partial hashed cue keeps about ten units, and some keep five.
- **C3a holds narrowly on `hashed-280`** (partial .980 against .984) and with room on
  `code-128`. **C3b holds**: about 2.3 ms a read at 100,000 and 3.7 at 1,000,000 through NumPy
  with the box under load; the scan takes 50 ms at 1,000,000 on `code-128` and 113 on
  `hashed-280`, and the address of units 4.9 and 6.0, 7 times its 100,000.
- **C2 holds**: with `apart` each fact holds a record of its own; without it a sharing pair is one
  record, so found reads about .98 and the founders fall under .99.
- **C4 and C5 hold at 100,000** (every reading 1.000 on two founders); 1,000,000 is unread.
- **Negatives expected:** k unrelated codes summed into one cue fail past two at 128 coordinates
  for the scan too (four: .86 for the scan, .53 for `pairs`; eight: .00), the located term of
  the first declaration: one address holds few superposed codes. Each part read by a head finds
  1.000.
- **`code-64` binds on the key's width** at 1,000,000: the scan misses noisy and partial cues
  there.
