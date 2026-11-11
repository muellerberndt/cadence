# Memory at scale: a large memory of encoded records

These instruments read [#288](https://github.com/muellerberndt/cadence/issues/288): how many
encoded records a memory holds while it gives the right one back, whether several heads return
several relevant records in one read, what stays when the slots run out, and what a read costs.
They measure settings of the library, an instrument-level candidate and independently scored
references against each other; they establish no default. The declared protocol is
[`PROTOCOL.md`](PROTOCOL.md).

## The instruments

| Instrument | Asks | Reads |
| --- | --- | --- |
| `retrieval.py` | N stored records, cues of five kinds, k heads | **found**: the right record holds more than half of the read's mass; first choice; mass |
| `used.py` | a brain taught on fresh facts, then N facts no lesson taught | found, and **used**: the brain's act is the record's content; the read's margins by source |
| `retention.py` | a budget of B slots, a stream of 4 B facts used by a Zipf law for 100 B moments, asked and then told, one occurrence in twenty after a change | the 10% most-used facts retrievable, the revised ones current, stale copies, rare facts held |
| `cost.py` | read, write and the rest of a moment against N; a read on a device | ms on one thread, bytes per record, the address map's bytes |
| `nets.py` | the transformer over the window | right by decoding |

`world.py` holds the records, cues and streams, `references.py` the exact nearest-neighbour store
and the eviction policies over exact keys, `library.py` the library's own recall memory as an arm,
and `bank.py` the candidate by its declared name. `meanings.py` and `meaning_keys.py` hold the
second declaration's facts, whose keys `cadence.Message` makes, and their retrieval reading. The library holds the candidate as
[`cd.LongTermMemory`](../../docs/memory.md#a-long-term-memory-of-records). `summarize.py` recomputes
every gate and table below from the receipts in [`results/`](results).

```bash
PYTHONPATH=src python benchmarks/memory_scale/retrieval.py --plan benchmarks/memory_scale/protocol/retrieval.json --out DIR/retrieval.jsonl --workers 6
PYTHONPATH=src python benchmarks/memory_scale/used.py --plan benchmarks/memory_scale/protocol/used.json --out DIR/used.jsonl --workers 6
PYTHONPATH=src python benchmarks/memory_scale/retention.py --plan benchmarks/memory_scale/protocol/retention.json --out DIR/retention.jsonl --workers 19
PYTHONPATH=src python benchmarks/memory_scale/cost.py --plan benchmarks/memory_scale/protocol/cost.json --out DIR/cost.jsonl --workers 1
python benchmarks/memory_scale/summarize.py --dir benchmarks/memory_scale/results
python -m pytest -q benchmarks/memory_scale
```

## The candidate

A long-term bank of records (a key with a value) for one stream, in four parts. Each stays inside
the boundary of #288: an address computed before a read, a competition among slots, the winners'
values held; no backward pass.

- **A sparse address picks candidates.** Every key is coded as pattern separation codes it: a
  fixed random projection onto 8,192 units of which the 64 most driven are the key's address.
  Each unit lists the slots whose address holds it. A cue's winners drive the slots, and the 256
  most driven go on. No key is read at this stage.
- **The recall memory's read picks among them**, each head reading its first choice among the
  slots the heads before it left, whole.
- **Depth and recent use decide what stays.** Every slot keeps three leaky sums and no count or
  time stamp: its depth (the reads' attention plus one per write, kept over 8 capacities of
  moments), its recent use (the reads' attention, kept over a sixteenth of a capacity) and how
  far it has faded, which grows each moment by `idle / (1 + (depth / 2) ** 4)` and which a read
  takes away. A full bank replaces the most faded slot.
- **A key told again revises its record.** A write whose own first choice holds its key at
  cosine .9 rewrites that slot.

## Readings

Declared on six fresh founders (2880 to 2885) under [`PROTOCOL.md`](PROTOCOL.md) at commit
`741161ba`, on one spot c7a.8xlarge, one thread per process; the device readings on one A10G. Each
cell is the range over the founders. Records are random unit codes of 64 coordinates, a quarter of
them in families of 8 at cosine .8.

### The gates

| Gate | Reading | Bar | Founders | Verdict |
| --- | --- | --- | --- | --- |
| R1 | found, exact cue, 10,000 records | .99 | 1.000 on 6 of 6 | **passes** |
| R2 | found, noisy cue, 10,000 records | .90 | .999 to 1.000, 6 of 6 | **passes** |
| R3 | 4 heads return 4 different relevant records, 10,000 records | .85 | 1.00 on 6 of 6 | **passes** |
| U1 | used at 10,000 records within .05 of used at 128 | | .99 to 1.00 against 1.00, 6 of 6 | **passes** |
| K1 | budget 1,000: the 10% most-used facts retrievable | .95 | .963 to .980, 6 of 6 | **passes** |
| K2 | budget 1,000: revised facts current | .90 | .971 to .981, 6 of 6 | **passes** |
| K3 | budget 1,000: above least-recently-used | | .963 to .980 against .782 to .823, 6 of 6 | **passes** |
| C1 | ms per read at 10,000 records on one core, against the rest of the moment | .120 | .245 to .251, 0 of 3 | **fails**: the address costs more than the budget |
| C2 | ms per read at 10,000 records on a device, against the rest of the moment | .203 | .119 to .123, 3 of 3 | **passes** |

### Capacity and retrieval

`found` by cue. `library` is the recall memory of a new brain at a capacity that holds the
records, `library@.05` the same at `recall_temperature=0.05`, `bank` the candidate, `bank-soft`
the candidate with a softmax at .05 over its candidates, `exact` the exact nearest-neighbour
store, the ceiling. `bank-dense`, the candidate with every slot scored, reads as `exact` does on
every row.

| Records | Arm | exact | noisy | partial | similar, noisy | 2 heads | 4 heads | 8 heads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | `exact` | 1.000 | 1.000 | 1.000 | .955 to .966 | 1.00 | 1.00 | 1.00 |
| 128 | `library` | .000 | .000 | .000 | .000 | .00 | .00 | .00 |
| 128 | `library@.05` | 1.000 | 1.000 | .990 to .996 | .734 to .788 | .00 | .00 | .00 |
| 128 | `bank` | 1.000 | 1.000 | 1.000 | .955 to .966 | 1.00 | 1.00 | 1.00 |
| 1,000 | `exact` | 1.000 | 1.000 | 1.000 | .956 to .971 | 1.00 | 1.00 | 1.00 |
| 1,000 | `library` | .000 | .000 | .000 | .000 | .00 | .00 | .00 |
| 1,000 | `library@.05` | 1.000 | .998 to 1.000 | .845 to .862 | .709 to .772 | .00 | .00 | .00 |
| 1,000 | `bank` | 1.000 | 1.000 | 1.000 | .956 to .971 | 1.00 | 1.00 | 1.00 |
| 10,000 | `exact` | 1.000 | 1.000 | .998 to 1.000 | .949 to .970 | 1.00 | 1.00 | 1.00 |
| 10,000 | `library` | .000 | .000 | .000 | .000 | .00 | .00 | .00 |
| 10,000 | `library@.05` | 1.000 | .944 to .958 | .294 to .343 | .626 to .641 | .00 | .00 | .00 |
| 10,000 | `bank` | 1.000 | .999 to 1.000 | .998 to .999 | .949 to .969 | 1.00 | 1.00 | .99 to 1.00 |
| 10,000 | `bank-soft` | 1.000 | .986 to .993 | .677 to .722 | .702 to .712 | .00 | .00 | .00 |
| 100,000 | `exact` | 1.000 | .991 to .999 | .990 to .998 | .958 to .973 | 1.00 | 1.00 | 1.00 |
| 100,000 | `library` | .000 | .000 | .000 | .000 | .00 | .00 | .00 |
| 100,000 | `library@.05` | 1.000 | .373 to .406 | .014 to .021 | .175 to .188 | .00 | .00 | .00 |
| 100,000 | `bank` | 1.000 | .986 to .997 | .984 to .993 | .956 to .972 | 1.00 | 1.00 | .99 to 1.00 |
| 100,000 | `bank-soft` | 1.000 | .900 to .921 | .396 to .441 | .593 to .615 | .00 | .00 | .00 |

The similar cue without noise reads 1.000 on every arm but `library` at every size, and .999 to
1.000 for the bank at 100,000.

**What binds, in the order it binds.**

1. **The read's spread.** The recall memory's first choice is the right record at every size
   (1.000 to 10,000 records, .991 to .999 on noisy cues at 100,000). At a new brain's temperature
   of .25 and keys of unit length the right record holds .27 of the read's mass at 128 records,
   .046 at 1,000, .005 at 10,000 and .001 at 100,000: the read holds a crowd. At .05 the mass on an
   exact cue is 1.000 to 10,000 and .996 at 100,000, and noisy and partial cues fall as the
   records grow.
2. **Several heads.** Nothing is taught here, so every head holds the same address. Under the
   soft law every such head's first choice is the same record: the share of heads that return a
   new relevant record is 1/k at every size, on both soft arms. Reading the first choice whole, k
   heads return k different records.
3. **The address code's width**, for a cue that superposes unrelated addresses. Two are found at
   1.00 to 10,000 records and .98 to 1.00 at 100,000. Four are found by the exact store itself at
   .85 to .89 among 128 records, .54 to .62 among 1,000, .21 to .27 among 10,000 and .04 to .07
   among 100,000; eight at .04 to .12 among 128 and .00 above. Each of k superposed addresses holds
   `1 / sqrt(k)` of the cue, and no read by dot product finds more at 64 coordinates.
4. **The coarse stage.** The right record is among the 256 candidates on .991 to .999 of the noisy
   cues and .989 to .995 of the partial ones at 100,000 records, and on .999 to 1.000 at 10,000.
   Against the exact store the candidate loses at most .010 on any single cue and founder at
   100,000 and .002 at 10,000. It binds for cues that superpose several addresses once the key is
   wide enough for the exact store to find them ([key width 256](#key-width-256)).
5. **Similar keys at the write.** A write that recognises a held key rewrites it. Among 10,000 and
   100,000 records with families at cosine .8, at most one record in 10,000 is taken for a sibling
   told again.

### Key width 256

Declared on founders 2880 to 2882 (`protocol/retrieval-wide.json`), the same records at 256
coordinates. The exact cue reads 1.000 on every row.

| Records | Arm | noisy | partial | similar, noisy | 4 superposed | 8 superposed | 8 heads |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1,000 | `exact` | 1.000 | 1.000 | 1.000 | 1.00 | .93 to .96 | 1.00 |
| 1,000 | `bank` | 1.000 | 1.000 | 1.000 | .98 to .99 | .54 to .61 | 1.00 |
| 10,000 | `exact` | 1.000 | 1.000 | 1.000 | 1.00 | .83 to .85 | 1.00 |
| 10,000 | `bank` | 1.000 | 1.000 | 1.000 | .88 to .93 | .07 to .09 | 1.00 |
| 100,000 | `exact` | 1.000 | 1.000 | 1.000 | 1.00 | .57 to .59 | 1.00 |
| 100,000 | `bank` | 1.000 | 1.000 | 1.000 | .60 to .73 | .00 | 1.00 |

A wider key holds more superposed addresses: the exact store finds four at 1.00 to 100,000
records and eight at .83 to .85 among 10,000. The candidate's address does not follow it there. A
record at cosine .5 or .35 to the cue shares too few of its winners with it, so for superposed cues
the coarse stage binds: four are found at .88 to .93 among 10,000 records and .60 to .73 among
100,000, eight at .07 to .09 and .00. A memory that scores every slot reads as the exact store
does, which the library's founder does to 16,384 records. Single cues and heads over one family
read 1.000 on both arms at every size. At this width a record takes 6.9 kB at 10,000 records and
5.3 at 100,000 with its address.

### Another key source (development)

The key source is a parameter of `world.codes`. A development reading on founders 0 to 2 (receipt
`results/dev-sparse.jsonl.gz`; nothing here is declared) draws sparse nonnegative codes with a
tenth of their units active, the shape a fixed or a learned sparse code gives, with the same
families.

| Key | Records | Arm | exact | noisy | partial | similar | records held apart |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 units, 6 active | 10,000 | `exact` | 1.000 | .994 to .997 | .386 to .395 | 1.000 | |
| 64 units, 6 active | 10,000 | `bank` | .999 to 1.000 | .994 to .997 | .386 to .395 | .966 to .977 | .9915 to .9922 |
| 64 units, 6 active | 100,000 | `exact` | 1.000 | .958 to .978 | .219 to .247 | 1.000 | |
| 64 units, 6 active | 100,000 | `bank` | .994 to .995 | .948 to .965 | .219 to .247 | .971 to .976 | .9891 to .9895 |
| 256 units, 25 active | 10,000 | `exact` | 1.000 | 1.000 | .998 to 1.000 | 1.000 | |
| 256 units, 25 active | 10,000 | `bank` | 1.000 | 1.000 | .997 to .998 | 1.000 | 1.0000 |
| 256 units, 25 active | 100,000 | `exact` | 1.000 | 1.000 | .998 to .999 | 1.000 | |
| 256 units, 25 active | 100,000 | `bank` | 1.000 | 1.000 | .990 to .996 | 1.000 | 1.0000 |

With 6 active units the code itself binds: half of a cue silent leaves about 3 units, and the exact
store finds the record on .39 of such cues among 10,000. Family members share most of their active
units, so their keys pass the revision's cosine of .9 and about one record in a hundred is taken
for a sibling told again. With 25 active units of 256 neither happens. What a key holds decides
both readings before the memory does.

### Found and used

A nursery of 1,500 rounds on fresh facts (4 to 16 facts, then four taught questions) brings every
arm's nursery questions to .97 or above. Then 400 questions about records no lesson taught, at
each size. `default` is the default composition with its memory at the capacity of the reading;
`content` the same with `recall_position=0, recall_temperature=0.05`; `bank` a brain without a
recall memory and the candidate beside it, whose read of every moment's address is held on input
coordinates of its own; `bank+notebook` the default composition at its 128 slots with the
candidate beside it.

| Arm | Records | first choice right | found | used | used where the first choice is right |
| --- | --- | --- | --- | --- | --- |
| `default` | 128 | .27 to .38 | .08 to .14 | .35 to .41 | .76 to .96 |
| `default` | 1,000 | .04 to .07 | .01 to .02 | .12 to .20 | .67 to 1.00 |
| `default` | 10,000 | .00 to .01 | .00 to .01 | .11 to .14 | 1.00 |
| `default` | 100,000 | .00 | .00 | .12 to .14 | 1.00 |
| `content` | 128 to 100,000 | 1.00 | 1.00 | 1.00 | 1.00 |
| `bank` | 128 | 1.00 | 1.00 | 1.00 | 1.00 |
| `bank` | 1,000 to 100,000 | 1.00 | 1.00 | .99 to 1.00 | .99 to 1.00 |
| `bank+notebook` | 128 to 100,000 | 1.00 | 1.00 | 1.00 | 1.00 |

Chance is .125. The default composition's read does not find a record by its content past the
distances its lessons reached, and where it does find one the brain uses it. The read's logit
margin, the right slot less the best other slot, by source (mean of the six founders; the address
map's own terms stay within .01):

| Records | key | position maps, event rows | position maps, cortex columns | position maps, bias |
| --- | --- | --- | --- | --- |
| 128 | +2.8 | -0.0 | -1.3 | -3.3 |
| 1,000 | +3.1 | +0.0 | -10.2 | -28.4 |
| 10,000 | +3.3 | +0.6 | -96.3 | -280.1 |
| 100,000 | +3.3 | -2.9 | -935.1 | -2778.0 |

The key favours the right slot at every size. The nursery's facts are a few moments old, so the
position maps learned to look near, and their bias and cortex columns outvote the key by a margin
that grows with the distance. The same terms carry the misses at 1,000 raw words (#266) and at a
duplicate's moment (#269). With the position term off and the temperature at .05 the memory finds
and uses every record to 100,000.

### Retention

Budget 1,000 at the end of the stream (100,000 moments). `top`: the share of the 10% most-used
facts that a read returns at their current content. `revised`: the same over the facts among them
that changed. `stale`: revised facts whose read holds an old content. The drifting stream
reshuffles the facts' ranks at its midpoint; `left` is the share held of the facts that were among
the most used before it and are not after it.

| Arm | top | revised | stale | rarest half held | questions answered | writes, budgets | drifting: top | left |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fade`, the candidate | .963 to .980 | .971 to .981 | .00 | .07 to .08 | .76 to .77 | 19.5 to 19.7 | .948 to .963 | .23 to .28 |
| `fade-soft`, softmax at .05 | .973 to .990 | .973 to .989 | .00 | .06 to .08 | .76 to .77 | 19.2 to 19.5 | .953 to .975 | .25 to .29 |
| `lru`, exact keys | .782 to .823 | .782 to .832 | .00 | .11 to .12 | .73 to .74 | 22.2 to 22.6 | .782 to .823 | .14 to .23 |
| `fifo`, exact keys | .660 to .703 | .670 to .715 | .00 | .13 to .14 | .70 | 26.3 to 26.7 | .665 to .698 | .17 to .24 |
| `random`, exact keys | .650 to .720 | .657 to .738 | .00 | .12 to .14 | .70 | 26.3 to 26.7 | .685 to .725 | .18 to .24 |
| `lfu`, exact counts | 1.000 | 1.000 | .00 | .00 to .01 | .78 | 17.6 to 17.8 | .848 to .887 | .98 to .99 |
| `ring`, the candidate with the fading off | .660 to .703 | .670 to .715 | .00 | .13 to .14 | .70 | 26.3 to 26.7 | .665 to .698 | .17 to .24 |
| `recent`, no depth | .790 to .820 | .788 to .828 | .00 | .11 to .12 | .73 to .74 | 22.3 to 22.8 | .785 to .823 | .16 to .24 |
| `cumulative`, the depth a whole sum | .955 to .978 | .960 to .979 | .00 | .06 to .07 | .76 to .77 | 19.4 to 19.8 | .405 to .480 | .94 to .97 |
| `append`, no revision | .230 to .253 | .203 to .236 | .56 to .62 | .04 to .05 | .12 to .13 | 97.9 to 98.0 | .278 to .360 | .14 to .22 |

Every arm passes 14 budgets of writes or more. At 128 slots `fade` reads .961 to 1.000 against
.765 to .863 for least-recently-used; at 10,000 slots, with the address picking its candidates,
.953 to .961 against .813 to .827, revised .965 to .972.

`top` over the stream, at 10 to 100 budgets of moments (mean of the six founders):

| Arm | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fade` | .900 | .941 | .961 | .961 | .967 | .966 | .975 | .970 | .970 | .970 |
| `lru` | .827 | .812 | .818 | .811 | .822 | .822 | .832 | .808 | .807 | .800 |
| `lfu` | .933 | .978 | .992 | .997 | .998 | .999 | 1.000 | 1.000 | 1.000 | 1.000 |

**What the readings say about the rule.**

- **Depth is what holds a fact.** Recent use alone reads as least-recently-used does. With depth
  a fact asked a few times fades slowly enough to reach its next use, and each use deepens it.
  Facts used often are gathered in the first 30 budgets of moments and held from there on.
- **The depth has to leak.** Kept as a whole sum, the form `memory_fade` (#253) gives the store's
  records, it holds the plain stream and fails the drifting one (.405 to .480): the facts that fell
  out of use keep their slots, as an exact count's do. Kept over 8 capacities of moments it holds
  both.
- **Revision keeps a fact current.** Without it every statement takes a slot, copies crowd the
  memory, and the read holds an old content on .56 to .62 of the revised facts. With it no fact
  holds two slots and no read is stale.
- **The read and the address make no difference to what is kept.** With the fading off the
  candidate reads as first-in-first-out over exact keys does, to the digit.
- An exact count is the upper reference on a plain stream, and it never forgets.

### Cost

One core of the c7a.8xlarge, the box running nothing else, mean of founders 2880 to 2882, ms.
`moment` is the default composition at key width 64 (an observation of 128), its memory filled to
N records: the read, the write and the rest (the settle and the brain's own work). The budget of
C1 is the rest at 128 records, .120 ms.

| Records | `moment`: read | write | rest | `moment-content`: read | `exact`: read | `bank`: read | write | `bank-dense`: read |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | .140 | .037 | .120 | .059 | .006 | .013 | .017 | .013 |
| 1,000 | .265 | .050 | .130 | .156 | .019 | .208 | .241 | .048 |
| 10,000 | 1.469 | .060 | .184 | 1.112 | .168 | .249 | .299 | .363 |
| 100,000 | 13.638 | .179 | .286 | 11.031 | 3.218 | .573 | .807 | 10.120 |

On the device's box (one A10G beside its own CPU; the budget of C2 is that CPU's rest at 128
records, .203 ms): the recall memory's read as one product on the device, float32, every slot
scored, takes .111 ms at 128 records, .112 at 1,000, .120 at 10,000 and .398 at 100,000.

- **The 128-slot default.** The recall memory's own read takes .140 ms at 128 slots, more than the
  rest of the moment (.120), and grows with the slots: 1.5 ms at 10,000 and 13.6 at 100,000, 8 and
  48 times the rest there. Its write stays under .2 ms.
- **The candidate.** Its read costs .25 ms at 10,000 records and .57 at 100,000, a sixth and a
  twenty-fourth of the recall memory's. It misses C1's budget by a factor of 2.1. The address is
  the cost: coding the cue onto 8,192 units and choosing its winners takes about .19 ms whatever
  the records, more than scoring 10,000 keys (.17 ms for the exact store). The address pays from
  about twice its units on: at 100,000 records the exact store takes 3.2 ms.
- **Bytes.** A slot of the recall memory holds a key, a value and a next, 3.1 kB at key width 64 in
  float64. The candidate holds 1.05 kB per record without its address, and with it 2.5 kB at 10,000
  records and 2.1 at 100,000.
- **The guard at 4,096 code coordinates.** At a new brain's 128 slots, by the observation's width:

| Observation | Address map of one head | kB per slot | Read, ms |
| --- | --- | --- | --- |
| 128 | 0.15 MB | 3.1 | .147 |
| 512 | 2.2 MB | 12.4 | .236 |
| 2,048 | 33.8 MB | 49.5 | .918 |
| 4,096 | 134.8 MB | 99.1 | 2.723 |

  The map grows with the square of the width and every slot with the width: at 4,096 coordinates
  10,000 records hold 1 GB. A large memory keys on a compact code.

**What a large memory needs.** A read by content alone, at a temperature that holds one record: the
position term and the spread bind first, and both are settings. Then, by size: to about twice the
address's units, scoring every slot is the cheapest read on one core; past that a sparse address in
front of the read; and on a device one product reads 100,000 records in .4 ms. Under a fixed budget
it needs a replacement rule that reads depth, and a write that revises.

### After the declaration

C1 named its own cause: the address. The library's `cd.LongTermMemory` takes that as its founder.
A memory of at most `direct` slots (twice the address's units, 16,384) scores every slot, and the
address picks candidates past that; its reads are `bank-dense`'s to that size and `bank`'s above
it, both read above. Read after the declaration on the same box, core and founders
([`after/cost.json`](after/cost.json)), ms: `memory` is the library's founder, `memory-float32`
the same with keys and values in float32, `bank` the declared arm on the library's code.

| Records | `moment`: rest | `memory`: read | write | `memory-float32`: read | `bank`: read | `exact`: read |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | .121 | .008 | .019 | .009 | .008 | .006 |
| 1,000 | .129 | .015 | .028 | .013 | .197 | .019 |
| 10,000 | .184 | .087 | .115 | .056 | .231 | .167 |
| 100,000 | .283 | .543 | .822 | .496 | .543 | 3.199 |

At 10,000 records the founder reads in .084 to .093 ms against a budget of
.120 to .121, on 3 of 3 founders, and in .056 ms in float32. This reading was
taken after C1 failed and is no declared pass. On the library's code the declared arm reads in
.231 ms at 10,000 records, where the declared code read .249.

### The transformer over the window

`nets.py`, declared on founders 2880 to 2882 on one A10G: 2 layers, 2 heads, width 256, 1,103,936
parameters, taught by backpropagation on record sets drawn anew for every lesson. An answer is
right when the stored content nearest to it is the right record's. That is decoding, a weaker test
than `found`: a read whose mass is spread can decode right.

| Taught at | Read at | Founders | exact | noisy | partial | similar | similar, noisy | Lessons | Seconds to teach |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 128 | 3 | 1.000 | .993 to .998 | .996 to 1.000 | 1.000 | .855 to .877 | 20,000 | 634 to 641 |
| 128 | 1,000 | 3 | 1.000 | .987 to .996 | .969 to .984 | 1.000 | .864 to .875 | 20,000 | 634 to 641 |
| 1,000 | 1,000 | 3 | 1.000 | .980 to .988 | .953 to .960 | 1.000 | .767 to .797 | 12,000 | 518 to 519 |
| 1,000 | 4,000 | 3 | 1.000 | .945 to .960 | .902 to .920 | 1.000 | .785 to .830 | 12,000 | 518 to 519 |

The transformer holds the windows it reads. A network taught at 128 records reads 1,000 with its
exact and noisy cues intact, and one taught at 1,000 reads 4,000 at .945 to .960 on noisy cues and
.902 to .920 on partial ones, where the candidate bank, by the stricter `found`, reads 10,000 at
.999 to 1.000 and .998 to .999. Full self-attention keeps N squared weights per head and layer:
400 MB in float32 at 10,000 records and 40 GB at 100,000, so it is not read there. The memories
above are taught nothing: their read is fixed, and what they hold is written once.

## The second declaration

[`PROTOCOL_2.md`](PROTOCOL_2.md), declared at commit `cb3c5f1a` before its runs: the cost gate for
the store as it stands, on founders 2886 to 2888, and facts whose keys and values
`cadence.Message` makes (#286) in place of the stand-ins above, on founders 2890 to 2895.

### The gates

| Gate | Reading | Bar | Founders | Verdict |
| --- | --- | --- | --- | --- |
| C1b | ms per read with its value at 10,000 records, the store as it stands, one core | .122 | .084 to .093, 3 of 3 | **passes** |
| M1 | found, exact key, 10,000 facts | .99 | .996 to 1.000, 6 of 6 | **passes** |
| M2 | found, noisy key, 10,000 facts | .90 | .852 to .908, 1 of 6 | **fails**: the key, the exact store misses with it |
| M3 | 4 heads return 4 different facts about the agent, 10,000 facts | .85 | .825 to .925, 3 of 6 | **fails**: the key, the exact store misses with it |
| M4 | budget 1,000: the 10% most-used facts retrievable | .95 | .968 to .978, 6 of 6 | **passes** |
| M5 | budget 1,000: revised facts current | .90 | .970 to .984, 6 of 6 | **passes** |
| M6 | budget 1,000: above least-recently-used |  | .968 to .978, 6 of 6 | **passes** |

### The cost gate for the store as it stands

`cd.LongTermMemory` scores every slot up to 16,384 records and reads through its address past
that. One process on a spot m7a.8xlarge that ran nothing else (the first reading's processor, an
AMD EPYC 9R14; the c7a.8xlarge first launched was reclaimed after five minutes and its rows are
not used), ms, mean of founders 2886 to 2888. `floor` is the keys' product with the cue and its
first choice alone, the least a read that scores every slot does in NumPy.

| Records | `moment`: read | rest | `memory`: read | write | `memory-float32`: read | `floor` | `floor32` | `exact`: read |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | .142 | .122 | .008 | .019 | .009 | .003 | .003 | .006 |
| 1,000 | .266 | .131 | .016 | .029 | .013 | .008 | .005 | .020 |
| 10,000 | 1.496 | .188 | .088 | .117 | .055 | .074 | .038 | .158 |
| 100,000 | 14.012 | .287 | .543 | .802 | .497 | 2.533 | .388 | 3.346 |

At 10,000 records the store reads in .084 to .093 ms against a budget of .122, on 3 of 3
founders: C1b passes, and with C2 the issue's cost box is met. The floor reads .071 to .079 ms,
so the store's read is within a fifth of the least such a read does, and the margin under the
budget is the floor's: by its growth with the records the product alone reaches the budget at
about 16,000 records. Past 16,384 the store reads through its address, .543 ms at 100,000
records where the product alone takes 2.533; that is above the budget, as the first
declaration read it.

### Facts keyed by held meanings

`meanings.py` holds the facts. The lexicon is the grammar chamber's concept moments at 512 noun
stems in two numbers and 128 verbs. A fact says of an agent and a verb which patient. Its key
says what it is about: `Message.key()` after the act and the agent are heard, about ten active
units of 280 in two groups of five. Its value says it: `Message.state` after the patient is heard
too. The code is #286's (4 phases, 5 winners, 280 units), each unit reading the chamber's share
of the channels. N facts are drawn about different agent-verb pairs of the lexicon's 131,072, so
two facts about one agent, or with one verb, share a group of five units. `exact` is an exact
store over the same keys, the ceiling; `memory` is `cd.LongTermMemory` as it stands;
`memory-same` recognises a key told again only where it is the same key (`revise=0.999`). Cues:
the fact's key; a noisy key (three in ten active units exchanged for silent ones); a partial key
(half of the coordinates silent); for 4 heads, the agent's moment alone.

| Facts | Arm | exact | noisy | partial | 4 heads | held apart |
| --- | --- | --- | --- | --- | --- | --- |
| 1,000 | `exact` | .998 to 1.000 | .950 to .984 | .796 to .870 | .91 to 1.00 |  |
| 1,000 | `memory` | .998 to 1.000 | .963 to .989 | .809 to .855 | .83 to 1.00 | .9990 to 1.0000 |
| 1,000 | `memory-same` | 1.000 | .963 to .989 | .809 to .855 | .83 to 1.00 | 1.0000 |
| 10,000 | `exact` | .995 to 1.000 | .850 to .894 | .597 to .655 | .82 to .93 |  |
| 10,000 | `memory` | .996 to 1.000 | .852 to .908 | .613 to .646 | .82 to .93 | .9951 to .9985 |
| 10,000 | `memory-same` | .996 to .999 | .852 to .909 | .614 to .645 | .83 to .92 | .9986 to .9995 |
| 100,000 | `exact` | .971 to .993 | .593 to .707 | .357 to .422 | .78 to .94 |  |
| 100,000 | `memory` | .961 to .991 | .567 to .706 | .335 to .437 | .77 to .96 | .9670 to .9872 |
| 100,000 | `memory-same` | .971 to .993 | .571 to .706 | .342 to .437 | .78 to .96 | .9812 to .9947 |
| 100,000 | `memory-direct` | .961 to .991 | .567 to .706 | .336 to .437 | .80 to .96 | .9670 to .9872 |

On the random stand-ins the candidate read 1.000, .999 to 1.000 and .998 to .999 at 10,000 records
and 1.000, .986 to .997 and .984 to .993 at 100,000, with 4 heads at 1.00.

**What the key holds.** On the exact store: the share of facts whose key is their own, and the
share of cues whose best overlap the right fact holds alone (the rest are ties).

| Facts | distinct keys | the right fact alone at the best overlap: exact | noisy | partial |
| --- | --- | --- | --- | --- |
| 1,000 | 1.0000 | .998 to 1.000 | .926 to .974 | .720 to .799 |
| 10,000 | .9986 to .9995 | .992 to .999 | .781 to .842 | .502 to .554 |
| 100,000 | .9806 to .9947 | .950 to .986 | .451 to .586 | .250 to .318 |

**The first term that binds is the key.**

- **An exact key** finds its fact as the stand-ins did to 10,000 facts (M1 holds). At 100,000 it
  reads .961 to .991, and the exact store .971 to .993: up to two facts in a hundred share their
  key with another fact, since some agents' codes are one code.
- **A noisy or a partial key** binds from the first thousand facts. A key holds two groups of
  five units, and each group is every sibling's too. A cue that loses units of one group ties
  with the siblings that share the other: among 10,000 facts the right fact holds the best overlap
  alone on .78 to .84 of the noisy cues and .50 to .55 of the partial ones, and the exact store
  finds no more than the memory does (M2 and M3 fail on both).
- **The write** binds second. At cosine .9 the store takes a sibling for the fact told again
  where two keys of ten units differ in one: among 100,000 facts it holds .967 to .987 of them
  apart, and .981 to .995 where only the same key counts, which is the share of distinct keys.
- **The address does not bind.** At 100,000 facts the store reads through its address, and on
  the mean of the six founders it reads within .01 of the exact store on every cue where only
  the same key counts. With every slot scored
  (`memory-direct`, the declared code on all six founders) the exact key reads the same to the
  digit on every founder, the noisy and partial keys within .002, and the agent cue within .035.
- **The form binds wholly.** The same act and agent heard as their words find their fact on
  .000 to .010 of the cues at every size: a word's code and its concept's are unrelated draws.
  #286 located that limit, and a learned map from words to concept codes is its to build.

**Beside the declared code** (`memory`):

| Code | Facts | exact | noisy | partial | 4 heads | distinct keys |
| --- | --- | --- | --- | --- | --- | --- |
| 280 units, the chamber's share of the channels (declared) | 10,000 | .996 to 1.000 | .852 to .908 | .613 to .646 | .82 to .93 | .9986 to .9995 |
| 280 units, the chamber's share of the channels (declared) | 100,000 | .961 to .991 | .567 to .706 | .335 to .437 | .77 to .96 | .9806 to .9947 |
| 70 units, the same share | 10,000 | .908 to .965 | .367 to .446 | .314 to .449 | .14 to .41 | .9716 to .9959 |
| 70 units, the same share | 100,000 | .723 to .862 | .094 to .174 | .124 to .204 | .15 to .39 | .8648 to .9567 |
| 280 units, the class's own fan-in | 10,000 | .942 to .995 | .468 to .583 | .369 to .443 | .42 to .80 | .9500 to .9988 |
| 280 units, the class's own fan-in | 100,000 | .770 to .940 | .146 to .195 | .149 to .203 | .45 to .77 | .8287 to .9851 |

A narrower code and a code whose units read one input each hold fewer distinct keys, and every
cue falls with them.

**Retention** (budget 1,000, 4,000 facts, 100,000 moments, a change of patient before one
occurrence in twenty):

| Arm | top | revised | stale | drifting: top | left |
| --- | --- | --- | --- | --- | --- |
| `memory`, the store as it stands | .968 to .978 | .970 to .984 | .00 | .938 to .970 | .27 to .31 |
| `memory-same`, the same key only | .965 to .980 | .970 to .982 | .00 | .927 to .970 | .27 to .32 |
| `lru` | .780 to .833 | .795 to .847 | .00 | .780 to .833 | .16 to .22 |
| `fifo` | .675 to .728 | .682 to .739 | .00 | .677 to .725 | .16 to .21 |
| `lfu`, exact counts | 1.000 | 1.000 | .00 | .843 to .890 | .99 to 1.00 |
| `ring`, the fading off | .677 to .730 | .685 to .739 | .00 | .677 to .723 | .16 to .21 |
| `append`, no revision | .190 to .215 | .160 to .189 | .60 to .62 | .290 to .367 | .25 to .33 |

The store keeps and revises facts keyed by held meanings as it did the stand-ins (M4 to M6 hold;
.963 to .980 and .971 to .981 there). With the 70-unit code it reads .938 to .973 and .943 to .973.

**Where the key is taken** (development founders 0 to 2, receipt `results/dev-whole-keys.jsonl.gz`;
nothing here is declared). A fact told under the key of its whole episode, the patient's code in
it, and asked with what it is about: the top facts are retrievable at .25 to .28, the
revised ones current at .22 to .23 and stale at .58 to .61, against .968 to
.978, .970 to .984 and .00 on the declared founders with the key taken before the patient is
heard. A changed fact has another key, takes
another slot, and leaves its old copy to tie with the new one.

## The third declaration: a brain that writes and reads its own records

[`PROTOCOL_3.md`](PROTOCOL_3.md), declared at commit `713f83f1` before its runs: the second
declaration's facts with a brain in place of the instrument (`brain_records.py`). The brain hears
every fact, keys the record by its own held message at the turn, writes it, reads it before the
settle and answers by itself (`Brain.compose(longterm=records)`). Every episode begins with
`reset()`, which empties the recall memory's slots, so the records are the only carrier from a fact
to its question. Records founders 2900 to 2905, on a spot c7i.8xlarge; cost founders 2906 to 2908,
one process on that box running nothing else, and on one A10G.

### The gates

| Gate | Reading | Bar | Founders | Verdict |
| --- | --- | --- | --- | --- |
| B1 | found, unrevised facts, 1,000 records | .99 | .9975 to 1.000, 6 of 6 | **passes** |
| B2 | found, unrevised facts, 10,000 records | .99 | .980 to 1.000, 5 of 6 | **passes** |
| B3 | used, unrevised, 10,000 records, and within .05 of 1,000 | .90 | .980 to 1.000 against .9975 to 1.000, 6 of 6 | **passes** |
| B4 | revised facts at 10,000: found current, stale, used | .95, .02, .90 | .990 to 1.000, .000, .9925 to 1.000; 6 of 6 | **passes** |
| B5 | `plain`, the brain without the option: used | at most .25 | .1225 to .155, 6 of 6 | **passes** |
| C3 | ms per moment at 10,000 records over that without the option, one stream, one core | at most 2 | 1.82 to 1.83, 3 of 3 | **passes** |
| D1 | the same on the device | at most 2 | 1.65 to 1.69, 3 of 3 | **passes** |

### Found and used

Each cell is the range over the six founders; 400 questions of each kind per founder.
Parameters: `records` 413,082, `records_h2` 438,302, `plain` 387,862. Every nursery ends at
.997 to 1.000 with records and .15 to .17 without.

| Records | Arm | Unrevised: found | used | Revised: found current | stale | used | Records held per stream |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1,000 | `records` | .9975 to 1.000 | .9975 to 1.000 | .9975 to 1.000 | .000 | .9975 to 1.000 | 997.7 to 1,000 |
| 1,000 | `records_h2` | .9975 to 1.000 | .9975 to 1.000 | .9975 to 1.000 | .000 | .995 to 1.000 | 997.7 to 1,000 |
| 1,000 | `plain` | .000 | .1025 to .155 | .000 | .000 | .1125 to .150 | |
| 10,000 | `records` | .980 to 1.000 | .980 to 1.000 | .990 to 1.000 | .000 | .9925 to 1.000 | 9,812 to 10,000 |
| 10,000 | `records_h2` | .980 to 1.000 | .9675 to .9975 | .990 to 1.000 | .000 | .985 to 1.000 | 9,812 to 10,000 |
| 10,000 | `plain` | .000 | .1025 to .1425 | .000 | .000 | .090 to .145 | |

The brain answers with what its read holds: with one head, on every founder, size and kind, `used`
is within .005 of `found`. A revised fact is current and never stale. A second head adds nothing here, where every
key is exact, and at 10,000 records its answers read up to .0125 below what it finds.

**The misses are the key's.** Every question hears what its fact did up to the turn, so a miss is
a key that does not name its fact alone. Read on the founders' own facts (four streams each):

- **Two facts with one key.** At founder 2901 3.7% to 3.9% of 10,000 facts share their whole key
  with another fact (two agents or two verbs whose codes are one code at 280 units over this
  lexicon); a question finds the record told last, and the stream holds 9,812 records for 10,000
  facts. Founders 2903 and 2905 have no shared key.
- **A key another contains.** A key whose units another record's key holds as well ties with it
  on the dot product the read scores by, and the first slot wins: .2% of the facts at 2901 and .6%
  at 2905, whose found reads .9925 with every key its own.

### Cost

ms per moment, mean of founders 2906 to 2908; the records filled by the store's own writes of facts'
keys and values (a contract test holds the keys equal to the brain's message's).

| Where | Streams | Records | Question moment | Fact moment |
| --- | --- | --- | --- | --- |
| one core | 1 | none (option off) | .767 | .768 |
| one core | 1 | 1,000 | .872 | .927 |
| one core | 1 | 10,000 | 1.068 | 1.398 |
| one core | 16 | none | 2.569 | 2.576 |
| one core | 16 | 1,000 | 3.531 | 4.274 |
| one core | 16 | 10,000 | 11.675 | 27.043 |
| A10G | 1 | none | 1.951 | 1.942 |
| A10G | 1 | 1,000 | 2.052 | 2.099 |
| A10G | 1 | 10,000 | 2.527 | 3.242 |
| A10G | 16 | none | 4.447 | 4.421 |
| A10G | 16 | 1,000 | 5.766 | 6.809 |
| A10G | 16 | 10,000 | 11.038 | 20.401 |

Over one stream the records cost .30 ms a question moment and .63 a fact moment at 10,000 records.
Over 16 streams the records are 16 stores read and written one after another on the host, and a
fact moment's write reads its store twice (the recognition, then the store's own revision check)
before it writes: at 10,000 records the records take 24.5 ms of a 27.0 ms moment, 10.5 times the
moment without the option. That cost is reported without a bar; one product across the streams' stores
and one read per write are what it asks for.

### Amendment 1: the many-stream write

[Amendment 1](PROTOCOL_3.md#amendment-1-the-many-stream-write-declared-10-november-2026-before-its-run),
declared at commit `c44abbcc` before its run: the streams' records are read, written and faded as
one bank (one product over every stream's records, the read before the settle as the write's
recognition, one fading pass), which a contract test holds to the stores read one by one to the
bit. Founders 2909 to 2911, one process on a spot c7i.8xlarge running nothing else; the records
filled by the records' own write.

| Gate | Reading | Bar | Founders | Verdict |
| --- | --- | --- | --- | --- |
| C4 | ms per fact moment over 16 streams at 10,000 records over that without the option, one core | at most 2 | 1.661 to 1.688, 3 of 3 | **passes** |
| C3, read again | ms per moment at 10,000 records over that without the option, one stream, one core | at most 2 | 1.254 to 1.261, 3 of 3 | passes |

ms per moment, mean of founders 2909 to 2911:

| Streams | Records | Question moment | Fact moment |
| --- | --- | --- | --- |
| 1 | none (option off) | .779 | .780 |
| 1 | 1,000 | .882 | .902 |
| 1 | 10,000 | .963 | .981 |
| 16 | none | 2.649 | 2.654 |
| 16 | 1,000 | 3.073 | 3.214 |
| 16 | 10,000 | 4.322 | 4.453 |

At 10,000 records over 16 streams the records add 1.80 ms to a fact moment and 1.67 to a question
moment, against 24.5 and 9.1 store by store; over one stream they add .20 and .18, against .63 and
.30. Founder 2909 holds 9,873 records for 10,000 facts: facts whose key another fact shares are one
record, the term the reading located at 2901.

### Amendment 2: the read scored by cosine

[Amendment 2](PROTOCOL_3.md#amendment-2-the-read-scored-by-cosine-declared-10-november-2026-before-its-run),
declared at commit `4049414e` before its run: the records' read and the write's recognition score
each record by the cosine of its key with the message's key, so a key that another record's key
contains reads its own record. Founders 2912 to 2917, the `records` arm alone, on a spot
c7i.8xlarge.

| Gate | Reading | Bar | Founders | Verdict |
| --- | --- | --- | --- | --- |
| B6 | found, unrevised facts, 10,000 records | .99, 6 of 6 | .9875 to 1.000, 5 of 6 | **fails** |
| B7 | the same over the questions whose fact's key is its own | .99, 6 of 6 | 1.000, 6 of 6 | **passes** |
| B1, read again | found, unrevised facts, 1,000 records | .99 | .995 to 1.000, 6 of 6 | passes |
| B3, read again | used, unrevised, 10,000 records, and within .05 of 1,000 | .90 | .9875 to 1.000, 6 of 6 | passes |
| B4, read again | revised facts at 10,000: found current, stale, used | .95, .02, .90 | .9975 to 1.000, .000, .9925 to 1.000; 6 of 6 | passes |

At 10,000 records a stream; 400 questions of each kind per founder, parameters 413,082, every
nursery ending at .9988 to 1.000:

| Founder | Facts whose key another fact has | Records held per stream | Unrevised: found | found, own key (questions) | used | Revised: found current | stale | used |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2912 | 1.20% | 9,938 to 9,943 | .995 | 1.000 (398) | .995 | .9975 | .000 | .9925 |
| 2913 | 0 | 10,000 | 1.000 | 1.000 (400) | 1.000 | 1.000 | .000 | .9975 |
| 2914 | 0 | 10,000 | 1.000 | 1.000 (400) | 1.000 | 1.000 | .000 | 1.000 |
| 2915 | 0 | 10,000 | 1.000 | 1.000 (400) | 1.000 | 1.000 | .000 | 1.000 |
| 2916 | .02% | 9,999 | 1.000 | 1.000 (400) | 1.000 | 1.000 | .000 | 1.000 |
| 2917 | 2.43% | 9,873 to 9,885 | .9875 | 1.000 (389) | .9875 | 1.000 | .000 | 1.000 |

At 1,000 records found reads .995 to 1.000 and 1.000 over the facts whose key is their own.

**B6 fails on the key's term, at founder 2917.** There 2.43% of a stream's 10,000 facts share their
whole key with another fact (two agents' or two verbs' codes are one code at 280 units over this
lexicon), so each such pair is one record, which the fact told last wrote. Eleven of its 400
unrevised questions ask about such a fact, and its 5 misses are among them; the 389 questions
about facts whose key is their own all find their record (B7). No score of the read tells two
facts with one key apart: the term is the key's, as at 2901 in the third declaration.

**The read's term is closed.** Over the facts whose key is their own every founder finds 1.000,
and the four founders with distinct keys or nearly so find 1.000 over all facts, where the third
declaration's reading by the dot product found .9925 to 1.000 on its two such founders. Used lies
within .005 of found on every founder and kind, and no revised fact is stale.

## The fourth declaration: a million records

Declared on [#310](https://github.com/muellerberndt/cadence/issues/310) under
[`PROTOCOL_4.md`](PROTOCOL_4.md) at commit `042cced6`, before the first run on its founders. It
reads the store alone at 100,000, 1,000,000 and 10,000,000 records: an address whose work grows
slower than the records, several records for one question, chains, and facts whose keys are one
key. One spot c7i.16xlarge, one thread per process; `summarize_million.py` recomputes every table
and gate below from the receipts (`results/fourth-*.jsonl.gz`).

**The address of pairs** (`LongTermMemory(pairs=16, expansion=16384, winners=256,
candidates=4096)`). A record is filed under every pair of its 16 most driven address units, in a
table of cells sized to the capacity. A cue's 256 winners name the pairs it reads, and the 4,096
slots found under the most pairs go to the read's competition. The store's address of units, as
it stands, lists every slot under each of 64 units, so a cue's units name about half of the slots
whatever their number. The scan scores every record: the control and the ceiling. Every arm reads
by cosine at temperature 0, so a read holds its first choice whole; found means that choice is
the record. Nothing is learned: the projection, the table and the key of parts are designed, and
each is read beside its control.

### The gates

| Gate | Reading | Bar | Result |
| --- | --- | --- | --- |
| C1 | pairs at 1,000,000, codes of 128 and the hashed form: exact; noisy and partial | .99; .95 | exact 1.000; codes .997 to 1.000 and .996 to 1.000; hashed .999 to 1.000 and .973 to .986; 12 of 12: **passes** |
| C3a | pairs against the scan at 1,000,000 on the same cues | within .01 | largest gap .004 (codes) and .010 (hashed, partial), 12 of 12: **passes** |
| C3b | ms per read at 1,000,000 over 100,000, one core, compiled and through NumPy | under 3 | pending: the timing cells run on the idle box |
| C2 | 1,000,000 facts, 4 in 100 sharing their key: found over all facts, with `apart` | .99, 12 of 12 | 1.000, 12 of 12: **passes** |
| C4a | 1,000,000 facts: the subject by 2, 4, 8 heads | .95 each | 1.000; 1.000; .994 to 1.000, 12 of 12: **passes** |
| C4b | 1,000,000 facts: four parts, a head each | .90 | 1.000, 12 of 12: **passes** |
| C5 | 100,000 facts: two and three hops, exact and noisy first cue | .95; .90 | 1.000 and .998 to 1.000, 12 of 12: **passes** |

### Capacity

`million.py`, founders 3100 to 3111, 1,000 cues of each kind per cell, 200 trials for several heads. Each cell is the range over the twelve founders; bytes, records scored, table entries read, multiply-accumulates and the projection's fixed weights are per record or per read, means over the founders.

| Form | Records | Arm | exact | noisy | partial | similar, noisy | bytes a record | records scored | entries read | multiply-accumulates | fixed weights |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `code-128` | 100,000 | `pairs` | 1.000 | .999 to 1.000 | 1.000 | .996 to 1.000 | 1,859 | 3,462 | 522,240 | 2,540,286 | 2,097,152 |
| `code-128` | 100,000 | `scan` | 1.000 | 1.000 | 1.000 | .996 to 1.000 | 772 | 100,000 | 0 | 12,800,000 | 0 |
| `code-128` | 100,000 | `units` | 1.000 | .998 to 1.000 | .999 to 1.000 | .996 to 1.000 | 1,869 | 256 | 50,001 | 1,081,344 | 1,048,576 |
| `code-128` | 1,000,000 | `pairs` | 1.000 | .997 to 1.000 | .996 to 1.000 | .995 to .998 | 1,783 | 4,096 | 522,240 | 2,621,440 | 2,097,152 |
| `code-128` | 1,000,000 | `scan` | 1.000 | 1.000 | 1.000 | .998 to 1.000 | 772 | 1,000,000 | 0 | 128,000,000 | 0 |
| `code-128` | 1,000,000 | `units` | 1.000 | .997 to 1.000 | .994 to 1.000 | .994 to .999 | 1,828 | 256 | 500,016 | 1,081,344 | 1,048,576 |
| `code-256` | 100,000 | `pairs` | 1.000 | 1.000 | 1.000 | 1.000 | 2,455 | 3,015 | 522,240 | 4,966,255 | 4,194,304 |
| `code-256` | 100,000 | `scan` | 1.000 | 1.000 | 1.000 | 1.000 | 1,284 | 100,000 | 0 | 25,600,000 | 0 |
| `code-256` | 100,000 | `units` | 1.000 | .999 to 1.000 | .999 to 1.000 | 1.000 | 2,423 | 256 | 50,007 | 2,162,688 | 2,097,152 |
| `code-256` | 1,000,000 | `pairs` | 1.000 | .999 to 1.000 | .998 to 1.000 | .998 to 1.000 | 2,304 | 4,096 | 522,240 | 5,242,880 | 4,194,304 |
| `code-256` | 1,000,000 | `scan` | 1.000 | 1.000 | 1.000 | 1.000 | 1,284 | 1,000,000 | 0 | 256,000,000 | 0 |
| `code-256` | 1,000,000 | `units` | 1.000 | .999 to 1.000 | .999 to 1.000 | .999 to 1.000 | 2,345 | 256 | 499,975 | 2,162,688 | 2,097,152 |
| `code-64` | 100,000 | `pairs` | 1.000 | .994 to .999 | .987 to .997 | .950 to .972 | 1,561 | 4,092 | 522,240 | 1,310,469 | 1,048,576 |
| `code-64` | 100,000 | `scan` | 1.000 | .995 to 1.000 | .988 to .997 | .952 to .972 | 516 | 100,000 | 0 | 6,400,000 | 0 |
| `code-64` | 100,000 | `units` | 1.000 | .988 to .997 | .984 to .993 | .950 to .971 | 1,592 | 256 | 49,999 | 540,672 | 524,288 |
| `code-64` | 1,000,000 | `pairs` | 1.000 | .970 to .989 | .960 to .984 | .939 to .957 | 1,523 | 4,096 | 522,240 | 1,310,720 | 1,048,576 |
| `code-64` | 1,000,000 | `scan` | 1.000 | .981 to .993 | .971 to .987 | .946 to .960 | 516 | 1,000,000 | 0 | 64,000,000 | 0 |
| `code-64` | 1,000,000 | `units` | 1.000 | .969 to .977 | .953 to .972 | .931 to .954 | 1,570 | 256 | 499,907 | 540,672 | 524,288 |
| `hashed-280` | 100,000 | `pairs` | 1.000 | 1.000 | .987 to .997 |  | 2,566 | 3,104 | 522,240 | 5,456,710 | 4,587,520 |
| `hashed-280` | 100,000 | `scan` | 1.000 | 1.000 | .990 to .999 |  | 1,380 | 100,000 | 0 | 28,000,000 | 0 |
| `hashed-280` | 100,000 | `units` | 1.000 | 1.000 | .987 to .993 |  | 5,270 | 256 | 78,793 | 2,365,440 | 2,293,760 |
| `hashed-280` | 1,000,000 | `pairs` | 1.000 | .999 to 1.000 | .973 to .986 |  | 2,401 | 4,096 | 522,240 | 5,734,400 | 4,587,520 |
| `hashed-280` | 1,000,000 | `scan` | 1.000 | 1.000 | .982 to .992 |  | 1,380 | 1,000,000 | 0 | 280,000,000 | 0 |
| `hashed-280` | 1,000,000 | `units` | 1.000 | .997 to 1.000 | .965 to .982 |  | 5,344 | 256 | 793,704 | 2,365,440 | 2,293,760 |

| Form | Records | Arm | family, 2, 4, 8 heads | summed, 2, 4, 8 | each by a head, 2, 4, 8 |
| --- | --- | --- | --- | --- | --- |
| `code-128` | 100,000 | `pairs` | 1.00; 1.00; 1.00 | .99 to 1.00; .68 to .79; .00 to .01 | 1.00; 1.00; 1.00 |
| `code-128` | 100,000 | `scan` | 1.00; 1.00; 1.00 | 1.00; .82 to .93; .00 to .01 | 1.00; 1.00; 1.00 |
| `code-128` | 100,000 | `units` | 1.00; 1.00; 1.00 | .99 to 1.00; .43 to .55; .00 | 1.00; 1.00; 1.00 |
| `code-128` | 1,000,000 | `pairs` | 1.00; 1.00; 1.00 | .99 to 1.00; .28 to .38; .00 | 1.00; 1.00; 1.00 |
| `code-128` | 1,000,000 | `scan` | 1.00; 1.00; 1.00 | 1.00; .62 to .74; .00 | 1.00; 1.00; 1.00 |
| `code-128` | 1,000,000 | `units` | 1.00; 1.00; 1.00 | .98 to 1.00; .11 to .20; .00 | 1.00; 1.00; 1.00 |
| `code-256` | 100,000 | `pairs` | 1.00; 1.00; 1.00 | 1.00; .79 to .88; .01 to .07 | 1.00; 1.00; 1.00 |
| `code-256` | 100,000 | `scan` | 1.00; 1.00; 1.00 | 1.00; 1.00; .53 to .67 | 1.00; 1.00; 1.00 |
| `code-256` | 100,000 | `units` | 1.00; 1.00; 1.00 | 1.00; .61 to .72; .00 to .01 | 1.00; 1.00; 1.00 |
| `code-256` | 1,000,000 | `pairs` | 1.00; 1.00; 1.00 | .99 to 1.00; .51 to .67; .00 to .01 | 1.00; 1.00; 1.00 |
| `code-256` | 1,000,000 | `scan` | 1.00; 1.00; 1.00 | 1.00; .99 to 1.00; .20 to .32 | 1.00; 1.00; 1.00 |
| `code-256` | 1,000,000 | `units` | 1.00; 1.00; 1.00 | .99 to 1.00; .33 to .38; .00 | 1.00; 1.00; 1.00 |
| `code-64` | 100,000 | `pairs` | 1.00; 1.00; 1.00 | .99 to 1.00; .01 to .07; .00 | 1.00; 1.00; 1.00 |
| `code-64` | 100,000 | `scan` | 1.00; 1.00; 1.00 | .99 to 1.00; .01 to .07; .00 | 1.00; 1.00; 1.00 |
| `code-64` | 100,000 | `units` | 1.00; 1.00; 1.00 | .99 to 1.00; .01 to .06; .00 | 1.00; 1.00; 1.00 |
| `code-64` | 1,000,000 | `pairs` | 1.00; 1.00; 1.00 | .98 to 1.00; .00 to .01; .00 | 1.00; 1.00; 1.00 |
| `code-64` | 1,000,000 | `scan` | 1.00; 1.00; 1.00 | .98 to 1.00; .00 to .02; .00 | 1.00; 1.00; 1.00 |
| `code-64` | 1,000,000 | `units` | 1.00; 1.00; 1.00 | .96 to .98; .00 to .01; .00 | 1.00; 1.00; 1.00 |

- **What binds at 1,000,000 is the key, then the address.** A partial hashed cue keeps about ten
  of its twenty units: the scan itself finds .982 to .992 of them, and the address of pairs loses
  .002 to .010 more, where the right record is not among its 4,096 candidates (it is on .985 to
  .994 of those cues). At 64 coordinates the key binds for every arm: the scan reads .981 to .993
  from a noisy cue and .971 to .987 from a partial one.
- **Several records.** k heads over a family, and a head for each of k codes, return k different
  relevant records at 1.00 for k = 2, 4 and 8, every arm and size. k codes summed into one cue bind
  on the key's width, as in the first declaration: four are found at .62 to .74 by the scan at 128
  coordinates and 1,000,000 records and at .28 to .38 through the address of pairs, which loses
  more of them; eight at .00 to .01 for every arm but the scan at 256 coordinates (.20 to .32).
- **Bytes.** A record costs its key and value (772 bytes for codes of 128 and values of 64 in
  float32), its sums (32 bytes) and, through the address of pairs, 960 bytes of table: 120 pairs at
  8 entries a cell of 64 bytes. The table is sized to the capacity and drops the entries a full
  cell does not take: 1.3% of them at 1,000,000 hashed keys and 2.3% at 1,000,000 codes of 128.

### Several records, questions of parts, chains

`facts.py`, founders 3112 to 3123. N / 8 entities and 64 relations, each a unit code of 64
coordinates; every entity is the subject of 8 facts with 8 different relations, and a fact's
value is its object's code, so what a read holds can be the next cue's subject. The key of parts
is the subject's code beside the relation's (128 coordinates); the control key is their unit sum
(64 coordinates). Each cell is the range over the twelve founders.

| Key | Facts | Arm | one fact, exact; noisy | the subject by 2; 4; 8 heads | parts, a head each, 2; 4; 8 | parts summed into one cue, 2; 4; 8 | chain from an exact cue, 2 and 3 hops | from a noisy cue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| parts | 100,000 | `pairs` | 1.000; .998 to 1.000 | 1.000; 1.000; 1.000 | 1.000; 1.000; 1.000 | .830 to .896; .014 to .028; .000 | 1.000 | .998 to 1.000 |
| parts | 100,000 | `scan` | 1.000; 1.000 | 1.000; 1.000; 1.000 | 1.000; 1.000; 1.000 | .814 to .870; .014 to .030; .000 | 1.000 | .998 to 1.000 |
| parts | 1,000,000 | `pairs` | 1.000; .980 to .992 | 1.000; 1.000; .994 to 1.000 | 1.000; 1.000; 1.000 | .806 to .862; .000 to .006; .000 | .998 to 1.000 | .978 to 1.000 |
| parts | 1,000,000 | `scan` | 1.000; .996 to 1.000 | 1.000; 1.000; 1.000 | 1.000; 1.000; 1.000 | .818 to .870; .000 to .010; .000 | 1.000 | .998 to 1.000 |
| summed | 100,000 | `pairs` | 1.000; .946 to .978 | 1.000; 1.000; .986 to 1.000 | 1.000; 1.000; 1.000 | .832 to .884; .000 to .002; .000 | 1.000 | .948 to .972 |
| summed | 100,000 | `scan` | 1.000; .946 to .978 | 1.000; 1.000; .990 to 1.000 | 1.000; 1.000; 1.000 | .836 to .884; .000 to .002; .000 | 1.000 | .950 to .972 |
| summed | 1,000,000 | `pairs` | .998 to 1.000; .862 to .892 | 1.000; 1.000; .946 to .968 | .998 to 1.000; .996 to 1.000; .994 to 1.000 | .700 to .752; .000; .000 | .996 to 1.000 | .848 to .892 |
| summed | 1,000,000 | `scan` | 1.000; .868 to .902 | 1.000; 1.000; .976 to .996 | 1.000; 1.000; 1.000 | .726 to .784; .000; .000 | 1.000 | .860 to .910 |

A chain of three hops reads as its two hops do: the miss is the first, noisy hop, and every later
hop's cue holds a value read whole.

- **Several facts through one cue bind.** Two parts summed are found at .81 to .90 and four at .00
  to .03, for the scan as well: one address holds few superposed codes, the first declaration's
  located term. A head per part finds every part to eight parts.
- **The key of parts carries the noise.** With the subject and the relation summed on one set of
  coordinates a noisy cue among 1,000,000 facts finds its fact at .86 to .90, the scan too; with
  the parts side by side, .980 to .992 (the scan .996 to 1.000).
- **The address against the scan** at 1,000,000 lies within .02 on every reading of the key of
  parts. The widest gap is the noisy single fact: 8 facts share each subject and about 15,600
  each relation, half a key apiece, and they crowd the candidates.

### The read's cost on one core

Pending: the declared timing cells (founders 3136 to 3138) run on a box that runs nothing else,
from 21:57 UTC. The capacity cells above were timed on a box shared with about forty other
processes: a read at 1,000,000 records took 6 to 18 ms through the address of pairs, 14 to 32 ms
through the address of units and 69 to 93 ms for the scan (codes of 128); 9 to 20, 17 to 41 and
155 to 297 ms (the hashed form). The address of pairs reads 522,240 table entries whatever the
records, so its cost stays nearly flat, while the address of units' grows with the records: at
100,000 records the address of units reads in about 1 ms and the address of pairs in 5 to 13 ms.

### Facts whose keys are one key

`facts.py`, founders 3124 to 3135 (the plain arm on 3124 to 3127), 1,000,000 facts with a key of
parts. A share .04 of the facts have a key another fact has: an entity takes another's code and
relations. Every fact has a source, a code of 32 coordinates drawn for it, which stands in for a
second code of the sentence. Facts are written one by one in a shuffled order, then 1 in 100 is
told again with another object; found is by value, the read holding the fact's current object.
The cells were run again from 19:40 UTC on the same plan and code after the queue's session ended
them while they wrote.

| Arm | found, all facts | sharing facts | the others | told again: current | stale | records for the facts | ms per write |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pairs` | .9782 to .9802 | .456 to .506 | 1.000 | 1.000 | .000 | .9800 | 6.564 to 6.706 |
| `pairs-apart` | 1.0000 | 1.000 | 1.000 | 1.000 | .000 | 1.0000 | 8.015 to 8.151 |

- **With `apart` every fact holds a record of its own.** A write whose key a record holds revises
  it only where the source's signatures agree (5 of 8 units), and a read that carries its source
  takes its own record among those tied at its best score: 1.000 over every kind of fact.
- **Without it** a pair of facts with one key is one record, written by the fact told last: the
  sharing facts read .46 to .51 and all facts .978 to .980.
- The signature costs 16 bytes a record and about 1.4 ms a write here, with sixteen writers on the
  box.

### Ten million records

Not read before the release. The declared cells (codes of 128 on founders 3100 and 3101, the
hashed form on 3100) were queued after the timing; the first, filling at 19:39 UTC, was ended with
the queue's session and runs again after the timing. One development cell (founder 0, codes of
128, `expansion=32768, winners=256, candidates=4096`, half the declared winners and candidates)
read exact 1.000, noisy .985 and partial .985 against the scan's 1.000, a read in 8.8 ms against
the scan's 500 ms; a full table drops 12.6% of its entries there, since pairs of units whose
weights point alike recur in many records.

## The fifth declaration: a brain with its own records at 100,000

Declared on #310 under [`PROTOCOL_5.md`](PROTOCOL_5.md) at `e60dbb04`, run under its Amendment 1
at `e75f56a5` (the instrument's own-key count by parts, after a first life that ended in that
count's memory before any question). The third declaration's chamber and brain with a lexicon of
256 stems and 256 verbs (131,072 agent-verb pairs, 1,284 inputs) and each stream's store through
the address of pairs (`longterm_pairs=16, longterm_expansion=16384, longterm_winners=256,
longterm_candidates=4096, longterm_fade=False`). A nursery of 1,500 rounds over 16 streams; the
reading tells 100,000 facts once each on one stream, a tenth again with another patient, and asks
400 questions of each kind without lessons (chance .125). Founders 3140 to 3151.

| Gate | Reading | Bar | Result |
| --- | --- | --- | --- |
| C6a | used, unrevised facts, 100,000 records | .95 | .960 to 1.000, 12 of 12: **passes** |
| C6b | revised facts: found current; stale; used | .95; under .02; .95 | .995 to 1.000; .000; .995 to 1.000, 12 of 12: **passes** |

| Founder | Records | Facts whose key another has | found | used | found, own key | revised: current | stale | used | ms a told moment | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3140 | 99,856 | 0.29% | 0.9975 | 0.9975 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 9.78 | 2,403,231 |
| 3141 | 99,546 | 0.91% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 9.66 | 2,403,231 |
| 3142 | 99,079 | 1.84% | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 9.57 | 2,403,231 |
| 3143 | 95,365 | 8.58% | 0.9600 | 0.9600 | 1.0000 | 0.9950 | 0.0000 | 0.9950 | 9.61 | 2,403,231 |
| 3144 | 99,400 | 1.20% | 0.9975 | 0.9950 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 9.55 | 2,403,231 |
| 3145 | 98,761 | 2.36% | 0.9900 | 0.9900 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 9.68 | 2,403,231 |
| 3146 | 99,228 | 1.54% | 0.9975 | 0.9975 | 1.0000 | 0.9975 | 0.0000 | 0.9975 | 9.76 | 2,403,231 |
| 3147 | 97,694 | 4.38% | 0.9700 | 0.9700 | 1.0000 | 0.9975 | 0.0000 | 0.9950 | 9.75 | 2,403,231 |
| 3148 | 97,494 | 5.00% | 0.9775 | 0.9775 | 1.0000 | 0.9950 | 0.0000 | 0.9950 | 9.59 | 2,403,231 |
| 3149 | 99,550 | 0.90% | 0.9975 | 0.9950 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 9.75 | 2,403,231 |
| 3150 | 99,254 | 1.49% | 0.9875 | 0.9875 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 9.67 | 2,403,231 |
| 3151 | 98,895 | 2.09% | 0.9850 | 0.9850 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 9.61 | 2,403,231 |

- **The misses are the key's.** Every question whose fact has a key of its own finds its record
  and is answered from it (1.000 on every founder). At 280 units over this lexicon some agents' or
  verbs' codes are one code, so 0.3% to 8.6% of a founder's facts share their whole key with
  another fact, and each such pair is one record, written by the fact told last: the third
  declaration's located term at ten times the records. The store's `apart` tells such facts apart
  where a write carries a source (C2); the brain's records pass none.
- **Used is found.** The answer lies within .0025 of the read on every founder and kind; a revised
  fact is current on .995 to 1.000 of questions and never stale.
- **Bytes and time.** About 14 kB a record, mostly its value (the heard moment of 1,284
  coordinates in float64); a told moment takes 9.6 to 9.8 ms on one core with the box shared.
- **Not read: 1,000,000 records a stream**, about 11 hours a founder at that cost.

## What these readings do not establish

- Any default. The candidate is an instrument-level store and the library's `LongTermMemory` a
  standalone one; a default moves only under the preservation panel and every gate of #256.
- A key that two wordings share. The first declaration's addresses are random codes and the
  second's are `cadence.Message`'s codes of concept moments, where a sentence heard as words has
  another key; a learned code, and one lexicon and one shape of fact beyond those read, are not
  read. `world.codes` and `meanings.py` take the key source as a parameter.
- A brain that writes its own long-term memory in a life: which moments are written, where in
  its message the key is taken, and a read that enters the settle through a region of its own.
  In `used.py` the bank's read is held on input coordinates, and in every reading the instrument
  tells the store the facts with their keys and values apart.
- A brain that uses what a message-keyed read holds. `used` is read on the stand-ins alone.
- Taught heads. The recall memory's heads come apart through lessons on their own address maps
  ([several heads](../../docs/memory.md#several-heads-opt-in)); those maps are at zero here, and
  untaught heads return different records only when each reads its first choice whole.
- The cost of a compiled read, in the first three declarations: every reading there is NumPy on
  one thread. The fourth times the address of pairs compiled and through NumPy.
- A learned address. The fourth declaration's address is a fixed projection and a table of pairs,
  designed; its keys are random codes and hashed codes that stand in for a sentence's settled
  state (#308), whose own keys are not read here.
- A brain that holds 1,000,000 records a stream: the fifth declaration reads 100,000, and a
  million facts of four moments take about 12 hours a founder at its cost.
- The controls of C7: a transformer that holds the records in its context, and a retriever with a
  reader. The scan is the retrieval ceiling over the same vectors; the first declaration's
  transformer over the window holds 4,000 records.
