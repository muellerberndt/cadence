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
and `bank.py` the candidate by its declared name. The library holds the candidate as
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
  the same key counts.
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

## What these readings do not establish

- Any default. The candidate is an instrument-level store and the library's `LongTermMemory` a
  standalone one; a default moves only under the preservation panel and every gate of #256.
- A key that holds meaning. The addresses are random codes; what a learned code or the message
  code of #286 does to the similarity between records is not read here. `world.codes` takes the
  key source as a parameter.
- A brain that writes its own long-term memory in a life: which moments are written, and a read
  that enters the settle through a region of its own. In `used.py` the bank's read is held on
  input coordinates and the instrument tells the bank the facts.
- Taught heads. The recall memory's heads come apart through lessons on their own address maps
  ([several heads](../../docs/memory.md#several-heads-opt-in)); those maps are at zero here, and
  untaught heads return different records only when each reads its first choice whole.
- The cost of a compiled read. Every reading is NumPy on one thread.
