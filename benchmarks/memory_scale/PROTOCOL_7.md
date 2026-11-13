# Records at any size, on disk: the seventh declaration

Declared 13 November 2026, before the first run on its founders. Issue:
[#310](https://github.com/muellerberndt/cadence/issues/310): the scope of its comment of 13
November (a store that grows past the machine's memory, read at 10,000,000 and 100,000,000
records; the law of the read's time and of the bytes per record; records added while the store
is read, with no rebuild and a bounded pause). It follows [the fourth declaration](PROTOCOL_4.md),
which read the store in memory at 1,000,000, and stands beside [the sixth](PROTOCOL_6.md), which
reads a brain with its own million. Instrument: `scale.py`; plans: [`protocol-7/`](protocol-7);
contracts: `test_memory_scale.py` and `tests/test_longterm_memory.py`; `summarize_scale.py`
recomputes every gate from the receipts.

## The store

One store, made the same way for every size:

```python
cadence.LongTermMemory(width, 64, capacity=134_217_727, pairs=16, expansion=65536, winners=256,
                       candidates=4096, cosine=True, fade=False, revise=0.0, dtype="float32",
                       grow=True, sure=0.999, path=folder)
```

- **`grow=True`**: the table of pairs starts at 64 cells and takes a cell at a time (linear
  hashing); the capacity, 2 to the 27th less one, is a bound.
- **`path=folder`**: every array is a mapped file on the box's NVMe disk.
- **`sure=0.999`**: a read of a held key ends at its record after 120 cells.
- **`expansion=65536`**: the address's units, set once for the bound. A record's place depends
  on them, so a store that is to reach 100,000,000 records is made with them from the start.
  The fourth declaration's stores, each made for one size, used 16,384 at 1,000,000.

Nothing is learned in this declaration. The designed parts are the fourth declaration's (the
fixed projection, the table of pairs, the centring) with the table's growth, the files and the
read that ends at its record; each is read beside the scan that scores every record.

## Records, keys and cues

The fourth declaration's forms, drawn by parts of 65,536 records so that no step holds a cell's
records in memory at once (`scale.draw_part`): `code-128` (unit codes of 128 coordinates in
float32, the first quarter of every part in families of 8 at cosine .8) and `hashed-280` (280
coordinates, 15 to 22 of them one). Values are unit codes of 64 coordinates. Cues as the fourth
declaration draws them: `exact`; `noisy` (cosine .71; a hashed key with three in ten of its
active units exchanged); `partial` (half of the coordinates silent); 1,000 of each kind a cell,
about records outside every family.

## The steps of a cell

Each step is a process of its own.

1. **build**: the records are written part by part with `fill(file=False)`. One `file()` files
   the first nine tenths onto an empty table, and a second files the last tenth onto the table
   as it stands, whose cells split in turn as they do for a write. The store is flushed and
   closed.
2. **read**: the page cache is emptied and the store opened again. Every cue is read once
   (`first`: found, ms, page faults that waited for the disk), then the first 300 of each kind
   again (`again`: their pages are in memory). Then 10,000 new records are written one by one,
   a read between every two writes: each write's pause, the cells the table took, whether the
   new records and the earlier ones are found.
3. **read under a memory limit** (10,000,000 and 100,000,000): the same step in a control
   group whose memory, page cache with it, is held to 8 GB (`MemoryMax`, no swap): 2.6 and 26
   times less than the store's bytes.
4. **scan**: the same cues by the read that scores every record, by parts over the store's keys.

## Gates

| Gate | Reading | Bar |
| --- | --- | --- |
| S1 | found on disk at 10,000,000 (12 founders) and at 100,000,000 (the founders read), `code-128`: from an exact cue; from a noisy and from a partial cue; each against the scan on the same cues | .99; .95; no more than .02 under the scan; on 10 of 12, and on every founder at 100,000,000 |
| S1, `hashed-280` | the same without a bar on the partial cue's own level (the scan's is the key's): exact; noisy; noisy and partial against the scan | .99; .95; no more than .02 under the scan |
| S2 | ms per read on one core with the read's pages in memory (`again`, mean of the three cue kinds), at N over at N / 10, for each tenfold from 100,000 to 100,000,000, `code-128` | under 2 at each tenfold, on the timing founders |
| S3 | the read under the 8 GB limit at 10,000,000 and 100,000,000: found as without the limit; ms per read of a held key (`exact`, first pass); ms per read of any cue (`noisy`, `partial`, first pass) | the same found; under 25 ms; under 500 ms |
| S4 | 10,000 records written one by one at every size, with and without the limit: the cells a write takes; the pause of a write; the new records found from their keys; the earlier records found | at most 20 cells a write; 99 of 100 writes under 100 ms and none over 1,000 ms; .99; as before the writes |
| S5 | bytes a record on the disk at 1,000,000, 10,000,000 and 100,000,000 | within 5 % of each other |

Reported beside the gates, without a bar: 100,000 records (the fixed projection of 33.5 MB
weighs on few records); the entries a full cell did not take; the seconds to build and the µs
per record to fill and file; the first pass's ms and page faults without the limit; the store in
memory (`grow=True` without `path`, and the fixed table of the fourth declaration) at 100,000
and 1,000,000; multiply-accumulates of a read (the address's projection and the candidates'
scores: no learned parameters).

**The timing** is read by cells of their own, each alone on a box that runs nothing else.

## Founders

Development founders 0 to 5 fixed the constants; none of the following was read before this
declaration.

| Run | Plan | Founders | Sizes |
| --- | --- | --- | --- |
| found | `found.json` | 3200 to 3211 | 100,000, 1,000,000 and 10,000,000, both key forms |
| a hundred million | `hundred-million.json` | 3212 | 100,000,000, `code-128` |
| timing | `timing.json` | 3213 to 3215 | 100,000, 1,000,000 and 10,000,000, `code-128`; 3212's cell at 100,000,000 |

One on-demand i4i.4xlarge in us-east-2 (8 cores, 16 threads, 128 GB of memory, one NVMe
instance store of 3.75 TB under XFS). A store of 100,000,000 `code-128` records holds about
211 GB.

## Expectations, stated before the runs

EXPECTATIONS
