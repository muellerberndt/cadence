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
cadence.LongTermMemory(width, 64, capacity=134_217_727, pairs=16, expansion=65536,
                       winners=384, candidates=8192, cosine=True, fade=False, revise=0.0,
                       dtype="float32", grow=True, sure=0.999, path=folder)
```

- **`grow=True`**: the table of pairs starts at 64 cells and takes a cell at a time (linear
  hashing); the capacity, 2 to the 27th less one, is a bound.
- **`path=folder`**: every array is a mapped file.
- **`sure=0.999`**: a read of a held key ends at its record after 120 cells.
- **`expansion=65536`**: the address's units, set once for the bound. A record's place depends
  on them, so a store that is to reach 100,000,000 records is made with them from the start.
  The fourth declaration's stores, each made for one size, used 16,384 at 1,000,000.
- **`winners=384, candidates=8192`**: a read takes the pairs of the cue's 384 most driven units
  (73,536 cells) and scores the 8,192 records found under the most of them. With the address's
  units spread over 65,536, the fourth declaration's 256 winners name too few of a noisy or
  partial cue's pairs at 10,000,000 records (development, below).

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
2. **read**: the store is opened again. Every cue is read once (`first`: found, ms, page faults
   that waited for the disk), then the first 300 of each kind again (`again`: their pages are in
   memory). On the shared box the store's files are read once into memory before the step. Then 10,000 new records are written one by one, a read between every two writes:
   each write's pause, the cells the table took, whether the new records and the earlier ones
   are found. Every read step adds its 10,000 records to the store.
3. **read under a memory limit**: the same step after the page cache is emptied, in a control
   group whose memory, page cache with it, is held to 8 GB (`MemoryMax`, no swap).
4. **scan**: the same cues by the read that scores every record, by parts over the store's keys.

## Two boxes

- **An on-demand r7i.8xlarge** (32 threads, 256 GB, a gp3 volume) runs the found cells beside the
  sixth declaration's brains: build (8 threads), read (one thread; the store's pages are in
  memory from the build), scan (8 threads), then the store is removed. It builds the store of
  100,000,000 records (16 threads) and copies its files to the lane's bucket before any read.
- **An on-demand i4i.4xlarge** (16 threads, 128 GB, an NVMe instance store under XFS) runs
  nothing else: it fetches that store and reads it under the limit, then without it, then scans
  it; and it builds and reads the timing cells, each read without the limit and under it.

## Gates

| Gate | Reading | Bar |
| --- | --- | --- |
| S1 | found at 10,000,000 (12 founders) and at 100,000,000 (founder 3212), `code-128`: from an exact cue; from a noisy and from a partial cue; noisy and partial against the scan on the same cues | .99; .95; no more than .02 under the scan; on 10 of 12, and at 100,000,000 |
| S1h | found at 10,000,000, `hashed-280`: exact; noisy; noisy and partial against the scan | .99; .95; no more than .02 under the scan; on 10 of 12 |
| S2 | ms per read on one core with the read's pages in memory (`again`, mean of the three cue kinds), at N over N / 10: 1,000,000 over 100,000 and 10,000,000 over 1,000,000 on each timing founder; 100,000,000 (3212) over 10,000,000 (each timing founder) | under 2 at each tenfold, on every timing founder |
| S3 | under the 8 GB limit, 10,000,000 (timing founders) and 100,000,000 (3212), `code-128`, first pass: found against the same store's read without the limit; ms per read of an exact cue; ms per read of a noisy and of a partial cue | within .01; under 25 ms; under 1,000 ms; on every cell |
| S4a | 10,000 records written one by one, every cell of every run, with and without the limit: cells a write takes; new records found from their keys; earlier records found | at most 20; .99; .99; on every cell |
| S4b | the same on the cells read alone (timing founders and 3212), with and without the limit: the pause of a write | 99 of 100 under 100 ms and none over 1,000 ms, on every cell |
| S5 | bytes a record on the disk, `code-128`, at 1,000,000, 10,000,000 and 100,000,000 | within 5 % of their mean |

Reported beside the gates, without a bar: 100,000 records (the fixed projection of 33.5 MB
weighs on few records); the entries a full cell did not take; the seconds to build and the µs
per record to fill and file; the first pass's ms and page faults without the limit;
multiply-accumulates of a read (the address's projection and the candidates' scores: no learned
parameters). The found cells' times are read on a box shared with the sixth declaration's
brains and are reported, never gated.

## Founders

Development founders 0 and 1 fixed the constants; none of the following was read before this
declaration.

| Run | Plan | Founders | Sizes | Box |
| --- | --- | --- | --- | --- |
| found | `found.json` | 3200 to 3211 | 100,000, 1,000,000 and 10,000,000, both key forms | r7i |
| a hundred million | `hundred-million.json` | 3212 | 100,000,000, `code-128` | built on the r7i, read on the i4i |
| timing | `timing.json` | 3213 to 3215 | 100,000, 1,000,000 and 10,000,000, `code-128` | i4i |

A store of 100,000,000 `code-128` records holds about 220 GB.

## Expectations, stated before the runs

EXPECTATIONS
