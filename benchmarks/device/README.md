# What a moment costs on a device

Two instruments for [#274](https://github.com/muellerberndt/cadence/issues/274), the whole lifecycle on the
device:

- [`ledger.py`](ledger.py), the ledger of a moment: every crossing between the host and the device that a
  `live`, `step`, `lesson` or `act` makes, by the place in the library that made it, with its bytes and time,
  beside the seconds of the moment.
- [`lives.py`](lives.py), seconds per life: many continuing lives of one topology, stepped in turn in one
  process, in a process pool on the CPU, or as the streams of one brain in one call per moment.

```sh
PYTHONPATH=src python benchmarks/device/ledger.py --backend mlx --brains chamber speech --rows 1 16
PYTHONPATH=src python benchmarks/device/ledger.py --backend torch:cuda --rows 1 16 256 --out ledger.jsonl
PYTHONPATH=src python benchmarks/device/ledger.py --backend torch:cuda --depth 4 --brains chamber --rows 16
PYTHONPATH=src python benchmarks/device/ledger.py --table ledger.jsonl
PYTHONPATH=src python benchmarks/device/lives.py --ways serial pool streams --backend torch:cuda --lives 12 128
```

The brains are the chamber's default brain, `Brain.compose(35, 18)` (251 neurons, 13,120 synapses), and the
speech size, `Brain.compose(3920, 37, modules=(512,))` (12,821 neurons, 6.3 million synapses). A reading of
`ledger.py` is two passes over moments of one brain: the first, untouched, gives the seconds of a moment, and
the second counts the crossings, each named by the innermost function of the library on the stack (`--depth`
names its callers too). A crossing's time includes the wait for the device's queued work, so a `wait` row
(`mx.eval`, `torch.cuda.synchronize`) and the first read after queued work hold the device's own time.

## Reading a ledger

A wait or a read of numbers is a synchronization: the host stops until the device has done everything queued
before it, and the overlap of host and device work ends there. On an Apple M4 one costs about 180
microseconds whatever the work ([which hardware](../../docs/backends.md#which-hardware)). The bytes that cross
are small: a moment of the chamber's brain moves 75 kB at one row and 330 kB at 16, a moment of the speech size
0.9 MB at one row and 14 MB at 16, mostly the observations. A moment pays for its count of synchronizations.
