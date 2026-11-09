# What a moment costs on a device

Two instruments for [#274](https://github.com/muellerberndt/cadence/issues/274), the whole lifecycle on the
device:

- [`ledger.py`](ledger.py), the ledger of a moment: every crossing between the host and the device that a
  `live`, `step`, `lesson` or `act` makes, by the place in the library that made it, with its bytes and time,
  beside the seconds of the moment.
- [`lives.py`](lives.py), seconds per life: many continuing lives of one topology, stepped in turn in one
  process, in a process pool on the CPU, as the streams of one brain in one call per moment, or as
  `cadence.Lives`, brains of their own in one call per moment.

```sh
PYTHONPATH=src python benchmarks/device/ledger.py --backend mlx --brains chamber speech --rows 1 16
PYTHONPATH=src python benchmarks/device/ledger.py --backend torch:cuda --rows 1 16 256 --out ledger.jsonl
PYTHONPATH=src python benchmarks/device/ledger.py --backend torch:cuda --depth 4 --brains chamber --rows 16
PYTHONPATH=src python benchmarks/device/ledger.py --table ledger.jsonl
PYTHONPATH=src python benchmarks/device/lives.py --ways serial pool streams --backend torch:cuda --lives 12 128
PYTHONPATH=src python benchmarks/device/lives.py --composition lives --teach 4 --ways lives streams pool \
    --backend torch:cuda --lives 12 128 1024
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

## On an NVIDIA A10G

One AWS g5.xlarge: an A10G and four vCPUs of an AMD EPYC 7R32, torch 2.12.1, float32 on the device, one thread
per process. Before is the library before the outcome's one read (main at a513b74), after is with it, and 0.94.0
is the release, whose default brain composes fewer memories, so its moments are another brain's.

Host reads per moment (the crossings that bring numbers to the host; each is a synchronization) and the
milliseconds spent in all crossings:

| Brain | Moment | Rows | 0.94.0 | Before | After |
| --- | --- | ---: | ---: | ---: | ---: |
| Chamber | `live` | 1 | 37, 3.5 ms | 28, 0.82 ms | 24, 0.76 ms |
| Chamber | `step` | 1 | 37, 3.6 ms | 27, 0.77 ms | 23, 0.73 ms |
| Chamber | `step` | 16 | 37, 3.8 ms | 27, 1.0 ms | 23, 0.95 ms |
| Chamber | `step` | 256 | 37, 5.3 ms | 31, 1.0 ms | 25, 0.87 ms |
| Chamber | `lesson` | 256 | 35, 1.4 ms | 30, 1.0 ms | 26, 0.94 ms |
| Speech | `live` | 1 | 37, 4.6 ms | 38, 3.6 ms | 32, 2.0 ms |
| Speech | `step` | 1 | 37, 4.5 ms | 36, 3.5 ms | 30, 2.0 ms |
| Speech | `step` | 16 | 37, 22.7 ms | 36, 15.1 ms | 30, 13.7 ms |
| Speech | `step` | 256 | not read | 36, 82.5 ms | 30, 80.8 ms |
| Speech | `lesson` | 256 | not read | 36, 14.6 ms | 32, 14.1 ms |

0.94.0's speech-size readings end at 16 rows, where its counting pass ran out of the device's memory. At 256
speech rows the time in crossings is the device's own work: the outcome's read waits for the reward's step
over 256 streams of 6.3 million eligibilities each. A crossing's bytes stay below 25 MB a moment, mostly the
observations.

Milliseconds per moment, before and after in turn for three rounds (the median of 30 moments of the chamber's
brain and 20 of the speech size each round, after 5 and 4 to warm), and one core of the same machine:

| Brain | Moment | Rows | One core | Before, three rounds | After, three rounds | Median, before to after |
| --- | --- | ---: | ---: | --- | --- | --- |
| Chamber | `live` | 1 | 3.01 | 8.43, 7.64, 7.55 | 7.70, 7.87, 7.93 | 7.64 to 7.87 |
| Chamber | `step` | 1 | 2.90 | 8.29, 7.43, 7.67 | 8.07, 7.51, 8.33 | 7.67 to 8.07 |
| Chamber | `step` | 16 | 6.64 | 9.94, 9.21, 9.51 | 9.43, 9.40, 9.33 | 9.51 to 9.40 |
| Chamber | `step` | 256 | 80.0 | 24.66, 23.52, 23.32 | 23.14, 25.04, 23.86 | 23.52 to 23.86 |
| Chamber | `lesson` | 16 | 5.54 | 7.45, 7.22, 7.44 | 6.93, 7.23, 7.04 | 7.44 to 7.04 |
| Chamber | `lesson` | 256 | 70.0 | 12.05, 11.32, 11.35 | 11.70, 11.38, 11.72 | 11.35 to 11.70 |
| Speech | `live` | 1 | 262 | 21.65, 20.99, 21.11 | 20.33, 20.44, 20.56 | 21.11 to 20.44 |
| Speech | `step` | 1 | 266 | 21.93, 20.80, 20.92 | 19.76, 20.51, 19.55 | 20.92 to 19.76 |
| Speech | `step` | 16 | 982 | 34.96, 35.47, 34.99 | 34.47, 35.90, 33.95 | 34.99 to 34.47 |
| Speech | `lesson` | 1 | 302 | 15.42, 14.84, 14.82 | 14.85, 14.23, 14.39 | 14.84 to 14.39 |
| Speech | `lesson` | 16 | 387 | 16.56, 17.06, 16.23 | 16.00, 16.09, 15.85 | 16.56 to 16.00 |

The speech size, whose critic and memories rest on the device, reads 1.5 to 5.5% faster; the chamber's brain, whose
memories stay on the host below their size, reads within the rounds' spread.

Whole lives under `backend="auto"` ([`life.py`](life.py)), seconds from before `Brain.compose` to the last moment,
before and after, and with a transport plan calibrated at the life's rows before its second moment:

| Life | `auto` | `auto` with a plan | One core |
| --- | --- | --- | ---: |
| Chamber, 256 streams, 300 moments of `step` | 9.36 to 9.09 | 9.48 to 9.40 | 23.5 |
| Speech, 1 stream, 100 moments | 8.99 to 8.71 | 9.05 to 8.86 | 9.40 for 20 moments |
| Speech, 16 streams, 50 moments | 8.57 to 8.40 | | |

A speech-size life on the device spends 5.0 seconds composing (2.9 on the CPU) and 1.7 in its first moment before
its moments run at 20 milliseconds; a chamber life of 256 streams spends 2.1 seconds in its first moment, where the
brain moves to the device.

## Lives

Seconds per life and moment of `lives.py` on the same machine, the chamber's default brain with its store
(`--composition default`), 20 moments after 3 to warm:

| Way | 12 lives | 128 lives |
| --- | ---: | ---: |
| `streams`, one brain on the A10G, one stream per life | 0.77 ms | 0.156 ms |
| `pool`, four CPU processes | 1.25 ms | 1.28 ms |
| `serial`, one core | 2.91 ms | 2.92 ms |
| `serial`, each brain on the A10G in turn | 7.75 ms | 7.59 ms |

## Lives of their own, on an NVIDIA L4

`--composition lives` (`episodic=False`: the default composition without the store, which `cadence.Lives`
holds), one AWS g6.xlarge (an NVIDIA L4 and four vCPUs of an AMD EPYC 7R13), 24 moments after 4 to warm, 12
after 4 at 512 and 1,024 lives. Every moment a `step` with the reward of the moment before:

| Way | 12 lives | 128 lives | 512 lives | 1,024 lives |
| --- | ---: | ---: | ---: | ---: |
| `lives`, `cadence.Lives` on the L4 | 1.18 ms | 0.118 ms | 0.043 ms | 0.049 ms |
| `streams`, one brain on the L4 | 0.619 ms | 0.119 ms | 0.045 ms | 0.034 ms |
| `pool`, four CPU processes | 0.874 ms | 0.876 ms |  |  |
| `serial`, one core | 2.35 ms | 2.37 ms |  |  |
| `serial`, each brain on the L4 in turn | 6.52 ms | 6.66 ms |  |  |

With a teacher's label at every fourth moment (`--teach 4`):

| Way | 12 lives | 128 lives | 512 lives | 1,024 lives |
| --- | ---: | ---: | ---: | ---: |
| `lives`, `cadence.Lives` on the L4 | 1.45 ms | 0.147 ms | 0.056 ms | 0.067 ms |
| `streams`, one brain on the L4 | 0.789 ms | 0.153 ms | 0.056 ms | 0.040 ms |
| `pool`, four CPU processes | 1.12 ms | 1.13 ms |  |  |
| `serial`, one core | 2.97 ms | 3.03 ms |  |  |
| `serial`, each brain on the L4 in turn | 7.86 ms | 7.98 ms |  |  |

`--composition plain` (`recall=False` as well) on the same card, `lives` and `streams`: 0.584 ms and 0.451 ms at
12 lives, 0.060 ms and 0.066 ms at 128, 0.021 ms and 0.027 ms at 1,024.

On an Apple M4 at a load of 4, `--composition lives`, at 12 and 128 lives, without and with the teacher:

| Way | 12 lives | 128 lives | 12, taught | 128, taught |
| --- | ---: | ---: | ---: | ---: |
| `streams`, one brain on MLX | 0.690 ms | 0.156 ms | 0.572 ms | 0.128 ms |
| `pool`, two CPU processes | 0.595 ms | 0.662 ms | 0.808 ms | 0.727 ms |
| `serial`, one core | 1.42 ms | 0.979 ms | 1.56 ms | 1.48 ms |
| `lives`, `cadence.Lives` on the torch CPU device | 1.17 ms | 1.21 ms | 1.48 ms | 1.32 ms |

The earlier reading of the plain composition on an L4 (`recall=False, episodic=False`, 20 moments after 3 to
warm) had `lives` at 0.638, 0.063, 0.024 and 0.021 ms and `streams` at 0.468, 0.069, 0.033 and 0.027 ms at 12,
128, 512 and 1,024 lives, a pool of four processes at 0.414 and 0.411 ms and one core at 1.12 ms.

## On an Apple M4

The same ledger on MLX, on an M4 that other work shared at a load of 7 to 10: a speech-size `step` or `live`
waits for the device 9 times a moment after where it waited 11 before (the critic's and the recall memory's
checks join the update's launch), and the chamber's brain, whose memories stay on the host, waits 5 times
before and after. The moments' times there move with the machine's load by more than the change.
