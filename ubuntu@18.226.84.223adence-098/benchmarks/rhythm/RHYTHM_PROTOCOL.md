# Steady-rhythm protocol, frozen 2026-10-04

This protocol is the bounded continuing-action chamber of
[issue 116](https://github.com/muellerberndt/cadence/issues/116) (roadmap row 06 of
[issue 109](https://github.com/muellerberndt/cadence/issues/109)). The machine-readable
declaration is [`protocol.json`](protocol.json); every run copies it and records the
SHA-256 of its exact bytes. Any command-line override marks a run `frozen_protocol:
false`. The frozen inputs are built by [`rhythm_inputs.py`](rhythm_inputs.py) before
any model runs; the driver is [`steady_rhythm.py`](steady_rhythm.py).

## Question and declared limit

Can one continuing `Brain.compose` life, after a finite number of lessons, alternate
two actions A, B, A, B while every observation it receives is identical, with the
working trace as the only carrier of phase inside the brain? The qualification
concerns a declared trajectory window of 64 events after a probe, scored per event
and per block of 8 events. A nonconstant orbit is the target; a held fixed point
(the same action at every event) scores zero alternation by construction.

The chamber measures the current System 1. A failure to alternate is a measured
limit of that System 1 and is reported with the same prominence as a success.
No mechanism, regulator, default or gene is added to make the task pass.

## Event time and physical cadence

One event is one accepted observation and its one free greedy `act`. A lesson at
that event belongs to the same event. Solver sweeps, residual checks, refused
attempts and checkpoint IO are work, not events. The brain has no clock input:
no slot number, timestamp or phase value enters any observation.

The paced runs declare a physical cadence of one event per 100 ms, with variants at
50 ms and 200 ms, and two slot disturbances at slot 10 of 32: an extra event (two
events due in one slot) and a skipped slot. Per event the driver records the
lateness of the act's start against its due time, the solve time, and whether the
act finished after the next distinct due time (a missed deadline). The same 32
events run unpaced under free-step budgets 1024 (reference), 256, 64 and 16, under
residual tolerances 0.003 (reference), 0.01 and 0.0001, and paced at 100 ms with one
spinning process per logical CPU. The per-event action sequence of every variant is
compared with the unpaced reference. Agreement with the slot-indexed ideal
alternation is reported beside agreement with the event-indexed one, so a slot
disturbance shows as an external phase shift of exactly one slot when the per-event
actions are unchanged.

## Fixed brain and learning law

Two recipes run on every founder. `selected` is `Brain.compose(4, 2, modules=(32,),
lateral=None, episodic=False, working_memory_decay=0.1, working_memory_amplitude=3.0)`
with the composed finite teaching law (`eta=0.5`, `eta_bias=0.05`, `momentum=0.9`,
`beta=0.1`, `temperature=0.2`, `free_steps=1024`, `nudged_steps=12`,
`tolerance=0.003`, `normalize=0`, `qualified=False`, `damping=3`). `compose_default`
differs in one gene, `working_memory_decay=0.2`, the constructor default; it is the
control recipe. `lateral=None` resolves to the default -0.5 for two actions. There is
no associative store, no reward, no actor eligibility and no observer.

The recipe was chosen on development seeds 0 to 5 over decays 0.0/0.1/0.2,
amplitudes 1.0/3.0, rates default/0.05n/0.02n/0.005n (n marks `normalize=0.99`) and
12 or 24 teaching bouts; the tables are in `results/development-*.json` and every
failed cell is kept. At 24 bouts the mean window alternation over both arms and six
seeds is 0.73 for the selected cell and 0.61 for the compose default; the best cell,
rate 0.005n at decay 0.1, scores 0.75, within the per-seed spread of the composed
law, which is kept for its fewer changed settings. Amplitude 1.0 cells range from
0.10 to 0.66; every cell is bimodal across seeds, with some founders at 1.00 and some
holding one action. Confirmation seeds 101 to 105 are fresh.

## Observations, rows and phase

Observations have four coordinates: drive, distractor, cue A, cue B. A drive event
sets coordinate 0 to 1.0. A cue event adds one cue coordinate; it is the first event
of a bout, and two of the four rows receive each cue by a frozen permutation. A pause
event is all zeros. A distractor event sets coordinate 1 alone. After the cue every
observation of a bout is identical for every row.

Four rows are four continuing streams of one brain with shared parameters and
per-row neural state and trace. The declared phase state is the row's working trace
(`Trace.trace`, `Trace.last`, `Trace.cold`) over the 32 association coordinates,
updated once per accepted event by the source law `trace = decay * trace +
(1 - decay) * h`. The warm neural state is a second retained quantity; the erased
and reset controls separate the two. Actuator events are the greedy motor choice,
A = 0 or B = 1, one per row per event.

## Teaching

Teaching runs 24 bouts of 12 events (288 events). The label at a cue event is the
cued action. At every later event the label is the opposite of the row's own last
executed action. The label depends on the brain's own witnessed actions and on no
clock. Two arms are run on separate founders:

- `every`: `Learner.step(brain.stimulus(x), labels)` on the live stimulus (sensory
  drive plus trace) before the free act of every event.
- `mismatch`: the free act first, then `Learner.step` on the pre-act stimulus for
  the rows whose act differed from the label.

Lessons use the finite teaching law; `Learner.step` does not advance the trace,
and only the free act does. A refused lesson is charged and skipped.

A matched control is trained beside the brain by the same arm: a linear softmax
policy over [1, observation, one-hot of its own previous action] with step 0.5,
taught online with the same label rule against its own previous action. It sees the
constant drive, the cue and its own previous output. It is a control, never the
answer path.

## Window, probe and controls

After teaching, one further bout begins: a cue event, four lead drive events, then
the probe. The probe is a complete `Brain.save` after the fourth lead act and before
the next act. The anchor of each row is its last executed action at the probe; the
ideal alternation continues the anchor. The remaining 64 events are the window.
Every branch receives the same 64 observations:

| Branch | Fork at the probe |
| --- | --- |
| intact | the live life continues |
| restored | `Brain.load(probe)` runs the window; its actions and its saved arrays after the window must equal the intact life's |
| erased | trace, last and cold zeroed once; warm neural state kept |
| shuffled | trace, last and cold transplanted from the row with the other window cue |
| reset | `Brain.reset()` once; parameters kept |
| static | trace erased before every act |
| static_cold | `Brain.reset()` before every act |
| flipflop | the matched control from its own saved state |
| random | frozen uniform-random actions |

No branch installs state into the live life. The shuffled branch is also scored
against the donor row's anchor, and the rows whose donor anchor equals their own
are listed, because a transplant between equal phases is uninformative.

## Disturbances

Each disturbance forks the probe and runs 8 drive events, the disturbance, then 32
drive events: pauses of 1, 2 and 4 zero-drive events and one distractor event. The
brain acts at every event, including pause and distractor events, and those actions
are recorded. The post window is scored against two anchors: `hold` continues the
last action before the disturbance, `continue` continues the last action executed
during it. Recovery is the number of post events before uninterrupted alternation
holds through the end of the post window, 0 for an unbroken rhythm, none when it
does not hold. The pause of two events saves a checkpoint after its first pause
event; a loaded twin runs the remaining events beside the original and must give
equal actions and equal saved arrays.

## Scoring

Per row over a window of T events with actions a_t and ideal alternation i_t:

| Measure | Definition |
| --- | --- |
| alternation rate | pairs with a_t != a_(t-1) over pairs with two executed actions |
| repeats | pairs with a_t == a_(t-1); each repeat is one phase slip |
| agreement | mean over events of a_t == i_t, refusals wrong |
| period | mean interval between successive A actions; 2.0 for a perfect rhythm |
| blocks | alternation rate per block of 8 events |
| refusals | events whose act refused |
| slot agreement | agreement with the ideal alternation indexed by physical slot |

Means over rows and founders are reported with every planned founder in the
denominator. A refused act is a missed action: no action, no trace update, the event
is consumed by the clock, and the life continues. A random policy is the baseline
for every claim; its expected alternation rate is 0.5.

## Caps, custody and receipt

One run has a wall cap of 900 s; past the cap the completed founders are written and
the process exits 3. Protocol copy, frozen inputs, every checkpoint, the per-call
settlement and teaching reports (`reports.jsonl`), per-founder rows (`runs.jsonl`)
and the summary receipt are written into the output directory. The summary is a
`cadence.Receipt` binding the copied driver, input module and imported library
sources; `--verify` recomputes the digest, the source bytes and every artifact hash.
Checkpoint files are compared by their saved arrays, since compressed archives carry
timestamps. Nothing larger than the receipts is committed.
