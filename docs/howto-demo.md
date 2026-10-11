# Build a demo that accepts a mechanism

A mechanism is accepted by a demo, under one rule: the page carries the whole
brain, the world, the instrument strip and a plain-words account, and a
three-way switch (the mechanism on; off, the brain below it; the hand-designed
control) makes the acceptance visible. The person is in the world. The mechanism
is accepted when the demo works with the switch on and fails with it off, on the
numbers in its report, at matched information and compute. This page is the
order of work; the pieces are in [a brain that reads itself](steering.md) and
[the belief patch](belief.md).

This recipe describes the sequential composition demos. A task-scoped acceptance
does not establish observer and observed activity settling jointly, or a depth
advantage over matched conventional learners; see the
[numerical contracts](contracts.md).

## 1. State the claim

Write down, before anything runs, what the steering patch reads and returns,
what it unlocks, the experiments, the control and the falsifier. A claim of the
shape "a large surprise below seizes the steering patch's state within one
decision; repetition without consequence habituates; a consequential event keeps
capturing" has a hand-set threshold rule as its control, with its constants as
genes if it wins, and is falsified if that rule matches capture and habituation
at matched compute.

## 2. Build the world

A world is a body and senses whose readings are declared functions of its state,
a schedule of events, and a target the brain should predict.

- **Senses as blocks.** One block of the cortex's port per sense, so each sense
  has a gain of its own.
- **Identity linearly reachable.** A steering patch can only learn to treat an
  event differently if what distinguishes it is in the readback. "The source is
  at one of these corners" is not learnable from a position pair; an axis that
  only the event of interest occupies is. Put the distinguishing channel in the
  reading and read it through `evidence=`.
- **Noise where the weighing must rest.** With two senses equally clean the
  day's weighing rests on whichever is more predictive and there is nothing left
  to capture. Give the sense that should dominate in the quiet an order of
  magnitude less noise, so the other can still pull under its own conditions. A
  probe of the day alone at two noise levels on two seeds is a receipt worth
  keeping.
- **Persistence units.** The surprise is measured in units of each sense's
  persistence error, the mean squared change of its compared channels from one
  quiet moment to the next. Compute them on the training streams and declare
  them with `set_implied_reading(implied, units)`.
- **The events.** Each event is `(kind, start, length)`; the consequential kind
  changes the world. A kind that arrives late measures novelty.
- **Seeds.** Campaign seeds and one design seed for the page, named as such.

## 3. Build the four arms

All four are constructions of one class:

```text
the mechanism   Steered(cortex, steering, Softmax(2, span), evidence=[sense])  the steering patch hears all channels
the brain below Steered(cortex, steering, Softmax(2, span))                    its port's mask hides the evidence
fixed gains     Steered(cortex)                                                no steering patch at all
the control     Steered(cortex, weighing=Rule(threshold_rule, macs=8))         the hand-designed rule
```

The genome is a dict of the hand-set values (the steering patch's size, damping,
span, port scale, rate scale, retention, and a `reads_*` flag per channel
group), and `genes(space)` gives `evolve` its mutation over it. Keep the cortex
identical across arms, and count every arm with `macs_per_moment()` and `cost`.
Those estimate dense forward work: add backward, write, callback and search
costs, and measure elapsed time, for a training comparison.

## 4. The day

Raise every arm by day on whole streams: batches of streams from a fresh
boundary (`run(o, a, y, rate=cap, state=brain._fresh(n))`), one admitted joint
step per chunk, validation on held-out streams every few epochs, the best
snapshot kept. Long days matter — a day cut to a fifth of the epochs leaves the
weighing on the wrong sense and misleads every pilot. Save the dawn brains
(`snapshot()`); they are what the page loads.

Keep training, calibration/validation and final test streams disjoint. Any
governor baseline or threshold is fixed on calibration data before the online
night, and test targets never choose it.

## 5. The night

The night is one long stream lived online: short chunks, a cap near a tenth of
the day's, the boundary carried (`run(..., state=None)` continues from the live
boundary), the cortex asleep (`learn_cortex=False`) while the steering patch
learns, or gated by a governor (`Life`). The day's cap on the night's chunks
wrecks online tracking, and a steering rate scale above one pins a sense. A
night too short forgets the ordinary case without keeping the consequential one,
so measure the length the exception needs. Record the gains, the readbacks, the
errors and the events.

## 6. Measure

- `orienting(gain, events)` for capture, latency, return and the habituation
  curve per kind; `dishabituation` around the consequential events; the held-out
  error per condition.
- The sign of "capture" depends on the dawn baseline: where a sense rests open,
  habituation is a closing of it at known events. Report the gain at the
  consequential event against the gain at the habituated one, and the error at
  the consequence, beside the signed capture.
- Ablations on the trained brain at test time: `ablation = "cut"` and
  `deaf = mask`.
- The night in batched epochs with the cortex asleep is the clean unlock: the
  mechanism against the brain below on the consequential event alone.

## 7. Select

`evolve(fitness, hand_set, mutate=genes(space), grow=grow, generations=,
population=, keep=)` with the hand-set genome as the lineage's first member, and
a random search over the same space at the same number of evaluations as the
second control, on held-out seeds. The fitness reads only what the genome cannot
reweight, and prices both compute and surprise; a price on compute alone
switches learning off. If the control's constants are genes and the control wins
on the total error, say so — an evolved threshold rule that wins the ordinary
error and loses the exception is the demo's answer, not a failure to report.

## 8. The page

Cadence ships no renderer: a page draws the brain from
[`record_settlements`](api.md#record-every-settling-step) and whatever atlas the
application owns.

The page is a Python server (the brains, the world, `/state`, `/atlas`,
`/weights`, `/control`) and one HTML file: the world on the left with the
person's controls in it, the whole brain drawn on the right, the instrument
strip, the tiles, one line to switch the brain, everything else in a collapsed
section. The world runs from the design seed's dawn brains so the page starts at
once. Every number the page quotes for the selected brain comes from the frame
on screen, not from the server's latest stat, since the picture trails the
server by the frames still queued. A stat that is not a number is sent as null:
the browser's parser rejects `NaN` and the whole state with it.

## 9. The check

A headless check drives the page through the scenario of its promise
(Playwright, software GL): it waits for the first live frame, selects each arm,
drives the events, reads the instrument tiles, takes a screenshot per arm and at
phone width, counts console errors and horizontal overflow, and writes
`receipts/page_check.json`. Run it right after a server start, so the driven
events are the night's first; a check begun mid-night gives small captures and
must not replace the receipt. Run it before every deploy.

## 10. The receipts and the report

`receipts/*.json` in canonical JSON (`canonical_json`), one per campaign (the
arms, the night school, the rate sweep, the pilot, the evolution, the page
check), each carrying the protocol, the library's version and commit, and the
numbers. The report states the acceptance sentence — the brain below fails at
matched compute and the brain with the mechanism passes, or not — the numbers
with their receipts, the three-way switch, what did not work, the limits and the
regime, and what the library should carry next. A list of what the demo had to
write around the library is the library's signal: two demos asking for the same
thing is a feature request.
