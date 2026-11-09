# Routes with a goal: declared protocol (cadence#257)

Declared on 2026-11-09, committed to the lane branch before the first life. The founders, speakers,
readings and gate below are fixed; a failed gate is reported as failed.

## The question

On the grammar chamber every sentence is a function of the meaning alone, so a word-by-word speaker
that reads well has all it needs, and no row showed planning paying. In the goal-route chamber
(`benchmarks/grammar/routes.py`) the first hop is said before the goal it must lead to: the start
has two neighbours and only one branch reaches the goal. Does a sliding planning window of three
(plan three words, say the first, hear it, plan again) say the first hop right more often than word
by word, on start-goal pairs no training episode joined?

## Speakers

- `word`: `Brain.compose(22, 10, modules=(64,), seed=founder, backend="cpu")`, the default
  composition, nothing else named.
- `sliding`: the same with `slots=3` and `3 * 10` actions; each lesson teaches the next three
  words.
- `lstm`, `transformer`: matched to `word`'s `parameters()`, online (one pass, the brain's order)
  and offline (to convergence), reported beside the brains with no gate.

## Life

3,000 batches of sixteen episodes, two and three hops in turn, from the founder's training stream;
the panels `held` (the founder's eight held-out start-goal pairs) and `trained` (fresh graphs on
trained pairs), 256 episodes each, read at the end, greedy and closed loop.

## Founders

Development (instrument checks only, no reading decides anything): founders 0 and 1, both speakers,
read before the declared run to check that the chamber is learnable within the life. Declared:
founders 9625700 to 9625705, which no lane has used.

## Gate

| Gate | Reading | Pass (planning pays) |
| --- | --- | --- |
| P1 | first-hop accuracy on `held` at 3,000 batches, `sliding` minus `word`, paired by founder | mean difference at least .05 and above two standard errors of the paired differences |
| P2 | the same on `trained` | reported, no bar |

Exact routes, valid routes and first-hop accuracy per hop count are reported for every speaker.

**Verdict.** P1 passing closes the narrowed acceptance with planning paying on this chamber, and the
sliding form is documented as the planning window's use. P1 failing is a measured negative on this
chamber, recorded on the issue.

## Compute

One AWS c7i.16xlarge, one process per life, one BLAS thread per process.

## Amendment before the declared run

The first development lives (founders 0 and 1, both speakers, the route said as `start a1 ... goal
end`) read first-hop accuracy .17 to .27 on both panels, below the .5 of a speaker that only picks a
neighbour of the start, so the chamber was not learnable within the life in that form. Each fact is
heard as `X near Y end`, so the place after `X` is two moments on. The route is said as the facts and
the grammar chamber's paths say it, `start near a1 near ... near goal end`, and `first_hop` is the
place after the first `near`. Nothing else changes; the development lives are read again before the
declared run.

## Second amendment before the declared run

Read again with `near`, the development lives at 3,000 batches (held-out pairs, founders 0 and 1):
word by word says a neighbour of the start on .74 and .80 of routes, the goal branch's on .32 and
.45, so it chooses between the two neighbours near chance, as the chamber intends; the sliding window
says a neighbour on .37 and .47 and is still rising (first hop .05, .14, .18 and .09, .19, .24 at
1,000, 2,000 and 3,000 batches). The matched transformer online reads .32 and .54. So that the window
is read once it has learned the routes' form, the declared life is 8,000 batches, read at 2,000,
4,000, 6,000 and 8,000, and P1 and P2 read the reading at 8,000. The controls' online pass takes the
same 8,000 batches. Development founders 0 and 1 live the declared life beside the declared founders
and are reported apart.

## Result

Read on one c7i.16xlarge, founders 9625700 to 9625705, 8,000 batches, 256 episodes a panel.

| Gate | Verdict | Reading |
| --- | --- | --- |
| P1, held-out pairs | fail | sliding minus word by word -.022 (s.e. .050): -.047 .090 -.105 -.184 .145 -.027 |
| P2, trained pairs | reported | -.076 (s.e. .028) |

| Speaker | First hop, held-out | The other neighbour | Exact routes | First hop, trained |
| --- | --- | --- | --- | --- |
| Word by word | .334 | .356 | .240 | .372 |
| Sliding window of three | .312 | .319 | .203 | .296 |
| LSTM online | .395 | .366 | .277 | .419 |
| Transformer online | .540 | .374 | .518 | .548 |
| LSTM offline (30 epochs) | .830 | .106 | .712 | .963 |
| Transformer offline (14 to 30 epochs) | .754 | .246 | .753 | .755 |

A measured negative: planning does not pay on this chamber. Both speakers say the goal branch's first
place as often as the other neighbour's, so neither reads the goal at the first hop; the window adds no
lookahead to the joint settle. The controls show the information suffices (offline .75 to .83 on
held-out pairs).
