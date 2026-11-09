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
