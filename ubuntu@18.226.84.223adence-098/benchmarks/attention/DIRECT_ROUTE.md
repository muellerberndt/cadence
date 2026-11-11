# Direct-route learned-code attention

The direct-route candidate is the first single-head component in this tree to
clear acquisition and causal relevance together. It is deliberately the
simplest version tested after the larger query-and-route design: one live
64-coordinate route, one born-zero 64 by 64 map, and no separately settling
query or query anchor.

For the already learned sensory meaning `h`, route `R`, local map `W`, answer
read `A`, and a fixed gate `g`, the live answer query is

```text
q = (1-g) h + g 6.75 R W
```

and the whole memory contribution is one energy:

```text
E = g 36 E_route(R, h) + (1-g) 3 ||R||² / 2 + E_answer(A, q)
```

Both agreement terms use the witnessed bank's null-inclusive softmax mixture.
The route therefore settles toward the record selected by the meaning already
present at the sensory boundary. `W` immediately turns that settling payload
into the address used by the answer read. Answer disagreement pushes back
through `W.T`, so selection and interpretation can revise one another inside
the same equilibrium. Free/positive/negative phases teach only `W` through its
local `R`-by-answer-address contrast. At `W=0` that contrast is already nonzero;
the mechanism does not need a random query initializer.

Gate off is the exact old held-meaning reader. The actual Cadence assay checks
bit identity by dispatching that arm through the captured old memory and
solver. The active arm adds 64 live coordinates per stream, not the 128 of the
separate-query control. The fixed route strength keeps the first meaning read
stable enough for the decoded address while preserving content feedback.

## Retained result

The [development receipt](results/direct-route-development.json) records one
founder, 64 local `W` lessons, and independent allowed and held panels. Both
panels score 100% answer accuracy. Across the combined causal panels, the
learned arm scores 100% answers and top-1 selection and 98.44% material target
mass. `W=0` falls to 33.85% answers; erasing the relation falls to 49.74%;
erasing the relevant record falls to 0%; erasing the unrelated record remains
at 100%. Changing either child's payload preserves 100% answers and top-1
selection, and answers follow the changed relevant payload.

Relevant payload can move the live query by 10.48% because it is part of the
current coupled interpretation. Irrelevant payload moves it by at most 5.44%
and changes no answer or selected record. This candidate therefore does not use
the older control's requirement that even relevant evidence leave the live
query fixed—that would test a frozen selector rather than coupled relevance.

The 64 lessons used 12,640 free, 2,784 positive, and 2,816 negative sweeps.
Every phase qualified. The learned map ended with Frobenius norm 0.26386 and
spectral norm 0.17013, below the retained cap.

## Scope

This is positive **component evidence**, not a supported `Brain.compose`
option, recall default, multi-head result, continuing-life result, or Stone Age
advantage. The retained measurement is one development founder. Independent
founders, lifecycle integration, saved continuation, matched old-memory costs,
and larger continuing environments remain required before promotion. The
earlier failed coupled-reader and parallel-head receipts remain valid controls.

The regression suite checks every analytical force against finite differences,
gate-off identity, a nonzero centered learning signal at `W=0`, and all retained
behavioral gates:

```sh
PYTHONPATH=src python -m pytest -q benchmarks/attention/test_direct_route.py
```
