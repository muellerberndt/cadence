# Named-fact selection

This bounded component instrument asks whether the existing composed System 1
can answer a later query using the relevant one of two earlier named facts.
It adds no mechanism or runtime default. It is an initial diagnostic, not an
attention implementation, a continuing-life result or sentence understanding.

A and B each have an independently chosen binary value. They appear in either
order, followed by identical neutral events. The query names A or B and supplies
no value, selected record, memory address or relevance mask. The neural arm uses
`Brain.compose(..., episodic=False)`: its learned graph and working trace, with
no episodic action vote. Teachers label only the final query. The existing
local learning rule changes the present interpretation under a held trace; it
does not add credit through the earlier operations that formed that trace.

The same learned brain persists through every trial. Its transient activity and
trace reset at each trial boundary. All sixteen combinations of facts, order and
query appear once per training round, with stream rows permuted independently.
Evaluation measures acquisition of these cases; it has no held-out compositional
generalization claim. Each free action checks the complete current neural
equations. Finite teaching retains its separate existing phase contract.

Three matched founders are assigned the same schedule:

- **trace:** learns with the facts absent at the query;
- **exposed:** learns with both witnessed facts additionally present at the
  query, independent of which fact is requested;
- **frozen:** receives the trace arm's observations with no parameter updates.

The exposed arm is an acquisition control, not a memory solution. Its overall,
unequal-fact and per-query accuracy must reach the declared 0.95 floor before a
failure of the trace arm can be attributed to retention or selection. Failed
control acquisition remains a reported failure; the instrument does not search
for settings until the gate passes. Uniform-random actions provide another
control, with their finite draw and analytical expected accuracy of 0.5 reported.

Evaluation saves one complete pre-query state containing all eight fact/order
histories. Both queries fork that same state. Separate branches erase the trace
while preserving the neural warm state. Evaluation operates on saved copies and
checks that the original acquired brain's complete checkpoint remains unchanged.
Counterfactual pairs change only the requested fact, only the other fact, only
the query, or presentation order. The receipt reports correctness beside changes
and invariances: a constant response can be invariant without knowing anything.
The unequal-fact subset is essential: always computing AND or OR scores 75% over
all cases while answering differing facts at only 50%.

From the repository root, with Cadence available on the Python path:

```sh
PYTHONPATH=src python benchmarks/relevance/selective_recall.py --out /tmp/relevance.json
PYTHONPATH=src python benchmarks/relevance/selective_recall.py --verify /tmp/relevance.json
```

The [declared protocol](protocol.json) uses two development founders, 128 rounds
and one neutral event, with a 60-second training deadline. The deadline is
checked between bounded trials and shared across all arms and founders; bounded
evaluation still runs. A capped run can have unequal exposure: compare training
completion and actual rounds before interpreting the arms as matched. These are not
fresh confirmation seeds. A smoke run can use `--rounds 1 --seeds 0`; overrides
are recorded and no longer count as the frozen protocol. `--neutral-events`
changes the common suffix, not an answer-bearing phase signal. Without `--out`,
the receipt is printed as JSON. Existing output files are never overwritten.

The receipt binds imported source bytes and protocol, records the Git revision,
resolved genes, actual training schedule, answers, refused work and solver/teacher
work. Verification recomputes scores and the acquisition gate from retained
answers as well as checking source and receipt digests. It verifies arithmetic
and custody, not independent behavioral reproduction. Counters cover solver and
teaching work; they do not separately meter encoding, trace updates or checkpoint
I/O and support no efficiency claim. Timing is host seconds; solver work is not
physical energy. There is no automatic promotion of a runtime
mechanism, default or general capability after this component check.
