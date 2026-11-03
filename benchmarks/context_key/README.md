# The context key: Will Glynn's Stone Age chains with credit

This chamber measures `memory_context_hold`
([#202](https://github.com/muellerberndt/cadence/issues/202)) against the episodic choice without
it, on three four-stage crafting chains ported from Will Glynn's
[cadence-teachers](https://github.com/WGlynn/cadence-teachers) (EXP-57 and EXP-66), with
credit. It establishes no default.

## The chains

| Chain | What the brain sees | Why it is hard |
| --- | --- | --- |
| `ambiguous` | the stages look alike, and a miss leaves the observation as it was | a retry is keyed in the context the miss moved on, so the records of the right action never meet the key they were written under |
| `observed` | every stage is observed | the founder key suffices; the hold must not cost it |
| `invisible` | the right action at one stage leaves the observation unchanged | the hold takes that right action out of the key as well |

Two signals teach: `natural` (one continuing stream through `live`, reward alone) and
`corrective` (the stage's right action taught where the brain missed). Arms: `founder`
(`memory_choice=True`), `unchanged` (with `memory_context_hold="unchanged"`), `cost` (misses
cost .05) and `punished` (the cost with `memory_context_hold="punished"`). A life of 120
episodes is read greedily (does it pass the chain exactly, and how many stages does it reach)
and sampled over 100 attempts; `first_try` holds the share of first decisions right at each
stage, after cadence-teachers' EXP-56.

**Reading.** Lives are paired by seed across arms; `--summary` pairs the rows of a results file
against `--against` (the founder by default): clean greedy passes per arm, the discordant pairs
and the exact two-sided McNemar test, and the sampled success with its standard error.

## Run

From the repository root:

```sh
PYTHONPATH=src python benchmarks/context_key/stone_chains.py --chain ambiguous \
    --arms founder unchanged --seeds 48 --out DIR/ambiguous.jsonl --workers 2
PYTHONPATH=src python benchmarks/context_key/stone_chains.py --chain invisible \
    --arms cost punished --seeds 24 --out DIR/invisible_cost.jsonl --workers 2
python benchmarks/context_key/stone_chains.py --summary DIR/ambiguous.jsonl
python -m pytest -q benchmarks/context_key
```

A life takes 5 to 30 seconds on one core.

## Readings on 0.93

Fresh seeds, declared before the run, 48 per signal (24 for the cost arms); the release repeats
a sample of these lives byte for byte.

| Chain | Arms | Clean greedy passes | Exact McNemar | Sampled success |
| --- | --- | --- | --- | --- |
| `ambiguous` | `unchanged` against `founder`, both signals | 35 of 96 against 6 (30 and 1 discordant) | p < .0001 | .77 against .50 |
| `observed` | `unchanged` against `founder` | 40 against 43 of 48 | p = .51 | .92 against .94 |
| `invisible` | `unchanged` against `founder` | 0 against 9 | | .32 against .44 |
| `ambiguous` | `punished` against `cost` | 15 against 13 of 48 | p = .75 | .46 against .46 |
| `invisible` | `punished` against `cost` | 20 against 13 of 48 | p = .039 | |

The hold passes the gate of #202 and keeps the fully observed chain; on the chain whose right
action leaves the observation unchanged it costs, which is where `"punished"` with a cost for
misses serves instead.
