# The index memory: learned addresses over the brain's own moments

Declared 16 October 2026, before the first declared run, for issues
[#199](https://github.com/muellerberndt/cadence/issues/199) and
[#201](https://github.com/muellerberndt/cadence/issues/201). Instrument: `index_read.py`
(the memory), `index_runs.py` (the lives); contracts: `test_index_read.py`. Companion to
[token binding](BINDING_PROTOCOL.md), [succession](SEQUENCE_PROTOCOL.md),
[chained recall](CHAIN_PROTOCOL.md) and [fresh bindings](../relevance/BINDINGS_PROTOCOL.md).

## Question

A store read by the present event generalizes to bindings never trained when the question
shares an event with the stored moment (token binding, succession), and fails where it shares
nothing (fresh bindings, finite recall). Learned reads tried so far either saturate (the
bilinear read) or complete on a stored moment nearly at random (a read competing inside the
settle), and lose part of what the fixed read has. Does a memory that addresses each moment before its
settle, by the event heard plus a learned map of the event and the cortex of the moment
before, trained by the address lesson, keep the stored-event read's binding and succession,
learn the addresses the present does not share, and hold in a continuing life and at camera
width?

## Mechanism

| Part | Definition |
| --- | --- |
| slot | one per accepted observation and stream; with `capacity`, the oldest dropped first |
| key `K_t` | the fixed part of the moment's address: `heard * x_t` (`x_t` the observation, or its sparse code at camera width) |
| address `q_t` | `K_t + x_t W_x + c_{t-1} W_c + b`, computed before the moment's settle; `c_{t-1}` the settled association activity of the moment before |
| read | `p = softmax_i(q_t . K_i / T - recency * age_i)`; recall drive `alpha sum_i p_i V_i`, next drive `alpha sum_i p_i N_i` on two source regions with one-way synapses into the association region (initial scale 1.2) |
| value, next | the event as witnessed; the next slot's value, zero until it comes |
| settle | the library's own; the recall and next synapses learn by its contrast like the sensory ones |
| address lesson | after a lesson, `d = W_{source->association} (c_+ - c_-)` per source; slot scores `(d_r . V_i + d_n . N_i) / |d|`; `p* = softmax(score / 0.1 + log p)`; the map takes `0.01 * (K* - q_t)` on `[x_t, c_{t-1}, 1]` at the taught rows, `K* = sum_i p*_i K_i` |
| sparse code (camera width) | 1024 code neurons, each reading 32 of the inputs with weights of variance 1/32; 24 winners kept; unit length |

## Arms

| Arm | Genes |
| --- | --- |
| `index` | alpha 3, T 0.25, heard 1, recency 0.25, next on, value the event, address 0.01, address temperature 0.1, address prior 1, context 0; capacity 0 (12 in the continuing task); the sparse code in the wide task only |
| `heard` | the same with address 0: the address stays the event heard |

## Tasks, readings and gates

Each life is read at every budget on forks; the reading asserts nothing changes the life.
Development founders 0 to 5; lessons judged by the act (`learning_surprise`).

| Task | Instrument | Budgets | Gate per founder at the last budget | Development passes at |
| --- | --- | --- | --- | --- |
| `token` | `token_binding`'s trials, panels and conditions | 256, 1024 | held-out at least .75; slots erased at most 1/8 + .15; a donor's slots followed on at least .75 | 5 of 6 |
| `fresh` | `relevance/bindings` (3 bits, m64, the trace arm) through this instrument's runner | 512, 2048 | the runner's gate: held-out exact at least .75 with both carriers erased at chance | 4 of 6 |
| `succession` | `sequence_binding`'s trials and panels | 256, 1024 | held-out at least .75; slots erased within .15 of 1/8 | 5 of 6 |
| `continuing` | token binding without a reset between trials; the panel read after two earlier trials of the same life | 256, 1024 | held-out at least .75; slots erased at most 1/8 + .15 | 5 of 6 |
| `wide` | token binding with each token a fixed patch of 400 of 12,824 inputs, noise pixels at .01 each frame | 256, 1024 | held-out at least .75; slots erased at most 1/8 + .15 | 5 of 6 |
| `chain` | `chain_binding`'s trials and panels | 1024, 4096 | two-held at least .60 and one-held at least .75 | 5 of 6 |

The `heard` arm runs on `token`, `fresh` and `succession` as the address lesson's control. The
comparisons (the fixed stores, the composed graph, the supervised GRU) are the retained rows of
those instruments' own readings, not rerun.

## Confirmation

Founders 400 to 411 stay unopened until the development readings are in and this protocol's
arms are frozen. A task whose development passes is confirmed on them with the same arm, budgets
and gate; it passes at 10 of 12 founders (`fresh`: 8 of 12).

## Sizing before this note (founders outside the declared set; not declared)

Sizing on founders outside the declared set fixed the arm's genes and three parts of the
mechanism, each from a failure of the form before it: the address is computed before the
settle (a read competing inside the settle from rest completed on a stored moment nearly at
random); the answer's address is named among the moments the address attended to (without that
prior the lesson named the moment whose own value is the answer); the keys hold the fixed part
of the address (keys written with the learned map made the lesson chase its own output in a
continuing life). The readings are retained outside this repository.

## Declared expectations

- `token`, `succession`: at the gate from 256 rounds on most founders; the slots erased at
  chance. The working trace carries some facts on its own, near the edge of the erased
  criterion.
- `fresh`: the `index` arm at the gate by 2048 rounds; the `heard` arm reads the mixture of both
  facts and stays below it on some founders.
- `continuing`, `wide`: at the gate if the address and the code carry the binding; a failure
  says which (the readings' erased conditions).
- `chain`: below the gate. The second link's address depends on the question marker and the
  recalled token together, a selection the linear address map makes only through conjunctions
  in the cortex of the moment before.

## What this does not claim

A bounded component on supplied event streams with lessons at the answer only. A pass
establishes that a memory of the brain's own moments, addressed before each settle by the event
heard plus a learned map trained by the address lesson, answers bindings and successions never
trained, learns an address the question does not share, holds in a continuing life of three
trials and at camera width, on the founders named. It establishes no chained inference, no
sentence understanding, no runtime default and nothing about C64 transcription or spoken words,
which wait on their own instruments.

## Commands

From the repository root, with Cadence on the Python path, one BLAS thread and at most two
workers; `DIR` lies outside the repository.

```sh
for task in token fresh succession continuing wide chain; do
  PYTHONPATH=src python benchmarks/binding/index_runs.py --task $task --out DIR/$task \
    --arms index --seeds 0 1 2 3 4 5 --workers 2
done
for task in token fresh succession; do
  PYTHONPATH=src python benchmarks/binding/index_runs.py --task $task --out DIR/$task-heard \
    --arms heard --seeds 0 1 2 3 4 5 --workers 2
done
python -m pytest -q benchmarks/binding
```

## Amendment 1, after the development readings (16 October 2026)

The development readings are retained as measured, outside this repository. The development
selected `token`, `succession`, `continuing` and `fresh`; their confirmation on founders 400
to 411 is declared, with the arm, budgets and gate above. `chain` and `wide` are kept as
declared. The camera task, declared before its first run:

| Item | `camera` |
| --- | --- |
| world | 12,824 inputs: a 128 x 100 image plus 24 body inputs; each token a contiguous 20 x 20 block on its own cell of the image; an event lights its tokens' blocks; .01 of the inputs lit at random each frame |
| retina | a fixed first stage between the pixels and the brain: 744 local pooling units (8 x 8 windows at stride 4, the drive the window's mean), the units at or above .5 kept (at most 128), the code scaled to unit length, the 24 body inputs passed through: 768 units; the cortex's observation and the memory's key are this code |
| brain | the composition of the token task on 768 inputs: 49,152 sensory synapses |
| genes | the `index` arm's with `heard` 1.5 (unit-length codes: a query shares most of its code with its fact's and little with the other fact's); `code` 0 |
| arms | `index` and `heard` |
| budgets, panels, gate | the `wide` task's: 256 and 1024 rounds; train, held-out, three facts, new name; held-out at least .75 with the slots erased at most 1/8 + .15; development 5 of 6, confirmation on founders 400 to 411 at 10 of 12 |

The retina is a construction of the task, as a seeing brain's first visual stage is of its
anatomy: the claim is that a memory keyed on a compressed code of the scene, read by a cortex
on the same code, binds what the scene shows; nothing here learns the code.

```sh
PYTHONPATH=src python benchmarks/binding/index_runs.py --task camera --out DIR/camera \
  --arms index heard --seeds 0 1 2 3 4 5 --workers 2
```

## Amendment 2, the generic code in place of the retina (16 October 2026)

The camera task's retina is a stage of the task's world. The library carries a generic code
only: a fixed sparse code of the whole observation with no spatial assumption (1,024 code
neurons, each reading 32 random inputs with fixed weights, the largest positive sums kept,
unit length). Declared before its first run:

| Item | `camera` with the generic code |
| --- | --- |
| observation | the generic code of the frame (1,024 units, fan-in 32, 96 winners) plus the 24 body inputs: 1,048 units, the cortex's observation and the memory's key alike (`--generic 96`) |
| everything else | amendment 1's: the `index` arm, budgets, panels, gate; development 5 of 6 on founders 0 to 5 |

The sizing selected 96 winners; the development on founders 0 to 5 selected the arm, and its
confirmation on founders 400 to 411 is declared with the same gate. The readings are retained
outside the repository.

```sh
PYTHONPATH=src python benchmarks/binding/index_runs.py --task camera --out DIR/camera-generic \
  --arms index --seeds 0 1 2 3 4 5 --generic 96 --workers 2
```

## Amendment 3, the input schedule, the address step and the attention record (17 October 2026)

Three changes to the instrument; no gene, panel, budget or gate changes.

| Item | Change |
| --- | --- |
| input schedule (`wide`, `camera`) | the training frames take their noise from the training stream's generator alone; each reading panel takes noise of its own, drawn afresh from the founder and the panel at every reading, so every reading and every condition (intact, slots erased, trace erased) sees the same frames, and a reading changes no later training frame (`test_index_runs.py`) |
| address step | each taught row's rate is `min(0.01, 1 / (z . z))` on `z = [x_t, c_{t-1}, 1]`, the rows' steps averaged as before; a step whose map or address is not finite is refused and counted (`address_refused`). The declared tasks' inputs (at most 27 values, or a unit-length code with at most 24 body inputs) and an association region of 64 neurons whose activations lie below one keep `0.01 * (z . z)` below 0.92, so no row of them reaches the bound and the step is the earlier one bit for bit |
| attention record | the memory keeps the attention each read used (`attended`), recency included, recorded before the moment's write; `fresh_competition` reports it in place of a weight recomputed without the recency charge. The scores and the gates do not read it |

The readings taken before this amendment are kept as measured. In them the reading panels of
`wide` and `camera` drew their noise from the training stream's generator, so their conditions
saw different noise and the training after a reading followed that reading's draws; and the
fresh-bindings `p_asked` and `decided` omit the recency charge. `wide` and `camera` are read
again under this schedule before their gates are cited.

## Amendment 4, the sparse code (17 October 2026)

The instrument's sparse code is the library's covering code; no gene, panel, budget or gate
changes. It replaces the sparse code of the mechanism table and amendment 2's generic code
(each code neuron reading random inputs with signed weights) for every reading taken after
this amendment.

| Item | Change |
| --- | --- |
| construction | `PatternSeparator(..., cover=2)`, as `Brain.compose(code=...)` builds it: each input is read through two rectified channels, ON `max(x, 0)` and OFF `max(-x, 0)`; each channel is dealt two distinct code neurons first, each code neuron's remaining fan-in is then filled with distinct channels at random, and the weights are positive, uniform in `(0, 1]` over the square root of the fan-in; the `code_winners` largest sums stay (24 in `wide`, 96 in the camera task's generic code) and the code is scaled to unit length |
| cover | 2: every channel is read by two code neurons at least |
| fan-in rule | `max(fan_in, ceil(2 * 2 * inputs / units))` channels per code neuron, at most `2 * inputs`: 51 in `wide` (12,824 inputs, 1,024 units, fan-in 32) and 50 in the camera task's generic code (12,800 pixels, 1,024 units, fan-in 32) |
| seed | `SeedSequence([founder, 7103]).generate_state(1)[0]`, a stream of its own, as the library draws the brain's code; the founder's own seed is the stream `develop` builds the brain from and the brain acts with |
| readings | `wide` and the camera task with the generic code (`--generic 96`) are read again on this code before their gates are cited; the camera task's retina does not read the code |

The readings taken before this amendment are kept as measured.

## Amendment 5, four options of the memory (17 October 2026)

Declared before the first run of any of them. The memory gains four options, each a gene whose
founder value is the `index` arm's, so every reading declared above is unchanged. The library's
held-input memory carries the same options as settings.

| Option | Gene (founder) | Definition |
| --- | --- | --- |
| address step | `address_step` (`"lms"`) | `"toward"`: the lesson's error is `K* - sum_i p_i K_i`, the named keys less the attended ones, in place of `K* - q_t`. To first order a step `s (K* - K_bar)` raises the named slots' logits against the attended ones by `s |K* - K_bar|^2 / T`; the error `K* - q_t` gives no such guarantee once the address lies away from the keys |
| recency | `recency` (0.25) | `None`: each slot's logit is lowered by `r log(1 + age_i)`, `r = heard^2 / T / log(1 + capacity)`: a full match of a unit-length key outweighs the age of every slot the capacity holds, and of two equal matches the younger keeps the larger weight by the factor `(1 + a_old)^r / (1 + a_young)^r` |
| writes | `write` (`"always"`) | `"aroused"`: a moment is written only for the rows whose act was sampled, or that the caller marks aroused |
| null slot | `null` (none) | a number: a slot with logit `null / T` joins the read and reads nothing; the slots' weights sum to one minus its share |

Under reward the library's memory also offers `address_reward` (0, off): each sampled act's
address step, the step a lesson naming that act would take, joins an eligibility trace, and the
temporal-difference error steps the map along it. This instrument teaches by lessons and does
not read it.

### Arms

| Arm | Genes |
| --- | --- |
| `index` | as declared: the control, read again on this instrument's code |
| `toward` | `index` with `address_step="toward"` |
| `horizon` | `index` with `recency=None` and capacity 64 (12 in the continuing task) |
| `null` | `index` with `null` at half the logit of a full match of a unit-length key: `heard^2 / 2`, 0.5 (1.125 in the camera task) |
| `improved` | `index` with `address_step="toward"`, `recency=None`, capacity 64 (12 in the continuing task) and `null` at `heard^2 / 2` |

### Component probes

`memory_probes.py` on founders 0 to 5, untrained, capacity 64, one-hot events of unit length:
what the next moment attends to, before it is written.

| Probe | Arms | Gate, on every founder |
| --- | --- | --- |
| age | `index`, `horizon`, `improved` | `horizon`, `improved`: the target holds the largest of the slots' weights at every age to 63 of 64; `index` reported |
| repeat | `index` with `write="aroused"` (the fact aroused, the questions calm), `index` | aroused writes: the fact holds at least .9 at every ask (1 to 8); `index` reported |
| no match | `index`, `null`, `improved` | `null`, `improved`: the null share at least .5 at the unrelated event and at the all-zero event; every arm: no slot above .5 |
| successor | every arm | known successor: cosine at least .8; unknown: norm at most a quarter of the known one's |
| density | every arm | no refused settle or lesson; every parameter finite |

### Binding chambers

Development founders 0 to 5; the tasks, budgets, gates and development pass counts declared
above. `named` is the fresh-bindings task with both facts' bits on shared coordinates
(`named_bindings.py`: the question shares nothing with either fact, and an even mixture of the
two facts cannot answer a bit where they differ), read through the fresh runner at 512, 2048
and 4096 rounds with the fresh gate, development 4 of 6.

| Arm | Tasks |
| --- | --- |
| `index` | token, succession, fresh, named, continuing, chain |
| `toward` | token, succession, fresh, named, continuing, chain |
| `horizon` | token, succession, continuing |
| `null` | token, fresh, named |
| `improved` | token, succession, fresh, named, continuing, chain, camera with the generic code |

### Selection and confirmation

An option is selected as a library founder when, on the development founders: (i) every task
the `index` arm passes at its development count, the option's arm passes too; (ii) on no task
does its mean held-out score over the founders at the last budget fall more than .05 below the
`index` arm's (token: also its `long_delay` and `restated` panels); (iii) its probe gate holds
on every founder. `chain` is read and reported; its gate decides nothing here. The selected
options are confirmed together, as the `improved` arm less the options not selected, on founders
400 to 411 with the same tasks, budgets and gates (10 of 12; `fresh` and `named` 8 of 12) and
the same comparison with the `index` arm. A selected option becomes a founder only with the
witnesses of the worlds, which are declared with their own instruments.

### Declared expectations

- `toward`: within the founders' spread of the `index` arm on every task, at least as fast on
  `named`.
- `horizon`: the age probe passes (it follows from the charge for unit-length keys); `restated`
  at or above the `index` arm's (ages one and three differ by `r log 2`, .66 at capacity 64,
  against .5 under the linear charge); `continuing` at the gate.
- `null`: the chambers hold; a full match keeps all but about a tenth of its weight.
- `improved`: the chambers hold, `camera` at the gate.
- `chain`: below the gate for every arm, as declared for the `index` arm.

### Commands

```sh
for task in token succession fresh named continuing chain; do
  PYTHONPATH=src python benchmarks/binding/index_runs.py --task $task --out DIR/$task \
    --arms index toward improved --seeds 0 1 2 3 4 5 --workers 2
done
for task in token succession continuing; do
  PYTHONPATH=src python benchmarks/binding/index_runs.py --task $task --out DIR/$task-horizon \
    --arms horizon --seeds 0 1 2 3 4 5 --workers 2
done
for task in token fresh named; do
  PYTHONPATH=src python benchmarks/binding/index_runs.py --task $task --out DIR/$task-null \
    --arms null --seeds 0 1 2 3 4 5 --workers 2
done
PYTHONPATH=src python benchmarks/binding/index_runs.py --task camera --out DIR/camera \
  --arms improved --seeds 0 1 2 3 4 5 --generic 96 --workers 2
PYTHONPATH=src python benchmarks/binding/memory_probes.py --out DIR/probes.json \
  --seeds 0 1 2 3 4 5 --arms '{"index": {}, "horizon": {"recency": null}, "null": {"null": 0.5},
  "improved": {"address_step": "toward", "recency": null, "null": 0.5}}'
```

## Amendment 6, the lives' parameters for the scaling family (17 October 2026)

The lives of `index_runs.py` and the vocabularies of `token_binding.py` and
`sequence_binding.py` take the parameters of [the scaling family](SCALING_PROTOCOL.md) as
arguments whose defaults are the declared values; no gene, panel, budget or gate of this
protocol changes. Every declared task reads the same at two budgets on one founder before and
after the change. The readings gain processor seconds (`cpu`, and `read_cpu` for the reading
alone) beside the wall seconds, and the fresh tasks' training record names why the life ended
(`ended`) and the brain's association width (`width`).

## Amendment 7, the context-window key (17 October 2026)

Declared before its first run. The memory gains a key option: `window` (founder 1) and
`window_decay` (founder 1.0). With `window` k the key's fixed part is the ordered window of the
stream's last k events, `[heard x_t, d heard x_{t-1}, ..., d^(k-1) heard x_{t-k+1}]`, each
position on coordinates of its own (zeros before a stream's first moment and after an episode's
end); the learned map still reads the present event. Where the same event recurs with different
continuations, the window separates its occurrences by their recent history.

| Arm | Genes |
| --- | --- |
| `index` | as declared, the control |
| `window2` | `index` with `window` 2 |
| `window4` | `index` with `window` 4 |

Development founders 0 to 5; the tasks, budgets and gates declared above: `succession` and
`chain` as the readings the window addresses, `token`, `fresh`, `named` and `continuing` for
preservation, and `camera` (the generic code) for `window2`. The selection rule of amendment 5
applies to each window arm against the `index` arm on the same founders; `chain` is read and
reported. Declared expectation: both window arms keep every gate the `index` arm passes; `chain`
stays below the gate. The window's main readings are outside this instrument (a music row loop and
a word stream) and are declared with their own instruments.

## Note, 21 October 2026: options the library removed

The library removed settings of its recall memory whose readings were negative or never
confirmed; release 0.84.3 of `cadence-net` is where they live on. This instrument keeps every
arm and probe above, with their receipts, as the record of those readings:

- `write="aroused"` (the `repeat` probe's arm; the library's `recall_write="aroused"`):
  neutral on every binding chamber.
- `recency=None` (the `horizon` and `improved` arms; the library's `recall_recency=None`):
  the power-law charge failed its confirmation on the camera task's founders.
- `value="cortex"` (the library's `recall_value="cortex"`): cortex values failed all four
  binding tasks (token .06, succession .07, fresh bindings .55, camera .00).
- `context`, `context_decay` (the library's `recall_context`, `recall_context_decay`): never
  read above zero in a chamber.
