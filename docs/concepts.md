# Concepts

## Owners, overlaps, settlement

A patch net is not a function that maps inputs to outputs. It is a set of owners, each
holding a patch of state, joined by overlaps that carry messages. There is no global step:
every owner repairs its own patch from its own state, its inbox, its clamp, and its bias.
Iterating that repair from rest is *settlement*, and the state the net rests in is what a
readout sees.

The graded rule is

    v <- v + dt * ( -v + inbox + clamp + bias - strength * a )
    inbox = sum over inbound overlaps of  gain * count * sign * exp(log_gain[pre]) * s[pre]
    s(v)  = rectified sigmoid, exactly zero at rest

The drive is absolute per contact: a hub integrates every contact it receives, and the
one global parameter is the drive of one contact. That follows the leaky integrate-and-fire
convention used for the fly brain, and it is why gains are small numbers.

## Why rest emits nothing

A plain sigmoid emits a few percent at rest. Multiplied by thousands of contacts on a hub
that leak ignites the net with no input at all. The activation is therefore re-based so
that an owner at exactly rest publishes exactly zero, in the engine's own precision. Rest
is then a fixed point of the whole net, and "nothing in, nothing out" is a testable fact.

## Why adaptation

A graded rule with one time constant converges to a fixed point under a constant clamp:
a posture, never a gait. Adaptation adds one slow variable per owner that follows its own
activation and subtracts from its own drive. With mutual inhibition, which every real
wiring has in abundance, that is the half-center oscillator, and the net can carry rhythm.
It is still owner-local: an owner reads only its own adaptation. It is off by default, and
a lane that turns it on says how it chose the two numbers.

## Why sparsity gates the gain

Raise the gain enough and any wiring runs away: a large fraction of owners saturate and
every readout lights. In that regime a protocol passes for reasons that have nothing to do
with the wiring, and a shuffled control passes just as well. So a gain is admissible only
while the net stays sparse under the training stimuli. The cap is declared, the table of
gains tried is recorded, and the same rule is applied to the control.

## Why float64

Owners sit on knife edges. In the fly brain, one motor neuron under one taste settled to
1.0 in one run and to 0.01 in another with the same wiring and clamp, because float32
summation order differed. Receipts are made on the float64 CPU backend for that reason.
Accelerated backends are for looking, and they come with a conformance number.

## Why a shuffled control

A protocol scored on a wiring alone measures the protocol as much as the wiring. The
control keeps everything about the wiring that is not the wiring: every count, every sign,
every owner's out-degree, every named set; only who talks to whom is destroyed. What the
wiring passes and the control fails is what the wiring predicted.

## Why receipts

Numbers in a notebook rot. A receipt is canonical JSON with its own digest, the digests of
the code and data that produced it, and enough stored readings that every pass flag can be
recomputed by the verifier. Editing any source file the receipt names invalidates it, on
purpose: a result belongs to the code that made it.

## What Cadence does not claim

Settling a measured wiring and passing held-out facts is evidence that the wiring carries
those facts under this rule. It is not a claim about biology beyond the scored predicates,
and nothing in the library ascribes experience to anything.
