"""Independent lives over one topology, stepped together (cadence#274, #167).

``Lives`` holds many composed brains of one anatomy, each with its own parameters, optimizer,
critic, eligibility, working trace and random state, one stream each. On the CPU each life is a
lone ``Brain``, stepped in turn. On a torch device the lives are the streams of one brain whose
parameters carry a lives axis: every settle reads each stream through its own life's synapses,
and every update moves each life on its own contrast and dopamine, so one call per moment settles
and teaches all of them. Each life follows the lone brain of its seed on the same device within
the solver's tolerance.

The lives run the composition's own equations through these parts:

- ``_LivesTorchKernel``: the settle's transport as one batched product per block of the layout,
  each life's dense block of its own synapses, and the contrast per stream;
- ``_LivesLearner``: a lesson's update and a reward's update per life, the momentum per life;
- ``_LivesActorCritic`` and ``_LivesCritic``: the critic's weights and bias per life, the
  dopamine and the actor's step per stream;
- ``_LivesRandom``: each life's own random generator, drawn as its lone brain draws it.

A composition holds the parts these cover: ``recall=False`` and ``episodic=False`` (the recall
memory's maps and the store's consolidated records are a brain's own and have no lives axis),
no arousal, maturity, learned code, apical compartment, rehearsal, carried gains, normalized or
surprise-judged teaching. ``Lives`` refuses any other.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from .brain import (
    _MLX_CHECK_FLOOR,
    BrainState,
    Equilibrium,
    _plan_structure,
    _TorchKernel,
    _transport_override,
    _TransportPlan,
)
from .brain import Brain as NeuralGraph
from .learning import Learner
from .plasticity import ActorCritic
from .resident import _CriticOnDevice

__all__ = ["Lives"]


class _RowBias:
    """A kernel's biases, one row per life, as its sweeps read them: ``bias[:, None]``, the drive
    that never moves, is ``(neurons, lives)`` there."""

    def __init__(self, rows: Any) -> None:
        self.rows = rows  # (lives, neurons)

    def __getitem__(self, key: Any) -> Any:
        return self.rows.T


class _LivesTorchKernel(_TorchKernel):
    """``_TorchKernel`` whose parameters carry a lives axis, one stream per life.

    ``scale`` is ``(lives, synapses)`` and ``bias_param`` ``(lives, neurons)``. A sweep's synaptic
    input is one batched product per block of the layout, each life's stream through its own
    dense block, and the contrast of a lesson or a reward is each stream's own. Every settle
    holds exactly one stream per life."""

    def __init__(
        self, connectome: Any, efficacy: Any, gain_pre: Any, bias: Any, neuron_model: Any,
        device: str | None, layout: Any, precision: str | None = None, *,
        reference: bool = False, parent: _TorchKernel | None = None,
    ) -> None:
        self.lives = int(efficacy.shape[0])
        self._dense: Any = None
        super().__init__(
            connectome, efficacy[0], gain_pre, bias[0], neuron_model, device, layout, precision,
            None, reference=reference, parent=parent,
        )
        torch = self.torch
        self._fused = None  # the neuron update operation by operation: it reads one bias per row
        self.plan = _TransportPlan(  # a plan without held products: the lives' blocks move each
            torch,
            _plan_structure(torch, connectome, None, self.device, layout, _transport_override(),
                            False),
            connectome.n, self.device, self.dtype,
        )
        self.flat = self.plan.values
        self.set_parameters(self._placed(efficacy), self._placed(bias))

    # -- the blocks

    def _blocks(self) -> list[tuple[int, int, int, int, Any]]:
        """Each block's bounds and its view, ``(lives, pre, post)``, of the dense buffer the
        parameters are written into in place (a captured sweep reads the same storage)."""
        if self._dense is None:
            torch, lay = self.torch, self.layout
            self._dense = torch.zeros(self.lives, lay.size, dtype=self.dtype, device=self.device)
            self._entry = torch.from_numpy(np.ascontiguousarray(lay.edge_index)).to(self.device)
            views = []
            for k in range(lay.pairs):
                a0, a1, b0, b1 = lay.bounds(k)
                view = self._dense[:, int(lay.offset[k]) : int(lay.offset[k + 1])]
                views.append((a0, a1, b0, b1, view.view(self.lives, a1 - a0, b1 - b0)))
            self._views = views
        return self._views

    def _synaptic_input(self, s: Any, out: Any) -> Any:
        """The transport of a neuron-major activation ``(neurons, lives)``: per block, every
        life's presynaptic row through its own block."""
        if s.shape[1] != self.lives:
            raise ValueError(f"a settle of these lives holds one stream per life ({self.lives})")
        torch = self.torch
        out.zero_()
        rows = s.T
        for a0, a1, b0, b1, block in self._blocks():
            product = torch.bmm(rows[:, a0:a1].unsqueeze(1), block).squeeze(1)
            out[b0:b1].add_(product.T)
        return out

    # -- each life stops where its own stream meets the tolerance

    def _configure(
        self, ws: Any, drive: Any, keep: Any, nudge: Any, tolerance: float | None, batch: int,
    ) -> Any:
        cfg = super()._configure(ws, drive, keep, nudge, tolerance, batch)
        held = getattr(ws, "lives_met", None)
        if held is None or held.shape[0] != batch:
            ws.lives_met = self.torch.zeros(batch, dtype=self.torch.bool, device=self.device)
        hold = getattr(self.root, "_hold", None)  # the lives a solve found settled already
        if hold is None:
            ws.lives_met.zero_()
        else:
            ws.lives_met.copy_(hold)
        return cfg

    def _update(
        self, ws: Any, cfg: Any, i: int, j: int, synaptic: Any, stage: int, standing: Any,
    ) -> None:
        """The sweep of every stream, then each life whose stream met the tolerance at an earlier
        sweep held where it met it, as its lone brain's settle stops there: a stream's own
        movement under the tolerance meets it, and the settle meets it when every life has."""
        super()._update(ws, cfg, i, j, synaptic, stage, standing)
        torch = self.torch
        met = ws.lives_met  # (lives,): met at an earlier sweep, or settled before the solve
        for slots in ((ws.v[i], ws.v[j]), (ws.s[i], ws.s[j]),
                      *(((ws.a[i], ws.a[j]),) if ws.a is not None else ())):
            torch.where(met[None, :], slots[0], slots[1], out=slots[1])
        torch.where(met, ws.change[i], ws.change[j], out=ws.change[j])
        if cfg.tolerance is None:
            return
        moved = ws.total.amax(dim=0)  # each stream's largest movement in this sweep
        met |= moved < cfg.tolerance
        torch.all(met, out=ws.flag[j])

    # -- parameters

    def _spread(self, values: Any, width: int) -> Any:
        """A parameter of one life given to every life, or the lives' own."""
        if values.ndim == 1:
            return values.unsqueeze(0).expand(self.lives, width).clone()
        return values

    def parameter_check(self, scale: Any, bias: Any) -> tuple[Any, ...]:
        torch = self.torch
        if tuple(scale.shape) != (self.lives, *self.gain_pre.shape) or tuple(bias.shape) != (
            self.lives, self.n,
        ):
            raise ValueError("lives' parameters are (lives, synapses) and (lives, neurons)")
        weights = (self.gain_pre * scale).to(self.dtype)
        live_bias = bias.to(self.dtype)
        finite = torch.stack(
            [torch.isfinite(value).all() for value in (scale, bias, weights, live_bias)]
        ).all()
        return weights, live_bias, None, finite, finite

    def set_parameters(
        self, scale: Any, bias: Any, *, checked: tuple[Any, ...] | None = None
    ) -> None:
        scale = self._spread(scale, int(self.gain_pre.shape[0]))
        bias = self._spread(bias, self.n)
        if checked is None:
            checked = self.parameter_check(scale, bias)
            flags = self.torch.stack([checked[3], checked[4]]).cpu().numpy()
            self.parameter_refusal(bool(flags[0]), bool(flags[1]))
        weights, live_bias = checked[:2]
        self.scale = scale
        self.bias_param = bias
        self._blocks()
        self._dense.zero_()
        self._dense.index_add_(1, self._entry, weights)
        self.bias = _RowBias(live_bias)
        self._version += 1

    def _float64(self) -> _TorchKernel:
        if self.dtype == self.torch.float64:
            return self
        ref = self._reference
        if ref is None:
            synapses = self._connectome.synapses
            ref = self._reference = _LivesTorchKernel(
                self._connectome, np.zeros((self.lives, synapses)), np.zeros(synapses),
                np.zeros((self.lives, self.n)), self.neuron_model, str(self.device),
                self.layout, "float64", reference=True,
            )
            ref._version = -1
        if ref._version != self._version:
            ref.set_parameters_from(self)
        return ref  # type: ignore[no-any-return]

    def stepped(self, neuron_model: Any) -> _TorchKernel:
        root = self.root
        if root is not self:
            return root.stepped(neuron_model)
        kernel = self._stepped.get(neuron_model)
        if kernel is None:
            kernel = _LivesTorchKernel(
                self._connectome, self.scale, self.gain_pre, self.bias_param, neuron_model,
                str(self.device), self.layout,
                "float32" if self.dtype == self.torch.float32 else "float64", parent=self,
            )
            kernel._version = self._version
            self._stepped[neuron_model] = kernel
        elif kernel._version != self._version or kernel.gain_pre is not self.gain_pre:
            kernel.set_parameters_from(self)
        return kernel

    def __deepcopy__(self, memo: dict[int, Any]) -> _LivesTorchKernel:
        twin = _LivesTorchKernel(
            self._connectome, self.host_scale(), self.gain_pre.cpu().double().numpy(),
            self.host_bias(), self.neuron_model, str(self.device), self.layout,
            "float32" if self.dtype == self.torch.float32 else "float64",
            reference=self._reference_only,
        )
        memo[id(self)] = twin
        return twin

    # -- contrasts, one per stream and so per life

    def contrast_tensors(self, s_plus: Any, s_minus: Any) -> tuple[Any, Any]:
        edges, neurons = self.contrast_rows(s_plus, s_minus)
        return edges.to(self.param_dtype), neurons.to(self.param_dtype)

    def contrast_edges(self, s_plus: Any, s_minus: Any, weights: Any = None) -> Any:
        edges = self.contrast_rows(s_plus, s_minus)[0]
        return edges if weights is None else edges * weights[:, None]


class _LivesGraph(NeuralGraph):
    """The lives' graph: a solve's chunk of sweeps holds the lives whose stream the residual
    check found settled, as each lone brain stops its own solve there."""

    def _equilibrate_local(
        self, drive: Any, *, budget: int, chunk: int, tolerance: float,
        state: BrainState | None, mask: Any, nudge: Any, stop_if_stalled: bool = False,
        fast: bool = False,
    ) -> Equilibrium:
        kernel = self._torch
        assert isinstance(kernel, _LivesTorchKernel)
        root = kernel.root
        current = self.settle_batch(drive, steps=0, state=state, mask=mask, nudge=nudge)
        reference: dict[str, bool] = (
            {"own_precision": True} if tolerance >= _MLX_CHECK_FLOOR else {}
        )
        error = self.residual(drive, current, mask=mask, nudge=nudge, **reference)
        used, checks, stagnation_checks, repeats = 0, 1, 0, 0
        while used < budget and not (error <= tolerance).all():
            if not np.isfinite(error).all() and any(
                not np.isfinite(value).all()
                for value in (current.v, current.activation, current.adaptation)
            ):
                break
            previous = current
            root._hold = kernel.torch.as_tensor(
                np.asarray(error <= tolerance), device=kernel.device
            )
            try:
                current = self.settle_batch(
                    drive, steps=min(chunk, budget - used), state=current, mask=mask, nudge=nudge
                )
            finally:
                root._hold = None
            used += current.steps
            error = self.residual(drive, current, mask=mask, nudge=nudge, **reference)
            checks += 1
            if stop_if_stalled and used < budget and not (error <= tolerance).all():
                stagnation_checks += 1
                repeats = repeats + 1 if self._repeated_state(previous, current) else 0
                if repeats >= 2:
                    break
        return Equilibrium(
            current, error, used, tolerance, residual_checks=checks,
            stagnation_checks=stagnation_checks,
        )


class _LivesLearner(Learner):
    """``Learner`` whose device update moves each life on its own contrast: the momentum is per
    life, and the masks, ties and bounds read the last axis."""

    def _update(
        self, free: BrainState, nudged: BrainState, opposite: BrainState | None = None,
        share: float = 1.0,
    ) -> dict[str, float]:
        self._validate_phases(free, nudged, opposite)
        cfg = self.config
        assert cfg.eta_bias is not None
        minus_state, span = (free, cfg.beta) if opposite is None else (opposite, 2.0 * cfg.beta)
        kernel = self._device_kernel(nudged, minus_state)
        if kernel is None:
            raise RuntimeError("the lives' phases rest on their device")
        edges, neurons = kernel.contrast_tensors(nudged.device["s"], minus_state.device["s"])
        edges, neurons = self._adaptive_device(kernel, edges / span, neurons / span)
        report = self._apply_device(kernel, share * cfg.eta * edges, share * cfg.eta_bias * neurons)
        self.contrast_updates += 1
        return report

    def _device_candidate(self, kernel: Any, delta_scale: Any, delta_bias: Any) -> dict[str, Any]:
        torch, cfg = kernel.torch, self.config
        lives = kernel.lives
        if (
            tuple(delta_scale.shape) != (lives, self.brain.connectome.synapses)
            or tuple(delta_bias.shape) != (lives, self.brain.connectome.n)
        ):
            raise ValueError("steps must be one per life and synapse and one per life and neuron")
        given = torch.isfinite(delta_scale).all() & torch.isfinite(delta_bias).all()
        ix = self._device_indices(kernel)
        d = delta_scale.clone()
        if ix["synapses"] is not None:
            d = d * ix["synapses"]
        if ix["rate"] is not None:
            d = d * ix["rate"]
        if ix["paired"] is not None:
            d[:, ix["paired"]] = 0.5 * (d[:, ix["paired"]] + d[:, ix["paired_reverse"]])
        if ix["members"] is not None:
            total = torch.zeros(lives, len(ix["member_count"]), dtype=d.dtype, device=d.device)
            total.index_add_(1, ix["member_groups"], d[:, ix["members"]])
            d[:, ix["members"]] = (total / torch.clamp(ix["member_count"], min=1.0))[
                :, ix["member_groups"]
            ]
        if ix["synapses"] is not None:
            d = d * ix["synapses"]
        tied = torch.isfinite(d).all()
        scale = kernel.scale + d
        db = delta_bias if ix["neurons"] is None else delta_bias * ix["neurons"]
        bias = kernel.bias_param + db
        bounded = torch.clamp(scale, -cfg.scale_cap, cfg.scale_cap)
        scale = (
            bounded if ix["synapses"] is None
            else torch.where(ix["synapses"].bool(), bounded, kernel.scale)
        )
        checked = kernel.parameter_check(scale, bias)
        zero = torch.zeros((), dtype=d.dtype, device=d.device)
        return {
            "scale": scale, "bias": bias, "checked": checked,
            "flags": (given, tied, checked[3], checked[4]),
            "steps": (d.abs().mean() if d.numel() else zero, db.abs().mean()),
        }


class _LivesCritic(_CriticOnDevice):
    """The critic on the device with one weight row and one bias per life."""

    def _bias(self) -> Any:
        return self.xp.here(np.asarray(self.owner.b_critic, dtype=float))

    def value(self, state: BrainState) -> np.ndarray:
        return self.xp.host(self.value_here(state))

    def value_here(self, state: BrainState) -> Any:
        part = self.xp.part(state, self.source)
        return (part * self.weights).sum(axis=1) + self._bias()

    def unread(self, delta: Any) -> tuple[Any, Any, Any]:
        trace = self.trace
        assert trace is not None
        xp = self.xp
        d = xp.here(delta) if isinstance(delta, np.ndarray) else xp.floats(delta)
        trace = trace / (1.0 + (trace * trace).sum(axis=1, keepdims=True))
        step = self.owner.config.eta_critic * (d[:, None] * trace)  # each life its own row
        weights = self.weights + step[:, :-1]
        return weights, xp.all_finite(weights), step[:, -1]

    def candidate(self, delta: np.ndarray) -> tuple[Any, Any]:
        weights, finite, step = self.unread(delta)
        read = self.xp.host(self.xp.cat([finite.reshape(1).to(step.dtype), step], 0))
        bias = np.asarray(self.owner.b_critic, dtype=float) + read[1:]
        if not read[0] or not np.isfinite(bias).all():
            raise ValueError("critic parameters must remain finite")
        return weights, bias


class _LivesActorCritic(ActorCritic):
    """``ActorCritic`` of lives, one stream per life: the critic rests on the device with a row
    per life, and an outcome steps each life on its own stream's eligibility and dopamine."""

    def reside(self, kernel: Any) -> None:
        if kernel is None:
            raise RuntimeError("the lives' critic rests on their device")
        if self._critic_device is not None:
            if not any(name in self.__dict__ for name in _CriticOnDevice.OWNED):
                return
            self.leave_device()
        self._critic_device = _LivesCritic(self, kernel)

    def _learn_device(
        self, kernel: Any, plus: BrainState, minus: BrainState, value: np.ndarray,
        reward: np.ndarray, done: np.ndarray, next_drive: np.ndarray,
        bootstrap: np.ndarray | None, observed: np.ndarray, warm: BrainState | None = None,
    ) -> dict[str, float]:
        """``ActorCritic._learn_device`` per life: each stream's step is its own life's, with no
        mean over the streams. The outcome's checks and the dopamine come to the host in one
        read before anything is installed."""
        torch = kernel.torch
        cfg = self.config
        span = 2.0 * self.learner.config.beta
        decay = cfg.gamma * cfg.lam
        batch = len(reward)
        if batch != kernel.lives or not observed.all():
            raise ValueError("an outcome of these lives has one observed stream per life")
        s_plus, s_minus = plus.device["s"], minus.device["s"]
        neurons = s_plus - s_minus
        synapses = self.synapses
        if self._trace_device is None or tuple(self._trace_device[0].shape) != (batch, synapses):
            self._trace_device = (
                torch.zeros((batch, synapses), dtype=neurons.dtype, device=neurons.device),
                torch.zeros_like(neurons),
            )
            self.trace = self.trace_bias = None
        trace, trace_bias = self._trace_device
        trace_bias = decay * trace_bias + neurons / span
        free = self._free
        assert free is not None
        self._critic_eligibility(free, batch)
        next_state = self._next_state(next_drive, done, warm)
        resting = self._critic_device
        assert isinstance(resting, _LivesCritic)
        exact = torch.float32 if trace.device.type == "mps" else torch.float64

        def here(values: Any) -> Any:
            return torch.as_tensor(np.asarray(values, dtype=float), dtype=exact,
                                   device=trace.device)

        following = resting.value_here(next_state).to(exact)
        ends = torch.as_tensor(np.asarray(done), device=trace.device)
        boot = torch.zeros_like(following) if bootstrap is None else here(bootstrap)
        td_here = here(reward) + cfg.gamma * torch.where(ends, boot, following) - here(value)
        delta_here = td_here
        if cfg.dopamine_cap > 0:
            delta_here = torch.clamp(delta_here, -cfg.dopamine_cap, cfg.dopamine_cap)
        critic_weights, critic_finite, critic_step = resting.unread(delta_here)
        d = delta_here.to(trace.dtype)
        self.updates += 1
        next_brain = self.learner.brain
        eta, eta_bias = self._rates()
        step_bias = eta_bias * (d[:, None] * trace_bias)
        step = decay * trace * d[:, None] + kernel.contrast_edges(s_plus, s_minus, d) / span
        update = self.learner._device_candidate(kernel, eta * step, step_bias)
        read = torch.cat([
            torch.stack([torch.isfinite(td_here).all().to(exact), critic_finite.to(exact)]
                        + [flag.to(exact) for flag in update["flags"]]
                        + [number.to(exact) for number in update["steps"]]),
            critic_step.to(exact), td_here, delta_here,
        ]).cpu().numpy().astype(float)
        if not read[0]:
            raise ValueError("TD errors must remain finite")
        critic_bias = np.asarray(self.b_critic, dtype=float) + read[8 : 8 + batch]
        if not read[1] or not np.isfinite(critic_bias).all():
            raise ValueError("critic parameters must remain finite")
        self.learner._device_refusal(read[2:6])
        td_error = read[8 + batch : 8 + 2 * batch]
        delta = read[8 + 2 * batch :]
        capped = self._capped(td_error, observed)
        report = self.learner._install_device(update, read[6:8])
        trace.mul_(decay)
        kernel.add_contrast(trace, s_plus, s_minus, span, None)
        self._accept_critic((critic_weights, critic_bias))
        if done.any():
            ended = torch.as_tensor(np.flatnonzero(done), device=trace.device)
            trace[ended] = 0.0
            trace_bias[ended] = 0.0
            self._forget_critic(done)
        self._trace_device = (trace, trace_bias)
        self._free = next_state
        self._free_brain = next_brain
        self._drive = next_drive
        self._pending = None
        self._pending_memory = None
        self._pending_hearing = None
        report["delta"] = float(np.abs(delta).mean())
        report["dopamine"] = float(delta.mean())
        report["td_error"] = float(np.abs(td_error).mean())
        report["capped"] = capped
        report["value"] = float(np.asarray(value).mean())
        report["free_steps"] = float(next_state.steps)
        report["saturation"] = self._saturation(free)
        report["trace"] = float(trace.abs().mean())
        return report


class _LivesRandom:
    """Each life's own generator, drawn as its lone brain draws it: one row per life."""

    def __init__(self, generators: Sequence[np.random.Generator]) -> None:
        self.generators = list(generators)

    def random(self, shape: Any) -> np.ndarray:
        shape = tuple(np.atleast_1d(shape))
        if not shape or shape[0] != len(self.generators):
            raise ValueError("a draw of these lives takes one row per life")
        return np.stack([g.random(shape[1:]) for g in self.generators])


_REFUSED = (
    ("recall", "recall=False: the recall memory's maps have no lives axis"),
    ("hippocampus", "episodic=False: the store's consolidated records have no lives axis"),
    ("arousal", "no arousal: Lives steps, acts and teaches; live is a lone brain's"),
    ("learned_code", "no learned code"),
    ("hearing", "no carried gains"),
    ("_episodic", "no memory_choice"),
)


def _supported(brain: Any) -> None:
    """Refuse a composition with a part the lives axis does not cover, naming it."""
    for name, what in _REFUSED:
        if getattr(brain, name, None) is not None:
            raise ValueError(f"Lives holds the parts its lives axis covers; compose with {what}")
    learner, graph = brain.learner, brain.brain
    cfg = learner.config
    if cfg.normalize or cfg.surprise or cfg.rehearsal or cfg.fast_settle or cfg.qualified:
        raise ValueError(
            "Lives teaches with the unnormalized, unjudged lesson: learning_normalize=0, "
            "learning_surprise=False, no rehearsal, fast_settle=False, qualified=False"
        )
    if graph.apical is not None:
        raise ValueError("Lives holds no apical compartment")
    agent = brain.basal_ganglia
    if agent.maturity is not None or agent.salience is not None:
        raise ValueError("Lives holds no maturity and no salience")


class Lives:
    """Many continuing lives of one composed anatomy, each a brain of its own, stepped together.

    ``Lives(inputs, actions, lives=128, backend="torch")`` composes ``lives`` brains of
    ``Brain.compose(inputs, actions, seed=seeds[i], recall=False, episodic=False, **options)``,
    one per seed (``range(lives)`` by default). Every operation takes one row of observations
    per life and returns one answer per life: ``act``, ``step``, ``lesson`` and ``learn`` mean
    what they mean on a lone brain with one stream, life by life. ``life(i)`` is life ``i`` as a
    lone brain on the CPU, with its parameters, optimizer and critic.

    On ``backend="cpu"`` each life is its own ``Brain`` and the lives run in turn, to the bit as
    lone brains. On ``backend="torch"`` (CUDA, or a torch CPU device) they are the streams of one
    brain whose parameters carry a lives axis, and each moment of all of them is one settle and
    one update; each life agrees with its lone brain on that device within the solver's
    tolerance. The lives share their anatomy and nothing they learn.
    """

    def __init__(
        self, inputs: int, actions: int, lives: int, *, seeds: Sequence[int] | None = None,
        backend: str = "cpu", device: str | None = None, precision: str | None = None,
        **options: Any,
    ) -> None:
        from .generic import Brain

        if lives < 1:
            raise ValueError("lives must be at least one")
        self.seeds = list(range(lives)) if seeds is None else [int(s) for s in seeds]
        if len(self.seeds) != lives:
            raise ValueError("one seed per life")
        if backend not in ("cpu", "torch"):
            raise ValueError("Lives runs on backend='cpu' or backend='torch'")
        options = {"recall": False, "episodic": False, **options}
        self.inputs, self.actions, self.backend = int(inputs), int(actions), backend
        self._options = options
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lone = [Brain.compose(inputs, actions, seed=s, backend="cpu", **options)
                    for s in self.seeds]
        _supported(lone[0])
        topology = lone[0].connectome
        for other in lone[1:]:
            if not (np.array_equal(other.connectome.pre, topology.pre)
                    and np.array_equal(other.connectome.post, topology.post)):
                raise ValueError("the lives' seeds compose different anatomies")
        if backend == "cpu":
            self.brains: list[Any] | None = lone
            self.brain: Any = None
            return
        self.brains = None
        place: dict[str, Any] = {"backend": "torch"}
        if device is not None:
            place["device"] = device
        if precision is not None:
            place["precision"] = precision
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            brain = Brain.compose(inputs, actions, seed=self.seeds[0], **place, **options)
        graph = brain.brain
        kernel = graph._torch
        assert kernel is not None
        if kernel.device.type == "mps":
            raise ValueError(
                "Lives checks its settles in float64 on its device: CUDA or the torch CPU device "
                "(MPS has no float64)"
            )
        efficacy = np.stack([b.brain.efficacy for b in lone])
        bias = np.stack([b.brain.bias for b in lone])
        lives_kernel = _LivesTorchKernel(
            graph.connectome, efficacy, kernel.gain_pre, bias, graph.neuron_model,
            str(kernel.device), graph.layout,
            "float32" if kernel.dtype == kernel.torch.float32 else "float64",
        )
        graph._torch = lives_kernel
        graph.__class__ = _LivesGraph
        for name in ("_efficacy", "_bias", "_weights_host", "_flat", "_csr"):
            if hasattr(graph, name):
                setattr(graph, name, None)
        learner = brain.learner
        learner.__class__ = _LivesLearner
        synapses, n = graph.connectome.synapses, graph.connectome.n
        learner.__dict__["_device_moments"] = None
        for name, width in (("velocity", synapses), ("velocity_bias", n),
                            ("second_moment", synapses), ("second_moment_bias", n)):
            learner.__dict__["_" + name] = np.zeros((lives, width))
        agent = brain.basal_ganglia
        agent.leave_device()
        agent.__class__ = _LivesActorCritic
        agent.w_critic = np.stack([b.basal_ganglia.w_critic for b in lone])
        agent.b_critic = np.array([b.basal_ganglia.b_critic for b in lone], dtype=float)
        agent.rng = _LivesRandom([b.basal_ganglia.rng for b in lone])
        self.brain = brain

    @property
    def lives(self) -> int:
        return len(self.seeds)

    def _rows(self, observations: Any) -> np.ndarray:
        x = np.asarray(observations, dtype=float)
        if x.shape != (self.lives, self.inputs):
            raise ValueError(f"one row of {self.inputs} observations per life ({self.lives})")
        return x

    def _each(self, name: str, x: np.ndarray, *rest: Any, **named: Any) -> list[Any]:
        assert self.brains is not None
        out = []
        for i, brain in enumerate(self.brains):
            args = [None if r is None else np.asarray(r)[i : i + 1] for r in rest]
            kwargs = {k: None if v is None else np.asarray(v)[i : i + 1]
                      for k, v in named.items()}
            out.append(getattr(brain, name)(x[i : i + 1], *args, **kwargs))
        return out

    def act(self, observations: Any, *, greedy: bool = False) -> np.ndarray:
        """Each life's action on its own row: ``Brain.act`` life by life."""
        x = self._rows(observations)
        if self.brains is not None:
            return np.concatenate([b.act(x[i : i + 1], greedy=greedy)
                                   for i, b in enumerate(self.brains)])
        return np.asarray(self.brain.act(x, greedy=greedy))

    def step(
        self, observations: Any, *, reward: Any = None, done: Any = None, teacher: Any = None,
    ) -> np.ndarray:
        """Each life's ``Brain.step`` on its own row, its own reward, its own ending and, with
        ``teacher``, its own label taught before the act."""
        x = self._rows(observations)
        if self.brains is not None:
            return np.concatenate(
                self._each("step", x, reward=reward, done=done, teacher=teacher)
            )
        return np.asarray(self.brain.step(x, reward=reward, done=done, teacher=teacher))

    def learn(self, reward: Any, done: Any, next_observations: Any) -> None:
        """Each life's ``Brain.learn`` from its own outcome."""
        x = self._rows(next_observations)
        if self.brains is not None:
            for i, b in enumerate(self.brains):
                b.learn(np.asarray(reward)[i : i + 1], np.asarray(done)[i : i + 1], x[i : i + 1])
            return
        self.brain.learn(reward, done, x)

    def lesson(self, observations: Any, labels: Any) -> None:
        """Each life's ``Brain.lesson`` of its own label on its own row."""
        x = self._rows(observations)
        y = np.asarray(labels)
        if self.brains is not None:
            for i, b in enumerate(self.brains):
                b.lesson(x[i : i + 1], y[i : i + 1])
            return
        self.brain.lesson(x, y)

    def parameters(self, i: int) -> dict[str, np.ndarray]:
        """Life ``i``'s learned parameters on the host, as a lone brain holds them: its
        ``efficacy`` and ``bias``, the critic's ``w_critic`` and ``b_critic``, and the optimizer's
        ``velocity`` and ``velocity_bias``."""
        if self.brains is not None:
            b = self.brains[i]
            return {
                "efficacy": np.array(b.brain.efficacy), "bias": np.array(b.brain.bias),
                "w_critic": np.array(b.basal_ganglia.w_critic),
                "b_critic": np.array(b.basal_ganglia.b_critic, dtype=float),
                "velocity": np.array(b.learner.velocity),
                "velocity_bias": np.array(b.learner.velocity_bias),
            }
        brain = self.brain
        kernel = brain.brain._torch
        agent = brain.basal_ganglia
        learner = brain.learner
        learner._host_moments()
        resting = agent._critic_device
        weights = (resting.xp.host(resting.weights) if resting is not None
                   else np.asarray(agent.w_critic))
        return {
            "efficacy": kernel.host_scale()[i], "bias": kernel.host_bias()[i],
            "w_critic": weights[i], "b_critic": np.asarray(agent.b_critic, dtype=float)[i],
            "velocity": np.array(learner.__dict__["_velocity"][i]),
            "velocity_bias": np.array(learner.__dict__["_velocity_bias"][i]),
        }

    def life(self, i: int) -> Any:
        """Life ``i`` as a lone brain on the CPU: the composition of its seed with its learned
        parameters, optimizer and critic; its stream starts afresh. Save it as any brain."""
        from .generic import Brain

        if self.brains is not None:
            return self.brains[i]
        held = self.parameters(i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lone = Brain.compose(self.inputs, self.actions, seed=self.seeds[i], backend="cpu",
                                 **self._options)
        lone.learner.brain = lone.learner.brain.with_parameters(
            efficacy=held["efficacy"], bias=held["bias"]
        )
        lone.learner.velocity = held["velocity"]
        lone.learner.velocity_bias = held["velocity_bias"]
        lone.learner.updates = self.brain.learner.updates
        lone.learner.contrast_updates = self.brain.learner.contrast_updates
        lone.basal_ganglia.updates = self.brain.basal_ganglia.updates
        lone.basal_ganglia.w_critic = held["w_critic"]
        lone.basal_ganglia.b_critic = float(held["b_critic"])
        return lone
