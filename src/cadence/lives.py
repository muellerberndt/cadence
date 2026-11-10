"""Independent lives over one topology, stepped together (cadence#274, #167).

``Lives`` holds many composed brains of one anatomy, each with its own parameters, optimizer,
critic, eligibility, recall memory, carried gains, working trace and random state, one stream
each. On the CPU each life is a lone ``Brain``, stepped in turn. On a torch device the lives are
the streams of one brain whose parameters carry a lives axis: every settle reads each stream
through its own life's synapses, and every update moves each life on its own contrast and
dopamine, so one call per moment settles and teaches all of them. Each life follows the lone
brain of its seed on the same device within the solver's tolerance.

The lives run the composition's own equations through these parts:

- ``_LivesTorchKernel``: the settle's transport as one batched product per block of the layout,
  each life's dense block of its own synapses, each life held where its own stream meets the
  tolerance, and the contrast per stream;
- ``_LivesGraph``: a solve that holds each life where its own equations hold, and a rescue on
  the damping ladder life by life;
- ``_LivesLearner``: a lesson's update and a reward's update per life; the momentum, the second
  moments and the counts of their bias correction per life; a lesson judged by surprise updates
  the lives whose own answer differed;
- ``_LivesRecall``: the recall memory's address and position maps per life (a stream's slots are
  its own already), the address lesson and the reward's step on each life's own row;
- ``_LivesHearing``: the carried gains, their second moments and counts per life;
- ``_LivesActorCritic`` and ``_LivesCritic``: the critic's weights and bias per life, the
  dopamine and the actor's step per stream;
- ``_LivesRandom``: each life's own random generator, drawn as its lone brain draws it.

A composition holds the parts these cover: the default composition without the store
(``episodic=False``: the store's consolidated records are a brain's own and have no lives axis),
with or without the recall memory and the carried gains, at any teaching point. ``Lives`` refuses
arousal, maturity, a learned or fixed code, an apical compartment, rehearsal, qualified teaching,
the efference copy, the heard history and a recall memory with several heads, a window, the
command, the read of when or tokens, each by name. A moment any life refuses is refused for
every life.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

import numpy as np

from .brain import (
    _MLX_CHECK_FLOOR,
    BrainState,
    Equilibrium,
    _check_ladder,
    _plan_structure,
    _SweepConfig,
    _TorchKernel,
    _transport_override,
    _TransportPlan,
)
from .brain import Brain as NeuralGraph
from .learning import _MOMENTS, _RELATIVE_TINY, LearnedState, Learner, _taught_report
from .plasticity import ActorCritic, _held_drive
from .resident import (
    _MAP_KINDS,
    TorchArrays,
    _CriticOnDevice,
    _forgotten,
    _Moment,
    _RecallOnDevice,
    arrays_of,
)
from .stream import Hearing

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

    def _blocks(self) -> list[tuple[Any, int, int, Any]]:
        """The layout's blocks grouped by the range they reach: per group the presynaptic neurons
        of its blocks side by side (a slice where they are one run, else their indices), the
        range reached and the group's view, ``(lives, pre, post)``, of the dense buffer the
        parameters are written into in place (a captured sweep reads the same storage). A sweep
        makes one batched product per range reached."""
        if self._dense is None:
            torch, lay = self.torch, self.layout
            assert lay is not None
            connectome = self._connectome
            by_post: dict[tuple[int, int], list[int]] = {}
            for k in range(lay.pairs):
                _, _, b0, b1 = lay.bounds(k)
                by_post.setdefault((b0, b1), []).append(k)
            block_of = np.asarray(lay.block_of_edge)
            entry = np.zeros(connectome.synapses, dtype=np.int64)
            spans, at = [], 0
            for (b0, b1), blocks in by_post.items():
                width, row, pre = b1 - b0, 0, []
                for k in blocks:
                    a0, a1, _, _ = lay.bounds(k)
                    inside = np.flatnonzero(block_of == k)
                    entry[inside] = (
                        at + (row + connectome.pre[inside] - a0) * width
                        + connectome.post[inside] - b0
                    )
                    pre.append(np.arange(a0, a1))
                    row += a1 - a0
                index = np.concatenate(pre)
                spans.append((index, b0, b1, at, row))
                at += row * width
            self._dense = torch.zeros(self.lives, at, dtype=self.dtype, device=self.device)
            self._entry = torch.from_numpy(entry).to(self.device)
            views = []
            for index, b0, b1, first, rows in spans:
                run = len(index) and bool((np.diff(index) == 1).all())
                place = slice(int(index[0]), int(index[-1]) + 1) if run else torch.from_numpy(
                    index
                ).to(self.device)
                view = self._dense[:, first : first + rows * (b1 - b0)]
                views.append((place, b0, b1, view.view(self.lives, rows, b1 - b0)))
            self._views = views
        return self._views

    def _synaptic_input(self, s: Any, out: Any) -> Any:
        """The transport of a neuron-major activation ``(neurons, lives)``: per range reached,
        every life's presynaptic row through its own blocks."""
        if s.shape[1] != self.lives:
            raise ValueError(f"a settle of these lives holds one stream per life ({self.lives})")
        torch = self.torch
        out.zero_()
        rows = s.T
        for place, b0, b1, block in self._blocks():
            product = torch.bmm(rows[:, place].unsqueeze(1), block).squeeze(1)
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
        # sweeps that hold a life are other operations than sweeps that hold none: another
        # captured graph, and ``_update`` reads the mark. The kernel keeps one configuration per
        # key; this is the lives' own of it, one per configuration and mark
        holds = hold is not None or tolerance is not None
        kept = ws.__dict__.setdefault("lives_configs", {})
        mine = kept.get((id(cfg), holds))
        if mine is None or mine[0] is not cfg:
            if len(kept) >= 64:
                kept.clear()
            mine = kept[(id(cfg), holds)] = (cfg, _SweepConfig(
                (cfg.key, holds), cfg.masked, cfg.keep, cfg.nudge, cfg.held, cfg.tolerance,
                cfg.quiet,
            ))
        return mine[1]

    def _update(
        self, ws: Any, cfg: Any, i: int, j: int, synaptic: Any, stage: int, standing: Any,
    ) -> None:
        """The sweep of every stream, then each life whose stream met the tolerance at an earlier
        sweep held where it met it, as its lone brain's settle stops there: a stream's own
        movement under the tolerance meets it, and the settle meets it when every life has."""
        super()._update(ws, cfg, i, j, synaptic, stage, standing)
        if not cfg.key[-1]:  # a fixed count of sweeps with no life settled: every life sweeps
            return
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
    """The lives' graph: a solve holds each life where its own stream's equations hold, as each
    lone brain stops its own solve there, and a rescue on the damping ladder runs life by life."""

    _steps_on_kernel = True  # a numerical copy at another step keeps this class and its kernel

    def _equilibrate_local(
        self, drive: Any, *, budget: int, chunk: int, tolerance: float,
        state: BrainState | None, mask: Any, nudge: Any, stop_if_stalled: bool = False,
        fast: bool = False,
    ) -> Equilibrium:
        """``NeuralGraph._equilibrate_local`` with every life whose residual the check found
        within the tolerance held through the next chunk of sweeps. With ``root._only`` set, the
        solve is that one life's: every other life is held as it is."""
        kernel = self._torch
        assert isinstance(kernel, _LivesTorchKernel)
        root: Any = kernel.root
        only = getattr(root, "_only", None)
        current = self.settle_batch(drive, steps=0, state=state, mask=mask, nudge=nudge)
        reference: dict[str, bool] = (
            {"own_precision": True} if tolerance >= _MLX_CHECK_FLOOR else {}
        )

        def settled(error: np.ndarray) -> np.ndarray:
            done = np.asarray(error <= tolerance)
            if only is not None:
                done = done.copy()
                done[np.arange(len(done)) != only] = True
            return done

        error = self.residual(drive, current, mask=mask, nudge=nudge, **reference)
        used, checks, stagnation_checks, repeats = 0, 1, 0, 0
        while used < budget and not settled(error).all():
            if not np.isfinite(error).all() and any(
                not np.isfinite(value).all()
                for value in (current.v, current.activation, current.adaptation)
            ):
                break
            previous = current
            done = settled(error)
            root._hold = kernel.torch.as_tensor(done, device=kernel.device) if done.any() else None
            try:
                current = self.settle_batch(
                    drive, steps=min(chunk, budget - used), state=current, mask=mask, nudge=nudge
                )
            finally:
                root._hold = None
            used += current.steps
            error = self.residual(drive, current, mask=mask, nudge=nudge, **reference)
            checks += 1
            if stop_if_stalled and used < budget and not settled(error).all():
                stagnation_checks += 1
                repeats = repeats + 1 if self._repeated_state(previous, current) else 0
                if repeats >= 2:
                    break
        return Equilibrium(
            current, error, used, tolerance, residual_checks=checks,
            stagnation_checks=stagnation_checks,
        )

    def equilibrate(
        self, drive: Any, *, budget: int = 512, chunk: int = 32, tolerance: float = 1e-5,
        state: BrainState | None = None, mask: Any = None, nudge: Any = None, damping: int = 0,
        fast: bool = False,
    ) -> Equilibrium:
        """``NeuralGraph.equilibrate``. With ``damping``, each life whose equations do not hold
        takes the ladder of halved steps by itself, every other life held, as its lone brain
        takes it: a life that stalls or qualifies moves on at its own sweep."""
        if not damping:
            return super().equilibrate(
                drive, budget=budget, chunk=chunk, tolerance=tolerance, state=state, mask=mask,
                nudge=nudge,
            )
        kernel = self._torch
        assert isinstance(kernel, _LivesTorchKernel)
        root: Any = kernel.root
        budget, chunk, damping = int(budget), int(chunk), int(damping)
        attempts = min(damping + 1, max(1, budget))
        _check_ladder(
            self.neuron_model, attempts - 1,
            "float32" if kernel.dtype == kernel.torch.float32 else "float64",
        )
        current = self.settle_batch(drive, steps=0, state=state, mask=mask, nudge=nudge)
        error = self.residual(drive, current, mask=mask, nudge=nudge)
        steps = checks = stagnation_checks = halvings = 0
        for life in np.flatnonzero(~np.asarray(error <= tolerance)):
            used = 0
            for halving in range(attempts):
                portion = (budget - used + attempts - halving - 1) // (attempts - halving)
                candidate = self if halving == 0 else self._with_dt(
                    float(np.ldexp(self.neuron_model.dt, -halving)))
                root._only = int(life)
                try:
                    phase = candidate._equilibrate_local(
                        drive, state=current, budget=portion, chunk=chunk, tolerance=tolerance,
                        mask=mask, nudge=nudge, stop_if_stalled=halving < attempts - 1,
                    )
                finally:
                    root._only = None
                used += phase.steps
                checks += phase.residual_checks
                stagnation_checks += phase.stagnation_checks
                current = phase.state
                found = phase.residual
                if halving:
                    found = self.residual(drive, current, mask=mask, nudge=nudge)
                    checks += 1
                halvings = max(halvings, halving)
                if found[life] <= tolerance or used >= budget or not np.isfinite(found[life]):
                    break
            steps = max(steps, used)
        error = self.residual(drive, current, mask=mask, nudge=nudge)
        return Equilibrium(
            current, error, steps, tolerance, residual_checks=checks + 2,
            damping_halvings=halvings, stagnation_checks=stagnation_checks,
        )


class _LivesLearner(Learner):
    """``Learner`` whose device update moves each life on its own contrast: the momentum, the
    second moments and their counts are per life, a lesson judged by surprise runs its phases
    on every life and updates the lives it taught, and the masks, ties and bounds read the last
    axis."""

    def _counts(self, lives: int) -> np.ndarray:
        """Each life's own count of lessons that updated it, the optimizer's bias correction."""
        held = self.__dict__.get("lives_contrast_updates")
        if held is None or len(held) != lives:
            held = self.__dict__["lives_contrast_updates"] = np.zeros(lives, dtype=np.int64)
        return cast(np.ndarray, held)

    def _applied(self, lives: int) -> np.ndarray:
        """Each life's own count of updates applied, lessons and outcomes."""
        held = self.__dict__.get("lives_updates")
        if held is None or len(held) != lives:
            held = self.__dict__["lives_updates"] = np.zeros(lives, dtype=np.int64)
        return cast(np.ndarray, held)

    def step(
        self, drive: Any, labels: np.ndarray | None, warm: BrainState | None = None,
        weight: np.ndarray | None = None, *, answer: np.ndarray | None = None,
        targets: np.ndarray | None = None,
    ) -> tuple[LearnedState, dict[str, float]]:
        """``Learner.step`` for one row per life. Under ``surprise`` the caller's ``answer``
        judges each life by its own row; the phases run on every life, since a settle of these
        lives holds one stream per life, and the update moves the lives whose answer differed.
        ``LearnedState.taught`` marks them, and the states hold every life's row."""
        cfg = self.config
        judged = None
        if cfg.surprise and labels is not None:
            if answer is None:
                raise ValueError("a lesson of lives judged by surprise takes its caller's answer")
            judged = self._differs(np.asarray(answer), self._labels(labels))
            if not judged.any():
                return self._untaught(judged)
        self.__dict__["_taught_lives"] = judged
        self.config = replace(cfg, surprise=False)  # every row runs; the update reads the mark
        try:
            learned, report = super().step(drive, labels, warm, weight, targets=targets)
        finally:
            self.config = cfg
            self.__dict__["_taught_lives"] = None
        if judged is not None:
            learned = LearnedState(learned.free, learned.nudged, learned.opposite, judged)
            report.update(_taught_report(judged, len(judged)))
        return learned, report

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
        taught = self.__dict__.get("_taught_lives")
        mask = None
        if taught is not None and not taught.all():
            mask = kernel.torch.as_tensor(np.asarray(taught, dtype=bool), device=kernel.device)
        edges, neurons = self._adaptive_lives(kernel, edges / span, neurons / span, mask)
        step, step_bias = share * cfg.eta * edges, share * cfg.eta_bias * neurons
        if mask is not None:  # a life the lesson did not teach moves nothing
            zero = kernel.torch.zeros((), dtype=step.dtype, device=step.device)
            step = kernel.torch.where(mask[:, None], step, zero)
            step_bias = kernel.torch.where(mask[:, None], step_bias, zero)
        applied = self._applied(kernel.lives).copy()
        report = self._apply_device(kernel, step, step_bias)
        rows = slice(None) if taught is None else np.asarray(taught, dtype=bool)
        self._counts(kernel.lives)[rows] += 1
        self.__dict__["lives_updates"] = applied  # an update of every life counted the taught
        applied[rows] += 1
        self.contrast_updates += 1
        return report

    def _install_device(self, candidate: dict[str, Any], steps: Any) -> dict[str, float]:
        report = super()._install_device(candidate, steps)
        kernel = self.brain._torch
        assert kernel is not None
        self._applied(kernel.lives)[:] += 1
        return report

    def _group_power(self, kernel: Any, values: Any, group: tuple[Any, ...], which: int) -> Any:
        """Each life's sum of ``values`` squared per group of a projection table: every group in
        a row of its own, summed in a fixed order, as the lone update sums it."""
        torch, lives = kernel.torch, kernel.lives
        index, size, _, place, matrix = group
        if place is None:
            power = torch.zeros(lives, int(size), dtype=values.dtype, device=values.device)
            return power.index_add_(1, index, values * values)
        rooms = self.__dict__.setdefault("_lives_rooms", {})
        key = (id(kernel), which)
        room = rooms.get(key)
        if room is None or room.shape != (lives, matrix.numel()):
            room = rooms[key] = torch.zeros(
                lives, matrix.numel(), dtype=values.dtype, device=values.device
            )
        room[:, place] = values * values
        return room.view(lives, *matrix.shape).sum(dim=2)

    def _adaptive_lives(self, kernel: Any, edges: Any, neurons: Any, mask: Any) -> tuple[Any, Any]:
        """``Learner._adaptive_device`` per life: each life's momentum and second moments, the
        bias correction at its own count, the relative floor over its own projections. A life
        ``mask`` leaves out keeps its history."""
        cfg = self.config
        if not (cfg.momentum or cfg.normalize):
            return edges, neurons
        torch = kernel.torch
        previous = self._moments_on_device(kernel)
        held = {"holder": kernel, **{name: previous[name] for name in _MOMENTS}}
        self.__dict__["_device_moments"] = held
        count = self._counts(kernel.lives) + 1

        def correction(factor: float) -> Any:
            return torch.as_tensor(
                1.0 - np.power(factor, count), dtype=kernel.param_dtype, device=kernel.device
            )[:, None]

        def kept(name: str, new: Any) -> Any:
            held[name] = new if mask is None else torch.where(mask[:, None], new, previous[name])
            return new

        raw_edges, raw_neurons = edges, neurons
        if cfg.momentum:
            m = cfg.momentum
            velocity = kept("velocity", previous["velocity"].mul(m).add_(edges, alpha=1.0 - m))
            velocity_bias = kept(
                "velocity_bias", previous["velocity_bias"].mul(m).add_(neurons, alpha=1.0 - m)
            )
            edges, neurons = velocity / correction(m), velocity_bias / correction(m)
        if cfg.normalize:
            rho = cfg.normalize
            second = kept("second_moment", previous["second_moment"].mul(rho).addcmul_(
                raw_edges, raw_edges, value=1.0 - rho
            ))
            second_bias = kept("second_moment_bias", previous["second_moment_bias"].mul(
                rho
            ).addcmul_(raw_neurons, raw_neurons, value=1.0 - rho))
            rms = (second / correction(rho)).sqrt()
            rms_bias = (second_bias / correction(rho)).sqrt()
            if cfg.normalize_relative:
                divisors = []
                for which, (values, group) in enumerate(zip(
                    (rms, rms_bias), self._projections_on_device(kernel), strict=True
                )):
                    power = self._group_power(kernel, values, group, which)
                    floor = (power / group[2]).sqrt()[:, group[0]]
                    divisors.append(values + cfg.normalize_relative * floor + _RELATIVE_TINY)
                edges, neurons = edges / divisors[0], neurons / divisors[1]
            else:
                edges = edges / (rms + cfg.normalize_floor)
                neurons = neurons / (rms_bias + cfg.normalize_floor)
        ix = self._device_indices(kernel)  # a rate of zero keeps the moments it held (``held``)
        for name, stopped in (("velocity", "held_synapses"), ("second_moment", "held_synapses"),
                              ("velocity_bias", "held_neurons"),
                              ("second_moment_bias", "held_neurons")):
            if ix[stopped] is not None and held[name] is not previous[name]:
                held[name] = torch.where(ix[stopped], previous[name], held[name])
        if not bool(torch.stack([torch.isfinite(held[name]).all() for name in _MOMENTS]).all()):
            raise ValueError("optimizer moments must remain finite")
        return edges, neurons

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


class _LivesArrays(TorchArrays):
    """The array namespace of a lives kernel: a read back through the synapses reads each
    stream through its own life's."""

    def read_back(
        self, graph: Any, ports: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, int]],
        movement: Any, width: int,
    ) -> list[Any]:
        """``movement @ W.T`` per port, each row through its own life's ``W``."""
        kernel = graph._torch
        torch, lives = self.torch, kernel.lives
        if movement.shape[0] != lives:
            raise ValueError("a read back of these lives takes one row per life")
        out = []
        for port in ports:
            placed = self.placed.get(id(port[0]))
            if placed is None:
                placed = self.placed[id(port[0])] = (
                    port[0], self.long(port[0]), self.long(port[1]), self.long(port[2]),
                )
            _, synapses, on_rows, on_cols = placed
            values = (kernel.gain_pre[synapses] * kernel.scale[:, synapses]).to(movement.dtype)
            w = torch.zeros((lives, port[3] * width), dtype=movement.dtype, device=self.device)
            if graph.layout.parallel_synapses:
                w.index_add_(1, on_rows * width + on_cols, values)
            else:
                w[:, on_rows * width + on_cols] = values
            w = w.view(lives, port[3], width)
            out.append(torch.bmm(w, movement[:, :, None])[:, :, 0])
        return out


def _finite_rows(torch: Any, *arrays: Any) -> Any:
    """Per life, whether every entry of its row of every array is finite."""
    return torch.stack([
        torch.isfinite(a).reshape(a.shape[0], -1).all(dim=1) for a in arrays
    ]).all(dim=0)


def _chosen(torch: Any, rows: Any, new: Any, old: Any) -> Any:
    """``new`` at the lives ``rows`` marks, ``old`` at the others."""
    return torch.where(rows.reshape(-1, *([1] * (new.ndim - 1))), new, old)


class _LivesRecall(_RecallOnDevice):
    """A recall memory's slots, maps and reads on the device, the maps one per life.

    A stream's slots, the cortex of the moment before and its reads are a stream's already
    (``_RecallOnDevice``); here every address and position map carries a lives axis first, a
    moment's address and position terms read each stream through its own life's maps, and the
    address lesson and the reward's step move each life's maps on its own row. The arrays rest
    on the device alone: ``Lives.parameters`` reads a life's maps."""

    def __init__(self, recall: Any, kernel: Any, lone: Sequence[Any]) -> None:
        super().__init__(recall, kernel)
        here = self.xp.here
        for name in _MAP_KINDS:
            if getattr(self, name):
                setattr(self, name, [tuple(
                    here(np.stack([np.asarray(getattr(one, name)[0][part]) for one in lone]))
                    for part in range(3)
                )])

    def give_back(self) -> None:
        raise RuntimeError(
            "the lives' recall memory rests on their device; Lives.parameters(i) and "
            "Lives.life(i) read a life's maps"
        )

    def maps_of(self, life: int) -> dict[str, tuple[np.ndarray, ...]]:
        """Life ``life``'s maps on the host, by kind."""
        host = self.xp.host
        return {
            name: tuple(host(a[life]) for a in getattr(self, name)[0])
            for name in _MAP_KINDS if getattr(self, name)
        }

    # -- each stream through its own life's maps

    def _term(self, maps: Any, x: Any, c: Any) -> Any:
        m_x, m_c, m = maps
        if m_x.ndim == 2:  # one number per life (the slope's lean), its bias ``(lives, 1)``
            return (x * m_x).sum(axis=1) + (c * m_c).sum(axis=1) + m[:, 0]
        bmm = self.xp.bmm
        return bmm(x[:, None, :], m_x)[:, 0] + bmm(c[:, None, :], m_c)[:, 0] + m

    def _moment(self, x: np.ndarray) -> _Moment:
        recall, xp = self.recall, self.xp
        inputs = self.kernel.observed(x)
        n = len(x)
        if n != self.kernel.root.lives:
            raise ValueError("a moment of these lives takes one row per life")
        c = self.previous if len(self.previous) == n else xp.zeros((n, len(recall.association)))
        key = recall.heard * inputs
        return _Moment(x, c, key + self._term(self.head_maps[0], inputs, c), key, inputs)

    def _lms(self, maps: Any, step: Any, rate: float, x: Any, c: Any) -> Any:
        """``_lms`` per life: each life's maps on its own row's step, no mean over the rows."""
        m_x, m_c, m = maps
        if m_x.ndim == 2:
            step = step[:, None]
            return (m_x + rate * x * step, m_c + rate * c * step, m + rate * step)
        return (
            m_x + rate * x[:, :, None] * step[:, None, :],
            m_c + rate * c[:, :, None] * step[:, None, :],
            m + rate * step,
        )

    def _position_step(self, found: tuple[Any, ...]) -> dict[str, Any]:
        """``_RecallOnDevice._position_step`` with the slope's rule read per life: past the
        table a life's lean takes no step, and a life's own slots decide it (the bank is as wide
        as the longest stream's)."""
        recall, xp = self.recall, self.xp
        stepped = super()._position_step(found)
        if not (self.slope_maps and self.grid_maps and self.slots > recall.position):
            return stepped
        _, x, c, _, _, _, target, prior, _, _ = found
        finite = xp.isfinite(prior)
        gradient = xp.where(finite, target - xp.where(finite, xp.exp(prior), 0.0), 0.0)
        energy = (x * x).sum(axis=1) + (c * c).sum(axis=1) + 1.0
        lean = gradient @ (self._ages() / recall.SLOPE)
        lean = lean * xp.minimum(xp.ones_like(energy), 1.0 / (recall.slope * energy))
        new = self._lms(self.slope_maps[0], lean, recall.slope, x, c)
        short = xp.flags(recall.held.sum(axis=1) <= recall.position)
        torch = self.kernel.torch
        stepped["slope_maps"] = tuple(
            _chosen(torch, short, a, b) for a, b in zip(new, self.slope_maps[0], strict=True)
        )
        return stepped

    def learn(self, graph: Any, learned: Any) -> dict[str, float]:
        """``Recall.learn`` per life: the phases hold every life's row, each taught life whose
        row moves steps its own maps, and a step that is not finite is refused for that life."""
        recall, xp = self.recall, self.xp
        torch = self.kernel.torch
        quiet = {"address_rows": 0.0, "address_error": 0.0, "address_refused": 0.0}
        if self.position_maps:
            quiet.update(position_rows=0.0, position_refused=0.0)
        moment = recall.last
        positions = bool(self.position_maps) and (recall.position_rate > 0 or recall.slope > 0)
        if (
            not (recall.address or positions)
            or moment is None
            or learned.free is None
            or learned.nudged is None
            or learned.nudged is learned.free
        ):
            return quiet
        lives = len(moment.address)
        taught = np.ones(lives, dtype=bool) if learned.taught is None else np.asarray(
            learned.taught, dtype=bool
        )
        if not self.slots or not taught.any() or self.streams != lives:
            return quiet
        minus = learned.free if learned.opposite is None else learned.opposite
        movement = self._movement(learned.nudged, minus)
        d_value, d_next = self._readbacks(graph, movement)
        found = self._row_steps(moment, np.arange(lives), d_value, d_next)
        moving = found[0] & xp.flags(taught)
        _, x, c, key, error, step = found[:6]
        zero = torch.zeros((), dtype=error.dtype, device=error.device)
        scalars = [moving.sum()]
        head = moved = None
        if recall.address:
            w_x, w_c, b = self._lms(self.head_maps[0], step, recall.address, x, c)
            address = key + self._term((w_x, w_c, b), x, c)
            moved = moving & _finite_rows(torch, w_x, w_c, b, address)
            head = tuple(
                _chosen(torch, moved, a, old)
                for a, old in zip((w_x, w_c, b), self.head_maps[0], strict=True)
            )
            scalars += [moved.sum(), torch.where(moved, error.abs().mean(dim=1), zero).sum()]
        placed: dict[str, tuple[Any, ...]] = {}
        if positions:
            stepped = self._position_step(found)
            sound = moving & _finite_rows(torch, *[a for maps in stepped.values() for a in maps])
            for name, maps in stepped.items():
                placed[name] = tuple(
                    _chosen(torch, sound, a, old)
                    for a, old in zip(maps, getattr(self, name)[0], strict=True)
                )
            scalars.append(sound.sum())
        read = xp.read(*scalars)
        report = dict(quiet)
        at = 1
        if head is not None:
            self.head_maps = [head]
            report["address_rows"] = float(read[at])
            report["address_error"] = float(read[at + 1] / read[at]) if read[at] else 0.0
            report["address_refused"] = float(read[0] - read[at])
            at += 2
        if positions:
            for name, maps in placed.items():
                setattr(self, name, [maps])
            report["position_rows"] = float(read[at])
            report["position_refused"] = float(read[0] - read[at])
        return report

    # -- reward trains each life's read

    def weighted(self, trace: Any, delta: Any) -> Any:
        """Each life's eligibility weighted by its own dopamine."""
        xp = self.xp
        d = xp.floats(delta) if isinstance(delta, type(trace)) else xp.here(
            np.asarray(delta, dtype=float)
        )
        return d[:, None, None] * trace

    def proposal_unread(self, step: Any) -> tuple[Any, ...] | None:
        """``_RecallOnDevice.proposal_unread`` of a step per life, ``(lives, inputs, columns)``."""
        recall, xp = self.recall, self.xp
        positions = bool(recall.position_reward) and bool(self.position_maps)
        if not (recall.address_reward or positions):
            return None
        w_x, w_c, b = self.head_maps[0]
        stepped, changed = [], []
        start = recall.address_width if recall._address_held or recall.address_reward else 0
        if recall.address_reward:
            moved = recall.address_reward * step[:, :, :recall.address_width]
            w_x = w_x + moved
            stepped.append(moved)
            changed.append(w_x)
        position_maps = anchor_maps = None
        if positions:
            count = recall.position_buckets
            part = recall.position_reward * step[:, :, start : start + count]
            on_x, on_c, on_b = self.position_maps[0]
            position_maps = [(on_x + part, on_c, on_b)]
            stepped.append(part)
            changed.append(position_maps[0][0])
            start += count
            if self.anchor_maps:
                part = recall.position_reward * step[:, :, start : start + count + 1]
                on_x, on_c, on_b = self.anchor_maps[0]
                anchor_maps = [(on_x + part, on_c, on_b)]
                stepped.append(part)
                changed.append(anchor_maps[0][0])
        every = [*changed, *stepped]
        norm = xp.sqrt(sum((part * part).sum() for part in stepped))
        return (w_x, w_c, b, xp.all_finite(*every), norm, every, position_maps, anchor_maps)


class _LivesHearing(Hearing):
    """``Hearing`` with one gain per life and route. A route's memory holds its gains as a
    ``(lives, 1)`` array on the device, which the memory's drive and the route's contrast read
    as they read one gain; the second moments and the counts are each life's own."""

    lives_gains: np.ndarray
    lives_second: np.ndarray
    lives_count: np.ndarray
    kernel: Any

    @classmethod
    def take(cls, hearing: Hearing, lone: Sequence[Hearing], kernel: Any) -> None:
        """Make ``hearing`` the lives': each life's gains, second moments and count from its
        lone brain's."""
        for route in hearing.routes:
            if route["slot"] is not None:
                raise ValueError("Lives holds the carried gains of the working trace and the read")
        hearing.__class__ = cls
        assert isinstance(hearing, cls)
        hearing.kernel = kernel
        hearing.lives_gains = np.array(
            [[one.gain(route) for route in one.routes] for one in lone], dtype=float
        )
        hearing.lives_second = np.stack([np.asarray(one.second, dtype=float) for one in lone])
        hearing.lives_count = np.array([one.count for one in lone], dtype=np.int64)
        hearing._place()

    def _place(self) -> None:
        """The gains where the device paths read them: ``(lives, 1)`` per route."""
        here = arrays_of(self.kernel).here
        for k, route in enumerate(self.routes):
            route["memory"].hearing = here(self.lives_gains[:, k : k + 1])

    def gain(self, route: dict[str, Any]) -> Any:
        return route["memory"].hearing

    def gains_of(self, life: int) -> dict[str, float]:
        """Life ``life``'s gains by route."""
        return {
            route["name"]: float(self.lives_gains[life, k]) for k, route in enumerate(self.routes)
        }

    def step(self, contrast: np.ndarray) -> None:
        raise RuntimeError("the lives' gains step per life (step_lives)")

    def step_lives(self, contrast: np.ndarray, rows: np.ndarray) -> None:
        """``Hearing.step`` for the lives ``rows``, each on its own row of ``contrast``."""
        self.lives_count[rows] += 1
        second = self.RHO * self.lives_second[rows] + (1.0 - self.RHO) * contrast**2
        correction = 1.0 - np.power(self.RHO, self.lives_count[rows])[:, None]
        moved = self.rate * contrast / (np.sqrt(second / correction) + self.FLOOR)
        bound = np.array([route["bound"] for route in self.routes])
        self.lives_gains[rows] = np.clip(self.lives_gains[rows] + moved, 0.0, bound)
        self.lives_second[rows] = second
        self._place()

    def learn(self, graph: Any, learned: Any, beta: float) -> None:
        """A lesson's step for the lives it taught; the phases hold every life's row."""
        if (
            not self.routes
            or learned.free is None
            or learned.nudged is None
            or learned.nudged is learned.free
        ):
            return
        minus_state = learned.free if learned.opposite is None else learned.opposite
        moved = self._movement_on_device(graph, learned.nudged, minus_state)
        if moved is None:
            raise RuntimeError("the lives' phases and memories rest on their device")
        rows = np.arange(len(moved)) if learned.taught is None else np.flatnonzero(
            np.asarray(learned.taught, dtype=bool)
        )
        if not len(rows):
            return
        span = beta if learned.opposite is None else 2.0 * beta
        contrast = self._contrast_on_device(graph, moved / span, None, len(moved))
        self.step_lives(contrast[rows], rows)


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

    def _hearing_step(self, delta: np.ndarray) -> None:
        """Each life's carried gains stepped on its own trace by its own dopamine."""
        if self.trace_hearing is not None and self.hearing is not None:
            self.hearing.step_lives(
                delta[:, None] * self.trace_hearing, np.arange(len(delta))
            )

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
        self._memory_eligibility(batch, decay)
        self._hearing_eligibility(batch, decay)
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
        # the recall memory's maps step on each life's own trace by its own dopamine
        unread = None
        traced, memory = self.trace_memory, self.memory
        if traced is not None and getattr(memory, "_device", None) is not None:
            unread = memory._device.proposal_unread(memory._device.weighted(traced, delta_here))
        one = torch.ones((), dtype=exact, device=trace.device)
        mapped = (one, 0.0 * one) if unread is None else (unread[3], unread[4])
        next_brain = self.learner.brain
        eta, eta_bias = self._rates()
        step_bias = eta_bias * (d[:, None] * trace_bias)
        step = decay * trace * d[:, None] + kernel.contrast_edges(s_plus, s_minus, d) / span
        update = self.learner._device_candidate(kernel, eta * step, step_bias)
        read = torch.cat([
            torch.stack([torch.isfinite(td_here).all().to(exact), critic_finite.to(exact),
                         mapped[0].to(exact), mapped[1].to(exact)]
                        + [flag.to(exact) for flag in update["flags"]]
                        + [number.to(exact) for number in update["steps"]]),
            critic_step.to(exact), td_here, delta_here,
        ]).cpu().numpy().astype(float)
        if not read[0]:
            raise ValueError("TD errors must remain finite")
        critic_bias = np.asarray(self.b_critic, dtype=float) + read[10 : 10 + batch]
        if not read[1] or not np.isfinite(critic_bias).all():
            raise ValueError("critic parameters must remain finite")
        if not read[2]:
            raise ValueError("the address map's reward step must be finite")
        self.learner._device_refusal(read[4:8])
        td_error = read[10 + batch : 10 + 2 * batch]
        delta = read[10 + 2 * batch :]
        capped = self._capped(td_error, observed)
        report = self.learner._install_device(update, read[8:10])
        trace.mul_(decay)
        kernel.add_contrast(trace, s_plus, s_minus, span, None)
        self._accept_critic((critic_weights, critic_bias))
        if unread is not None:
            memory.accept((*unread[:3], float(read[3]), *unread[6:]))
            report["address_reward_step"] = float(read[3])
        self._hearing_step(delta)
        if done.any():
            ended = torch.as_tensor(np.flatnonzero(done), device=trace.device)
            trace[ended] = 0.0
            trace_bias[ended] = 0.0
            self._forget_critic(done)
            if self.trace_memory is not None:
                self.trace_memory = _forgotten(self.trace_memory, done)
            if self.trace_hearing is not None:
                self.trace_hearing[done] = 0.0
        self._trace_device = (trace, trace_bias)
        self._free = next_state
        self._free_brain = next_brain
        self._drive = _held_drive(next_drive)
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


def _supported(brain: Any) -> None:
    """Refuse a composition with a part the lives axis does not cover, naming it."""

    def refuse(what: str) -> ValueError:
        return ValueError(f"Lives holds the parts its lives axis covers: {what}")

    if brain.hippocampus is not None:
        raise refuse(
            "compose with episodic=False (the store's consolidated records have no lives axis)"
        )
    for name, what in (
        ("arousal", "no arousal (Lives steps, acts and teaches; live is a lone brain's)"),
        ("learned_code", "no learned code"),
        ("_episodic", "no memory_choice"),
        ("efference", "no efference copy"),
        ("history", "no heard history"),
    ):
        if getattr(brain, name, None) is not None:
            raise refuse(what)
    if getattr(brain, "_coded_keys", False):
        raise refuse("no code of the observation (a code is drawn from a brain's seed)")
    recall = brain.recall
    if recall is not None and not recall.resides:
        raise refuse(
            "a recall memory of one head, without a window, the command, the read of when, "
            "tokens, their gain or the cortex at unit norm"
        )
    hearing = brain.hearing
    if hearing is not None and any(route["slot"] is not None for route in hearing.routes):
        raise refuse("the carried gains of the working trace and the read")
    learner, graph = brain.learner, brain.brain
    cfg = learner.config
    if cfg.rehearsal or cfg.qualified:
        raise refuse("no rehearsal (learning_rehearsal=0) and no qualified teaching")
    if graph.apical is not None:
        raise refuse("no apical compartment")
    agent = brain.basal_ganglia
    if agent.maturity is not None or agent.salience is not None:
        raise refuse("no maturity and no salience")
    if agent.config.entropy:
        raise refuse("no actor entropy (actor_entropy=0)")


class Lives:
    """Many continuing lives of one composed anatomy, each a brain of its own, stepped together.

    ``Lives(inputs, actions, lives=128, backend="torch")`` composes ``lives`` brains of
    ``Brain.compose(inputs, actions, seed=seeds[i], episodic=False, **options)``, the default
    composition without the store, one per seed (``range(lives)`` by default). Every operation
    takes one row of observations per life and returns one answer per life: ``act``, ``step``,
    ``lesson`` and ``learn`` mean what they mean on a lone brain with one stream, life by life.
    ``life(i)`` is life ``i`` as a lone brain on the CPU, with everything it learned.

    On ``backend="cpu"`` each life is its own ``Brain`` and the lives run in turn, to the bit as
    lone brains. On ``backend="torch"`` (CUDA where there is a card, else the torch CPU device, or
    the ``device`` named) they are the streams of one brain whose parameters carry a lives axis,
    and each moment of all of them is one settle and one update; each life agrees with its lone
    brain on that device within the solver's tolerance. The lives share their anatomy and nothing
    they learn. A moment any life refuses (a settle that does not qualify, a step that is not
    finite) is refused for every life.
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
        if not isinstance(inputs, (int, np.integer)):
            raise ValueError("Lives takes a vector of inputs, its length")
        options = {"episodic": False, **options}
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
        if device is None:  # CUDA where there is a card, else the torch CPU device (never MPS)
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
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
        agent.b_critic = np.array(  # type: ignore[assignment]  # one bias per life
            [b.basal_ganglia.b_critic for b in lone], dtype=float
        )
        agent.rng = _LivesRandom(  # type: ignore[assignment]  # each life's own generator
            [b.basal_ganglia.rng for b in lone]
        )
        # the memories' device paths read each stream through its own life's synapses and maps
        lives_kernel.__dict__["_arrays"] = _LivesArrays(lives_kernel)
        if brain.recall is not None:
            brain.recall._device = _LivesRecall(
                brain.recall, lives_kernel, [b.recall for b in lone]
            )
        if brain.hearing is not None:
            _LivesHearing.take(
                brain.hearing, [cast(Hearing, b.hearing) for b in lone], lives_kernel
            )
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

    def reset(self) -> None:
        """Every life's stream starts afresh, as ``Brain.reset``; what the lives learned stays."""
        if self.brains is not None:
            for brain in self.brains:
                brain.reset()
            return
        self.brain.reset()

    def parameters(self, i: int) -> dict[str, Any]:
        """Life ``i``'s learned parameters on the host, as a lone brain holds them: its
        ``efficacy`` and ``bias``; the critic's ``w_critic`` and ``b_critic``; the optimizer's
        ``velocity``, ``velocity_bias``, ``second_moment`` and ``second_moment_bias`` and its
        counts ``updates`` and ``contrast_updates``; with a recall memory ``recall``, its maps
        by kind; with carried gains ``carried_gains`` by route, their ``carried_second`` and
        ``carried_count``."""
        if self.brains is not None:
            b = self.brains[i]
            learner = b.learner
            out = {
                "efficacy": np.array(b.brain.efficacy), "bias": np.array(b.brain.bias),
                "w_critic": np.array(b.basal_ganglia.w_critic),
                "b_critic": np.array(b.basal_ganglia.b_critic, dtype=float),
                **{name: np.array(getattr(learner, name)) for name in _MOMENTS},
                "updates": int(learner.updates),
                "contrast_updates": int(learner.contrast_updates),
            }
            if b.recall is not None:
                out["recall"] = {
                    name: tuple(np.array(a) for a in getattr(b.recall, name)[0])
                    for name in _MAP_KINDS if getattr(b.recall, name)
                }
            if b.hearing is not None:
                out["carried_gains"] = b.hearing.gains()
                out["carried_second"] = np.array(b.hearing.second, dtype=float)
                out["carried_count"] = int(b.hearing.count)
            return out
        brain = self.brain
        kernel = brain.brain._torch
        agent = brain.basal_ganglia
        learner = brain.learner
        moments = learner._moments_on_device(kernel)
        resting = agent._critic_device
        weights = (resting.xp.host(resting.weights[i]) if resting is not None
                   else np.asarray(agent.w_critic)[i])
        out = {
            "efficacy": kernel.scale[i].detach().cpu().double().numpy(),
            "bias": kernel.bias_param[i].detach().cpu().double().numpy(),
            "w_critic": weights, "b_critic": np.asarray(agent.b_critic, dtype=float)[i],
            **{name: moments[name][i].detach().cpu().double().numpy() for name in _MOMENTS},
            "updates": int(learner._applied(self.lives)[i]),
            "contrast_updates": int(learner._counts(self.lives)[i]),
        }
        if brain.recall is not None:
            out["recall"] = brain.recall._device.maps_of(i)
        if brain.hearing is not None:
            out["carried_gains"] = brain.hearing.gains_of(i)
            out["carried_second"] = brain.hearing.lives_second[i].copy()
            out["carried_count"] = int(brain.hearing.lives_count[i])
        return out

    def life(self, i: int) -> Any:
        """Life ``i`` as a lone brain on the CPU: the composition of its seed with its learned
        parameters, optimizer, critic, recall maps, carried gains and random state; its stream
        starts afresh. Save it as any brain."""
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
        for name in _MOMENTS:
            setattr(lone.learner, name, held[name])
        lone.learner.updates = held["updates"]
        lone.learner.contrast_updates = held["contrast_updates"]
        lone.basal_ganglia.updates = self.brain.basal_ganglia.updates
        lone.basal_ganglia.w_critic = held["w_critic"]
        lone.basal_ganglia.b_critic = float(held["b_critic"])
        lone.basal_ganglia.rng.bit_generator.state = (
            self.brain.basal_ganglia.rng.generators[i].bit_generator.state
        )
        if lone.recall is not None:
            for name, maps in held["recall"].items():
                setattr(lone.recall, name, [maps])
        if lone.hearing is not None:
            for route in lone.hearing.routes:
                lone.hearing.set(route, held["carried_gains"][route["name"]])
            lone.hearing.second = held["carried_second"]
            lone.hearing.count = held["carried_count"]
        return lone
