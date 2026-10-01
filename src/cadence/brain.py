"""Run joint repair, admit labeled targets and preserve brain continuation.

Qualification means constrained stationarity of the declared nonlinear energy,
not a unique equilibrium, zero prediction error or a demonstrated depth benefit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib.resources import files
from types import MappingProxyType

from . import _repair
from ._validation import canonical, integer, number, strict_json
from .ports import _values

SCHEMA = "population-brain/1"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
IMPLEMENTATION = MappingProxyType(
    {
        name: hashlib.sha256(files(__package__).joinpath(name).read_bytes()).hexdigest()
        for name in (
            "brain.py",
            "cortex.py",
            "column.py",
            "ports.py",
            "_repair.py",
            "_validation.py",
            "_tensor.py",
        )
    }
)


def _coverage(graph, output_indices):
    """Count potential sensor paths through coupled patch components.

    State/error contacts carry returning energy influence. Fixed input samples
    attach to components but never join otherwise independent patches.
    """
    parents = list(range(graph.n_patches))
    sizes = [1] * graph.n_patches

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for kind, source, target in graph.edges:
        if kind != "input":
            left, right = root(source), root(target)
            if left != right:
                if sizes[left] < sizes[right]:
                    left, right = right, left
                parents[right] = left
                sizes[left] += sizes[right]
    sensors = {}
    for kind, source, target in graph.edges:
        if kind == "input":
            sensors.setdefault(root(target), set()).add(source)
    coverage = tuple(len(sensors.get(root(i), ())) for i in range(graph.n_patches))
    connected = sum(sizes[i] for i in {root(i) for i in output_indices})
    return coverage, connected


class SettlementError(RuntimeError):
    """A population solve did not meet its full stationarity threshold."""


class Brain:
    """A compiled layout with private parameters, live state and atomic admission.

    Construct with ``Cortex.build`` or ``Brain.from_snapshot``. Query methods
    freeze parameters; ``step`` can retain qualified live state. ``observe``
    jointly repairs live state and local relations under output targets, then
    commits only a qualified complete proposal. Targets are actual witnesses by
    default; derived estimates must use ``source="estimate"``. ``observe_batch``
    repairs private experience states and shared parameters, preserving live
    activity. These methods do not assign reward or temporal credit themselves.
    """

    def __init__(self, builder, graph, weights, population_ranges, layout):
        self._config = MappingProxyType(dict(builder.config))
        self._graph = graph
        self._inputs = tuple(builder._inputs)
        self._populations = tuple(builder._populations)
        self._outputs = tuple(builder._outputs)
        self._population_ranges = population_ranges
        self._layout = canonical(layout)
        self._weights = tuple(weights)
        self._biases = (0.0,) * graph.n_patches
        self._state = (0.0,) * graph.n_patches
        self._event_id = -1
        self._event_digest = None
        self._admissions = 0
        self._engine = None
        self._fingerprint = hashlib.sha256(
            canonical(
                {"layout": layout, "config": dict(self.config), "edges": graph.edges}
            ).encode()
        ).hexdigest()

    @property
    def config(self):
        """Read-only resolved configuration, bound into checkpoints."""
        return self._config

    @property
    def graph(self):
        """Immutable topology with input, state and residual connection kinds."""
        return self._graph

    @property
    def state(self):
        """Immutable current live patch values, in layout declaration order."""
        return self._state

    @property
    def weights(self):
        """Immutable local relation coefficients, aligned with ``graph.edges``."""
        return self._weights

    @property
    def biases(self):
        """Immutable local prediction offsets, one per patch."""
        return self._biases

    def _mapping(self, supplied, expected, *, partial=False):
        if not isinstance(supplied, Mapping):
            raise ValueError("Values must map layout names or references to samples")
        expected = {node.name: node for node in expected}
        result = {}
        for key, value in supplied.items():
            name = key if isinstance(key, str) else getattr(key, "name", None)
            if (
                not isinstance(name, str)
                or name not in expected
                or (not isinstance(key, str) and key is not expected[name])
            ):
                raise ValueError(
                    f"Unknown or foreign layout reference {name!r}; "
                    f"expected names: {', '.join(expected) or '(none)'}"
                )
            if name in result:
                raise ValueError(f"Duplicate values for {name!r}")
            result[name] = value
        if not partial and result.keys() != expected.keys():
            missing = ", ".join(name for name in expected if name not in result)
            raise ValueError(
                f"Supply every declared sensor exactly once; missing: {missing}"
            )
        return result

    def _arguments(self, inputs, targets=None, interventions=None):
        supplied = self._mapping(inputs, self._inputs)
        flat = tuple(
            v
            for source in self._inputs
            for v in _values(supplied[source.name], source.shape, source.name)
        )
        clamps = {}

        def add(indices, values, name):
            for index, value in zip(indices, values, strict=True):
                if abs(value) > self.config["state_bound"]:
                    raise ValueError(
                        f"{name!r} clamp {value} exceeds state_bound="
                        f"{self.config['state_bound']}; scale targets/interventions "
                        "to the model's state range"
                    )
                if index in clamps and clamps[index] != value:
                    raise ValueError(
                        f"{name!r} conflicts with another clamp on patch {index}"
                    )
                clamps[index] = value

        if targets is not None:
            supplied = self._mapping(targets, self._outputs, partial=True)
            for output in self._outputs:
                if output.name in supplied:
                    indices = tuple(
                        self._population_ranges[output.reads.name][i]
                        for i in output.indices
                    )
                    add(
                        indices,
                        _values(supplied[output.name], output.shape, output.name),
                        output.name,
                    )
        if interventions is not None:
            supplied = self._mapping(interventions, self._populations, partial=True)
            for population in self._populations:
                if population.name in supplied:
                    add(
                        self._population_ranges[population.name],
                        _values(
                            supplied[population.name],
                            (population.patches,),
                            population.name,
                        ),
                        population.name,
                    )
        return flat, clamps

    def _solve(self, inputs, clamps, *, learn, budget, batch_size=1):
        config = self.config
        budget = (
            config["settle_budget"] if budget is None else integer(budget, "budget")
        )
        if config["device"] != "python" and self._engine is None:
            from ._tensor import TensorEngine

            self._engine = TensorEngine(self.graph, config["device"], config["dtype"])
        result = _repair.settle(
            self.graph,
            inputs,
            self._state * batch_size,
            self._weights,
            self._biases,
            clamps=clamps,
            learn=learn,
            budget=budget,
            tolerance=config["tolerance"],
            state_prior=config["state_prior"],
            parameter_prior=config["parameter_prior"],
            state_bound=config["state_bound"],
            parameter_bound=config["parameter_bound"],
            step=config["step"],
            backtracks=config["backtracks"],
            _engine=self._engine,
            _batch_size=batch_size,
        )
        result["outputs"] = self._outputs_from(result["state"])
        return result

    def settle(self, inputs, *, targets=None, interventions=None, budget=None):
        """Query a full coupled solve without changing live or durable state.

        Optional target/intervention clamps are hypothetical diagnostics here;
        they never become learning evidence. Outputs are flat tuples in declared
        shape order. Inspect ``qualified`` before using them.
        """
        flat, clamps = self._arguments(inputs, targets, interventions)
        return self._solve(flat, clamps, learn=False, budget=budget)

    def predict(self, inputs, *, budget=None):
        """Return qualified output values without admitting experience."""
        result = self.settle(inputs, budget=budget)
        if not result["qualified"]:
            raise SettlementError(
                f"Population solve refused: {result['reason']}; "
                f"stationarity={result['stationarity']:.6g}, "
                f"tolerance={self.config['tolerance']:.6g}, "
                f"sweeps={result['sweeps']}. "
                "Use settle() to inspect work and diagnostic outputs."
            )
        return result["outputs"]

    def step(self, inputs, *, targets=None, interventions=None, budget=None):
        """Retain qualified activity with frozen relations and optional clamps.

        Targets and interventions condition this solve as in ``settle``. They
        are not teaching evidence and do not advance admission/event history.
        Qualification covers every eligible free state under these clamps;
        a goal-clamped output is not a prediction of that outcome. Clamps apply
        only to this call. Refusal or invalid arguments retain no proposed state.
        """
        result = self.settle(
            inputs, targets=targets, interventions=interventions, budget=budget
        )
        if result["qualified"]:
            self._state = tuple(result["state"])
        return {**result, "accepted": result["qualified"]}

    def observe(self, inputs, targets, *, event_id=None, budget=None, source="witness"):
        """Jointly repair and atomically retain a labeled input/target experience.

        Targets clamp at least one declared output. ``source="witness"`` labels
        actual observations; ``source="estimate"`` labels derived teaching values,
        not observed facts. Both use identical repair. The caller supplies this
        provenance label; the brain does not independently authenticate it.
        Results include ``source``, including refusals and duplicate retries.

        Ordered nonnegative event IDs recognize an identical latest retry with
        the same source; changed or older IDs are rejected. Omitting the ID
        allocates the next one only on acceptance.
        A refusal leaves live state, parameters and event ownership unchanged.
        """
        if not isinstance(source, str) or source not in ("witness", "estimate"):
            raise ValueError("source must be 'witness' or 'estimate'")
        if not isinstance(targets, Mapping) or not targets:
            raise ValueError("Observation requires at least one output target")
        flat, clamps = self._arguments(inputs, targets)
        event_id, digest, duplicate = self._event(
            [flat, sorted(clamps.items())], event_id, source
        )
        if duplicate:
            return duplicate
        result = self._solve(flat, clamps, learn=True, budget=budget)
        if result["qualified"]:
            self._state = tuple(result["state"])
            self._admit(result, event_id, digest)
        return {
            **result,
            "accepted": result["qualified"],
            "duplicate": False,
            "event_id": event_id,
            "source": source,
        }

    def observe_batch(self, examples, *, event_id=None, budget=None, source="witness"):
        """Jointly learn a batch with private activities and shared parameters.

        ``examples`` is a nonempty finite sequence of ``(inputs, targets)``
        pairs using the same boundaries as ``observe``. Every row starts from
        the same retained activity; output targets clamp its private state.
        Repair minimizes mean example energy plus one fixed parameter anchor
        penalty. Shared parameters commit only when the whole batch qualifies.
        The live activity is preserved; rows do not form a temporal sequence.

        One batch owns one event ID. Latest-event retries must match all rows
        in the same order. Invalid or refused batches change nothing. Returned
        ``states``, ``outputs``, ``predictions`` and ``errors`` contain one entry
        per example; other solve diagnostics describe the complete batch.
        Batch grouping changes the learning objective versus serial admission.

        ``source="witness"`` labels actual observations; ``source="estimate"``
        labels derived targets for every row, using the same repair. The source
        is caller-declared provenance, not authentication. It is bound into the
        event identity and returned for accepted, refused and duplicate results.
        """
        if not isinstance(source, str) or source not in ("witness", "estimate"):
            raise ValueError("source must be 'witness' or 'estimate'")
        if (
            isinstance(examples, (str, bytes))
            or not isinstance(examples, Sequence)
            or not examples
        ):
            raise ValueError("examples must be a nonempty finite sequence of pairs")
        records = []
        for index, pair in enumerate(examples):
            try:
                if (
                    isinstance(pair, (str, bytes))
                    or not isinstance(pair, Sequence)
                    or len(pair) != 2
                ):
                    raise ValueError("Each example must be an (inputs, targets) pair")
                if not isinstance(pair[1], Mapping) or not pair[1]:
                    raise ValueError("Supply at least one output target")
                flat, clamps = self._arguments(*pair)
                records.append((flat, sorted(clamps.items())))
            except ValueError as error:
                raise ValueError(f"examples[{index}]: {error}") from error
        size, width = len(records), self.graph.n_patches
        event_id, digest, duplicate = self._event(["batch", records], event_id, source)
        if duplicate:
            return {**duplicate, "batch_size": size}
        inputs = tuple(v for flat, _ in records for v in flat)
        clamps = {
            row * width + i: value
            for row, (_, fixed) in enumerate(records)
            for i, value in fixed
        }
        result = self._solve(inputs, clamps, learn=True, budget=budget, batch_size=size)
        for field, target in (
            ("state", "states"),
            ("predictions", "predictions"),
            ("errors", "errors"),
        ):
            values = result.pop(field)
            result[target] = tuple(
                tuple(values[row * width : (row + 1) * width]) for row in range(size)
            )
        result["outputs"] = tuple(self._outputs_from(s) for s in result["states"])
        if result["qualified"]:
            self._admit(result, event_id, digest)
        return {
            **result,
            "accepted": result["qualified"],
            "duplicate": False,
            "event_id": event_id,
            "batch_size": size,
            "source": source,
        }

    def _outputs_from(self, state):
        return {
            output.name: tuple(
                state[self._population_ranges[output.reads.name][i]]
                for i in output.indices
            )
            for output in self._outputs
        }

    def _event(self, payload, event_id, source):
        """Validate an atomic event and identify a latest-event retry."""
        if source == "estimate":
            payload = {"source": source, "content": payload}
        event_id = (
            self._event_id + 1 if event_id is None else integer(event_id, "event_id")
        )
        try:
            canonical(event_id)
        except ValueError as error:
            raise ValueError(
                "event_id must be representable as a JSON integer"
            ) from error
        digest = hashlib.sha256(canonical(payload).encode()).hexdigest()
        if event_id <= self._event_id:
            if event_id == self._event_id and digest == self._event_digest:
                return (
                    event_id,
                    digest,
                    {
                        "accepted": False,
                        "qualified": True,
                        "duplicate": True,
                        "event_id": event_id,
                        "source": source,
                    },
                )
            raise ValueError(
                "Event is older than, or conflicts with, the latest admitted event"
            )
        return event_id, digest, None

    def _admit(self, result, event_id, digest):
        self._weights = tuple(result["weights"])
        self._biases = tuple(result["biases"])
        self._event_id, self._event_digest = event_id, digest
        self._admissions += 1

    def inspect(self):
        """Return owned layout data, graph counts and potential input paths.

        Each output's ``sensor_coverage_by_coordinate`` counts input samples
        in its coupled patch component. ``output_connected_patches`` counts
        patches in any output component. These are structural possibilities,
        not measured influence: weights or saturation can suppress a path.
        """
        layout = json.loads(self._layout)
        for population in layout["populations"]:
            population["role"] = "observer" if population["observes"] else "processing"
            population["indices"] = tuple(self._population_ranges[population["name"]])
        used = {source for kind, source, _ in self.graph.edges if kind == "input"}
        indices = [
            tuple(self._population_ranges[o["reads"]][i] for i in o["indices"])
            for o in layout["outputs"]
        ]
        coverage, connected = _coverage(self.graph, {i for row in indices for i in row})
        for output, row in zip(layout["outputs"], indices, strict=True):
            output["sensor_coverage_by_coordinate"] = tuple(coverage[i] for i in row)
        return {
            **layout,
            "config": dict(self.config),
            "patches": self.graph.n_patches,
            "input_samples": self.graph.n_inputs,
            "connections": len(self.graph.edges),
            "edges": self.graph.edges,
            "observed_fields": ("state", "prediction_error"),
            "sensor_coverage": len(used),
            "output_connected_patches": connected,
            "fingerprint": self._fingerprint,
            "implementation": dict(IMPLEMENTATION),
            "admissions": self._admissions,
            "last_event_id": self._event_id,
        }

    def snapshot(self):
        """Serialize the entire layout and continuation state as validated JSON."""
        result = canonical(
            {
                "schema": SCHEMA,
                "implementation": dict(IMPLEMENTATION),
                "config": dict(self.config),
                "layout": json.loads(self._layout),
                "fingerprint": self._fingerprint,
                "state": self._state,
                "weights": self._weights,
                "biases": self._biases,
                "event_id": self._event_id,
                "event_digest": self._event_digest,
                "admissions": self._admissions,
            }
        )
        if len(result.encode()) > MAX_CHECKPOINT_BYTES:
            raise ValueError("Checkpoint exceeds text-size budget")
        return result

    @classmethod
    def from_snapshot(cls, text, *, device=None, dtype=None):
        """Validate continuation, optionally selecting a different execution device.

        Overrides are applied only after validating the original checkpoint.
        Changing device without a dtype chooses that device's default precision.
        The retained values stay unchanged; later numerical trajectories can differ.
        """
        # Restore uses the ordinary builder; defer the import to avoid a cycle.
        from .cortex import Cortex

        data = strict_json(text, MAX_CHECKPOINT_BYTES)
        fields = {
            "schema",
            "implementation",
            "config",
            "layout",
            "fingerprint",
            "state",
            "weights",
            "biases",
            "event_id",
            "event_digest",
            "admissions",
        }
        if (
            not isinstance(data, dict)
            or set(data) != fields
            or data["schema"] != SCHEMA
        ):
            raise ValueError("Unsupported population checkpoint")
        if data["implementation"] != IMPLEMENTATION:
            raise ValueError("Checkpoint repair/layout implementation mismatch")
        try:
            builder = Cortex(**data["config"])
            if set(data["config"]) != set(builder.config):
                raise ValueError("Checkpoint must include the complete configuration")
            nodes = {}
            layout = data["layout"]
            if set(layout) != {"inputs", "populations", "outputs"}:
                raise ValueError("Invalid layout fields")
            if any(
                not isinstance(data[key], list)
                for key in ("state", "biases", "weights")
            ):
                raise ValueError("Checkpoint parameters must be arrays")
            patch_count = sum(
                integer(record["patches"], "patches", 1)
                for record in layout["populations"]
            )
            if patch_count != len(data["state"]) or patch_count != len(data["biases"]):
                raise ValueError("State length does not match declared patch count")
            for record in layout["inputs"]:
                node = builder.input(**record)
                nodes[node.name] = node
            for record in layout["populations"]:
                if set(record) != {"name", "patches", "inputs", "observes"}:
                    raise ValueError("Invalid population fields")
                inputs = tuple(nodes[n] for n in record["inputs"])
                observes = tuple(nodes[n] for n in record["observes"])
                method = builder.observer if observes else builder.column
                extra = {"observes": observes} if observes else {}
                node = method(
                    record["name"], patches=record["patches"], inputs=inputs, **extra
                )
                nodes[node.name] = node
            for record in layout["outputs"]:
                if set(record) != {"name", "shape", "reads", "indices"}:
                    raise ValueError("Invalid output fields")
                builder.output(
                    record["name"],
                    shape=record["shape"],
                    reads=nodes[record["reads"]],
                    indices=record["indices"],
                )
            brain = builder._compile(
                min(builder.config["max_connections"], len(data["weights"]))
            )
            if brain._fingerprint != data["fingerprint"]:
                raise ValueError("Checkpoint layout/configuration fingerprint mismatch")
            initial_weights = brain.weights
            for key, length, bound in (
                ("state", brain.graph.n_patches, brain.config["state_bound"]),
                ("weights", len(brain.graph.edges), brain.config["parameter_bound"]),
                ("biases", brain.graph.n_patches, brain.config["parameter_bound"]),
            ):
                raw = data[key]
                if len(raw) != length:
                    raise ValueError(f"Invalid {key} length")
                values = tuple(number(v, key) for v in raw)
                if any(abs(v) > bound for v in values):
                    raise ValueError(f"Invalid {key} bound")
                setattr(brain, "_" + key, values)
            event_id = integer(data["event_id"], "event_id", -1)
            admissions = integer(data["admissions"], "admissions")
            digest = data["event_digest"]
            if (admissions == 0) != (event_id == -1) or admissions > event_id + 1:
                raise ValueError("Invalid event ownership")
            if event_id == -1:
                if digest is not None:
                    raise ValueError("Unexpected event digest")
                if brain.weights != initial_weights or any(brain.biases):
                    raise ValueError(
                        "Retained parameters changed without an admitted event"
                    )
            elif (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid event digest")
            brain._event_id, brain._event_digest, brain._admissions = (
                event_id,
                digest,
                admissions,
            )
            if device is not None or dtype is not None:
                config = dict(brain.config)
                if device is not None:
                    config.update(device=device, dtype=dtype)
                else:
                    config["dtype"] = dtype
                brain._config = Cortex(**config).config
                brain._fingerprint = hashlib.sha256(
                    canonical(
                        {
                            "layout": layout,
                            "config": dict(brain.config),
                            "edges": brain.graph.edges,
                        }
                    ).encode()
                ).hexdigest()
            return brain
        except (KeyError, TypeError, OverflowError, AttributeError) as error:
            raise ValueError("Malformed population checkpoint") from error

    def restore(self, text):
        """Atomically replace continuation state for this exact layout/config."""
        proposal = type(self).from_snapshot(text)
        if proposal._fingerprint != self._fingerprint:
            raise ValueError("Cannot restore a different population layout")
        self._state, self._weights, self._biases = (
            proposal.state,
            proposal.weights,
            proposal.biases,
        )
        self._event_id = proposal._event_id
        self._event_digest = proposal._event_digest
        self._admissions = proposal._admissions
