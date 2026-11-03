"""Isolated readers for retained campaigns; live prerequisite guards stay strict.

Only freshly loaded checker modules receive these dependency views. No public
class, runtime module, receipt, model parameter or saved policy is changed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from cadence.receipts import Receipt

# The names a retained protocol or record gives the brain's genes where this library spells
# them otherwise: the campaign protocols of 0.88 to 0.92 name the learner's softmax temperature
# ``learning_temperature``, one spelling since 0.93.0, and carry the constants of the composed
# brain since 0.93.0 at their founders (``CONSTANTS``: no gene since). A retained protocol is
# bound by its receipts and keeps its bytes; ``genes`` reads it for ``Brain.compose`` and reads
# a record's resolved genes for a comparison with ``describe()["genes"]``.
RENAMED = {"learning_temperature": "temperature"}
CONSTANTS = {"learning_beta": 0.1, "learning_momentum": 0.9, "actor_dopamine_cap": 1.0,
             "actor_eligibility_steps": 12, "arousal_need_scale": 0.1, "memory_decay": 0.9,
             "recall_heard": 1.0, "recall_address_temperature": 0.1, "recall_window_decay": 1.0}


def genes(brain):
    """``brain`` (a protocol's brain or a record's resolved genes) under the current names: a
    constant at its founder is left out, at another value refused."""
    if "learning_temperature" in brain and "temperature" in brain:
        raise ValueError("a brain names the learner's softmax temperature once")
    out = {}
    for name, value in brain.items():
        if name in CONSTANTS:
            if value != CONSTANTS[name]:
                raise ValueError(
                    f"{name}={value!r} is a constant of the composed brain since 0.93.0, "
                    f"{CONSTANTS[name]!r}"
                )
            continue
        out[RENAMED.get(name, name)] = value
    return out


class _ConfigView:
    """The actual learner config as a retained record of 0.86.0 to 0.92.1 wrote it: with the
    numerical policy those releases carried (``initial_damping``, retired in 0.93.0) at its
    neutral zero."""
    def __init__(self, actual):
        self.actual = actual

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def to_dict(self):
        return {**self.actual.to_dict(), "initial_damping": 0}


class _FounderView:
    """Read-only expected-record view, used only inside one offline arithmetic check: the
    actual brain as the retained record wrote it, with the retired numerical policy at its
    neutral zero where the record carries it."""
    def __init__(self, actual, record):
        self.actual, self.record = actual, record
        self.learner = actual.learner
        if "initial_damping" in record["learning_config"]:
            self.learner = _LearnerView(actual.learner)

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def describe(self):
        value = self.actual.describe()
        if "learning_initial_damping" in self.record["resolved"]["genes"]:
            value["genes"]["learning_initial_damping"] = 0
        return value


class _LearnerView:
    def __init__(self, actual):
        self.actual, self.config = actual, _ConfigView(actual.config)

    def __getattr__(self, name):
        return getattr(self.actual, name)


class Session:
    """A private module tree, with exact historical source bytes for each nested receipt."""
    def __init__(self, root, resolve, reference):
        self.root, self.resolve = Path(root), resolve
        self.reference = Receipt.read(reference)
        valid, message = Receipt.verify(reference, sources=resolve(self.reference))
        if not valid:
            raise ValueError("historical reference custody differs: " + message)
        self.stack, self.installed = [], set()
        self.receipts = SimpleNamespace(
            read=Receipt.read, build=Receipt.build, verify=self.verify_receipt)

    @property
    def current(self):
        return self.stack[-1] if self.stack else self.reference

    def bound_paths(self, paths):
        recorded = dict(self.resolve(self.current))
        return [(name, recorded.get(name, path)) for name, path in paths]

    def dependency_problem(self, path, receipt):
        if receipt.digest == self.current.digest:
            return None
        try:
            relative = Path(path).resolve().relative_to(self.root).as_posix()
        except ValueError:
            return "historical dependency is outside the retained repository"
        parent = {row["path"]: row["sha256"] for row in self.current.source["files"]}
        if (relative not in parent
                or parent[relative] != hashlib.sha256(Path(path).read_bytes()).hexdigest()):
            return "historical dependency bytes differ from the parent receipt"
        return None

    def verify_receipt(self, path, *, sources=None, check=None):
        receipt = Receipt.read(path)
        if sources is not None and [name for name, _ in sources] != [
                row["path"] for row in receipt.source["files"]]:
            return False, "historical source inventory differs"
        problem = self.dependency_problem(path, receipt)
        if problem:
            return False, problem
        try:
            bound = self.resolve(receipt)
        except (ValueError, OSError) as error:
            return False, str(error)
        self.stack.append(receipt)
        try:
            return Receipt.verify(path, sources=bound, check=check)
        finally:
            self.stack.pop()

    def load(self, filename):
        spec = importlib.util.spec_from_file_location(
            "offline_attention_" + Path(filename).stem,
            self.root / "benchmarks/attention" / filename)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if filename in ("minimal_binding_campaign.py", "single_relevance_campaign.py",
                        "neutral_variation_campaign.py"):
            capacity = {"minimal_binding_campaign.py": "minimal-binding-capacity.json",
                        "single_relevance_campaign.py": "single-relevance-capacity.json",
                        "neutral_variation_campaign.py":
                            "neutral-variation-capacity.json"}[filename]
            module.configure(self.root, self.root / "benchmarks/attention/results" / capacity)
        expected = ({module.KIND} if filename == "heads_continuation.py" else {
            stage["receipt_kind"] for stage in module.declared()["stages"].values()})
        if self.reference.kind not in expected:
            raise ValueError("historical reference kind differs from its checker")
        self.install(module)
        return module

    def install(self, module):
        if id(module) in self.installed:
            return
        self.installed.add(id(module))
        if hasattr(module, "Receipt"):
            module.Receipt = self.receipts
        for name in ("relevance", "admission"):
            child = getattr(module, name, None)
            if child is not None and hasattr(child, "__file__"):
                self.install(child)
        if hasattr(module, "context"):
            context = module.context()
            for name in ("owner", "base", "campaign", "admission", "relevance"):
                child = getattr(context, name, None)
                if child is not None:
                    self.install(child)
        if hasattr(module, "h1_verifier"):
            original_h1 = module.h1_verifier

            def h1_verifier():
                child = original_h1()
                self.install(child)
                return child

            module.h1_verifier = h1_verifier
        if hasattr(module, "sources"):
            original_sources = module.sources
            module.sources = lambda *args, **kwargs: self.bound_paths(
                original_sources(*args, **kwargs))
        if hasattr(module, "numerical_check"):
            original_numerical = module.numerical_check
            module.numerical_check = lambda result, protocol: original_numerical(
                result, protocol, historical=True)
        if Path(module.__file__).name == "single_relevance_campaign.py":
            original_check, factory = module.check, module.make_brain
            original_panel_check = module.panel_check

            def panel_check(panel, causal=False, *, brain=None, arm=None):
                # Native boundary checks depend on the actual oracle class.
                actual = brain.actual if isinstance(brain, _FounderView) else brain
                return original_panel_check(panel, causal, brain=actual, arm=arm)

            module.panel_check = panel_check

            def check(body):
                enclosing_factory = module.make_brain
                records = {(row["arm"], row["seed"]): row for row in body["runs"]
                           if row.get("status") != "unrun_budget"}
                for row in records.values():
                    for values, key in ((row["learning_config"], "initial_damping"),
                                        (row["resolved"]["genes"], "learning_initial_damping")):
                        if key in values and (type(values[key]) is not int or values[key] != 0):
                            return "historical numerical policy is not the neutral default"
                module.make_brain = lambda arm, seed: _FounderView(
                    factory(arm, seed), records[(arm, seed)])
                try:
                    return original_check(body)
                finally:
                    module.make_brain = enclosing_factory

            module.check = check
        if hasattr(module, "verify"):
            original_verify = module.verify

            def verify(path):
                if path is None:
                    return original_verify(path)
                receipt = Receipt.read(path)
                problem = self.dependency_problem(path, receipt)
                if problem:
                    return False, problem
                self.stack.append(receipt)
                try:
                    return original_verify(path)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    return False, str(error)
                finally:
                    self.stack.pop()

            module.verify = verify
