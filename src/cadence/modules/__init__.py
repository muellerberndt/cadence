"""Trained modules: regions that arrive with their synapses trained and are grafted into a brain.

A module file is a NumPy archive of arrays and one JSON record, with no pickled object in it:

- the arrays of the brain the module was trained in (a ``Brain.save`` checkpoint, its floats in
  32 bits);
- ``module``, the record: the format (``cadence-module/1``), the module's name, the library
  version that wrote it, its regions and their sizes, which of them are inputs their own senses
  drive and which a host reads, a record of the module's own (``interface``), a card (what it
  was trained on and its readings) and a checksum of every array.

``load_file`` reads a file (``cadence.load_module``), ``names`` and ``load`` the modules the
library ships beside this file, and ``save`` (``Brain.save_module``) writes one. Loading runs
no code from the file: it checks the record and every checksum and refuses a file by what is
wrong with it. ``Brain.graft(module)`` grafts a module into a brain that holds its regions, and
``Brain.compose(..., trained=[module])`` builds a brain around it. Both keep the module's
synapses as they are unless asked otherwise.

A host holds the module's whole trained brain, every population of it as ``name/...``; the
record's ``regions`` are the ones it describes. A hosted module hears nothing of its host: the
host reads what the module publishes through one-way synapses and no synapse leads into the
module, so its answers in a host are its answers alone, and ``freeze=False`` moves nothing in
it. ``graft(module)`` with ``joints="zero"`` (the founder) starts the host's synapses onto the
module at zero, also where the host had learned to read an older file; ``joints="kept"`` keeps
them. A brain composed around a module records no construction, so ``grow`` refuses it.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT = "cadence-module/1"
_HERE = Path(__file__).resolve().parent


@dataclass
class Module:
    """A trained module as its file holds it: ``record`` and the trained brain's ``arrays``."""

    record: dict[str, Any]
    arrays: dict[str, np.ndarray]
    _brains: dict[tuple[Any, ...], Any] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return str(self.record["name"])

    @property
    def regions(self) -> dict[str, int]:
        """Every region of the module and its size, in the trained brain's order."""
        return {str(k): int(v) for k, v in self.record["regions"].items()}

    @property
    def inputs(self) -> list[str]:
        """The regions the module's own senses drive: inputs of a brain that holds it."""
        return [str(name) for name in self.record["inputs"]]

    @property
    def published(self) -> list[str]:
        """The regions a host brain reads."""
        return [str(name) for name in self.record["published"]]

    @property
    def interface(self) -> dict[str, Any]:
        """The module's own record of what its regions stand for."""
        return dict(self.record.get("interface") or {})

    @property
    def card(self) -> dict[str, Any]:
        return dict(self.record.get("card") or {})

    @property
    def width(self) -> int:
        """The inputs the module adds to a brain's observation."""
        sizes = self.regions
        return sum(sizes[name] for name in self.inputs)

    def brain(self, *, backend: Any = None, device: str | None = None,
              precision: str | None = None) -> Any:
        """The brain the module was trained in, loaded from the file's arrays: on the CPU
        unless a backend is named. Every call gives a brain of its own, as the file holds it:
        lessons given to one never reach another."""
        import copy

        from ..generic import Brain

        key = (backend, device, precision)
        if key not in self._brains:
            with tempfile.TemporaryDirectory(prefix="cadence-module-") as folder:
                path = Path(folder) / "brain.npz"
                held: dict[str, Any] = dict(self.arrays)
                with path.open("wb") as handle:
                    np.savez(handle, **held)
                self._brains[key] = Brain.load(path, backend=backend, device=device,
                                               precision=precision)
        return copy.deepcopy(self._brains[key])

    def columns(self, brain: Any, prefix: str | None = None) -> np.ndarray:
        """Where the module's inputs lie in ``brain``'s observation: one column an input
        neuron, the input regions in order. ``prefix`` names the module's populations in the
        brain (``prefix/region``; the module's name unless given, none for ``""``)."""
        populations = brain.connectome.populations
        lead = self.name if prefix is None else prefix
        lead = f"{lead}/" if lead else ""
        for name in self.inputs:
            if lead + name not in populations:
                raise ValueError(f"this brain has no population {lead + name!r}: it does not "
                                 f"hold the module {self.name!r}")
        sensory = np.asarray(brain.sensory_index, dtype=np.intp)
        column = np.full(int(sensory.max()) + 1, -1, dtype=np.intp)
        column[sensory] = np.arange(len(sensory), dtype=np.intp)
        neurons = np.concatenate([np.asarray(populations[lead + name], dtype=np.intp)
                                  for name in self.inputs])
        return np.asarray(column[neurons])


def _checksum(array: np.ndarray) -> str:
    held = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(f"{held.dtype.str}{held.shape}".encode())
    digest.update(held.tobytes())
    return digest.hexdigest()


def save(brain: Any, path: str | Path, *, name: str, inputs: Any, published: Any,
         regions: Any = None, interface: dict[str, Any] | None = None,
         card: dict[str, Any] | None = None) -> Path:
    """Write ``brain``'s trained regions as a module file and return its path.

    ``regions`` names the populations the module is made of (the inputs, the published and
    every region between them; where omitted, ``inputs`` and ``published`` alone), ``inputs``
    those its own senses drive and ``published`` those a host reads. ``interface`` is the
    module's own record and ``card`` says what it was trained on and reads. The brain is saved
    as ``Brain.save`` saves it, its streams at rest and without per-synapse steps of its own,
    with floats in 32 bits and synapse indices in 32 bits."""
    import copy

    from .. import __version__

    path = Path(path)
    if not isinstance(name, str) or not name or "/" in name:
        raise ValueError("a module needs a nonempty name without '/'")
    if getattr(brain, "learned_code", None) is not None:
        raise ValueError("a module takes a brain without a learned code in this version: the "
                         "code's dictionary does not keep its unit lengths in 32 bits")
    populations = brain.connectome.populations
    inputs, published = [str(n) for n in inputs], [str(n) for n in published]
    named = [str(n) for n in regions] if regions is not None else []
    order = list(dict.fromkeys([*inputs, *named, *published]))
    for region in order:
        if region not in populations:
            raise ValueError(f"this brain has no population {region!r}")
    if not inputs or not published:
        raise ValueError("a module names the inputs its senses drive and the regions it publishes")
    sensory = set(np.asarray(brain.sensory_index).tolist())
    for region in inputs:
        if not set(populations[region]) <= sensory:
            raise ValueError(f"{region!r} is no input of this brain")
    resting = copy.deepcopy(brain)
    resting.reset()
    resting.learner.synapse_rate = None  # a module's steps are its host's (``graft``)
    with tempfile.TemporaryDirectory(prefix="cadence-module-") as folder:
        whole = Path(folder) / "brain.npz"
        resting.save(whole, compressed=False)
        with np.load(whole, allow_pickle=False) as held:
            arrays = {key: held[key] for key in held.files}
    for key, value in arrays.items():
        if value.dtype == np.float64:
            arrays[key] = value.astype(np.float32)
        elif value.dtype == np.int64 and value.size > 64 and np.abs(value).max(initial=0) < 2**31:
            arrays[key] = value.astype(np.int32)
    record = {
        "format": FORMAT, "name": name, "library": __version__,
        "regions": {region: len(populations[region]) for region in order},
        "inputs": inputs, "published": published, "neurons": int(brain.connectome.n),
        "interface": interface or {}, "card": card or {},
        "checksums": {key: _checksum(value) for key, value in sorted(arrays.items())},
    }
    written: dict[str, Any] = {"module": np.array(json.dumps(record)), **arrays}
    with path.open("wb") as handle:
        np.savez_compressed(handle, **written)
    return path


def load_file(path: str | Path) -> Module:
    """Read a module file. Nothing in the file is run: the archive is read without pickled
    objects, the record is checked field by field and every array against its checksum, and a
    file that fails is refused by what is wrong with it."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as held:
            arrays = {key: held[key] for key in held.files}
    except Exception:  # a text file, a pickled object, a broken archive: nothing of it is run
        raise ValueError(f"{path.name} is no module file: a module file is a NumPy archive of "
                         "arrays and one JSON record, with no pickled object") from None
    if "module" not in arrays:
        raise ValueError(f"{path.name} is no module file: it holds no 'module' record")
    try:
        record = json.loads(str(arrays.pop("module")))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path.name}: the module record is no JSON ({error})") from None
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        found = record.get("format") if isinstance(record, dict) else None
        raise ValueError(f"{path.name}: a module file states format {FORMAT!r}, got {found!r}")
    for key, kind in (("name", str), ("library", str), ("regions", dict), ("inputs", list),
                      ("published", list), ("checksums", dict)):
        if not isinstance(record.get(key), kind):
            raise ValueError(f"{path.name}: the module record holds no {key!r}")
    regions = record["regions"]
    for region in (*record["inputs"], *record["published"]):
        if region not in regions:
            raise ValueError(f"{path.name}: the record names {region!r}, which is no region of it")
    for region, size in regions.items():
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError(f"{path.name}: region {region!r} has no size")
    sums = record["checksums"]
    if set(sums) != set(arrays):
        odd = sorted(set(sums) ^ set(arrays))[0]
        raise ValueError(f"{path.name}: array {odd!r} is in the file or in the record alone")
    for key, value in arrays.items():
        if _checksum(value) != sums[key]:
            raise ValueError(f"{path.name}: array {key!r} differs from its checksum")
    for key in ("brain", "meta"):
        if key not in arrays:
            raise ValueError(f"{path.name}: the file holds no trained brain ({key!r} is missing)")
    held = json.loads(str(arrays["meta"])).get("populations", {})
    for region, size in regions.items():
        if region not in held or len(held[region]) != size:
            raise ValueError(f"{path.name}: the trained brain has no region {region!r} of "
                             f"{size} neurons")
    return Module(record, arrays)


def names() -> list[str]:
    """The modules the library ships, by name."""
    return sorted(p.stem for p in _HERE.glob("*.npz"))


def load(name: str) -> Module:
    """A module the library ships, by name (``names``)."""
    path = _HERE / f"{name}.npz"
    if "/" in str(name) or not path.is_file():
        raise ValueError(f"the library ships no module {name!r}: it ships {names()}")
    return load_file(path)


def host(module: Module, brain: Any) -> dict[str, str]:
    """How ``module``'s regions are named in ``brain``: each region and the population that
    holds it, by the module's name (``name/region``) or by the region's own name. A brain that
    lacks a region, or holds it at another size, is refused by that region's name."""
    populations = brain.connectome.populations
    out: dict[str, str] = {}
    for region, size in module.regions.items():
        mine = next((n for n in (f"{module.name}/{region}", region) if n in populations), None)
        if mine is None:
            raise ValueError(f"this brain has no region {region!r} for the module "
                             f"{module.name!r}: it needs {module.regions}")
        if len(populations[mine]) != size:
            raise ValueError(f"this brain's {mine!r} holds {len(populations[mine])} neurons and "
                             f"the module's {region!r} {size}")
        out[region] = mine
    return out


__all__ = ["FORMAT", "Module", "host", "load", "load_file", "names", "save"]
