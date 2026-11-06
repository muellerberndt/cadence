"""Source and protocol custody for current experiments and explicitly historical receipts.

Historical verification checks the exact registered receipt and its recorded bytes. It
never admits that receipt as a current experiment or reruns its training.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import types
import warnings
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

import cadence as cd
from cadence.receipts import source_manifest

ENVIRONMENT = ("CADENCE_FUSED", "CADENCE_RECALL_THREADS", "NUMBA_NUM_THREADS",
               "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")

def sources(root: Path, *scripts: str) -> list[tuple[str, Path]]:
    """Bind every library module, package metadata, this helper and the experiment scripts.

    Also reject a different installed implementation: paths may differ for an installed
    wheel, but its executed library bytes must equal the source tree being recorded.
    """
    names = {"pyproject.toml", "benchmarks/provenance.py", *scripts}
    package = Path(cd.__file__).resolve().parent
    for path in sorted((root / "src/cadence").glob("*.py")):
        if (package / path.name).read_bytes() != path.read_bytes():
            raise ValueError(f"executed library differs from benchmark source: {path.name}")
        names.add(path.relative_to(root).as_posix())
    return [(name, root / name) for name in sorted(names)]


def environment() -> dict[str, Any]:
    """The effective numerical mode and dependencies, without importing optional devices."""
    packages = {}
    for name in ("numpy", "numba", "llvmlite", "scipy", "torch", "mlx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    configuration = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(configuration):
        warnings.filterwarnings("ignore", message="Install `pyyaml` for better output")
        np.show_config()
    return {
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "effective_fused": bool(importlib.import_module("cadence.brain")._FUSED),
        "numpy_configuration": configuration.getvalue(),
        "environment": {name: os.environ.get(name) for name in ENVIRONMENT},
    }


def metadata(body: dict[str, Any], schema: str, configuration: dict[str, Any]) -> str | None:
    """Check exact declared settings and well-formed execution metadata before arithmetic."""
    try:
        cd.canonical_json(body)  # every recorded number must be finite
        if body["schema"] != schema:
            return "wrong schema"
        if cd.canonical_json(body["configuration"]) != cd.canonical_json(configuration):
            return "configuration differs from the declared protocol"
        execution = body["execution"]
        if set(execution) != {"python", "platform", "packages", "effective_fused",
                              "numpy_configuration", "environment"}:
            return "incomplete execution identity"
        if any(not isinstance(execution[key], str) or not execution[key]
               for key in ("python", "platform", "numpy_configuration")):
            return "invalid execution identity"
        if type(execution["effective_fused"]) is not bool or not isinstance(
            execution["packages"].get("numpy"), str
        ):
            return "invalid numerical mode or dependencies"
        if set(execution["environment"]) != set(ENVIRONMENT):
            return "incomplete numerical environment"
        if execution["environment"]["CADENCE_FUSED"] == "0" and execution["effective_fused"]:
            return "inconsistent fused mode"
    except (KeyError, TypeError, ValueError, AttributeError):
        return "incomplete or invalid protocol metadata"
    return None


def seeds(values: Any, minimum: int = 1) -> bool:
    """A declared distinct, nonnegative founder census; booleans are not seed integers."""
    return (isinstance(values, list) and len(values) >= minimum
            and all(type(value) is int and value >= 0 for value in values)
            and len(set(values)) == len(values))


def record(root: Path, *scripts: str):
    """Bind each worker result before it can be merged under another source or recipe."""
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            identity = source_manifest(sources(root, *scripts))
            execution = environment()
            result = function(*args, **kwargs)
            if identity != source_manifest(sources(root, *scripts)):
                raise RuntimeError("benchmark sources changed during the measurement")
            return {**result, "source": identity, "execution": execution}
        return measured
    return decorate


def worker(row: dict[str, Any], root: Path, *scripts: str) -> str | None:
    """A merged worker must have used the current complete implementation."""
    try:
        if row["source"] != source_manifest(sources(root, *scripts)):
            return "worker source identity differs"
        return metadata({"schema": "worker", "configuration": {},
                         "execution": row["execution"]}, "worker", {})
    except (KeyError, ValueError, TypeError):
        return "missing or invalid worker identity"


def _registered(root: Path, name: str) -> tuple[dict[str, Any], cd.Receipt]:
    index = json.loads((root / "benchmarks/historical-receipts.json").read_text())
    if name not in index:
        raise ValueError("unregistered historical receipt")
    entry = index[name]
    return entry, cd.Receipt.read(root / name)


def historical_sources(root: Path, receipt: cd.Receipt) -> list[tuple[str, Path]]:
    out = []
    for row in receipt.source["files"]:
        name, digest = PurePosixPath(row["path"]), row["sha256"]
        if (name.is_absolute() or ".." in name.parts or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid historical source identity")
        snapshot = root / "benchmarks/attention/results/sources" / (digest + ".txt")
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError("historical source digest differs")
        out.append((str(name), snapshot))
    return out


def historical_check(root: Path, name: str, body: dict[str, Any]) -> str | None:
    """Run the registered version's arithmetic, under that version's actual evidence scope."""
    entry, receipt = _registered(root, name)
    bound = dict(historical_sources(root, receipt))
    checker = entry["checker"]
    code = bound[checker].read_bytes()
    if hashlib.sha256(code).hexdigest() != entry["checker_sha256"]:
        raise ValueError("historical checker identity differs")
    module = types.ModuleType("cadence_historical_" + Path(checker).stem)
    module.__file__ = str(root / checker)
    sys.modules[module.__name__] = module
    exec(compile(code, module.__file__, "exec"), module.__dict__)
    return module.check(body)


def verify_historical(root: Path, path: Path, name: str) -> tuple[bool, str]:
    """Verify only an exact registered historical result, never a current admission claim."""
    try:
        entry, _ = _registered(root, name)
        receipt = cd.Receipt.read(path)
        if receipt.digest != entry["digest"] or receipt.kind != entry["kind"]:
            return False, "receipt is not the registered historical result"
        return cd.Receipt.verify(path, sources=historical_sources(root, receipt),
                                 check=lambda body: historical_check(root, name, body))
    except (KeyError, ValueError, OSError, TypeError) as error:
        return False, str(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    name = args.historical.resolve().relative_to(root).as_posix()
    valid, message = verify_historical(root, args.historical, name)
    print("Historical custody and recorded arithmetic:", message)
    raise SystemExit(0 if valid else 1)
