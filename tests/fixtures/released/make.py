"""Write the released files: every brain of the panel each release composes, saved by that
release, with its continuation after a load as that release reads it.

    python tests/fixtures/released/make.py --scratch DIR v0.99.0 v0.99.1 v0.100.0 v0.101.0

Each tag's ``src`` is extracted with ``git archive`` into ``DIR`` (outside the repository), and
an isolated child interpreter (``-I``) with that source first on its path writes ``<tag>/<brain>.npz`` here and
the brain's continuation, once on the NumPy solver (``CADENCE_FUSED=0``) and once on the
compiled kernel where Numba is installed, at one thread. ``released.json`` gathers them: the
tags named replace their entries, and the entries of other tags stay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

CHILD = r"""
import json, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
sys.path[:0] = [sys.argv[4], sys.argv[3]]
import cadence as cd, panel
assert Path(cd.__file__).is_relative_to(sys.argv[4]), cd.__file__
out, write = Path(sys.argv[1]), sys.argv[2] == "write"
from cadence import brain as kernel
report = {"version": cd.__version__, "solver": "fused" if kernel._FUSED else "numpy", "brains": {}}
for name in panel.names(cd.__version__):
    path = out / f"{name}.npz"
    if write:
        brain = panel.compose(name)
        panel.life(brain, name)
        brain.save(path)
    loaded = cd.Brain.load(path)
    acts = panel.continuation(loaded, name)
    report["brains"][name] = {**panel.record(path), "acts": acts, "learned": panel.learned(loaded)}
print(json.dumps(report))
"""


def extract(tag: str, scratch: Path) -> Path:
    target = scratch / tag
    if not (target / "src" / "cadence").is_dir():
        target.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(
            ["git", "-C", str(ROOT), "archive", tag, "src"], capture_output=True, check=True
        ).stdout
        tar = target / "src.tar"
        tar.write_bytes(archive)
        with tarfile.open(tar) as held:
            held.extractall(target, filter="data")
        tar.unlink()
    return target / "src"


def run(source: Path, out: Path, write: bool, fused: bool) -> dict:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "CADENCE_FUSED": "1" if fused else "0",
    }
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            CHILD,
            str(out),
            "write" if write else "read",
            str(HERE),
            str(source),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        raise SystemExit(done.stderr)
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("tags", nargs="+")
    args = parser.parse_args()
    written = HERE / "released.json"
    table: dict = json.loads(written.read_text()) if written.exists() else {}
    for tag in args.tags:
        source = extract(tag, args.scratch)
        out = HERE / tag
        out.mkdir(exist_ok=True)
        numpy = run(source, out, write=True, fused=False)
        fused = run(source, out, write=False, fused=True)
        table[numpy["version"]] = {
            "tag": tag,
            "written_by": f"cadence-net {numpy['version']} from the tag's src, on the NumPy "
            "solver (CADENCE_FUSED=0), one thread",
            "brains": {
                name: {
                    "format": entry["format"],
                    "features": entry["features"],
                    "acts": {"numpy": entry["acts"], "fused": fused["brains"][name]["acts"]},
                    "learned": {
                        "numpy": entry["learned"],
                        "fused": fused["brains"][name]["learned"],
                    },
                }
                for name, entry in numpy["brains"].items()
            },
        }
        print(tag, numpy["solver"], fused["solver"], len(numpy["brains"]), "brains")
    (HERE / "released.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
