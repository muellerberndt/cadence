"""Alternate source revisions in fresh subprocesses; preserve every outcome and wall time."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run import digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    protocol_path = here / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    roots = {"baseline": args.baseline.resolve(), "candidate": args.candidate.resolve()}
    revisions = {}
    for name, root in roots.items():
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        if dirty:
            raise RuntimeError(f"{name} checkout must be clean and committed: {dirty}")
        revisions[name] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    jobs = []
    for seed in protocol["seeds"]:
        for backend in protocol["backends"]:
            order = ["baseline", "candidate"] if seed % 2 == 0 else ["candidate", "baseline"]
            jobs.extend(("nursery", seed, backend, name) for name in order)
    for backend in protocol["backends"]:
        jobs.extend(("transport", None, backend, name) for name in ("candidate", "baseline"))
    for backend in protocol["backends"]:
        jobs.extend(("bottleneck", None, backend, name) for name in ("baseline", "candidate"))
    frozen = {
        "revisions": revisions,
        "protocol": protocol,
        "jobs": jobs,
        "harness_sha256_lf": {p.name: digest(p) for p in sorted(here.glob("*.py"))},
        "protocol_sha256_lf": digest(protocol_path),
    }
    write(args.out / "campaign.json", frozen)
    outcomes = []
    environment = dict(os.environ, PYTHONUTF8="1")
    for index, (kind, seed, backend, name) in enumerate(jobs):
        key = f"{index:02d}-{kind}-{name}-{backend}" + (f"-seed{seed}" if seed is not None else "")
        folder = args.out / key
        cmd = [
            sys.executable,
            "-X",
            "utf8",
            str(here / ("run.py" if kind == "nursery" else f"{kind}.py")),
            "--source-root",
            str(roots[name]),
            "--out",
            str(folder),
            "--backend",
            backend,
            "--protocol",
            str(protocol_path),
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        print(f"[{index + 1}/{len(jobs)}] {key}", flush=True)
        started = time.perf_counter()
        with (args.out / f"{key}.log").open("x", encoding="utf-8") as log:
            process = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False
            )
        row = {
            "directory": key,
            "kind": kind,
            "seed": seed,
            "backend": backend,
            "revision": name,
            "exit_code": process.returncode,
            "process_seconds": time.perf_counter() - started,
            "command": cmd,
        }
        outcomes.append(row)
        write(args.out / "outcomes.json", outcomes)
        print(f"  exit={process.returncode}, wall={row['process_seconds']:.2f}s", flush=True)
    unchanged = frozen["harness_sha256_lf"] == {
        p.name: digest(p) for p in sorted(here.glob("*.py"))
    } and frozen["protocol_sha256_lf"] == digest(protocol_path)
    for name, root in roots.items():
        unchanged = (
            unchanged
            and revisions[name]
            == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        )
    write(args.out / "completion.json", {"harness_and_revisions_unchanged": unchanged})
    return int(not unchanged or any(row["exit_code"] != 0 for row in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
