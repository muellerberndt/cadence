"""Run the single declared four-life eligibility development comparison, then stop."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import key_door as kd

PROTOCOL = Path(__file__).with_name("protocol-eligibility-development.json")


def commands(directory: Path, protocol: dict) -> list[tuple[str, list[str]]]:
    """The bounded declaration is checked before any life is run."""
    base = Path(__file__).with_name(protocol["base_protocol"])
    if (
        protocol["base_protocol"] != "protocol-3.json"
        or protocol["base_protocol_sha256"] != kd.FROZEN_PROTOCOL_SHA256
        or hashlib.sha256(base.read_bytes()).hexdigest() != protocol["base_protocol_sha256"]
        or protocol["arms"] != ["blind"]
        or protocol["seeds"] != [2, 3]
        or protocol["delays"] != [5]
        or protocol["episodes_per_rule"] != 500
        or protocol["rules"] != ["chest", "lamp", "chest"]
        or protocol["cases"] != {"control": 0.95, "candidate": 0.98}
        or protocol["maximum_lives"] != 4
        or protocol["gene"] != "operating_point.lam"
    ):
        raise ValueError("this helper admits only the declared four-life comparison")
    return [
        (
            name,
            [
                sys.executable,
                str(Path(kd.__file__).resolve()),
                "--protocol",
                str(base.resolve()),
                "--arms",
                "blind",
                "--seeds",
                "2",
                "3",
                "--delays",
                "5",
                "--point",
                json.dumps({"lam": value}),
                "--workers",
                "1",
                "--out",
                str((directory / f"{name}.json.gz").resolve()),
            ],
        )
        for name, value in protocol["cases"].items()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(PROTOCOL.read_text())
    plan = commands(args.out, protocol)
    args.out.mkdir(parents=True, exist_ok=False)
    source_files = [
        *kd.sources(),
        ("eligibility_development.py", Path(__file__).resolve()),
        (PROTOCOL.name, PROTOCOL),
    ]
    manifest = kd.source_manifest(source_files)
    declaration = {
        "protocol": protocol,
        "source": manifest,
        "commands": plan,
        "confirmation": False,
        "maximum_lives": 4,
    }
    (args.out / "declaration.json").write_text(kd.canonical_json(declaration) + "\n")
    results = []
    for name, command in plan:
        with (args.out / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        path = args.out / f"{name}.json.gz"
        valid, reason = kd.verify(path, current=True)
        results.append(
            {
                "case": name,
                "exit_code": result.returncode,
                "verified": valid,
                "reason": reason,
                "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                if path.exists()
                else None,
            }
        )
        print(json.dumps(results[-1]), flush=True)
        if result.returncode or not valid:
            break
    if manifest != kd.source_manifest(source_files):
        raise RuntimeError("a source changed during the bounded development comparison")
    (args.out / "verification.json").write_text(kd.canonical_json(results) + "\n")
    return (
        0
        if len(results) == 2 and all(r["verified"] and r["exit_code"] == 0 for r in results)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
