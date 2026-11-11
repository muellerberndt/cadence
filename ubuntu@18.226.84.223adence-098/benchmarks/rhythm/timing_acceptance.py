"""Predeclared event-time acceptance, independently recomputed from recorded actions/times."""

from __future__ import annotations

import numpy as np


def alternation(actions) -> list[float]:
    values = np.asarray(actions, dtype=int)
    assert values.ndim == 2 and values.shape[1] == 4 and len(values) >= 2
    assert np.isin(values, [-1, 0, 1]).all()
    changed = (values[1:] != values[:-1]) & (values[1:] >= 0) & (values[:-1] >= 0)
    return changed.mean(axis=0).tolist()


def recovery(actions) -> list[int | None]:
    values = np.asarray(actions, dtype=int)
    out = []
    for row in values.T:
        good = (row[1:] != row[:-1]) & (row[1:] >= 0) & (row[:-1] >= 0)
        bad = np.flatnonzero(~good)
        out.append(0 if not len(bad) else None if bad[-1] == len(good) - 1 else int(bad[-1] + 1))
    return out


def timing_readings(variant: dict) -> dict:
    data = variant["timing"]
    due, begin, end, deadline = (
        np.asarray(data[key], dtype=float)
        for key in ("due_ms", "begin_ms", "end_ms", "deadline_ms")
    )
    assert len(due) == len(begin) == len(end) == len(deadline) == len(variant["actions"])
    assert len(due) >= 3 and all(np.isfinite(a).all() for a in (due, begin, end, deadline))
    assert due[0] == 0 and np.all(np.diff(due) >= 0)
    assert np.all(begin >= due - 1e-6) and np.all(end >= begin)
    distinct = np.unique(due)
    intervals = np.diff(distinct)
    assert len(intervals) and np.all(intervals > 0)
    expected_deadline = np.array(
        [
            distinct[distinct > t][0] if np.any(distinct > t) else distinct[-1] + intervals[-1]
            for t in due
        ]
    )
    assert np.array_equal(deadline, expected_deadline)
    misses = int(np.sum(end > deadline))
    assert variant["missed_deadlines"] == misses
    lateness = begin - due
    periods = []
    for actions in np.asarray(variant["actions"]).T:
        issued = np.flatnonzero(actions == 0)
        actual, scheduled = np.diff(end[issued]), np.diff(due[issued])
        periods.append(
            {
                "event_period": float(np.mean(np.diff(issued))) if len(issued) > 1 else None,
                "observed_ms": actual.tolist(),
                "scheduled_ms": scheduled.tolist(),
                "max_error_ms": float(np.max(np.abs(actual - scheduled))) if len(actual) else None,
            }
        )
    return {
        "missed_deadlines": misses,
        "max_lateness_ms": float(lateness.max()),
        "max_lateness_fraction": float(np.max(lateness / (deadline - due))),
        "phase_drift_ms": float(np.ptp(end - due)),
        "phase_drift_fraction": float(np.ptp(end - due) / intervals.min()),
        "periods": periods,
    }


def founder_gate(run: dict, protocol: dict) -> dict:
    gate = protocol["timing_acceptance"]
    window, bounds = run["window"], gate["bounds"]
    branches = window["branches"]
    intact = np.asarray(branches["intact"]["actions"])
    rates = alternation(intact)
    untaught = alternation(branches["untaught"]["actions"])
    random = alternation(branches["random"]["actions"])
    block = protocol["window"]["block"]
    learned = all(
        rate >= bounds["alternation_min"]
        and rate - old >= bounds["untaught_margin"]
        and rate - chance >= bounds["random_margin"]
        for rate, old, chance in zip(rates, untaught, random, strict=True)
    )
    stable = all(
        min(alternation(intact[i : i + block])) >= bounds["block_alternation_min"]
        for i in range(0, len(intact), block)
    )
    cues = np.asarray(window["cues"])
    first, second = np.flatnonzero(cues == 0), np.flatnonzero(cues == 1)
    permutation = np.empty(4, dtype=int)
    permutation[first], permutation[second] = second, first
    history = (
        np.array_equal(branches["shuffled"]["actions"], intact[:, permutation])
        and np.array_equal(branches["erased"]["actions"], branches["reset"]["actions"])
        and max(alternation(branches["static"]["actions"])) == 0
        and max(alternation(branches["static_cold"]["actions"])) == 0
        and min(alternation(branches["flipflop"]["actions"])) >= bounds["alternation_min"]
    )
    disturbances = {}
    expected_names = {f"pause{n}" for n in protocol["disturbances"]["pauses"]} | {"distractor"}
    assert set(run["disturbances"]) == expected_names
    for name, value in run["disturbances"].items():
        after = np.asarray(value["brain"]["actions"])[-protocol["disturbances"]["post"] :]
        recovered = recovery(after)
        retained = alternation(after[-bounds["retention_events"] :])
        disturbances[name] = {
            "recovery_events": recovered,
            "retention_alternation": retained,
            "passed": all(n is not None and n <= bounds["recovery_events"] for n in recovered)
            and min(retained) >= bounds["alternation_min"],
        }
    custody = (
        window["continuation"]["actions_equal"]
        and window["continuation"]["saved_arrays_equal"]
        and np.array_equal(branches["intact"]["actions"], branches["restored"]["actions"])
        and run["disturbances"]["pause2"]["custody"]["actions_equal"]
        and run["disturbances"]["pause2"]["custody"]["saved_arrays_equal"]
    )
    cadence = run["cadence"]
    variants = cadence["variants"] if cadence else {}
    required = {
        "reference",
        "paced_regular",
        "paced_extra",
        "paced_skipped",
        "paced_regular_load",
        "paced_ordered",
        "paced_shuffled_time",
    }
    required.update(f"paced_regular{v}" for v in protocol["event"]["cadence_variants_ms"])
    required.update(f"budget_{v}" for v in protocol["cadence"]["budgets"])
    required.update(f"tolerance_{v}" for v in protocol["cadence"]["tolerances"])
    timing, consistent = {}, set(variants) == required
    if consistent:
        reference = np.asarray(variants["reference"]["actions"])
        for name, value in variants.items():
            actions = np.asarray(value["actions"])
            if (
                name.startswith("budget_")
                and value["free_steps"] < bounds["minimum_accepted_budget"]
            ):
                continue  # recorded stress case outside the predeclared supported envelope
            consistent = consistent and np.array_equal(actions, reference[: len(actions)])
            consistent = consistent and not bool(np.any(actions < 0))
            if value["paced"]:
                reading = timing_readings(value)
                reading["passed"] = (
                    reading["missed_deadlines"] <= bounds["missed_deadlines"]
                    and reading["max_lateness_fraction"] <= bounds["lateness_fraction"]
                    and reading["phase_drift_fraction"] <= bounds["phase_drift_fraction"]
                    and all(
                        p["event_period"] == 2
                        and p["max_error_ms"] is not None
                        and p["max_error_ms"] <= bounds["period_error_ms"]
                        for p in reading["periods"]
                    )
                )
                timing[name] = reading
        ordered = np.asarray(variants["paced_ordered"]["timing"]["due_ms"])
        shuffled = np.asarray(variants["paced_shuffled_time"]["timing"]["due_ms"])
        consistent = consistent and np.array_equal(
            np.sort(np.diff(ordered)), np.sort(np.diff(shuffled))
        )
        consistent = consistent and not np.array_equal(ordered, shuffled)
    work = [run["teaching"]["work"], window["lead_work"]]
    work += [v["work"] for v in branches.values() if v["work"] is not None]
    work += [v["brain"]["work"] for v in run["disturbances"].values()]
    no_refusals = all(w["action_refusals"] == 0 and w["teacher_refusals"] == 0 for w in work)
    passed = bool(
        learned
        and stable
        and history
        and custody
        and consistent
        and no_refusals
        and all(d["passed"] for d in disturbances.values())
        and timing
        and all(t["passed"] for t in timing.values())
    )
    return {
        "seed": run["seed"],
        "learned": learned,
        "stable": stable,
        "history_controls": bool(history),
        "continuation": bool(custody),
        "event_identity": bool(consistent),
        "no_refusals": no_refusals,
        "alternation": rates,
        "untaught_alternation": untaught,
        "random_alternation": random,
        "disturbances": disturbances,
        "timing": timing,
        "passed": passed,
    }


def evaluate(runs: list[dict], protocol: dict, declaration: dict, capped: bool) -> dict:
    gate = protocol["timing_acceptance"]
    expected = {
        (seed, recipe, arm)
        for seed in declaration["seeds"]
        for recipe in protocol["run_recipes"]
        for arm in protocol["run_arms"]
    }
    keys = [(r["seed"], r["recipe"], r["arm"]) for r in runs]
    complete = len(keys) == len(set(keys)) and set(keys) == expected and not capped
    primary = [
        r for r in runs if r["recipe"] == gate["primary_recipe"] and r["arm"] == gate["primary_arm"]
    ]
    founders = [founder_gate(run, protocol) for run in primary]
    confirmed = (
        declaration["frozen_protocol"]
        and not declaration["overrides"]
        and declaration["seeds"] == protocol["seeds"]["confirmation"]
        and declaration["recipes"] == protocol["run_recipes"]
        and declaration["arms"] == protocol["run_arms"]
        and declaration["cadence_runs"]
    )
    passed = (
        complete and confirmed and sum(f["passed"] for f in founders) >= gate["required_founders"]
    )
    return {
        "complete": complete,
        "confirmation_admitted": bool(confirmed),
        "founders": founders,
        "joint_passes": sum(f["passed"] for f in founders),
        "passed": bool(passed),
        "scope": "Event-driven period-two activity on the declared clock and machine; "
        "not an endogenous physical oscillator",
    }


def verify_work(work: dict, records: list[dict]) -> None:
    """Rebuild charged work from individual accepted and refused solve reports."""
    expected = dict.fromkeys(work, 0)
    for record in records:
        seconds = record["call_seconds"]
        assert np.isfinite(seconds) and seconds >= 0
        expected["calls_seconds"] += seconds
        if record["operation"] == "act":
            if "action_cpu_seconds" in work:
                cpu_seconds = record["process_cpu_seconds"]
                assert (
                    isinstance(cpu_seconds, (int, float))
                    and np.isfinite(cpu_seconds)
                    and cpu_seconds >= 0
                )
                expected["action_cpu_seconds"] += cpu_seconds
            accepted = record["answer"] is not None
            assert accepted == bool(record["qualified"])
            expected["action_attempts"] += 1
            expected["action_refusals"] += not accepted
            expected["action_sweeps"] += record["steps"]
            expected["action_row_sweeps"] += record["rows"] * record["steps"]
            expected["action_residual_checks"] += record["residual_checks"]
            expected["action_damping_halvings"] += record["damping_halvings"]
            expected["trace_row_updates"] += record["rows"] if accepted else 0
        else:
            assert record["operation"] == "teach"
            expected["teacher_attempts"] += 1
            expected["teacher_refusals"] += not record["accepted"]
            expected["teacher_presentations"] += record["attempted_presentations"]
            expected["teacher_sweeps"] += record["total_steps"]
            expected["teacher_row_sweeps"] += record["total_row_sweeps"]
            expected["teacher_budget_exhausted"] += int(record.get("free_budget_exhausted", 0))
    assert expected == work, "charged work differs from individual reports"


def verify_body(body: dict, directory) -> None:
    """Verify the new protocol's census, event counts, work and saved continuation independently."""
    import json

    protocol, declaration, runs = body["protocol"], body["declaration"], body["runs"]
    revision = declaration.get("instrument_revision", 1)
    assert revision in (1, 2)
    if revision == 2:
        assert declaration["threads"]["VECLIB_MAXIMUM_THREADS"] == "1"
    assert json.loads((directory / "declaration.json").read_text()) == declaration
    assert (
        declaration["frozen_protocol"] == body["frozen_protocol"] == (not declaration["overrides"])
    )
    if declaration["frozen_protocol"]:
        assert json.loads((directory / "protocol.json").read_text()) == protocol
    assert [
        json.loads(line) for line in (directory / "runs.jsonl").read_text().splitlines()
    ] == runs
    planned = [
        [s, r, a]
        for s in declaration["seeds"]
        for r in declaration["recipes"]
        for a in declaration["arms"]
    ]
    completed = [[r["seed"], r["recipe"], r["arm"]] for r in runs]
    assert len({tuple(k) for k in planned}) == len(planned)
    assert body["planned_founders"] == planned and body["completed_founders"] == completed
    assert completed == planned[: len(completed)]
    assert body["capped"] or completed == planned
    reports = [json.loads(line) for line in (directory / "reports.jsonl").read_text().splitlines()]
    assert all([r["seed"], r["recipe"], r["arm"]] in completed for r in reports)
    for run in runs:
        folder = directory / f"seed{run['seed']}-{run['recipe']}-{run['arm']}"
        records = [
            r
            for r in reports
            if (r["seed"], r["recipe"], r["arm"]) == (run["seed"], run["recipe"], run["arm"])
        ]
        consumed = []

        def branch(stage, name, value, actions=None, records=records, consumed=consumed):
            chosen = [r for r in records if (r["stage"], r["branch"]) == (stage, name)]
            consumed.extend(chosen)
            if revision == 2:
                assert "action_cpu_seconds" in value["work"]
            verify_work(value["work"], chosen)
            issued = [
                r["answer"] if r["answer"] is not None else [-1] * 4
                for r in chosen
                if r["operation"] == "act"
            ]
            assert issued == (value["actions"] if actions is None else actions)

        branch("teaching", "brain", run["teaching"])
        assert (
            len(run["teaching"]["actions"])
            == protocol["teaching"]["bouts"] * protocol["teaching"]["events_per_bout"]
        )
        window = run["window"]
        branch("window", "lead", {"work": window["lead_work"], "actions": window["lead_actions"]})
        assert len(window["lead_actions"]) == protocol["window"]["lead"] + 1
        assert set(window["branches"]) == set(protocol["controls"])
        for name, value in window["branches"].items():
            assert len(value["actions"]) == protocol["window"]["events"]
            if value["work"] is not None:
                branch("window", name, value, value.get("lead_actions", []) + value["actions"])
        for name, value in run["disturbances"].items():
            branch(name, "brain", value["brain"])
            expected = protocol["disturbances"]["pre"] + protocol["disturbances"]["post"]
            expected += (
                int(name[5:])
                if name.startswith("pause")
                else protocol["disturbances"]["distractor_events"]
            )
            assert len(value["brain"]["actions"]) == expected
            if name == "pause2":
                suffix = value["brain"]["actions"][protocol["disturbances"]["pre"] + 1 :]
                # The twin's reports retain its actual actions even if continuation failed.
                twin = [
                    r["answer"] if r["answer"] is not None else [-1] * 4
                    for r in records
                    if r["stage"] == name and r["branch"] == "twin"
                ]
                branch(name, "twin", {"work": value["custody"]["twin_work"], "actions": twin})
                assert value["custody"]["actions_equal"] == (twin == suffix)
        with np.load(directory / f"inputs-seed{run['seed']}.npz", allow_pickle=False) as frozen:
            assert window["branches"]["random"]["actions"] == frozen["random/window"].tolist()
            for name, value in run["disturbances"].items():
                assert value["random"]["actions"] == frozen[f"random/disturbance/{name}"].tolist()
            if run["cadence"] is not None:
                for name, value in run["cadence"]["variants"].items():
                    branch("cadence", name, value)
                    if value["paced"]:
                        prefix = f"cadence/{value['schedule']}"
                        assert value["timing"]["due_ms"] == frozen[prefix + "/due_ms"].tolist()
                        assert value["slots"] == frozen[prefix + "/slot"].tolist()
                        timing_readings(value)
                        assert value["burners"] == (
                            declaration["burners"] if name == "paced_regular_load" else 0
                        )
                    else:
                        assert (
                            name == "reference"
                            and len(value["actions"]) == protocol["cadence"]["events"] + 1
                        )
        assert len(consumed) == len(records), "unassigned solve reports"
        for first, second, claim in (
            (
                "intact-after.npz",
                "restored-after.npz",
                window["continuation"]["saved_arrays_equal"],
            ),
            (
                "pause2-after.npz",
                "pause2-twin-after.npz",
                run["disturbances"]["pause2"]["custody"]["saved_arrays_equal"],
            ),
        ):
            with (
                np.load(folder / first, allow_pickle=False) as a,
                np.load(folder / second, allow_pickle=False) as b,
            ):
                same = set(a.files) == set(b.files) and all(
                    np.array_equal(a[k], b[k]) for k in a.files
                )
                assert same == claim
        operations = run["checkpoint_io"]
        assert {o["file"] for o in operations if o["operation"] == "write"} == {
            p.name for p in folder.iterdir()
        }
        assert len([o for o in operations if o["operation"] == "compare"]) == 2
        for operation in operations:
            assert np.isfinite(operation["seconds"]) and operation["seconds"] >= 0
            paths = operation.get("files", [operation.get("file")])
            assert operation["bytes"] == sum((folder / p).stat().st_size for p in paths)
        world = json.loads((folder / "probe-world.json").read_text())
        assert world["next_event"] == protocol["window"]["lead"] + 1
        assert world["anchor"] == window["anchor"]
        world = json.loads((folder / "pause2-world.json").read_text())
        assert world["next_event"] == protocol["disturbances"]["pre"] + 1
    assert body["timing_acceptance"] == evaluate(runs, protocol, declaration, body["capped"])
