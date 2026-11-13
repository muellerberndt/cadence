"""The text parrot's runs: a founder's life in an arm with every reading, the controls, the tables.

    python textgates.py life <founder> <arm> <out_dir>        # pretraining, G1-G5, introspection
    python textgates.py control <founder> <kind> <out_dir>    # lstm-online, lstm-offline,
                                                              # transformer-online,
                                                              # transformer-offline, readout, table
    python textgates.py read <out_dir> [arm ...]              # the tables (markdown)

``ARMS`` names the brains: ``d<regions>a<apical>`` (``ear``: the words read reach the
association region alone; ``s``: sleep between stages).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import textchamber as tc  # noqa: E402
import textparrot as tp  # noqa: E402

ARMS: dict[str, dict[str, Any]] = {}
for regions, widths in ((2, [160, 128]), (4, [128] * 4), (6, [128] * 6)):
    for apical, tag in ((0.0, "0"), (0.25, "25"), (0.5, "5")):
        ARMS[f"d{regions}a{tag}"] = {"modules": widths, "apical": apical}
ARMS["d4a25ear"] = {"modules": [128] * 4, "apical": 0.25, "ear_entry": "association"}
ARMS["d4a5ear"] = {"modules": [128] * 4, "apical": 0.5, "ear_entry": "association"}
ARMS["d4a25s"] = {"modules": [128] * 4, "apical": 0.25, "sleep_scale": 0.5}
ARMS["d6a5s"] = {"modules": [128] * 6, "apical": 0.5, "sleep_scale": 0.5}
ARMS["d6a5ears"] = {"modules": [128] * 6, "apical": 0.5, "ear_entry": "association",
                    "sleep_scale": 0.5}
ARMS["d2a25all"] = {"modules": [160, 128], "apical": 0.25, "learning_surprise": False}

EXAM = ("colour", "relation", "where", "which_shape", "which_colour", "which_size",
        "which_material", "which_food", "can_eat", "know_shape", "know_food")
SENTENCES = ("describe", "describe_rel")
CAPS = {"words": 200, "properties": 600, "relations": 600, "questions": 400, "sentences": 300,
        "scenes": 600}
if os.environ.get("TEXTPARROT_CAPS"):  # a run's own caps, as JSON
    CAPS = {**CAPS, **json.loads(os.environ["TEXTPARROT_CAPS"])}


def readings(policy: Any, layout: tc.Layout, seed: int, *, per: int = 128) -> dict[str, Any]:
    """G1-G4: the exam forms and sentences on fresh scenes, the held-out splits, introspection."""
    import textintrospect as ti

    rng = np.random.default_rng([seed, 2])
    out: dict[str, Any] = {}
    t = time.time()
    out["G1"] = tp.read(policy, layout, rng, EXAM, per, split="test")
    out["G2"] = tp.read(policy, layout, rng, SENTENCES, per, split="test")
    out["G3"] = {
        "pairs": tp.read(policy, layout, rng, ("colour", "where", "which_colour", "describe"),
                         per, split="pairs"),
        "relations": tp.read(policy, layout, rng, ("relation", "describe_rel"), per,
                             split="relations"),
        "depth": tp.read(policy, layout, rng, ("where", "can_eat"), per, split="depth"),
    }
    out["owner"] = ti.owner_exam(policy)
    if hasattr(policy, "brain"):
        pr = ti.probes(policy, np.random.default_rng([seed, 3]))
        out["G4"] = {"probes": pr["accuracy"], "base_rate": pr["base_rate"],
                     "new_objects": pr["new_objects"],
                     "neighbours": ti.neighbours(policy,
                                                 np.random.default_rng([seed, 4]))["nearest"],
                     "word_alone": ti.word_alone(policy, pr, np.random.default_rng([seed, 8])),
                     "silencing": ti.silencing(policy, pr, np.random.default_rng([seed, 5])),
                     "changed_scene": ti.changed_scene(policy, np.random.default_rng([seed, 6])),
                     "regions": ti.regions(policy, np.random.default_rng([seed, 7]))}
    out["read_seconds"] = round(time.time() - t, 1)
    return out


def life(seed: int, arm: str, out_dir: Path, *, pistachio: bool = True) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{arm}_{seed}"
    log = open(f"{stem}.log", "a")
    genome = ARMS[arm]
    policy, layout, record = tp.life(seed, genome=genome, caps=CAPS, log=log)
    record.update(arm=arm, genome=genome, parameters=policy.parameters(),
                  synapses=int(len(policy.brain.connectome.pre)),
                  macs=policy.macs(), lessons=policy.lessons, refused=policy.refused)
    policy.brain.save(f"{stem}.npz")
    Path(f"{stem}.json").write_text(json.dumps(record, indent=1, default=float))
    record["readings"] = readings(policy, layout, seed)
    Path(f"{stem}.json").write_text(json.dumps(record, indent=1, default=float))
    if pistachio:
        import textpistachio as pi

        got = pi.new_word(policy, np.random.default_rng([seed, 9]))
        got.pop("parrot")
        record["readings"]["G5"] = got
    Path(f"{stem}.json").write_text(json.dumps(record, indent=1, default=float))
    print(json.dumps({"done": arm, "seed": seed}), file=log, flush=True)
    return record


POINTED = ("colour", "relation", "where", "can_eat")


def pointed(seed: int, arm: str, out_dir: Path, per: int = 128) -> dict[str, Any]:
    """The secondary reading (amendment 4, no gate): the forms that name a thing, read from the
    saved brain with the trainer pointing at it (the focus given, the parrot's own words), beside
    the free read's share of searches that held the named thing alone."""
    from cadence import Brain

    layout = tc.Layout.founder(seed)
    policy = tp.BrainPolicy.from_brain(Brain.load(out_dir / f"{arm}_{seed}.npz", backend="cpu"),
                                       layout)
    rng = np.random.default_rng([seed, 2])  # the readings' own stream, as the free read's
    out = {"G1": tp.read(policy, layout, rng, POINTED, per, mode="dagger"),
           "G2": tp.read(policy, layout, rng, ("describe_rel",), per, mode="dagger"),
           "G3": {"pairs": tp.read(policy, layout, rng, ("colour", "where"), per, split="pairs",
                                   mode="dagger"),
                  "relations": tp.read(policy, layout, rng, ("relation", "describe_rel"), per,
                                       split="relations", mode="dagger"),
                  "depth": tp.read(policy, layout, rng, ("where", "can_eat"), per,
                                   split="depth", mode="dagger")},
           "held_free": {f: v["held"] for f, v in tp.read(policy, layout, rng, POINTED, per)
                         .items()}}
    Path(out_dir / f"pointed_{arm}_{seed}.json").write_text(json.dumps(out, indent=1))
    return out


def plan_of(record: dict[str, Any]) -> list[tuple[str, tuple[str, ...], int]]:
    """The brain's curriculum as it lived it: each stage's forms and blocks."""
    forms = dict(tp.STAGES)
    out = []
    for st in record["stages"]:
        blocks = st.get("blocks", st["mastered"] or (st["curve"][-1]["block"] if st["curve"]
                                                     else 0))
        out.append((st["stage"], forms[st["stage"]], int(blocks)))
    return out


def control(seed: int, kind: str, out_dir: Path, brain_record: Path | None,
            arm: str = "d4a25") -> dict[str, Any]:
    import textcontrols as tcl

    out_dir.mkdir(parents=True, exist_ok=True)
    layout = tc.Layout.founder(seed)
    if brain_record is not None and brain_record.exists():
        rec = json.loads(brain_record.read_text())
        plan, parameters = plan_of(rec), int(rec["parameters"])
    else:  # the arm's parameter count, from its newborn brain; half the caps
        plan = [(s, f, CAPS[s] // 2) for s, f in tp.STAGES]
        parameters = tp.BrainPolicy(layout, ARMS[arm], seed).parameters()
    exchanges = sum(b for _, _, b in plan) * 64
    t = time.time()
    if kind.startswith("newborn"):  # the arm's brain untaught: the probes' and codes' baseline
        policy: Any = tp.BrainPolicy(layout, ARMS[kind.split(":", 1)[1] if ":" in kind else arm],
                                     seed)
    elif kind in ("lstm-online", "transformer-online"):
        policy = tcl.online(kind.split("-")[0], layout, parameters, seed, plan)
    elif kind in ("lstm-offline", "transformer-offline"):
        policy = tcl.offline(kind.split("-")[0], layout, parameters, seed,
                             max(640, min(exchanges, 20000)))
    else:
        policy = tcl.plain(kind, layout, seed, plan)
    seconds = time.time() - t
    macs = getattr(policy, "macs", 0)
    record: dict[str, Any] = {"kind": kind, "seed": seed, "parameters": policy.parameters(),
                              "updates": getattr(policy, "updates", 0),
                              "macs": float(macs() if callable(macs) else macs),
                              "train_seconds": seconds, "plan": plan}
    record["readings"] = readings(policy, layout, seed)
    name = kind.replace(":", "-")
    Path(out_dir / f"{name}_{seed}.json").write_text(json.dumps(record, indent=1, default=float))
    return record


# --- the tables ------------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def summary(rec: dict[str, Any]) -> dict[str, float]:
    r = rec.get("readings", {})
    out: dict[str, float] = {}
    if "G1" in r:
        out["exam"] = _mean([v["right"] for v in r["G1"].values()])
        for f in ("colour", "relation", "where", "which_shape"):
            out[f] = r["G1"][f]["right"]
        out["describe"] = r["G2"]["describe"].get("parsed", float("nan"))
        out["describe_rel"] = r["G2"]["describe_rel"].get("parsed", float("nan"))
        g3 = r["G3"]
        out["pairs"] = _mean([v.get("parsed", v["right"]) for v in g3["pairs"].values()])
        out["held_rel"] = _mean([v.get("parsed", v["right"]) for v in g3["relations"].values()])
        out["held_depth"] = _mean([v["right"] for v in g3["depth"].values()])
    if "G4" in r:
        out["probes"] = min(r["G4"]["probes"].values())
        out["new_obj"] = _mean(list(r["G4"]["new_objects"].values()))
        out["word_alone"] = r["G4"]["word_alone"]["moment_0"]["right"]
        out["same_thing"] = r["G4"]["word_alone"]["moment_0"]["same_thing"]
    if "pairs" in out:
        parts = [out["pairs"], out["held_rel"], out["held_depth"]] + (
            [out["new_obj"]] if "new_obj" in out else [])
        out["G3"] = _mean(parts)
    if "G5" in r:
        learned = r["G5"]["learned"]
        out["G5_trials"] = float(learned["trial"]) if learned else float("nan")
    out["parameters"] = float(rec.get("parameters", float("nan")))
    out["macs"] = float(rec.get("macs", float("nan")))
    if "stages" in rec:
        out["blocks"] = float(rec["blocks"])
        sat = [s.get("saturation", {}).get("max_efficacy", float("nan")) for s in rec["stages"]]
        out["max_eff"] = float(np.nanmax(sat)) if sat else float("nan")
    return out


def table(out_dir: Path, names: list[str] | None = None) -> str:
    rows: dict[str, list[dict[str, float]]] = {}
    for p in sorted(out_dir.glob("*.json")):
        name = p.stem.rsplit("_", 1)[0]
        if names and name not in names:
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "readings" not in rec:
            continue
        rows.setdefault(name, []).append(summary(rec))
    cols = ["exam", "colour", "relation", "where", "which_shape", "describe", "describe_rel",
            "G3", "pairs", "held_rel", "held_depth", "probes", "new_obj", "word_alone",
            "same_thing", "G5_trials", "blocks", "max_eff", "parameters", "macs"]
    lines = ["| arm | n | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for name, rs in rows.items():
        cells = []
        for c in cols:
            v = _mean([r[c] for r in rs if c in r and r[c] == r[c]])
            cells.append("" if v != v else (f"{v:.3g}" if c in ("parameters", "macs") else
                                             f"{v:.2f}" if c not in ("blocks", "G5_trials")
                                             else f"{v:.0f}"))
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _records(out_dir: Path, name: str) -> list[dict[str, Any]]:
    out = []
    for p in sorted(out_dir.glob(f"{name}_*.json")):
        if p.stem.rsplit("_", 1)[0] != name:
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "readings" in rec:
            out.append(rec)
    return out


def detail(out_dir: Path, arm: str, newborn: str | None = None) -> str:
    """G4 and G5 in detail for ``arm`` (beside its newborn's records where named): probes,
    a word alone, silencing, a changed scene, neighbours, the regions, the pistachios, and the
    owner's exam of the first founder."""
    recs = _records(out_dir, arm)
    born = _records(out_dir, newborn) if newborn else []
    lines = [f"{arm}: {len(recs)} founders" + (f", newborn {len(born)}" if born else "")]

    def m(rs: list[dict[str, Any]], *path: str) -> float:
        vals = []
        for r in rs:
            v: Any = r["readings"]
            try:
                for k in path:
                    v = v[k]
            except (KeyError, TypeError):
                continue
            vals.append(float(v))
        return _mean(vals)

    lines.append("| probe | " + " | ".join(ti_props) + " |")
    lines.append("|---|" + "---|" * len(ti_props))
    for label, rs in (("brain", recs), ("newborn", born)):
        if rs:
            lines.append(f"| {label} | " + " | ".join(f"{m(rs, 'G4', 'probes', p):.2f}"
                                                     for p in ti_props) + " |")
            lines.append(f"| {label}, new kinds | " + " | ".join(
                f"{m(rs, 'G4', 'new_objects', p):.2f}" for p in ti_props) + " |")
    lines.append("")
    lines.append("| a word alone | round | food | red | mean | same thing |")
    lines.append("|---|---|---|---|---|---|")
    for label, rs in (("brain", recs), ("newborn", born)):
        for mo in ("moment_0", "moment_1"):
            if rs:
                b = [m(rs, "G4", "word_alone", mo, "balanced", p) for p in ("round", "food", "red")]
                lines.append(f"| {label}, {mo} | " + " | ".join(f"{x:.2f}" for x in b)
                             + f" | {m(rs, 'G4', 'word_alone', mo, 'right'):.2f}"
                             + f" | {m(rs, 'G4', 'word_alone', mo, 'same_thing'):.2f} |")
    lines.append("")
    lines.append("| silenced | form | before | after | spared before / after |")
    lines.append("|---|---|---|---|---|")
    for prop in ("round", "red", "food"):
        if not recs or "silencing" not in recs[0]["readings"].get("G4", {}):
            break
        row = recs[0]["readings"]["G4"]["silencing"][prop]
        spared = [k[:-7] for k in row if k.endswith("_before") and k != "before"]
        sp = ", ".join(f"{s} {m(recs, 'G4', 'silencing', prop, s + '_before'):.2f} / "
                       f"{m(recs, 'G4', 'silencing', prop, s + '_after'):.2f}" for s in spared)
        lines.append(f"| {prop} | {row['form']} | {m(recs, 'G4', 'silencing', prop, 'before'):.2f}"
                     f" | {m(recs, 'G4', 'silencing', prop, 'after'):.2f} | {sp} |")
    lines.append("")
    lines.append(f"changed scene: where {m(recs, 'G4', 'changed_scene', 'where'):.2f}, colour "
                 f"{m(recs, 'G4', 'changed_scene', 'colour'):.2f}")
    near = [(r["readings"]["G4"]["neighbours"]["lemon"][0],
             r["readings"]["G4"]["neighbours"]["pen"][0]) for r in recs if "G4" in r["readings"]]
    import textworld as tw

    good = sum(tw.KINDS[a][0] == "round" or tw.KINDS[a][1] == "fruit" for a, _ in near)
    good_pen = sum(tw.KINDS[b][0] == "long" for _, b in near)
    lines.append(f"neighbours: lemon nearest a round or fruit kind on {good} of {len(near)}, "
                 f"pen nearest a long kind on {good_pen} of {len(near)} ({near[:3]})")
    lines.append("")
    if recs and "regions" in recs[0]["readings"].get("G4", {}):
        regs = list(recs[0]["readings"]["G4"]["regions"])
        lines.append("| region | colour decoded (name / colour / shape question) | shape decoded "
                     "(name / colour / shape) | kind decoded (name) | switch colour | "
                     "switch shape |")
        lines.append("|---|---|---|---|---|---|")
        for reg in regs:
            def q(qn: str, key: str, reg: str = reg) -> float:
                return m(recs, "G4", "regions", reg, qn, key)
            lines.append(
                f"| {reg} | {q('name', 'colour_decoded'):.2f} / {q('colour', 'colour_decoded'):.2f}"
                f" / {q('shape', 'colour_decoded'):.2f} | {q('name', 'shape_decoded'):.2f} / "
                f"{q('colour', 'shape_decoded'):.2f} / {q('shape', 'shape_decoded'):.2f} | "
                f"{q('name', 'kind_decoded'):.2f} | "
                f"{m(recs, 'G4', 'regions', reg, 'switch', 'colour'):+.3f} | "
                f"{m(recs, 'G4', 'regions', reg, 'switch', 'shape'):+.3f} |")
    lines.append("")
    g5 = [r["readings"].get("G5") for r in recs if r["readings"].get("G5")]
    if g5:
        learned = [g["learned"]["trial"] for g in g5 if g["learned"]]
        lines.append(f"G5: learned on {len(learned)} of {len(g5)} founders (trials {learned}); "
                     f"inherit round {_mean([g['inherit']['which_shape'] for g in g5]):.2f}, "
                     f"food {_mean([g['inherit']['which_food'] for g in g5]):.2f}; "
                     f"old names after {_mean([g['old_names_after'] for g in g5]):.2f}")
    if recs:
        lines.append("")
        lines.append(f"owner's room, founder {recs[0]['seed']}:")
        for x in recs[0]["readings"]["owner"]:
            lines.append(f"- {x['question']} -> {x['said']} (teacher: {x['want']})")
    return "\n".join(lines)


ti_props = ("round", "red", "food", "left", "near")


def sweep(out_dir: Path) -> str:
    """The development sweep per arm (mean over its founders): each stage's last read and its
    blocks (to mastery, or the cap), the largest efficacy at the stage's end, G3, and where in
    the depth the colour and the shape are held (the decoded share under the colour and the shape
    question in the lowest and the association region) with the question's switch."""
    arms: dict[str, list[dict[str, Any]]] = {}
    for p in sorted(out_dir.glob("*.json")):
        name = p.stem.rsplit("_", 1)[0]
        if name not in ARMS:
            continue
        try:
            rec = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "readings" in rec:
            arms.setdefault(name, []).append(rec)
    stages = [s for s, _ in tp.STAGES]
    head = ["arm", "n"] + [f"{s} (blocks)" for s in stages] + ["G3", "max eff", "params",
                                                                 "MACs"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, recs in arms.items():
        cells = [name, str(len(recs))]
        for st in stages:
            reads, blocks = [], []
            for r in recs:
                e = next((x for x in r["stages"] if x["stage"] == st), None)
                if e is None:
                    continue
                if e["curve"]:
                    reads.append(e["curve"][-1]["read"])
                blocks.append(e.get("blocks", 0))
            cells.append(f"{_mean(reads):.2f} ({_mean(blocks):.0f})")
        sm = [summary(r) for r in recs]
        cells.append(f"{_mean([s.get('G3', float('nan')) for s in sm]):.2f}")
        cells.append(f"{_mean([s.get('max_eff', float('nan')) for s in sm]):.2f}")
        cells.append(f"{_mean([float(r['parameters']) for r in recs]):.3g}")
        cells.append(f"{_mean([float(r['macs']) for r in recs]):.3g}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    head = ["arm", "region", "colour decoded: colour q / shape q", "shape decoded: colour q / "
            "shape q", "switch colour", "switch shape"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, recs in arms.items():
        regs = recs[0]["readings"].get("G4", {}).get("regions", {})
        for reg in regs:
            def q(qn: str, key: str, reg: str = reg,
                  recs: list[dict[str, Any]] = recs) -> float:
                return _mean([r["readings"]["G4"]["regions"][reg][qn][key] for r in recs])

            def sw(key: str, reg: str = reg, recs: list[dict[str, Any]] = recs) -> float:
                return _mean([r["readings"]["G4"]["regions"][reg]["switch"][key] for r in recs])
            lines.append(f"| {name} | {reg} | {q('colour', 'colour_decoded'):.2f} / "
                         f"{q('shape', 'colour_decoded'):.2f} | {q('colour', 'shape_decoded'):.2f}"
                         f" / {q('shape', 'shape_decoded'):.2f} | {sw('colour'):+.3f} | "
                         f"{sw('shape'):+.3f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["life", "control", "read", "detail", "pointed", "sweep"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--no-pistachio", action="store_true")
    ap.add_argument("--brain-record")
    ap.add_argument("--arm", default="d4a25")
    a = ap.parse_args()
    if a.what == "life":
        seed, arm, out = int(a.args[0]), a.args[1], Path(a.args[2])
        life(seed, arm, out, pistachio=not a.no_pistachio)
    elif a.what == "control":
        seed, kind, out = int(a.args[0]), a.args[1], Path(a.args[2])
        control(seed, kind, out, Path(a.brain_record) if a.brain_record else None, a.arm)
    elif a.what == "sweep":
        print(sweep(Path(a.args[0])))
    elif a.what == "pointed":
        pointed(int(a.args[0]), a.args[1], Path(a.args[2]))
    elif a.what == "detail":
        print(detail(Path(a.args[0]), a.args[1], a.args[2] if len(a.args) > 2 else None))
    else:
        print(table(Path(a.args[0]), a.args[1:] or None))


if __name__ == "__main__":
    main()
