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
ARMS["d2a25sur"] = {"modules": [160, 128], "apical": 0.25, "learning_surprise": True}

EXAM = ("colour", "relation", "where", "which_shape", "which_colour", "which_size",
        "which_material", "which_food", "can_eat", "know_shape", "know_food")
SENTENCES = ("describe", "describe_rel")
CAPS = {"words": 200, "properties": 400, "relations": 400, "questions": 300, "sentences": 300,
        "scenes": 400}
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
                  macs=policy.macs(), lessons=policy.lessons)
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


def plan_of(record: dict[str, Any]) -> list[tuple[str, tuple[str, ...], int]]:
    """The brain's curriculum as it lived it: each stage's forms and blocks."""
    forms = dict(tp.STAGES)
    out = []
    for st in record["stages"]:
        blocks = st.get("blocks", st["mastered"] or (st["curve"][-1]["block"] if st["curve"]
                                                     else 0))
        out.append((st["stage"], forms[st["stage"]], int(blocks)))
    return out


def control(seed: int, kind: str, out_dir: Path, brain_record: Path | None) -> dict[str, Any]:
    import textcontrols as tcl

    out_dir.mkdir(parents=True, exist_ok=True)
    layout = tc.Layout.founder(seed)
    if brain_record is not None and brain_record.exists():
        rec = json.loads(brain_record.read_text())
        plan, parameters = plan_of(rec), int(rec["parameters"])
    else:
        plan = [(s, f, CAPS[s] // 2) for s, f in tp.STAGES]
        parameters = 160000
    exchanges = sum(b for _, _, b in plan) * 64
    t = time.time()
    if kind in ("lstm-online", "transformer-online"):
        policy: Any = tcl.online(kind.split("-")[0], layout, parameters, seed, plan)
    elif kind in ("lstm-offline", "transformer-offline"):
        policy = tcl.offline(kind.split("-")[0], layout, parameters, seed,
                             max(640, min(exchanges, 20000)))
    else:
        policy = tcl.plain(kind, layout, seed, plan)
    seconds = time.time() - t
    record: dict[str, Any] = {"kind": kind, "seed": seed, "parameters": policy.parameters(),
                              "updates": getattr(policy, "updates", 0),
                              "macs": getattr(policy, "macs", 0), "train_seconds": seconds,
                              "plan": plan}
    record["readings"] = readings(policy, layout, seed)
    Path(out_dir / f"{kind}_{seed}.json").write_text(json.dumps(record, indent=1, default=float))
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
            "pairs", "held_rel", "held_depth", "probes", "new_obj", "word_alone", "same_thing",
            "G5_trials",
            "blocks", "max_eff", "parameters", "macs"]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["life", "control", "read"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--no-pistachio", action="store_true")
    ap.add_argument("--brain-record")
    a = ap.parse_args()
    if a.what == "life":
        seed, arm, out = int(a.args[0]), a.args[1], Path(a.args[2])
        life(seed, arm, out, pistachio=not a.no_pistachio)
    elif a.what == "control":
        seed, kind, out = int(a.args[0]), a.args[1], Path(a.args[2])
        control(seed, kind, out, Path(a.brain_record) if a.brain_record else None)
    else:
        print(table(Path(a.args[0]), a.args[1:] or None))


if __name__ == "__main__":
    main()
