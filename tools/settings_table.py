"""The settings table of docs/tune.md, generated from the library.

    python tools/settings_table.py            # print the table
    python tools/settings_table.py --write    # write it into docs/tune.md between its markers

Every name ``Brain.compose`` takes and a user can set is found by probing: each candidate from
the library's own tables is passed alone to a small brain with a value no validator accepts, and
a name refused as unknown is not taken. Names refused at any value (the learner fields a teaching
point sets), retired names and ``image`` (the picture learned fields read, set by ``inputs``) are
left out. Each setting's value is a new brain's: what ``describe()`` reports for brains composed
with each part, else the founders the library holds (``Recall``, the long-term records, the held
message, the learned code, maturity, the configurations), else the default of the signature.
"""

from __future__ import annotations

import inspect
import sys
import warnings
from collections.abc import Iterable
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TUNE = ROOT / "docs" / "tune.md"
BEGIN, END = "<!-- settings-table:begin -->", "<!-- settings-table:end -->"


def _cadence() -> Any:
    sys.path.insert(0, str(ROOT / "src"))
    import cadence

    return cadence


class _Sentinel:
    """A value no validator accepts."""


def candidates() -> set[str]:
    """Every name the library's own tables could make ``Brain.compose`` take."""
    cd = _cadence()
    from cadence import _configuration as cfg
    from cadence import generic as g
    from cadence.code import LearnedCode
    from cadence.stream import Recall

    out: set[str] = set()
    for function in (cd.Brain.compose, cd.Brain.__init__):
        out |= {
            name
            for name, parameter in inspect.signature(function).parameters.items()
            if parameter.kind is not parameter.VAR_KEYWORD and name != "self"
        }
    for prefix, members in cfg._FIELDS.items():
        out |= {f"{prefix}_{name}" for name in members}
    out |= {"temperature", "recall_founder", "bound", "voice", "learning", "reward", "arousal"}
    out |= {f"recall_{name}" for name in Recall.founders()}
    for table in ("OPTIONS", "CONSTRUCTION", "CONSTRUCTION_OPTIONS", "RUNTIME"):
        out |= {f"recall_{name}" for name in getattr(Recall, table, ())}
    out |= {f"learned_code_{name}" for name in LearnedCode.GENES}
    out |= {f"longterm_{name}" for name in g._LONGTERM_FOUNDERS}
    out |= {f"message_{name}" for name in g._MESSAGE_FOUNDERS}
    out |= set(g._EPISODIC_SETTINGS) | set(g._MATURITY_FOUNDERS) | set(g._CONSTRUCTION)
    out |= set(g._COMPOSED_SETTINGS)
    return out


def settable() -> list[str]:
    """The names ``Brain.compose`` takes and a user can set, sorted."""
    cd = _cadence()
    from cadence import _configuration as cfg

    left_out = set(cfg.RETIRED) | set(cfg._TAUGHT) | {"image", "inputs", "actions"}
    taken = {"inputs", "actions"}
    for name in sorted(candidates() - left_out):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cd.Brain.compose(inputs=4, actions=2, **{name: _Sentinel()})
        except TypeError as error:
            message = str(error)
            if "unknown" in message or "unexpected keyword" in message:
                continue
        except Exception:  # noqa: BLE001 - any refusal of the value means the name is read
            pass
        taken.add(name)
    retired_recall = {f"recall_{name}" for name in _recall_retired()}
    return sorted(taken - retired_recall)


def _recall_retired() -> Iterable[str]:
    _cadence()
    from cadence.stream import Recall

    return getattr(Recall, "RETIRED", {})


# Brains composed with each part, whose ``describe()`` holds a new brain's value of that part's
# settings: (the composition, the prefixes read from it); a name the composition sets is not read
PROBES: tuple[tuple[dict[str, Any], tuple[str, ...]], ...] = (
    ({"inputs": 4, "actions": 2}, ("",)),
    (
        {
            "inputs": 4,
            "actions": 2,
            "arousal": True,
            "maturity": True,
            "memory_choice": True,
            "efference_amplitude": 1.0,
        },
        ("arousal_", "maturity_", "memory_", "efference_"),
    ),
    (
        {
            "inputs": 8,
            "actions": 2,
            "message": 12,
            "message_boundary": 7,
            "message_boundary_holds": True,
            "longterm": 8,
            "message_read": True,
        },
        ("message_", "longterm_"),
    ),
    ({"inputs": 8, "actions": 2, "learned_code": 16, "attention": True}, ("learned_code_",)),
)

# What a value means where the number alone says too little
NOTES: dict[str, str] = {
    "inputs": "a width, or an image shape `(height, width)`",
    "actions": "the count of motor neurons",
    "senses": "unset (`{name: width}` declares senses by name, in place of `inputs`)",
    "recall": "on up to 4,096 code coordinates (`False` builds none)",
    "recall_founder": "unset (`True` takes the memory's own founders)",
    "episodic": "`True` (`False` builds no associative store)",
    "lateral": "unset, which is `0.0`: no lateral synapses",
    "backend": '`"auto"`: chosen by size at the first moment',
    "arousal": "off (`True` is the founders, a mapping patches them)",
    "surprise": "off (`True` is the founders, a mapping names genes)",
    "learning": "the composed learner (a `LearnerConfig` replaces it whole)",
    "reward": "the composed actor (an `ActorCriticConfig` replaces it whole)",
    "teaching": '`"deep"` (`"founder"` with `recall=False`)',
    "trained": "unset (module names, module files or a map)",
    "bound": "unset (a count of records, with `senses=` and `code=`)",
    "voice": "unset (a declared sense, with `bound=`)",
    "code": "unset: the observation itself",
    "connectome": "unset: the composed layout",
    "history_decays": "`(0.0,)`: the moment before (`()` builds none)",
    "efference_amplitude": "`0.0`: none (a positive amplitude at construction adds the copy)",
    "dense_limit": "unset: 2,048, or the blocks' width in a brain with the memory",
}

# Each family's guide, in the order the table lists them: (names, title, link)
FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("inputs actions senses seed", "start here", "start.md#1-a-brain-has-senses-and-actions"),
    ("slots lateral", "motor competition", "#operating-point-and-motor-competition"),
    ("modules observers apical tied scales connectome", "the brain's layout", "#brain"),
    ("features field learned_fields ports fields_motor", "image inputs", "senses.md#image-inputs"),
    ("code learned_code learned_code_fields attention", "codes", "senses.md#the-observation-code"),
    ("trained", "modules", "modules.md#modules-parts-that-arrive-as-files"),
    ("teaching temperature learning learning_", "teaching", "teach.md#choosing-rates-and-steps"),
    ("reward actor_", "reward", "reward.md#settings"),
    ("arousal arousal_", "routine and repair", "start.md#routine-and-repair-live"),
    ("maturity maturity_", "maturity", "reward.md#maturity-opt-in"),
    ("surprise", "surprise", "start.md#surprise-and-the-focus-it-draws-opt-in"),
    ("working_memory_ carried_gain", "the working trace", "memory.md#working-trace"),
    ("history_", "the heard history", "memory.md#heard-history"),
    ("efference_", "the efference copy", "memory.md#efference-copy"),
    ("recall recall_", "the recall memory", "memory.md#recall-settings"),
    (
        "episodic consolidation memory_",
        "the associative and episodic memory",
        "memory.md#associative-store-the-hippocampus",
    ),
    ("message message_", "the held message", "memory.md#held-message"),
    ("longterm longterm_", "long-term records", "memory.md#long-term-records"),
    ("bound voice", "records of bound moments", "memory.md#records-of-bound-moments-opt-in"),
    ("backend device precision dense_limit", "backends", "backends.md#which-backend-should-i-use"),
)


def _family(name: str) -> int:
    for index, (members, _, _) in enumerate(FAMILIES):
        for member in members.split():
            if name == member or (member.endswith("_") and name.startswith(member)):
                return index
    return len(FAMILIES)


def values(names: Iterable[str]) -> dict[str, Any]:
    """A new brain's value of each setting."""
    cd = _cadence()
    from cadence import _configuration as cfg
    from cadence import generic as g
    from cadence.code import LearnedCode
    from cadence.stream import Recall

    found: dict[str, Any] = {}
    for probe, prefixes in PROBES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            genes = cd.Brain.compose(**probe).describe()["genes"]
        for name, value in genes.items():
            if name not in probe and name.startswith(prefixes):
                found.setdefault(name, value)
    tables = {
        "recall_": {**Recall.founders(), **Recall.options()},
        "longterm_": g._LONGTERM_FOUNDERS,
        "message_": g._MESSAGE_FOUNDERS,
        "learned_code_": LearnedCode.founders(),
    }
    for prefix, table in tables.items():
        for name, value in table.items():
            found.setdefault(prefix + name, value)
    for name, value in g._MATURITY_FOUNDERS.items():
        found.setdefault(name, value)
    for prefix, kind in cfg._KINDS.items():
        for field in fields(kind):
            if field.default is not MISSING:  # a plain default
                found.setdefault(f"{prefix}_{field.name}", field.default)
    for function in (cd.Brain.compose, cd.Brain.__init__):
        for name, parameter in inspect.signature(function).parameters.items():
            if parameter.default is not parameter.empty:
                found.setdefault(name, parameter.default)
    return {name: found.get(name) for name in names}


def construction(names: Iterable[str]) -> dict[str, bool]:
    """Whether each setting is fixed at construction (``retune`` refuses it)."""
    cd = _cadence()
    from cadence import generic as g

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        brain = cd.Brain.compose(inputs=4, actions=2, arousal=True, maturity=True)
    out = {}
    for name in names:
        if name in g._CONSTRUCTION or name in ("senses", "recall_founder", "arousal"):
            out[name] = True
            continue
        try:
            brain.retune(**{name: _Sentinel()})
            out[name] = False
        except Exception as error:  # noqa: BLE001 - the refusal's words say which
            out[name] = "construction" in str(error)
    return out


def _shown(name: str, value: Any) -> str:
    if name in NOTES:
        return NOTES[name]
    if value is None:
        return "unset"
    if isinstance(value, str):
        return f'`"{value}"`'
    if isinstance(value, list):
        value = tuple(value)
    return f"`{value!r}`"


def rows(names: list[str] | None = None, per_row: int = 5) -> list[str]:
    """The table's rows: each family split into what construction fixes and what ``retune``
    changes, a few settings a row."""
    names = settable() if names is None else names
    value = values(names)
    fixed = construction(names)
    order = sorted(names, key=lambda name: (_family(name), name))
    out = [
        "| Setting | In a new brain | Changed at | Guide |",
        "| --- | --- | --- | --- |",
    ]
    for index in range(len(FAMILIES) + 1):
        members = [name for name in order if _family(name) == index]
        title, link = FAMILIES[index][1:] if index < len(FAMILIES) else ("the API", "api.md")
        for when in (True, False):
            group = [name for name in members if fixed[name] is when]
            chunks = [[name] for name in group if name in NOTES]
            plain = [name for name in group if name not in NOTES]
            chunks += [plain[start : start + per_row] for start in range(0, len(plain), per_row)]
            for chunk in chunks:
                settings = ", ".join(f"`{name}`" for name in chunk)
                shown = "; ".join(_shown(name, value[name]) for name in chunk)
                out.append(
                    f"| {settings} | {shown} | {'construction' if when else 'retune'} "
                    f"| [{title}]({link}) |"
                )
    return out


def table_names(text: str) -> set[str]:
    """The settings the table between the markers names, from its first column."""
    body = text.split(BEGIN, 1)[1].split(END, 1)[0]
    names: set[str] = set()
    for line in body.splitlines():
        if line.startswith("| `"):
            first = line.split("|")[1]
            names |= {part.strip().strip("`") for part in first.split(",")}
    return names


def write() -> None:
    text = TUNE.read_text()
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    TUNE.write_text(head + BEGIN + "\n" + "\n".join(rows()) + "\n" + END + tail)


if __name__ == "__main__":
    if "--write" in sys.argv[1:]:
        write()
    else:
        print("\n".join(rows()))
