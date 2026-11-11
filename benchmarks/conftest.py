"""A chamber runs from its own directory, and its modules import their siblings by plain names
(``import world``). One pytest process collects every chamber, so two directories that each
hold a ``world.py`` would hand each other their modules in the order pytest reaches them (#306).
While a test module is collected and while each test runs, the plain names of the chamber's own
modules name its own files: a sibling module another chamber loaded under that name is set aside
and handed back when that chamber's tests run."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent
_ASIDE: dict[tuple[Path, str], ModuleType] = {}  # (directory, name): a chamber's module set aside


def _chamber(path: Path) -> Path | None:
    """The directory of the benchmark module at ``path``, or None outside the benchmarks."""
    directory = Path(path).resolve().parent
    return directory if ROOT in directory.parents else None


def _own_names(directory: Path) -> None:
    """Point each plain module name of ``directory`` at that directory's own file."""
    names = {p.stem for p in directory.glob("*.py")} - {"conftest", "__init__"}
    for name in names:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if module is not None and origin:
            home = Path(origin).resolve().parent
            if home == directory or ROOT not in home.parents:
                continue  # its own, or no chamber's (a library or the standard library)
            _ASIDE[(home, name)] = sys.modules.pop(name)
        aside = _ASIDE.pop((directory, name), None)
        if aside is not None and name not in sys.modules:
            sys.modules[name] = aside
    here = str(directory)
    if here in sys.path and sys.path[0] != here:
        # a chamber that put itself on the path finds its own files first
        sys.path.remove(here)
        sys.path.insert(0, here)


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    if isinstance(collector, pytest.Module):
        directory = _chamber(collector.path)
        if directory is not None:
            _own_names(directory)
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    directory = _chamber(item.path)
    if directory is not None:
        _own_names(directory)
    return (yield)
