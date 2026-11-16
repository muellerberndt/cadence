"""The settings table of docs/tune.md names every setting ``Brain.compose`` takes, and no other.

The table is written by ``tools/settings_table.py``; after a new setting, run
``python tools/settings_table.py --write``. The source distribution ships neither ``tools/``
nor the guides' generator, so the test skips there.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "settings_table.py"


@pytest.fixture(scope="module")
def table():
    if not SCRIPT.exists():
        pytest.skip("tools/ is not part of the source distribution")
    spec = importlib.util.spec_from_file_location("settings_table", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_table_names_every_setting_compose_takes(table):
    named = table.table_names((ROOT / "docs" / "tune.md").read_text())
    settable = set(table.settable())
    missing = sorted(settable - named)
    assert not missing, f"run python tools/settings_table.py --write: missing {missing}"


def test_the_table_names_no_setting_compose_refuses(table):
    named = table.table_names((ROOT / "docs" / "tune.md").read_text())
    settable = set(table.settable())
    extra = sorted(named - settable)
    assert not extra, f"the table names settings compose does not take: {extra}"
