"""Keep source and distribution identities consistent before building a wheel."""

import tomllib
from pathlib import Path

import cadence


def test_package_version_matches_distribution_metadata():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert cadence.__version__ == metadata["project"]["version"]


def test_documented_installation_matches_distribution_identity():
    root = Path(__file__).resolve().parents[1]
    assert f"cadence-net=={cadence.__version__}" in (root / "README.md").read_text()
