"""Keep source and distribution identities consistent before building a wheel."""

import re
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

import cadence


def test_package_version_matches_distribution_metadata():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    assert cadence.__version__ == metadata["project"]["version"]


def test_release_documentation_matches_package_identity():
    root = Path(__file__).resolve().parents[1]
    for name in ("README.md", "docs/QUICKSTART.md", "docs/MIGRATION_060.md"):
        document = (root / name).read_text()
        assert f"cadence-net=={cadence.__version__}" in document, name


def test_pypi_readme_links_resolve_without_a_repository_base():
    root = Path(__file__).resolve().parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text())
    readme = (root / metadata["project"]["readme"]).read_text()
    targets = re.findall(r"\]\(([^)]+)\)", readme)
    assert targets
    for target in targets:
        assert target.startswith("#") or urlsplit(target).scheme == "https", target
