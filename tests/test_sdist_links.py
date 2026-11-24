"""Every local link of a page the source distribution ships leads to a file it ships."""

from __future__ import annotations

import re
import tomllib
from fnmatch import fnmatchcase
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SDIST = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
    "tool"]["hatch"]["build"]["targets"]["sdist"]


def shipped(relative: str) -> bool:
    """Whether the sdist's lists hold the file at ``relative``: under a path of ``only-include``
    and kept by ``exclude``, its patterns read as git's (one without a slash at any depth, the
    last that matches deciding, ``!`` keeping the file)."""
    if not any(relative == kept or relative.startswith(kept + "/")
               for kept in SDIST["only-include"]):
        return False
    kept = True
    for pattern in SDIST.get("exclude", []):
        glob = pattern.removeprefix("!")
        if "/" in glob:
            hit = fnmatchcase(relative, glob) or (
                glob.startswith("**/") and fnmatchcase(relative, glob[3:]))
        else:
            hit = any(fnmatchcase(part, glob) for part in relative.split("/"))
        if hit:
            kept = pattern.startswith("!")
    return kept


def test_the_pages_the_sdist_ships_link_only_what_it_ships():
    pages = [ROOT / name for name in ("README.md", "MANIFESTO.md", "CONTRIBUTING.md", "AGENTS.md")]
    pages += sorted((ROOT / "docs").rglob("*.md"))
    left_out = []
    for page in pages:
        if not shipped(page.relative_to(ROOT).as_posix()):
            continue
        for target in re.findall(r"\[[^\]\n]*\]\(([^\s)]+)\)", page.read_text(encoding="utf-8")):
            path = unquote(target).partition("#")[0]
            if re.match(r"[a-z]+:", target) or not path:
                continue
            destination = (page.parent / path).resolve()
            if ROOT != destination and ROOT not in destination.parents:
                held = False  # a file outside the project
            elif destination.is_dir():  # a folder is there where one of its files is
                held = any(shipped(file.relative_to(ROOT).as_posix())
                           for file in destination.rglob("*") if file.is_file())
            else:
                held = shipped(destination.relative_to(ROOT).as_posix())
            if not held:
                left_out.append(f"{page.relative_to(ROOT)}: {target} is not in the sdist")
    assert not left_out, "\n".join(left_out)
