"""Run the reader's actual introductory snippets and check local documentation links."""

import re
import runpy
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
# A block right after this marker is illustrative (it reads data the page cannot build) and is
# not run; every other ``python`` block on a page is.
NOT_RUN = "<!-- not-run"
BLOCK = re.compile(r"(<!-- not-run[^\n]*-->\n)?```python\n(.*?)```", re.S)
# pages whose blocks need an optional dependency
REQUIRES = {"docs/backends.md": "torch", "docs/population.md": "torch"}


def snippets(page: str) -> list[str]:
    return [code for marker, code in BLOCK.findall((ROOT / page).read_text()) if not marker]


def test_continuing_brain_example():
    """Exercise real outcomes and saved feedback, including the private graph control."""
    example = runpy.run_path(str(ROOT / "examples/continuing_brain.py"))
    report = example["run"]()
    assert report["saved_feedback_continuation_identical"]
    assert report["free_queries_preserved_live_checkpoint"]
    assert report["associative_writes"] == report["executed_transitions"]
    for outcome in report["outcomes"]:
        # Validate the witness independently of the example's label/execute helpers.
        cue = outcome["cue"]
        changed = outcome["phase"] in {"changed_environment", "repair", "continued_use"}
        expected = cue if changed else 1 - cue
        assert outcome["observed_target"] == expected
        assert outcome["reward"] == float(outcome["action"] == expected)
        if outcome["next_teacher"] is not None:
            assert outcome["reward"] == 0.0
            assert outcome["phase"] in {"bootstrap", "repair"}
            assert outcome["next_cue"] == cue
            assert outcome["next_teacher"] == expected
        else:
            assert outcome["next_cue"] == 1 - cue
        free = outcome["free_answer"]
        assert free["scope"] == "free_answer" and free["operation"] == "act"
        assert free["qualified"] and free["max_residual"] <= free["tolerance"]
        assert free["steps"] <= free["budget"]
        work = outcome["step_settlement_work"]
        assert work["sweeps"] >= free["steps"] + work["nudged_sweeps"]
    for previous, current in zip(report["outcomes"], report["outcomes"][1:], strict=False):
        if previous["phase"] != current["phase"]:
            # The first scored action of a new phase must not carry a teacher.
            assert previous["next_teacher"] is None
    for stage in report["stages"]:
        rows = [row for row in report["outcomes"] if row["phase"] == stage["phase"]]
        assert stage["executed_transitions"] == len(rows)
        assert stage["correct_actions"] == sum(row["reward"] for row in rows)
        assert stage["corrective_lessons"] == sum(row["next_teacher"] is not None for row in rows)
        assert stage["free_answer_sweeps"] == sum(row["free_answer"]["steps"] for row in rows)
        if stage["phase"] in {"unchanged_environment", "changed_environment", "continued_use"}:
            assert stage["corrective_lessons"] == 0
            # No teacher does not mean free of reward eligibility work.
            assert sum(row["step_settlement_work"]["nudged_sweeps"] for row in rows) > 0


PAGES = [
    "README.md",
    "docs/world-model.md",
    "docs/patchnet.md",
    "docs/temporal.md",
    "docs/architecture.md",
    "docs/temporal-memory.md",
    "docs/quickstart.md",
    "docs/concepts.md",
    "docs/memory.md",
    "docs/continuous.md",
    "docs/tasks.md",
    "docs/reward.md",
    "docs/certificate.md",
    "docs/cortex.md",
    "docs/brain.md",
    "docs/context.md",
    "docs/episodic.md",
    "docs/evolution.md",
    "docs/build.md",
    "docs/belief.md",
    "docs/steering.md",
    "docs/recursive-settlement.md",
    "docs/recursive-training.md",
    "docs/api.md",
    "docs/backends.md",
    "docs/interaction.md",
    "docs/learning.md",
    "docs/partitioned.md",
    "docs/planning.md",
    "docs/population.md",
    "docs/protocols.md",
    "docs/receipts.md",
    "docs/record-patch.md",
]


@pytest.mark.parametrize("page", PAGES)
def test_introductory_python_snippets(page, tmp_path, monkeypatch):
    if page in REQUIRES:
        pytest.importorskip(REQUIRES[page])
    monkeypatch.chdir(tmp_path)
    blocks = snippets(page)
    assert blocks, f"{page} has no runnable python block; drop it from PAGES"
    # a reader runs the page as a script of their own; ``__file__`` names it
    script = tmp_path / "documentation_example.py"
    script.write_text("\n\n".join(blocks))
    namespace = {"__name__": "documentation_example", "__file__": str(script)}
    for index, code in enumerate(blocks):
        exec(compile(code, f"{page}:python-block-{index + 1}", "exec"), namespace)
    if page in {"README.md", "docs/quickstart.md", "docs/world-model.md", "docs/brain.md"}:
        assert namespace["brain"].last_arousal is not None
    if page == "docs/quickstart.md":
        assert namespace["brain"].learner.updates > 0
        assert len(namespace["phases"]) == 2
        assert (namespace["continued"] == namespace["replayed"]).all()
        assert namespace["brain"].arousal.to_dict() == namespace["resumed"].arousal.to_dict()
        assert namespace["brain"].last_arousal == namespace["resumed"].last_arousal


def test_every_python_block_is_run_or_marked_illustrative():
    """A new page with python blocks must join PAGES, or mark each block ``<!-- not-run -->``."""
    # The preserved engine has its own executable-documentation parametrization.
    # Read that owner's actual inventory so adding an untested page still fails.
    experimental = runpy.run_path(str(ROOT / "tests/equilibrium/test_documentation.py"))
    covered = set(PAGES) | {path.relative_to(ROOT).as_posix() for path in experimental["DOCUMENTS"]}
    unrun = [
        page.relative_to(ROOT).as_posix()
        for page in sorted((ROOT / "docs").rglob("*.md"))
        if snippets(page.relative_to(ROOT).as_posix())
        and page.relative_to(ROOT).as_posix() not in covered
    ]
    assert not unrun, f"python blocks never executed: {unrun}"


def test_local_documentation_links_resolve():
    problems = []
    for page in [
        ROOT / "README.md",
        ROOT / "MANIFESTO.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "AGENTS.md",
        *sorted((ROOT / "docs").rglob("*.md")),
    ]:
        for target in re.findall(r"\[[^\]\n]*\]\(([^\s)]+)\)", page.read_text()):
            if re.match(r"[a-z]+:", target):
                continue
            path, _, anchor = unquote(target).partition("#")
            destination = (page.parent / path).resolve() if path else page
            if not destination.exists():
                problems.append(f"{page.name}: missing {target}")
            elif anchor and destination.suffix == ".md":
                content = destination.read_text()
                headings = re.findall(r"^#+\s+(.+)$", content, re.M)
                slugs = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
                if anchor not in slugs and f'id="{anchor}"' not in content:
                    problems.append(f"{page.name}: missing anchor {target}")
    assert not problems, "\n".join(problems)


def test_minimal_install_ci_guides_exist():
    """The installed-wheel smoke must not depend on a retired source guide."""
    workflow = ROOT / ".github/workflows/ci.yml"
    if not workflow.exists():
        pytest.skip("CI workflows are not included in the source distribution")
    pages = re.findall(r'"(docs/[^"\n]+\.md)"', workflow.read_text())
    assert pages, "The minimal-install job must exercise documented examples"
    assert all((ROOT / page).is_file() for page in pages)
