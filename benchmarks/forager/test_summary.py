"""The forager chamber's summary: a gate's verdict is read on the declared panel alone."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import summary  # noqa: E402

RECEIPT = Path(__file__).with_name("results") / "declared.json"


def lives() -> list[dict]:
    return json.loads(RECEIPT.read_text())


def read(rows: list[dict], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """What the summary prints for ``rows``, given as one file."""
    path = tmp_path / "lives.json"
    path.write_text(json.dumps(rows))
    assert summary.main([str(path)]) == 0
    return capsys.readouterr().out


def test_the_declared_receipt_keeps_its_verdicts_and_less_one_life_is_incomplete(tmp_path, capsys):
    assert summary.main([str(RECEIPT)]) == 0
    whole = capsys.readouterr().out
    assert "F1: 12 of 12 -> holds" in whole and "F2: 6 of 12 -> fails" in whole
    assert "incomplete" not in whole and "not counted" not in whole
    less = [life for life in lives() if (life["founder"], life["arm"]) != (1011105, "both")]
    out = read(less, tmp_path, capsys)
    assert "F1: 11 of 11 -> incomplete" in out and "F2: 6 of 11 -> incomplete" in out
    assert "holds" not in out and "fails" not in out
    assert out.count("not given: 1011105\n") == 1  # the founder whose pair is missing, by name
    assert "| 1011105 | code |" in out  # its other lives stay in the table


def test_one_founder_of_the_declared_panel_reads_incomplete_with_its_table(tmp_path, capsys):
    out = read([life for life in lives() if life["founder"] == 1011100], tmp_path, capsys)
    assert "F1: 1 of 1 -> incomplete" in out and "F2: 1 of 1 -> incomplete" in out
    assert "holds" not in out and "fails" not in out
    assert "not given: " + ", ".join(str(f) for f in range(1011101, 1011112)) + "\n" in out
    assert "| 1011100 | both |" in out and "both (1): taught 0.704" in out
    assert "own (no bar): both above code on 1 of 1" in out


def test_a_second_life_of_a_founders_arm_is_refused(tmp_path, capsys):
    rows = lives()
    again = dict(rows[0], mean=dict(rows[0]["mean"], teacher=0.9))  # founder 1011100, code
    path = tmp_path / "twice.json"
    path.write_text(json.dumps([*rows, again]))
    with pytest.raises(SystemExit, match="founder 1011100 already has a life of the arm 'code'"):
        summary.main([str(path)])
    with pytest.raises(SystemExit, match="declared.json: founder 1011100 already has a life"):
        summary.main([str(RECEIPT), str(RECEIPT)])  # the same receipt given twice
    assert capsys.readouterr().out == ""  # refused before a table is printed


def test_founders_outside_the_declared_panel_are_summarised_and_counted_in_no_verdict(
        tmp_path, capsys):
    # two development founders, 0 and 1, with the readings of two declared lives
    development = [dict(life, founder=life["founder"] - 1011100)
                   for life in lives() if life["founder"] < 1011102]
    out = read(development, tmp_path, capsys)
    assert "F1: 2 of 2 -> incomplete" in out and "F2: 1 of 2 -> incomplete" in out
    assert "holds" not in out and "fails" not in out and "| 0 | code |" in out
    out = read([*lives(), *development], tmp_path, capsys)
    assert "F1: 12 of 12 -> holds" in out and "F2: 6 of 12 -> fails" in out
    assert "not counted, outside the declared founders: 0, 1\n" in out
    assert "own (no bar): both above code on 12 of 12" in out
    assert "| 1 | both |" in out and "code (14):" in out  # they stay in the tables


@pytest.mark.parametrize("rows", [[], lives()[:1]])
def test_zero_complete_pairs_still_have_two_incomplete_verdicts(rows, tmp_path, capsys):
    out = read(rows, tmp_path, capsys)
    for name in ("F1", "F2"):
        assert f"{name}: 0 of 0 -> incomplete" in out
