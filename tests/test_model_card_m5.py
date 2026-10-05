"""J9: every number in the M5 card matches the committed reports (the M4 card's checks)."""

import json
from pathlib import Path

import pytest

from tests.test_model_card_m4 import DECIMAL, MINUS, lookup, rows, shown

DOC = Path(__file__).parent.parent / "docs" / "models" / "m5.md"
REPORTS = Path(__file__).parent.parent / "reports"


def test_every_table_value_matches_its_report_path() -> None:
    table = rows(DOC)
    assert len(table) >= 30
    reports = {
        name: json.loads((REPORTS / name).read_text(encoding="utf-8"))
        for name in {r["file"] for r in table}
    }
    wrong = [
        f"{r['file']}:{r['path']} doc {r['value'].strip()} report "
        f"{shown(lookup(reports[r['file']], r['path']), r['value'].strip())}"
        for r in table
        if shown(lookup(reports[r["file"]], r["path"]), r["value"].strip()) != r["value"].strip()
    ]
    assert wrong == []


def test_every_decimal_in_the_prose_is_a_table_value() -> None:
    prose = DOC.read_text(encoding="utf-8").split("## Numbers")[0].replace(MINUS, "-")
    values = {r["value"].strip().lstrip("-") for r in rows(DOC)}
    missing = sorted({n for n in DECIMAL.findall(prose) if n.lstrip("-") not in values})
    assert missing == []


@pytest.mark.parametrize(
    "term", ["Projected roster", "Oracle roster", "Team residual", "Rest difference"]
)
def test_context_defines_the_m5_terms(term: str) -> None:
    context = (Path(__file__).parent.parent / "docs" / "CONTEXT.md").read_text(encoding="utf-8")
    assert f"**{term}**:" in context
