"""K9: every number in the M7 card matches the committed reports (the M4/M5 card checks).

Report keys hold dots (checkpoint fractions "0.25", variant "sim_inflate_1.5"), so the card's
paths separate keys with "/".
"""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.test_model_card_m4 import DECIMAL, MINUS, shown

REPO = Path(__file__).parent.parent
DOC = REPO / "docs" / "models" / "m7.md"
ROW = re.compile(
    r"^\| (?P<label>[^|]+) \| (?P<value>[^|]+) \| `(?P<file>[\w.]+):(?P<path>[\w./-]+)` \|$"
)


def rows() -> list[dict[str, str]]:
    return [
        m.groupdict()
        for line in DOC.read_text(encoding="utf-8").splitlines()
        if (m := ROW.match(line))
    ]


def lookup(report: Any, path: str) -> Any:
    for key in path.split("/"):
        report = report[int(key)] if isinstance(report, list) else report[key]
    return report


def test_every_table_value_matches_its_report_path() -> None:
    table = rows()
    assert len(table) >= 40
    reports = {
        name: json.loads((REPO / "reports" / name).read_text(encoding="utf-8"))
        for name in {r["file"] for r in table}
    }
    wrong = []
    for r in table:
        value = r["value"].strip().replace(MINUS, "-")
        got = shown(lookup(reports[r["file"]], r["path"]), value)
        if got != value:
            wrong.append(f"{r['file']}:{r['path']} doc {value} report {got}")
    assert wrong == []


def test_every_decimal_in_the_prose_is_a_table_value() -> None:
    prose = DOC.read_text(encoding="utf-8").split("## Numbers")[0].replace(MINUS, "-")
    values = {r["value"].strip().replace(MINUS, "-").lstrip("-") for r in rows()}
    missing = sorted({n for n in DECIMAL.findall(prose) if n.lstrip("-") not in values})
    assert missing == []


@pytest.mark.parametrize("term", ["Checkpoint", "Strength posterior", "Cut line", "Simulation log"])
def test_context_defines_the_m7_terms(term: str) -> None:
    context = (REPO / "docs" / "CONTEXT.md").read_text(encoding="utf-8")
    assert f"**{term}**:" in context
