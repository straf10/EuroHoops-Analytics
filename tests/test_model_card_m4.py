"""I9: every number in the M4 card and the entity data doc matches the committed reports."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import REPO

DOCS = (REPO / "docs" / "models" / "m4.md", REPO / "docs" / "data" / "entity.md")
ROW = re.compile(
    r"^\| (?P<label>[^|]+) \| (?P<value>[^|]+) \| `(?P<file>[\w.]+):(?P<path>[\w.]+)` \|$"
)
DECIMAL = re.compile(r"-?\d+\.\d{3,}")
MINUS = "\u2212"


def lookup(report: Any, path: str) -> Any:
    for key in path.split("."):
        report = report[int(key)] if isinstance(report, list) else report[key]
    return report


def shown(value: Any, text: str) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, int) and "." not in text:
        return str(value)
    decimals = len(text.split(".")[1]) if "." in text else 0
    return f"{float(value):.{decimals}f}"


def rows(doc: Path) -> list[dict[str, str]]:
    return [
        m.groupdict()
        for line in doc.read_text(encoding="utf-8").splitlines()
        if (m := ROW.match(line))
    ]


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_every_table_value_matches_its_report_path(doc: Path) -> None:
    table = rows(doc)
    assert len(table) >= 10
    reports = {
        name: json.loads((REPO / "reports" / name).read_text(encoding="utf-8"))
        for name in {r["file"] for r in table}
    }
    wrong = [
        f"{r['file']}:{r['path']} doc {r['value'].strip()} report "
        f"{shown(lookup(reports[r['file']], r['path']), r['value'].strip())}"
        for r in table
        if shown(lookup(reports[r["file"]], r["path"]), r["value"].strip()) != r["value"].strip()
    ]
    assert wrong == []


@pytest.mark.parametrize("doc", DOCS, ids=lambda d: d.name)
def test_every_decimal_in_the_prose_is_a_table_value(doc: Path) -> None:
    prose = doc.read_text(encoding="utf-8").split("## Numbers")[0].replace(MINUS, "-")
    values = {r["value"].strip().lstrip("-") for r in rows(doc)}
    missing = sorted({n for n in DECIMAL.findall(prose) if n.lstrip("-") not in values})
    assert missing == []
