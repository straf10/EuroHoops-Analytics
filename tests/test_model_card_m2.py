"""F10: every number in the M2 model card matches the committed report JSON."""

import json
import re
from typing import Any

from tests.conftest import REPO

CARD = REPO / "docs" / "models" / "m2.md"
ROW = re.compile(
    r"^\| (?P<label>[^|]+) \| (?P<value>[^|]+) \| `(?P<file>[\w.]+):(?P<path>[\w.]+)` \|$"
)
DECIMAL = re.compile(r"-?\d+\.\d{3,}")
REPORTS = ("backtest_m2.json", "shots.json", "free_throws.json", "m2_teams.json", "m2_players.json")


def lookup(report: Any, path: str) -> Any:
    for key in path.split("."):
        report = report[int(key)] if isinstance(report, list) else report[key]
    return report


def shown(value: Any, text: str) -> str:
    """The report value formatted like the card shows it (same number of decimals)."""
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    decimals = len(text.split(".")[1]) if "." in text else 0
    return f"{float(value):.{decimals}f}"


def card_rows() -> list[dict[str, str]]:
    return [
        m.groupdict()
        for line in CARD.read_text(encoding="utf-8").splitlines()
        if (m := ROW.match(line))
    ]


def test_every_table_value_matches_its_report_path() -> None:
    reports = {
        name: json.loads((REPO / "reports" / name).read_text(encoding="utf-8")) for name in REPORTS
    }
    rows = card_rows()
    assert len(rows) > 150
    wrong = []
    for r in rows:
        card = r["value"].strip()
        report = shown(lookup(reports[r["file"]], r["path"]), card)
        if report != card:
            wrong.append(f"{r['file']}:{r['path']} card {card} report {report}")
    assert wrong == []


def test_every_decimal_in_the_prose_is_a_table_value() -> None:
    text = CARD.read_text(encoding="utf-8")
    prose = text.split("## Numbers")[0]
    # the prose writes negatives with a typographic minus, which DECIMAL reads unsigned
    table_values = {r["value"].strip().lstrip("-") for r in card_rows()}
    missing = sorted(
        {
            n
            for n in DECIMAL.findall(prose)
            if n not in table_values and n.lstrip("-") not in table_values
        }
    )
    assert missing == []


def test_the_card_covers_every_required_section() -> None:
    text = CARD.read_text(encoding="utf-8")
    for heading in (
        "## Data and exclusions",
        "## Features, and why no identity",
        "## Variants and the search",
        "## Results",
        "## Calibration",
        "## Free throws (F-e)",
        "## Team shot quality",
        "## Player shot-making and stability",
        "## Shot charts",
        "## Limitations and what failed",
    ):
        assert heading in text, heading
