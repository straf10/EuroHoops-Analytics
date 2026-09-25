"""Committed backtest reports keep every value they had before the weeks 3-5 additions."""

import json
from typing import Any

import pytest

from tests.conftest import FIXTURES, REPO

BEFORE = FIXTURES / "reports_week3"


def missing_or_changed(old: Any, new: Any, path: str = "") -> list[str]:
    """Paths of ``old`` whose value differs in ``new``; keys only in ``new`` are additions."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [
            diff
            for key, value in old.items()
            for diff in (
                missing_or_changed(value, new[key], f"{path}.{key}")
                if key in new
                else [f"{path}.{key} (missing)"]
            )
        ]
    return [] if old == new else [f"{path}: {old!r} -> {new!r}"]


@pytest.mark.parametrize("name", sorted(p.name for p in BEFORE.glob("*.json")))
def test_backtest_reports_reproduce_every_old_value(name: str) -> None:
    old = json.loads((BEFORE / name).read_text(encoding="utf-8"))
    new = json.loads((REPO / "reports" / name).read_text(encoding="utf-8"))
    assert missing_or_changed(old, new) == []


def test_the_diff_catches_a_moved_value_and_a_dropped_key() -> None:
    old = {"a": {"b": 1, "c": [1, 2]}, "d": 3}
    assert missing_or_changed(old, {"a": {"b": 1, "c": [1, 2], "e": 9}, "d": 3}) == []
    assert missing_or_changed(old, {"a": {"b": 2, "c": [1, 2]}}) == [
        ".a.b: 1 -> 2",
        ".d (missing)",
    ]


def test_three_reports_are_frozen() -> None:
    assert sorted(p.name for p in BEFORE.glob("*.json")) == [
        "backtest_elo.json",
        "backtest_elo_gbl.json",
        "backtest_elo_history.json",
    ]
