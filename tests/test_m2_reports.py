"""The committed M2 reports: the F-k verdict recomputed from the players report's own numbers,
intervals present, and the gate block consistent with its numbers (F5, F7)."""

import json
from typing import Any

import pytest

from eurohoops.eval.shot_making import stability_verdict
from eurohoops.eval.shot_metrics import calibrated
from tests.conftest import REPO


def _report(name: str) -> dict[str, Any]:
    path = REPO / "reports" / name
    if not path.exists():
        pytest.fail(f"{name} is missing: run the M2 pipeline")
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return report


def test_players_verdict_is_the_f_k_rule_on_the_reported_numbers() -> None:
    stability = _report("m2_players.json")["stability"]
    y2y = stability["year_to_year"]["shrunk_shot_making"]
    assert stability["verdict"] == stability_verdict(y2y["ci90"][0], stability["split_half"]["r"])
    for key in ("shrunk_shot_making", "raw_efg_pct", "raw_fg_pct"):
        block = stability["year_to_year"][key]
        assert block["ci90"][0] <= block["r"] <= block["ci90"][1]
        assert block["n"] == stability["year_to_year"]["pairs"] > 0


def test_players_have_intervals_and_the_minimum_sample() -> None:
    report = _report("m2_players.json")
    players = report["players"]
    assert players
    for p in players:
        assert p["fga"] >= report["shown_min_fga"]
        assert p["ci90"][0] <= p["shrunk"] <= p["ci90"][1]
        assert abs(p["shrunk"]) <= abs(p["raw"]) + 1e-9  # shrinkage toward 0


def test_gate_block_matches_its_numbers() -> None:
    report = _report("backtest_m2.json")
    gate = report["gate"]
    diff = gate["log_loss_challenger_minus_baseline"]["mean"]
    variants = report["variants"]
    validation = {v: variants[v]["validation"]["log_loss"] for v in variants}
    assert diff == pytest.approx(
        validation[gate["challenger"]] - validation[gate["baseline"]], abs=2e-6
    )
    assert gate["beats_baseline"] == (diff < 0)
    assert gate["calibrated"] == calibrated(variants[gate["chosen"]]["validation"])
    assert gate["passed"] == (gate["calibrated"] and gate["beats_baseline"])
    cv = {v: variants[v]["cv"]["log_loss"] for v in variants}
    assert gate["challenger"] == min(("lgbm", "lgbm_iso"), key=cv.__getitem__)
    assert gate["baseline"] == min(("spline", "spline_iso"), key=cv.__getitem__)
