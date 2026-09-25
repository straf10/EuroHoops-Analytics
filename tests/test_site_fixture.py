"""``tests/fixtures/site.json`` feeds the CI web build. It is regenerated here from the real
``site_data`` on synthetic games, so it can never drift from the schema the page reads.

Refresh it after a schema change: ``UPDATE_SITE_FIXTURE=1 uv run pytest tests/test_site_fixture.py``
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from eurohoops.config import Backtest, Grid
from eurohoops.eval.backtest import load_tuned_model, run_backtest, win_probabilities
from eurohoops.eval.scorecard import build_scorecard
from eurohoops.live_m1 import M1_LOG_COLUMNS
from eurohoops.models.elo import prepare, replay
from eurohoops.predict import LOG_COLUMNS, TIME_FORMAT
from eurohoops.publish import Section, site_data
from tests.conftest import FIXTURES, make_games, teams_table

FIXTURE = FIXTURES / "site.json"
NOW = datetime(2027, 9, 30, 8, tzinfo=UTC)  # before the synthetic 2027 season tips off
SPEC = Backtest(
    Path("report.json"),
    warmup=(2022,),
    tuning=(2023,),
    test=(2024,),
    grid=Grid(k=(20.0, 30.0), hca=(60.0, 90.0), reversion=(0.25,)),
)


def logged(
    games: pd.DataFrame, model_path: Path, seasons: tuple[int, ...], upcoming: int
) -> pd.DataFrame:
    """Rows stamped 10 h before tip-off: ``seasons`` plus the first ``upcoming`` 2027 games."""
    model = load_tuned_model(model_path)
    probs = win_probabilities(replay(prepare(games), model.params))
    diffs = replay(prepare(games), model.params)
    chosen = games["season"].isin(seasons) | (
        (games["season"] == 2027) & (games["game_code"] <= upcoming)
    )
    rows = [
        {
            "game_id": g.game_id,
            "season": g.season,
            "round": g.round,
            "phase": g.phase,
            "tipoff_utc": g.tipoff_utc.strftime(TIME_FORMAT),
            "home": g.home,
            "away": g.away,
            "p_home": f"{p:.4f}",
            "exp_margin": f"{d / model.margin_scale:.2f}",
            "model": "elo",
            "model_version": model.version(),
            "predicted_at_utc": (g.tipoff_utc - timedelta(hours=10)).strftime(TIME_FORMAT),
        }
        for g, p, d, keep in zip(games.itertuples(), probs, diffs, chosen, strict=True)
        if keep
    ]
    return pd.DataFrame(rows, columns=list(LOG_COLUMNS))


def m1_log(elo_log: pd.DataFrame) -> pd.DataFrame:
    """M1 rows for the same games: a slightly sharper call, a Student-t margin."""
    p = elo_log["p_home"].astype(float)
    return elo_log.assign(
        p_home=(0.5 + 1.1 * (p - 0.5)).clip(0.02, 0.98).round(4),
        exp_total=160.0,
        margin_sigma=10.4,
        margin_df=7.0,
        total_sigma=16.5,
        model="m1",
        model_version="0.2.0+m1.fixture",
    )[list(M1_LOG_COLUMNS)]


def section(tmp: Path, key: str, title: str, gbl_like: bool, seasons: tuple[int, ...]) -> Section:
    games = make_games({s: True for s in range(2022, 2027)} | {2027: False}, gbl_like=gbl_like)
    report = run_backtest(games, SPEC)
    report_path = tmp / f"{key}_report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    log = logged(games, report_path, seasons, upcoming=4)
    log_path = tmp / f"{key}_log.csv"
    log.to_csv(log_path, index=False)
    model = load_tuned_model(report_path)
    m1_path = None
    if key == "euroleague":  # GBL keeps the no-M1-log state
        m1_path = tmp / f"{key}_m1_log.csv"
        m1_log(log).to_csv(m1_path, index=False)
    card = build_scorecard(log_path, games, model, NOW, m1_log_path=m1_path)
    names = dict(teams_table(games).itertuples(index=False))
    return Section(
        key, title, log.astype({"p_home": float}), card, report, games, names, model, 2027, 2022
    )


def fixture_data(tmp: Path) -> dict[str, Any]:
    sections = [
        # 60 scored games: the rolling-50 line has 11 points.
        section(tmp, "euroleague", "EuroLeague", False, (2025, 2026)),
        # 30 scored games: below the window, the honest empty state.
        section(tmp, "gbl", "Greek Basket League", True, (2026,)),
    ]
    return site_data(sections, NOW)


def test_committed_site_fixture_matches_site_data(tmp_path: Path) -> None:
    fresh = json.dumps(fixture_data(tmp_path), indent=2) + "\n"
    if os.environ.get("UPDATE_SITE_FIXTURE"):
        FIXTURE.write_text(fresh, encoding="utf-8", newline="\n")
    assert FIXTURE.read_text(encoding="utf-8") == fresh


def test_the_fixture_exercises_every_state_the_page_draws() -> None:
    el, gbl = json.loads(FIXTURE.read_text(encoding="utf-8"))["competitions"]
    assert len(el["scorecard"]["rolling"]["series"]) == 60 - 50 + 1
    assert gbl["scorecard"]["rolling"]["series"] == []
    for comp in (el, gbl):
        assert len(comp["upcoming"]) == 4
        assert comp["results"]
        assert len(comp["backtest"]["test"]["elo"]["reliability"]) == 10
        assert comp["backtest"]["test"]["elo"]["margin_crps"] is not None
        assert comp["scorecard"]["elo"]["ece"] is not None
    assert el["scorecard"]["m1"]["n"] == 60 and el["scorecard"]["m1"]["log_loss"] is not None
    assert gbl["scorecard"]["m1"] is None
