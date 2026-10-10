"""Load validation: per-season flags, the published window, special seasons, the error policy and
the shape of ``validation.json`` (pinned by ``tests/fixtures/validation.json``)."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args, get_type_hints

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from eurohoops import cli
from eurohoops.publish import DISPLAY_CODES
from eurohoops.stats import export
from eurohoops.stats.box import BoxGames, build_box_games
from eurohoops.stats.export import Inputs, load_cached_games
from eurohoops.stats.shots import Shots, build_shots
from eurohoops.stats.validate import (
    Flag,
    SeasonReport,
    Severity,
    Special,
    Validation,
    published_seasons,
    report_lines,
    validate,
    write_validation,
)
from tests.conftest import FIXTURES

RAW = FIXTURES / "stats_raw"
NOW = datetime(2026, 10, 10, 8, 41, tzinfo=UTC)
runner = CliRunner()


@pytest.fixture(scope="module")
def inputs() -> Inputs:
    games, teams = load_cached_games(RAW)
    return Inputs(
        games=games,
        names=dict(teams.itertuples(index=False)),
        box=build_box_games(RAW, games),
        shots=build_shots(RAW, games),
        codes=DISPLAY_CODES["euroleague"],
        live_season=2026,
    )


def _row(report: Validation, season: int) -> SeasonReport:
    (row,) = (s for s in report["seasons"] if s["season"] == season)
    return row


def _codes(report: Validation, season: int) -> dict[str, Flag]:
    return {f["code"]: f for f in _row(report, season)["flags"]}


def test_clean_data_counts_every_season_and_raises_no_warning(inputs: Inputs) -> None:
    report = validate(inputs, NOW, published_seasons(inputs))
    assert report["built"] == "2026-10-10T08:41Z"
    assert [s["season"] for s in report["seasons"]] == [2010, 2024]
    row = _row(report, 2024)
    games = inputs.games[inputs.games["season"] == 2024]
    assert row["label"] == "2024-25" and row["published"]
    assert row["games_scheduled"] == len(games)
    assert row["games_played"] == int(games["played"].sum())
    assert row["box_missing"] == 0 and row["shots_games"] > 0
    assert all(f["severity"] == "info" for s in report["seasons"] for f in s["flags"])
    assert not any(is_warning for _, is_warning in report_lines(report))


def test_the_window_matches_the_site_and_covers_unpublished_seasons(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export, "SITE_SEASONS", 1)
    assert published_seasons(inputs) == [2024]
    report = validate(inputs, NOW, published_seasons(inputs))
    assert [(s["season"], s["published"]) for s in report["seasons"]] == [
        (2010, False),
        (2024, True),
    ]
    site = export.build_payloads(inputs, NOW)["meta.json"]["seasons"]
    assert [s["season"] for s in site] == [2024]


def test_unplayed_games_warn_unless_live_or_special(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    games = inputs.games.copy()
    games.loc[games.index[:3], "played"] = False
    cut = replace(inputs, games=games)
    first = int(games.loc[games.index[0], "season"])
    flag = _codes(validate(cut, NOW, [2024]), first)["games_unplayed"]
    assert flag["severity"] == "warn" and flag["message"] == "3 scheduled games never played."
    live = _codes(validate(replace(cut, live_season=first), NOW, [2024]), first)
    assert live["games_unplayed"]["severity"] == "info"
    assert "still to be played" in live["games_unplayed"]["message"]
    monkeypatch.setattr("eurohoops.stats.validate.SPECIAL_SEASONS", {first: ("k", "explained")})
    report = validate(cut, NOW, [2024])
    assert _codes(report, first)["games_unplayed"]["severity"] == "info"
    assert _row(report, first)["special"] == {"kind": "k", "note": "explained"}
    assert _codes(report, first)["special_season"]["message"] == "explained"


def test_missing_box_scores_and_shots_warn(inputs: Inputs) -> None:
    gone = inputs.box.players["game_id"].iloc[0]
    season = int(inputs.box.players["season"].iloc[0])
    missing = pd.DataFrame({"season": [season], "game_id": [gone], "reason": ["no cached box"]})
    box = BoxGames(inputs.box.players, inputs.box.teams, missing)
    cut = replace(
        inputs, box=box, shots=Shots(inputs.shots.table, inputs.shots.coverage.assign(games=0))
    )
    report = validate(cut, NOW, [season])
    flags = _codes(report, season)
    assert flags["box_missing"]["message"] == "1 played game has no box score."
    assert flags["box_missing"]["severity"] == "warn"
    assert flags["shots_missing"]["severity"] == "warn"
    assert "played games have no shot locations" in flags["shots_missing"]["message"]
    assert _row(report, season)["box_missing"] == 1 and _row(report, season)["shots_games"] == 0


def test_unplaced_shots_are_info(inputs: Inputs) -> None:
    coverage = inputs.shots.coverage.assign(unplaced=1500)
    report = validate(replace(inputs, shots=Shots(inputs.shots.table, coverage)), NOW, [2024])
    flag = _codes(report, 2024)["shots_unplaced"]
    assert flag["severity"] == "info" and flag["message"] == "1,500 shots have no court position."


def test_schema_breaks_warn_when_unpublished_and_error_when_published(inputs: Inputs) -> None:
    bad = inputs.games.drop(columns=["round"])
    report = validate(replace(inputs, games=bad), NOW, [2024])
    assert _codes(report, 2010)["schema_mismatch"]["severity"] == "warn"
    flag = _codes(report, 2024)["schema_mismatch"]
    assert flag["severity"] == "error" and "missing columns round" in flag["message"]

    odd = inputs.box.players.assign(sec=-5)
    cut = replace(inputs, box=BoxGames(odd, inputs.box.teams, inputs.box.missing))
    message = _codes(validate(cut, NOW, [2024]), 2024)["schema_mismatch"]["message"]
    assert message.startswith("The player box line data does not match") and "sec" in message

    no_season = inputs.shots.table.drop(columns=["season"])
    cut = replace(inputs, shots=Shots(no_season, inputs.shots.coverage))
    message = _codes(validate(cut, NOW, [2024]), 2024)["schema_mismatch"]["message"]
    assert "missing columns season" in message


def test_a_published_season_without_played_games_is_an_error(inputs: Inputs) -> None:
    games = inputs.games.assign(played=False)
    report = validate(replace(inputs, games=games), NOW, [2024])
    assert _codes(report, 2024)["no_games_played"]["severity"] == "error"
    assert "no_games_played" not in _codes(report, 2010)


def test_write_validation_prints_warnings_and_exits_nonzero_only_on_error(
    inputs: Inputs, tmp_path: Path
) -> None:
    lines: list[tuple[str, bool]] = []

    def echo(text: str, err: bool = False) -> None:
        lines.append((text, err))

    cut = replace(inputs, shots=Shots(inputs.shots.table, inputs.shots.coverage.assign(games=0)))
    write_validation(tmp_path, cut, NOW, echo)
    warnings = [text for text, err in lines if err]
    assert warnings and all(w.startswith("warning: ") for w in warnings)
    assert any("no shot locations" in w for w in warnings)
    assert sum(not err for _, err in lines) == 2  # one summary line per season
    assert (tmp_path / "validation.json").exists()

    bad = replace(inputs, games=inputs.games.drop(columns=["round"]))
    with pytest.raises(typer.Exit) as exc:
        write_validation(tmp_path / "bad", bad, NOW, echo)
    assert exc.value.exit_code == 1
    assert (tmp_path / "bad" / "validation.json").exists()  # written before the failure


def test_cli_writes_the_json_and_fails_after_writing_it_on_a_published_schema_break(
    inputs: Inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "data" / "raw" / "euroleague"
    for path in RAW.rglob("*.gz"):
        target = raw / path.relative_to(RAW)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    ok = runner.invoke(cli.app, ["export-stats", "--from-cache", "--out", "ok"])
    assert ok.exit_code == 0, ok.output
    written = json.loads((tmp_path / "ok" / "validation.json").read_text(encoding="utf-8"))
    assert [s["season"] for s in written["seasons"]] == [2010, 2024]
    assert (tmp_path / "ok" / "seasons").exists()

    broken = replace(inputs, games=inputs.games.drop(columns=["round"]))
    monkeypatch.setattr(cli, "stats_inputs", lambda store: broken)
    failed = runner.invoke(cli.app, ["export-stats", "--from-cache", "--out", "bad"])
    assert failed.exit_code == 1
    assert (tmp_path / "bad" / "validation.json").exists()


def _check(value: Any, hint: Any) -> None:
    """``value`` has the keys and types of the TypedDict or type ``hint``."""
    if hasattr(hint, "__required_keys__"):
        hints = get_type_hints(hint)
        assert set(value) == set(hints), set(value) ^ set(hints)
        for key, sub in hints.items():
            _check(value[key], sub)
    elif getattr(hint, "__origin__", None) is list:
        (item,) = get_args(hint)
        assert isinstance(value, list)
        for element in value:
            _check(element, item)
    elif hint == Severity:
        assert value in get_args(Severity)
    elif type(None) in get_args(hint):
        if value is not None:
            _check(value, next(a for a in get_args(hint) if a is not type(None)))
    else:
        assert type(value) is hint, (value, hint)


def test_the_fixture_and_the_output_have_the_pinned_shape(inputs: Inputs) -> None:
    fixture = json.loads((FIXTURES / "validation.json").read_text(encoding="utf-8"))
    _check(fixture, Validation)
    assert [s["season"] for s in fixture["seasons"]] == list(range(2019, 2026))
    assert {s["published"] for s in fixture["seasons"][:2]} == {False}
    assert set(fixture["seasons"][0]["special"]) == set(Special.__annotations__)
    report: Any = validate(inputs, NOW, published_seasons(inputs))
    _check(report, Validation)
    assert set(report) == set(fixture)
    assert set(report["seasons"][0]) == set(fixture["seasons"][0])
