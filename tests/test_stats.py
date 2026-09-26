"""Stats-site export: box and shot parsing, hex bins, payloads, the CLI and the committed fixture.

``tests/fixtures/stats_raw`` is a slice of the real raw cache (rounds 1-6 of 2024-25 and round 1
of 2010-11, schedules trimmed to those games). ``tests/fixtures/stats`` is the exporter's output
on it; the CI web build reads it. Refresh it after a payload change:
``UPDATE_STATS_FIXTURE=1 uv run pytest tests/test_stats.py``
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from eurohoops import cli
from eurohoops.config import MART_PATH
from eurohoops.marts import write_tables
from eurohoops.publish import DISPLAY_CODES
from eurohoops.stats.box import build_box_games, game_lines, seconds
from eurohoops.stats.export import (
    PLAYER_FIELDS,
    Inputs,
    _player_frame,
    build_payloads,
    display_name,
    load_cached_games,
    player_slugs,
    season_label,
    slugify,
    write_stats,
)
from eurohoops.stats.shots import (
    BANDS,
    HEX_RADIUS_M,
    SQRT3,
    band_counts,
    bands,
    build_shots,
    game_shots,
    hex_cells,
    hexbins,
    period,
)
from eurohoops.stats.twins import (
    COUNT_FIELDS,
    ZONES,
    distances,
    features,
    match,
    shot_zones,
    twins_payload,
)
from tests.conftest import FIXTURES

RAW = FIXTURES / "stats_raw"
OUT = FIXTURES / "stats"
NOW = datetime(2026, 9, 26, 8, tzinfo=UTC)
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


@pytest.fixture(scope="module")
def files(inputs: Inputs) -> dict[str, dict[str, Any]]:
    return build_payloads(inputs, NOW)


def test_seconds_parses_box_minutes() -> None:
    assert seconds("05:04") == 304
    assert seconds("200:00") == 12000
    assert seconds("DNP") is None


def test_period_counts_overtimes_after_the_fourth_quarter() -> None:
    assert [period(m) for m in (0, 1, 10, 11, 40, 41, 45, 46)] == [1, 1, 1, 2, 4, 5, 5, 6]


def test_bands_split_twos_by_distance_and_threes_at_eight_metres() -> None:
    distance = pd.Series([0.5, 2.0, 4.0, 6.0, 7.0, 9.0])
    value = pd.Series([2, 2, 2, 2, 3, 3])
    assert list(bands(distance, value)) == ["rim", "short", "mid", "long2", "three", "deep3"]


def test_hex_cells_round_trip_through_cell_centres() -> None:
    q = pd.Series([0, 1, -3, 4, 2])
    r = pd.Series([0, 2, 5, -1, 7])
    x = HEX_RADIUS_M * SQRT3 * (q + r / 2)
    y = HEX_RADIUS_M * 1.5 * r
    # nudge every point off-centre, well inside the cell
    got_q, got_r = hex_cells(x + 0.1, y - 0.1)
    assert list(got_q) == list(q) and list(got_r) == list(r)


def test_hexbins_group_by_key_and_count_points() -> None:
    shots = pd.DataFrame(
        {
            "player_id": ["A", "A", "B"],
            "x": [0.0, 0.05, 0.0],
            "y": [0.0, 0.05, 0.0],
            "made": [True, False, True],
            "value": [2, 2, 3],
        }
    )
    assert hexbins(shots, "player_id") == {"A": [[0, 0, 2, 1, 2]], "B": [[0, 0, 1, 1, 3]]}
    assert hexbins(shots) == {"": [[0, 0, 3, 2, 5]]}
    assert hexbins(shots.iloc[0:0]) == {}


def test_band_counts_list_every_band() -> None:
    shots = pd.DataFrame({"band": ["rim", "rim", "three"], "made": [True, False, True]})
    counts = band_counts(shots)
    assert list(counts) == list(BANDS)
    assert counts["rim"] == [2, 1] and counts["three"] == [1, 1] and counts["deep3"] == [0, 0]


def test_names_and_slugs() -> None:
    assert display_name("DE COLO, NANDO") == "Nando De Colo"
    assert display_name("MCCALEBB, BO") == "Bo McCalebb"
    assert display_name("O'NEAL, SHAQ") == "Shaq O'Neal"
    assert display_name("TEODOSIC") == "Teodosic"
    assert display_name("WALKER IV, LONNIE") == "Lonnie Walker IV"
    assert display_name("IVANOV, IVO") == "Ivo Ivanov"
    assert display_name("SHORTS, TJ") == "TJ Shorts"
    assert display_name("MACON JR, DARYL") == "Daryl Macon Jr"
    assert display_name("MC LEOD, KEITH") == "Keith Mc Leod"
    assert display_name("TY, SY") == "Sy Ty"
    assert slugify("Nikola Mirotić") == "nikola-mirotic"
    assert player_slugs({"P1": "John Smith", "P2": "John Smith", "P3": "Ana Ivić"}) == {
        "P1": "john-smith-p1",
        "P2": "john-smith-p2",
        "P3": "ana-ivic",
    }
    assert season_label(2024) == "2024-25" and season_label(2099) == "2099-00"


def test_game_lines_reject_placeholders_and_foreign_teams() -> None:
    game = {"season": 2024, "game_id": "E2024_1", "home": "AAA", "away": "BBB"}
    assert "placeholder" in str(game_lines({"Stats": []}, game))
    box = json.loads((FIXTURES / "box_E2023_1.json").read_text(encoding="utf-8"))
    assert "are not the game's" in str(
        game_lines(box, game | {"home_score": 1, "away_score": 0, "neutral": False})
    )


def test_missing_box_scores_are_listed_not_dropped(tmp_path: Path) -> None:
    games, _ = load_cached_games(RAW)
    built = build_box_games(tmp_path, games.head(3))  # an empty cache
    assert built.players.empty and built.teams.empty
    assert list(built.missing["reason"].unique()) == ["no cached box score (ingest --details)"]


def test_player_lines_add_up_to_the_team_box(inputs: Inputs) -> None:
    players = inputs.box.players.groupby(["game_id", "team"])[["pts", "fg2a", "ast"]].sum()
    teams = inputs.box.teams.set_index(["game_id", "team"])[["pts", "fg2a", "ast"]]
    pd.testing.assert_frame_equal(players.sort_index(), teams.sort_index(), check_names=False)
    # five players on court: player possessions add up to five times the game's, even where
    # the box records more minutes than were played (E2010_1)
    per_team = inputs.box.players.groupby(["game_id", "team"])["poss"].sum()
    game_poss = inputs.box.teams.groupby("game_id")["poss"].mean()
    ratio = per_team / game_poss.reindex(per_team.index.get_level_values(0)).to_numpy()
    assert ratio.between(5 - 1e-9, 5 + 1e-9).all()


def test_every_shot_is_a_box_field_goal_attempt(inputs: Inputs) -> None:
    shots = inputs.shots.table.groupby(["game_id", "team"]).size()
    box = inputs.box.teams.set_index(["game_id", "team"])
    fga = (box["fg2a"] + box["fg3a"]).reindex(shots.index)
    unplaced = int(inputs.shots.coverage["unplaced"].sum())
    assert int((fga - shots).sum()) == unplaced
    assert (fga >= shots).all()
    three = inputs.shots.table["value"] == 3
    assert inputs.shots.table.loc[three, "band"].isin(["three", "deep3"]).all()


def test_game_shots_skip_free_throws_and_unplaced_rows() -> None:
    game = {"season": 2024, "game_id": "E2024_1", "home": "AAA", "away": "BBB"}
    row = {"ID_PLAYER": "P1", "POINTS": 2, "MINUTE": 3, "FASTBREAK": "1", "SECOND_CHANCE": "0"}
    feed = {
        "Rows": [
            row | {"ID_ACTION": "2FGM", "TEAM": "AAA ", "COORD_X": 100, "COORD_Y": 0},
            row | {"ID_ACTION": "FTM", "TEAM": "AAA", "COORD_X": -1, "COORD_Y": -1},
            row | {"ID_ACTION": "3FGA", "TEAM": "BBB", "COORD_X": 0, "COORD_Y": 0},
            row | {"ID_ACTION": "2FGA", "TEAM": "BBB", "COORD_X": -1, "COORD_Y": -1},
            row | {"ID_ACTION": "2FGM", "TEAM": "ZZZ", "COORD_X": 5, "COORD_Y": 5},
        ]
    }
    shots, unplaced = game_shots(feed, game)
    assert unplaced == 2
    assert [(s["team"], s["opponent"], s["fastbreak"], s["coord_x"]) for s in shots] == [
        ("AAA", "BBB", True, 100.0)
    ]


def test_payload_windows_and_bands(files: dict[str, dict[str, Any]]) -> None:
    season = files["seasons/2024/players.json"]
    assert season["fields"] == list(PLAYER_FIELDS)
    gp = PLAYER_FIELDS.index("gp")
    for row in season["players"]:
        played = row["totals"]["season"][gp]
        assert ("last5" in row["totals"]) == (played > 5)
        assert "last10" not in row["totals"]  # six rounds only
        assert list(row["bands"]) == list(BANDS)
    pts = PLAYER_FIELDS.index("pts")
    tops = [r["totals"]["season"][pts] for r in season["players"]]
    assert tops == sorted(tops, reverse=True)


def test_splits_add_up_to_the_season_rows(files: dict[str, dict[str, Any]]) -> None:
    splits = files["splits.json"]
    assert splits["fields"] == list(PLAYER_FIELDS)
    seasons = {s["season"] for s in files["meta.json"]["seasons"]}
    for season in seasons:
        rows = [r for r in splits["rows"] if r[1] == season]
        for player in files[f"seasons/{season}/players.json"]["players"]:
            own = [r for r in rows if r[0] == player["id"]]
            assert [r[2] for r in own] == player["teams"]  # clubs in the order he joined them
            summed = [round(sum(r[4][i] for r in own), 1) for i in range(len(PLAYER_FIELDS))]
            # possessions are rounded to 0.1 in each split and in the season row
            assert summed == pytest.approx(player["totals"]["season"], abs=0.1 * len(own))
            assert all(r[3] for r in own) or not player["dorsal"]
    assert {r[2] for r in splits["rows"]} <= set(files["meta.json"]["teams"])


def test_payload_uses_display_codes_and_flags_early_coordinates(
    files: dict[str, dict[str, Any]],
) -> None:
    meta = files["meta.json"]
    assert "PAO" in meta["teams"] and "PAN" not in meta["teams"]
    flags = {s["season"]: s["coords_validated"] for s in meta["seasons"]}
    assert flags == {2010: False, 2024: True}
    teams = files["seasons/2024/teams.json"]
    assert sum(t["w"] for t in teams["teams"]) == sum(t["l"] for t in teams["teams"]) == 54
    shots = files["seasons/2024/shots.json"]
    assert set(shots["teams"]) == {t["code"] for t in teams["teams"]}
    league_att = sum(cell[2] for cell in shots["league"])
    assert league_att == sum(a for a, _ in teams["league"]["bands"].values())
    games = files["seasons/2024/games.json"]
    assert len(games["games"]) == 54
    assert {line[0] for log in games["logs"].values() for line in log} <= set(games["games"])
    index = files["players.json"]["players"]
    assert len({p["slug"] for p in index}) == len(index)


def test_attempts_decode_to_the_shot_bins_and_bands(files: dict[str, dict[str, Any]]) -> None:
    attempts = files["seasons/2024/attempts.json"]
    shots, teams = files["seasons/2024/shots.json"], files["seasons/2024/teams.json"]
    bits = attempts["bits"]
    flags = pd.Series(np.array(attempts["flags"], dtype=np.int64))

    def bit(name: str, width: int = 1) -> pd.Series:
        return (flags // 2 ** bits[name]) % 2**width

    frame = pd.DataFrame(
        {
            "x": pd.Series(attempts["x"]) / 100,
            "y": pd.Series(attempts["y"]) / 100,
            "made": bit("made"),
            "value": 2 + bit("three"),
            "band": [BANDS[b] for b in bit("band", 3)],
            "period": flags // 2 ** bits["period"],
            "team": [attempts["teams"][i] for i in attempts["team"]],
        }
    )
    columns = ("x", "y", "flags", "player", "team", "opp")
    assert len({len(attempts[c]) for c in columns}) == 1
    assert frame["period"].between(1, 8).all()
    assert hexbins(frame)[""] == shots["league"]  # centimetres keep every shot in its hex
    assert hexbins(frame, "team") == {c: v["taken"] for c, v in shots["teams"].items()}
    counts = frame.groupby("band")["made"].agg(["size", "sum"])
    assert {b: [int(n), int(m)] for b, (n, m) in counts.iterrows()} == {
        b: v for b, v in teams["league"]["bands"].items() if v[0]
    }


def test_shot_zones_split_bands_by_side_and_corner() -> None:
    shots = pd.DataFrame(
        {
            "x": [0.2, -2.0, 0.0, 2.0, -6.8, 6.8, 0.5, 0.0],
            "y": [0.1, 0.5, 4.0, 0.3, 0.5, 0.5, 7.0, 9.0],
            "band": ["rim", "short", "mid", "short", "three", "three", "three", "deep3"],
        }
    )
    assert list(shot_zones(shots)) == [
        "rim",
        "short_l",
        "mid_c",
        "short_r",
        "corner3_l",
        "corner3_r",
        "three_c",
        "deep3",
    ]


def test_features_and_match() -> None:
    counts = np.zeros((2, len(COUNT_FIELDS)))
    counts[0, [ZONES.index("rim"), ZONES.index("three_c")]] = [30, 10]
    counts[0, COUNT_FIELDS.index("made_rim")] = 20
    counts[0, COUNT_FIELDS.index("exp_rim")] = 20  # makes exactly what the league expects
    counts[0, [COUNT_FIELDS.index(f) for f in ("fga", "fg3a", "fta")]] = [40, 10, 20]
    feats = features(counts)
    assert np.allclose((feats[0, : len(ZONES)] ** 2).sum(), 1.0)
    assert np.allclose(feats[0, len(ZONES) : len(ZONES) + len(BANDS)], 0.0)
    assert np.allclose(feats[0, -2:], [0.25, 0.5])
    assert np.allclose(feats[1], 0.0)  # no attempts: every feature 0, no division by zero
    assert distances(feats[0], feats[:1])[0] == 0.0
    assert match(np.array([0.0]))[0] == 100.0


def test_twins_add_up_and_are_always_other_players(inputs: Inputs) -> None:
    frame = _player_frame(inputs)
    twins = twins_payload(
        frame, inputs.shots.table, inputs.codes, min_pool_att=20, min_window_att=5
    )
    assert twins["pool"] and twins["players"]
    by_key = {(r[0], r[1]): r for r in twins["pool"]}
    table = inputs.shots.table
    for (pid, season), row in by_key.items():
        taken = table[(table["player_id"] == pid) & (table["season"] == season)]
        assert sum(row[4][: len(ZONES)]) == len(taken) >= 20
        made = [row[4][COUNT_FIELDS.index(f"made_{b}")] for b in BANDS]
        assert made == [int(taken.loc[taken["band"] == b, "made"].sum()) for b in BANDS]
    for pid, windows in twins["players"].items():
        assert set(windows) <= set(twins["windows"])
        for w in windows.values():
            assert sum(w["counts"][: len(ZONES)]) >= 5
            scores = [m for _, m in w["twins"]]
            assert scores == sorted(scores, reverse=True) and all(0 < m <= 100 for m in scores)
            assert all(twins["pool"][i][0] != pid for i, _ in w["twins"])
        season = windows.get("season")
        if season and (pid, season["to"]) in by_key:
            assert season["counts"] == by_key[(pid, season["to"])][4]


def test_write_stats_replaces_old_seasons(tmp_path: Path) -> None:
    (tmp_path / "seasons" / "1999").mkdir(parents=True)
    write_stats(tmp_path, {"meta.json": {"a": 1}, "seasons/2024/x.json": {"b": [1, 2]}})
    assert not (tmp_path / "seasons" / "1999").exists()
    assert (tmp_path / "seasons/2024/x.json").read_text(encoding="utf-8") == '{"b":[1,2]}\n'


def _repo_layout(tmp: Path) -> Path:
    raw = tmp / "data" / "raw" / "euroleague"
    for path in RAW.rglob("*.gz"):
        target = raw / path.relative_to(RAW)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return raw


def test_cli_exports_from_the_cache_and_from_the_marts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "utc_now", lambda: NOW)
    _repo_layout(tmp_path)
    cached = runner.invoke(cli.app, ["export-stats", "--from-cache", "--out", "cached"])
    assert cached.exit_code == 0, cached.output
    assert "2 seasons" in cached.output and "0 played games without a box score" in cached.output

    games, teams = load_cached_games(Path("data/raw/euroleague"))
    MART_PATH.parent.mkdir(parents=True)
    write_tables(
        MART_PATH,
        {
            "games": games.assign(competition="euroleague"),
            "teams": teams.assign(competition="euroleague"),
        },
    )
    marts = runner.invoke(cli.app, ["export-stats", "--out", "marts"])
    assert marts.exit_code == 0, marts.output
    for path in (tmp_path / "cached").rglob("*.json"):
        assert (
            path.read_bytes()
            == (tmp_path / "marts" / path.relative_to(tmp_path / "cached")).read_bytes()
        )


def test_committed_stats_fixture_matches_the_export(
    files: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    fresh = tmp_path / "stats"
    write_stats(fresh, files)
    if os.environ.get("UPDATE_STATS_FIXTURE"):
        write_stats(OUT, files)
    written = sorted(p.relative_to(fresh).as_posix() for p in fresh.rglob("*.json"))
    committed = sorted(p.relative_to(OUT).as_posix() for p in OUT.rglob("*.json"))
    assert written == committed
    for name in written:
        assert (OUT / name).read_text(encoding="utf-8") == (fresh / name).read_text(
            encoding="utf-8"
        )
