"""I6: season rates, team nets, and dual/mover pair tables for M4."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from eurohoops.models.box_impact import STAT_COLUMNS
from eurohoops.models.translation_pairs import (
    MAX_OTHER,
    MIN_FROM,
    MIN_TO,
    build_pairs,
    season_rates,
    team_net,
)

# --- Hand fixture (two leagues, three seasons, one dual + one mover each way) ---------------
# Seasons 2019 (prior nets + early mover source), 2020, 2021.
# Minutes thresholds: 300 min = 18_000 sec. One game row per person-season unless noted.
#
# Team nets (relative = raw − league mean; two teams → mean 0 → relative = raw):
#   2019 GBL AAA vs BBB, 100–80 in 100 poss → AAA +20, BBB −20
#   2019 EL  XXX vs YYY,  90–80 in  80 poss → XXX +12.5, YYY −12.5
#   2020 GBL AAA vs BBB, 110–90 in 100 poss → AAA +20, BBB −20
#   2020 EL  XXX vs YYY, 110–70 in 100 poss → XXX +40, YYY −40
#
# Dual P:DUAL in 2021: GBL AAA (18_000 sec, 200 poss, 50 pts) and EL XXX (18_000, 180, 40).
# Hand rates: gbl pts 100*50/200 = 25; el pts 100*40/180 = 200/9.
# Hand team_gap: net(XXX,2020) minus net(AAA,2020) = 40 - 20 = 20.
#
# gbl_to_el P:MOVEG: GBL BBB 2020 (18_000, 150 poss, 30 pts) to EL YYY 2021 (18_000, 160, 48).
# Hand rates: gbl pts 20; el pts 30.
# Hand team_gap: net(YYY,2020) minus net(BBB,2019) = -40 - (-20) = -20.
#
# el_to_gbl P:MOVEE: EL XXX 2020 (18_000, 100 poss, 25 pts) to GBL AAA 2021 (18_000, 120, 36).
# Hand rates: el pts 25; gbl pts 30.
# Hand team_gap: net(XXX,2019) minus net(AAA,2020) = 12.5 - 20 = -7.5.
#
# Early gbl_to_el P:EARLY (for leakage later_season < 2021): GBL AAA 2019 to EL YYY 2020.
# later_season = 2020 (rates not asserted in the hand example).


def _zero_stats(**overrides: int) -> dict[str, int]:
    base = {stat: 0 for stat in STAT_COLUMNS}
    base.update(overrides)
    return base


def _player_row(
    *,
    season: int,
    game_id: str,
    team: str,
    player_id: str,
    sec: int,
    poss: float,
    **stats: int,
) -> dict[str, object]:
    return {
        "season": season,
        "game_id": game_id,
        "team": team,
        "player_id": player_id,
        "sec": sec,
        "poss": float(poss),
        **_zero_stats(**stats),
    }


def _team_row(
    *,
    competition: str,
    season: int,
    game_id: str,
    team: str,
    opponent: str,
    points: int,
    opp_points: int,
    poss_game: float,
) -> list[dict[str, object]]:
    """Both sides of one game (opp_points is the other team's points)."""
    return [
        {
            "competition": competition,
            "season": season,
            "game_id": game_id,
            "team": team,
            "opponent": opponent,
            "points": points,
            "poss_game": float(poss_game),
        },
        {
            "competition": competition,
            "season": season,
            "game_id": game_id,
            "team": opponent,
            "opponent": team,
            "points": opp_points,
            "poss_game": float(poss_game),
        },
    ]


def _xwalk() -> pd.DataFrame:
    rows = [
        ("P:DUAL", "gbl", "G_DUAL"),
        ("P:DUAL", "euroleague", "P000001"),
        ("P:MOVEG", "gbl", "G_MOVEG"),
        ("P:MOVEG", "euroleague", "P000002"),
        ("P:MOVEE", "gbl", "G_MOVEE"),
        ("P:MOVEE", "euroleague", "P000003"),
        ("P:EARLY", "gbl", "G_EARLY"),
        ("P:EARLY", "euroleague", "P000004"),
    ]
    return pd.DataFrame(
        [
            {
                "person_id": pid,
                "competition": comp,
                "source_id": sid,
                "method": "rule",
                "score": 1.0,
            }
            for pid, comp, sid in rows
        ]
    )


def _hand_player_games() -> tuple[pd.DataFrame, pd.DataFrame]:
    gbl = pd.DataFrame(
        [
            # Early mover source (2019)
            _player_row(
                season=2019,
                game_id="G2019E",
                team="AAA",
                player_id="G_EARLY",
                sec=18_000,
                poss=100.0,
                pts=10,
            ),
            # gbl_to_el source
            _player_row(
                season=2020,
                game_id="G2020M",
                team="BBB",
                player_id="G_MOVEG",
                sec=18_000,
                poss=150.0,
                pts=30,
                ast=15,
            ),
            # Dual GBL
            _player_row(
                season=2021,
                game_id="G2021D",
                team="AAA",
                player_id="G_DUAL",
                sec=18_000,
                poss=200.0,
                pts=50,
                fg2a=40,
            ),
            # el_to_gbl target
            _player_row(
                season=2021,
                game_id="G2021E",
                team="AAA",
                player_id="G_MOVEE",
                sec=18_000,
                poss=120.0,
                pts=36,
                tov=6,
            ),
        ]
    )
    el = pd.DataFrame(
        [
            # Early mover target
            _player_row(
                season=2020,
                game_id="E2020E",
                team="YYY",
                player_id="P000004",
                sec=18_000,
                poss=110.0,
                pts=22,
            ),
            # el_to_gbl source
            _player_row(
                season=2020,
                game_id="E2020M",
                team="XXX",
                player_id="P000003",
                sec=18_000,
                poss=100.0,
                pts=25,
                stl=5,
            ),
            # Dual EL
            _player_row(
                season=2021,
                game_id="E2021D",
                team="XXX",
                player_id="P000001",
                sec=18_000,
                poss=180.0,
                pts=40,
                fg3a=30,
            ),
            # gbl_to_el target
            _player_row(
                season=2021,
                game_id="E2021M",
                team="YYY",
                player_id="P000002",
                sec=18_000,
                poss=160.0,
                pts=48,
                blk=8,
            ),
        ]
    )
    return gbl, el


def _hand_team_games() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows += _team_row(
        competition="gbl",
        season=2019,
        game_id="TG19G",
        team="AAA",
        opponent="BBB",
        points=100,
        opp_points=80,
        poss_game=100.0,
    )
    rows += _team_row(
        competition="euroleague",
        season=2019,
        game_id="TG19E",
        team="XXX",
        opponent="YYY",
        points=90,
        opp_points=80,
        poss_game=80.0,
    )
    rows += _team_row(
        competition="gbl",
        season=2020,
        game_id="TG20G",
        team="AAA",
        opponent="BBB",
        points=110,
        opp_points=90,
        poss_game=100.0,
    )
    rows += _team_row(
        competition="euroleague",
        season=2020,
        game_id="TG20E",
        team="XXX",
        opponent="YYY",
        points=110,
        opp_points=70,
        poss_game=100.0,
    )
    # 2021 team games exist but must not enter prior-season gaps for later_season <= 2021 pairs
    rows += _team_row(
        competition="gbl",
        season=2021,
        game_id="TG21G",
        team="AAA",
        opponent="BBB",
        points=200,
        opp_points=50,
        poss_game=100.0,
    )
    rows += _team_row(
        competition="euroleague",
        season=2021,
        game_id="TG21E",
        team="XXX",
        opponent="YYY",
        points=200,
        opp_points=50,
        poss_game=100.0,
    )
    return pd.DataFrame(rows)


def _hand_rates_and_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    xwalk = _xwalk()
    gbl, el = _hand_player_games()
    rates = pd.concat(
        [season_rates(gbl, "gbl", xwalk), season_rates(el, "euroleague", xwalk)],
        ignore_index=True,
    )
    pairs = build_pairs(rates, team_net(_hand_team_games()))
    return rates, pairs


def test_season_rates_equal_100_sum_over_sum() -> None:
    xwalk = _xwalk()
    gbl, _ = _hand_player_games()
    # Two games for one person so the sum/sum is not a single-row tautology.
    extra = _player_row(
        season=2020,
        game_id="G2020M2",
        team="BBB",
        player_id="G_MOVEG",
        sec=3_000,
        poss=50.0,
        pts=10,
        ast=5,
    )
    games = pd.concat([gbl, pd.DataFrame([extra])], ignore_index=True)
    rates = season_rates(games, "gbl", xwalk)
    row = rates[(rates["person_id"] == "P:MOVEG") & (rates["season"] == 2020)].iloc[0]
    # sec = 18000+3000 → minutes = 350; poss = 150+50 = 200; pts = 30+10 = 40; ast = 15+5 = 20
    assert row["minutes"] == pytest.approx(350.0)
    assert row["poss"] == pytest.approx(200.0)
    assert row["rate_pts"] == pytest.approx(100.0 * 40 / 200)
    assert row["rate_ast"] == pytest.approx(100.0 * 20 / 200)
    for stat in STAT_COLUMNS:
        if stat in {"pts", "ast"}:
            continue
        assert row[f"rate_{stat}"] == pytest.approx(0.0)


def test_primary_team_is_most_seconds_then_smallest_code() -> None:
    xwalk = pd.DataFrame(
        [
            {
                "person_id": "P:T",
                "competition": "gbl",
                "source_id": "G_T",
                "method": "rule",
                "score": 1.0,
            }
        ]
    )
    # More seconds on ZZZ than AAA.
    games = pd.DataFrame(
        [
            _player_row(
                season=2020, game_id="a", team="AAA", player_id="G_T", sec=8_000, poss=50.0, pts=1
            ),
            _player_row(
                season=2020, game_id="b", team="ZZZ", player_id="G_T", sec=10_000, poss=60.0, pts=2
            ),
        ]
    )
    assert season_rates(games, "gbl", xwalk).iloc[0]["team"] == "ZZZ"
    # Equal seconds: smallest code wins (AAA < BBB).
    tied = pd.DataFrame(
        [
            _player_row(
                season=2021, game_id="c", team="BBB", player_id="G_T", sec=9_000, poss=40.0, pts=1
            ),
            _player_row(
                season=2021, game_id="d", team="AAA", player_id="G_T", sec=9_000, poss=40.0, pts=1
            ),
        ]
    )
    assert season_rates(tied, "gbl", xwalk).iloc[0]["team"] == "AAA"


def test_unmapped_player_id_raises() -> None:
    xwalk = _xwalk()
    games = pd.DataFrame(
        [
            _player_row(
                season=2020,
                game_id="x",
                team="AAA",
                player_id="MISSING",
                sec=100,
                poss=10.0,
                pts=1,
            )
        ]
    )
    with pytest.raises(ValueError, match="MISSING"):
        season_rates(games, "gbl", xwalk)


def test_team_net_hand_arithmetic() -> None:
    net = team_net(_hand_team_games())
    g19 = net[(net["competition"] == "gbl") & (net["season"] == 2019)].set_index("team")["net"]
    assert g19["AAA"] == pytest.approx(20.0)
    assert g19["BBB"] == pytest.approx(-20.0)
    e19 = net[(net["competition"] == "euroleague") & (net["season"] == 2019)].set_index("team")[
        "net"
    ]
    assert e19["XXX"] == pytest.approx(12.5)
    assert e19["YYY"] == pytest.approx(-12.5)
    e20 = net[(net["competition"] == "euroleague") & (net["season"] == 2020)].set_index("team")[
        "net"
    ]
    assert e20["XXX"] == pytest.approx(40.0)
    assert e20["YYY"] == pytest.approx(-40.0)


def test_hand_example_one_dual_and_one_mover_each_way() -> None:
    _, pairs = _hand_rates_and_pairs()
    assert sorted(pairs["pair_type"].tolist()) == ["dual", "el_to_gbl", "gbl_to_el", "gbl_to_el"]
    dual = pairs[pairs["pair_type"] == "dual"].iloc[0]
    assert dual["person_id"] == "P:DUAL"
    assert (int(dual["season_gbl"]), int(dual["season_el"]), int(dual["later_season"])) == (
        2021,
        2021,
        2021,
    )
    assert dual["minutes_gbl"] == pytest.approx(300.0)
    assert dual["minutes_el"] == pytest.approx(300.0)
    assert dual["team_gbl"] == "AAA"
    assert dual["team_el"] == "XXX"
    assert dual["rate_gbl_pts"] == pytest.approx(25.0)
    assert dual["rate_el_pts"] == pytest.approx(200.0 / 9.0)
    assert dual["rate_gbl_fg2a"] == pytest.approx(100.0 * 40 / 200)
    assert dual["rate_el_fg3a"] == pytest.approx(100.0 * 30 / 180)
    assert dual["team_gap"] == pytest.approx(20.0)

    g2e = pairs[(pairs["pair_type"] == "gbl_to_el") & (pairs["person_id"] == "P:MOVEG")].iloc[0]
    assert (int(g2e["season_gbl"]), int(g2e["season_el"]), int(g2e["later_season"])) == (
        2020,
        2021,
        2021,
    )
    assert g2e["rate_gbl_pts"] == pytest.approx(20.0)
    assert g2e["rate_el_pts"] == pytest.approx(30.0)
    assert g2e["rate_gbl_ast"] == pytest.approx(100.0 * 15 / 150)
    assert g2e["rate_el_blk"] == pytest.approx(100.0 * 8 / 160)
    assert g2e["team_gap"] == pytest.approx(-20.0)

    e2g = pairs[pairs["pair_type"] == "el_to_gbl"].iloc[0]
    assert e2g["person_id"] == "P:MOVEE"
    assert (int(e2g["season_gbl"]), int(e2g["season_el"]), int(e2g["later_season"])) == (
        2021,
        2020,
        2021,
    )
    assert e2g["rate_el_pts"] == pytest.approx(25.0)
    assert e2g["rate_gbl_pts"] == pytest.approx(30.0)
    assert e2g["rate_el_stl"] == pytest.approx(100.0 * 5 / 100)
    assert e2g["rate_gbl_tov"] == pytest.approx(100.0 * 6 / 120)
    assert e2g["team_gap"] == pytest.approx(-7.5)

    # Sorted by (later_season, person_id, pair_type)
    keys = list(zip(pairs["later_season"], pairs["person_id"], pairs["pair_type"], strict=True))
    assert keys == sorted(keys)


def _rates_for_minutes(
    *,
    person: str,
    gbl_s: float | None,
    el_s: float | None,
    gbl_s1: float | None = None,
    el_s1: float | None = None,
    season: int = 2020,
) -> pd.DataFrame:
    """Minimal season-rate rows for threshold tests (team/rates arbitrary)."""

    def row(comp: str, seas: int, minutes: float) -> dict[str, object]:
        base: dict[str, object] = {
            "person_id": person,
            "competition": comp,
            "season": seas,
            "team": "T",
            "minutes": minutes,
            "poss": 100.0,
        }
        for stat in STAT_COLUMNS:
            base[f"rate_{stat}"] = 1.0
        return base

    rows: list[dict[str, object]] = []
    if gbl_s is not None:
        rows.append(row("gbl", season, gbl_s))
    if el_s is not None:
        rows.append(row("euroleague", season, el_s))
    if gbl_s1 is not None:
        rows.append(row("gbl", season + 1, gbl_s1))
    if el_s1 is not None:
        rows.append(row("euroleague", season + 1, el_s1))
    return pd.DataFrame(rows)


_EMPTY_TEAM_COLS = [
    "competition",
    "season",
    "game_id",
    "team",
    "opponent",
    "points",
    "poss_game",
]


def _empty_net() -> pd.DataFrame:
    return team_net(pd.DataFrame(columns=_EMPTY_TEAM_COLS))


def test_minutes_thresholds_at_boundaries() -> None:
    empty_net = _empty_net()

    # Dual: 299 fails, 300 passes.
    assert build_pairs(
        _rates_for_minutes(person="P:A", gbl_s=MIN_FROM - 1, el_s=MIN_FROM), empty_net
    ).empty
    assert build_pairs(
        _rates_for_minutes(person="P:A", gbl_s=MIN_FROM, el_s=MIN_FROM - 1), empty_net
    ).empty
    dual = build_pairs(_rates_for_minutes(person="P:A", gbl_s=MIN_FROM, el_s=MIN_FROM), empty_net)
    assert list(dual["pair_type"]) == ["dual"]

    # gbl_to_el: other-league minutes 100 fails, 99 passes; to-side 299 fails, 300 passes.
    assert build_pairs(
        _rates_for_minutes(person="P:B", gbl_s=MIN_FROM, el_s=MAX_OTHER, el_s1=MIN_TO),
        empty_net,
    ).empty
    assert build_pairs(
        _rates_for_minutes(person="P:B", gbl_s=MIN_FROM, el_s=MAX_OTHER - 1, el_s1=MIN_TO - 1),
        empty_net,
    ).empty
    g2e = build_pairs(
        _rates_for_minutes(person="P:B", gbl_s=MIN_FROM, el_s=MAX_OTHER - 1, el_s1=MIN_TO),
        empty_net,
    )
    assert list(g2e["pair_type"]) == ["gbl_to_el"]
    # Absent other-league minutes count as 0.
    g2e_absent = build_pairs(
        _rates_for_minutes(person="P:C", gbl_s=MIN_FROM, el_s=None, el_s1=MIN_TO), empty_net
    )
    assert list(g2e_absent["pair_type"]) == ["gbl_to_el"]

    # el_to_gbl boundaries.
    assert build_pairs(
        _rates_for_minutes(person="P:D", el_s=MIN_FROM, gbl_s=MAX_OTHER, gbl_s1=MIN_TO),
        empty_net,
    ).empty
    e2g = build_pairs(
        _rates_for_minutes(person="P:D", el_s=MIN_FROM, gbl_s=MAX_OTHER - 1, gbl_s1=MIN_TO),
        empty_net,
    )
    assert list(e2g["pair_type"]) == ["el_to_gbl"]


def test_missing_prior_season_team_net_counts_as_zero() -> None:
    rates = _rates_for_minutes(person="P:Z", gbl_s=MIN_FROM, el_s=None, el_s1=MIN_TO)
    pairs = build_pairs(rates, _empty_net())
    assert pairs.iloc[0]["team_gap"] == pytest.approx(0.0)


def test_leakage_pairs_with_later_season_before_t_unchanged() -> None:
    xwalk = _xwalk()
    gbl, el = _hand_player_games()
    net = team_net(_hand_team_games())
    rates = pd.concat(
        [season_rates(gbl, "gbl", xwalk), season_rates(el, "euroleague", xwalk)],
        ignore_index=True,
    )
    before = build_pairs(rates, net)
    t = 2021
    early = before[before["later_season"] < t].copy()
    assert not early.empty  # P:EARLY with later_season 2020

    el_edit = el.copy()
    gbl_edit = gbl.copy()
    for frame, seasons in ((el_edit, el_edit["season"] >= t), (gbl_edit, gbl_edit["season"] >= t)):
        for stat in STAT_COLUMNS:
            frame.loc[seasons, stat] = frame.loc[seasons, stat] * 3

    rates_after = pd.concat(
        [season_rates(gbl_edit, "gbl", xwalk), season_rates(el_edit, "euroleague", xwalk)],
        ignore_index=True,
    )
    after = build_pairs(rates_after, net)
    early_after = after[after["later_season"] < t]
    pd.testing.assert_frame_equal(
        early.sort_values(["person_id", "pair_type"]).reset_index(drop=True),
        early_after.sort_values(["person_id", "pair_type"]).reset_index(drop=True),
    )


def test_two_runs_byte_identical() -> None:
    _, first = _hand_rates_and_pairs()
    _, second = _hand_rates_and_pairs()
    assert first.to_csv(index=False) == second.to_csv(index=False)
    # Also via an in-memory round-trip so dtypes cannot silently drift.
    buf = io.StringIO(first.to_csv(index=False))
    assert pd.read_csv(buf).to_csv(index=False) == first.to_csv(index=False)
