"""Rest features (M5): hand-derived fixtures, leakage under edits from a game on, determinism."""

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from eurohoops.models.rest import REST_SCHEMA, rest_features

COLUMNS = [
    "game_id",
    "season",
    "phase",
    "round",
    "tipoff_utc",
    "home",
    "away",
    "home_score",
    "away_score",
    "played",
    "forfeit",
    "neutral",
]
FEATURES = ["days_rest", "short_rest", "games_last_7d", "other_comp_prev"]


def _games(rows: list[tuple[str, str, str, str, bool]]) -> pd.DataFrame:
    """Rows of (game_id, tip-off 'YYYY-MM-DD HH:MM' UTC, home, away, played)."""
    frame = pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "season": 2026,
            "phase": "RS",
            "round": range(1, len(rows) + 1),
            "tipoff_utc": pd.to_datetime([r[1] for r in rows], utc=True),
            "home": [r[2] for r in rows],
            "away": [r[3] for r in rows],
            "home_score": [80 if r[4] else 0 for r in rows],
            "away_score": [75 if r[4] else 0 for r in rows],
            "played": [r[4] for r in rows],
            "forfeit": False,
            "neutral": False,
        }
    )
    return frame[COLUMNS]


def _row(out: pd.DataFrame, game_id: str, side: str) -> pd.Series:
    rows = out[(out["game_id"] == game_id) & (out["side"] == side)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_pan_week_crosses_competitions() -> None:
    # EuroLeague Thursday 19:00, then GBL Saturday 17:00 (2026-10-08 is a Thursday).
    euro = _games([("E1", "2026-10-08 19:00", "PAN", "OLY", True)])
    gbl = _games([("G1", "2026-10-10 17:00", "00000001", "00000009", False)])
    out = rest_features(gbl, euro, {"PAN": "00000001"})
    pan = _row(out, "G1", "home")
    # Thu 19:00 -> Sat 17:00 = 1 day 22 hours = 46 / 24 days.
    assert pan["days_rest"] == pytest.approx(46 / 24)
    assert pan["short_rest"]
    assert pan["games_last_7d"] == 1
    assert pan["other_comp_prev"]
    assert pan["team"] == "00000001"
    # The opponent maps to no club: nothing before the opener.
    opp = _row(out, "G1", "away")
    assert (opp["days_rest"], opp["games_last_7d"]) == (7.0, 0)


def test_games_last_7d_counts_both_competitions() -> None:
    euro = _games([("E1", "2026-10-01 19:00", "PAN", "OLY", True)])
    gbl = _games(
        [
            ("G1", "2026-10-04 17:00", "00000001", "00000009", True),
            ("G2", "2026-10-08 17:00", "00000001", "00000009", False),
        ]
    )
    pan = _row(rest_features(gbl, euro, {"PAN": "00000001"}), "G2", "home")
    # Window [Oct 1 17:00, Oct 8 17:00): E1 (Oct 1 19:00) and G1 are in; 2 games.
    assert pan["games_last_7d"] == 2
    # Previous game is G1, 4 days earlier: not short, same competition.
    assert pan["days_rest"] == pytest.approx(4.0)
    assert not pan["short_rest"]
    assert not pan["other_comp_prev"]


def test_season_opener_has_no_previous_game() -> None:
    games = _games([("G1", "2026-10-04 17:00", "A", "B", False)])
    out = rest_features(games, None, {})
    for side in ("home", "away"):
        row = _row(out, "G1", side)
        assert row["days_rest"] == 7.0
        assert not row["short_rest"]
        assert row["games_last_7d"] == 0
        assert not row["other_comp_prev"]


def test_postponed_game_does_not_count() -> None:
    games = _games(
        [
            ("G1", "2026-10-01 17:00", "A", "B", True),
            ("G2", "2026-10-03 17:00", "A", "C", False),  # postponed, never played
            ("G3", "2026-10-05 17:00", "A", "D", False),
        ]
    )
    row = _row(rest_features(games, None, {}), "G3", "home")
    # Only G1 counts: 4 days back, one game in the window.
    assert row["days_rest"] == pytest.approx(4.0)
    assert row["games_last_7d"] == 1
    assert not row["short_rest"]


def test_forfeit_counts_when_played() -> None:
    games = _games(
        [
            ("G1", "2026-10-03 17:00", "A", "B", True),
            ("G2", "2026-10-05 17:00", "A", "C", False),
        ]
    )
    games.loc[0, "forfeit"] = True
    row = _row(rest_features(games, None, {}), "G2", "home")
    assert row["days_rest"] == pytest.approx(2.0)
    assert row["short_rest"]
    assert row["games_last_7d"] == 1


def test_team_outside_club_map_ignores_other() -> None:
    euro = _games([("E1", "2026-10-09 19:00", "PAN", "OLY", True)])
    gbl = _games([("G1", "2026-10-10 17:00", "00000001", "00000002", False)])
    # Only 00000001 is mapped (from PAN); 00000002 has no EuroLeague link.
    out = rest_features(gbl, euro, {"PAN": "00000001"})
    assert _row(out, "G1", "home")["games_last_7d"] == 1
    away = _row(out, "G1", "away")
    assert (away["days_rest"], away["games_last_7d"], away["other_comp_prev"]) == (7.0, 0, False)
    # Without any map, nobody links.
    bare = rest_features(gbl, euro, {})
    assert _row(bare, "G1", "home")["games_last_7d"] == 0


def test_cap_at_seven_days() -> None:
    games = _games(
        [
            ("G1", "2026-10-01 17:00", "A", "B", True),
            ("G2", "2026-10-20 17:00", "A", "B", False),
            ("G3", "2026-10-08 17:00", "C", "D", True),
            ("G4", "2026-10-15 17:00", "C", "D", False),
        ]
    )
    out = rest_features(games, None, {})
    # G1 -> G2: 19 days, capped to 7.0; the game is outside the 7-day window.
    capped = _row(out, "G2", "home")
    assert capped["days_rest"] == 7.0
    assert capped["games_last_7d"] == 0
    # G3 -> G4: exactly 7 days: days_rest 7.0, and the half-open window [t - 7d, t) holds G3.
    exact = _row(out, "G4", "home")
    assert exact["days_rest"] == 7.0
    assert exact["games_last_7d"] == 1


def test_tie_at_same_tipoff_does_not_count() -> None:
    games = _games(
        [
            ("G1", "2026-10-03 17:00", "A", "B", True),
            ("G2", "2026-10-03 17:00", "A", "C", False),
        ]
    )
    row = _row(rest_features(games, None, {}), "G2", "home")
    assert (row["days_rest"], row["games_last_7d"]) == (7.0, 0)


def _schedule(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    gbl_teams = [f"G{i}" for i in range(6)]
    el_teams = [f"E{i}" for i in range(6)]
    club_map = {"E0": "G0", "E1": "G1", "E2": "G2"}

    def make(prefix: str, teams: list[str], n: int) -> pd.DataFrame:
        rows = []
        for i in range(n):
            home, away = rng.choice(teams, size=2, replace=False)
            hours = int(rng.integers(0, 60 * 24))
            tip = pd.Timestamp("2026-10-01", tz="UTC") + pd.Timedelta(hours=hours)
            rows.append(
                (
                    f"{prefix}{i}",
                    tip.strftime("%Y-%m-%d %H:%M"),
                    home,
                    away,
                    bool(rng.random() < 0.85),
                )
            )
        return _games(rows)

    return make("G", gbl_teams, 40), make("E", el_teams, 40), club_map


def _edit_from(
    frame: pd.DataFrame, cutoff: pd.Timestamp, keep_id: str, rng: np.random.Generator, tag: str
) -> pd.DataFrame:
    """Delete, move and add games, all at or after ``cutoff`` (``keep_id`` itself is untouched)."""
    edited = frame.copy()
    late = (edited["tipoff_utc"] >= cutoff) & (edited["game_id"] != keep_id)
    drop = late & (rng.random(len(edited)) < 0.3)
    edited = edited[~drop].copy()
    move = (
        (edited["tipoff_utc"] >= cutoff)
        & (edited["game_id"] != keep_id)
        & (rng.random(len(edited)) < 0.5)
    )
    shift = pd.to_timedelta(rng.integers(0, 72, size=int(move.sum())), unit="h")
    edited.loc[move, "tipoff_utc"] = edited.loc[move, "tipoff_utc"] + shift
    teams = pd.concat([frame["home"], frame["away"]]).unique()
    added = []
    for k in range(5):
        home, away = rng.choice(teams, size=2, replace=False)
        tip = cutoff + pd.Timedelta(hours=int(rng.integers(0, 96)))
        added.append((f"{tag}{k}", tip.strftime("%Y-%m-%d %H:%M"), home, away, True))
    edited = pd.concat([edited, _games(added)], ignore_index=True)
    edited["tipoff_utc"] = pd.to_datetime(edited["tipoff_utc"], utc=True)
    return edited


def test_features_ignore_games_at_or_after_tipoff() -> None:
    rng = np.random.default_rng(20261005)
    gbl, euro, club_map = _schedule(rng)
    inverse = {v: k for k, v in club_map.items()}
    for _ in range(25):
        # Either competition is the target; g is kept, all games from its tip-off on are edited.
        target_is_gbl = bool(rng.random() < 0.5)
        # club_map is EL code -> GBL code (GBL target); the EuroLeague target needs the inverse.
        games, other, cmap = (gbl, euro, club_map) if target_is_gbl else (euro, gbl, inverse)
        g = games.iloc[int(rng.integers(0, len(games)))]
        cutoff = g["tipoff_utc"]
        base = rest_features(games, other, cmap)
        new_games = _edit_from(games, cutoff, g["game_id"], rng, "NG")
        new_other = _edit_from(other, cutoff, "", rng, "NO")
        after = rest_features(new_games, new_other, cmap)
        pd.testing.assert_frame_equal(
            base[base["game_id"] == g["game_id"]].reset_index(drop=True),
            after[after["game_id"] == g["game_id"]].reset_index(drop=True),
        )


def test_features_ignore_scores() -> None:
    rng = np.random.default_rng(7)
    gbl, euro, club_map = _schedule(rng)
    base = rest_features(gbl, euro, club_map)
    noisy_gbl, noisy_euro = gbl.copy(), euro.copy()
    for frame in (noisy_gbl, noisy_euro):
        frame["home_score"] = rng.integers(0, 200, size=len(frame))
        frame["away_score"] = rng.integers(0, 200, size=len(frame))
    pd.testing.assert_frame_equal(base, rest_features(noisy_gbl, noisy_euro, club_map))


def test_output_follows_games_order_and_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    gbl, euro, club_map = _schedule(rng)
    first = rest_features(gbl, euro, club_map)
    second = rest_features(gbl, euro, club_map)
    pd.testing.assert_frame_equal(first, second)
    assert first.to_csv(index=False) == second.to_csv(index=False)
    assert list(first["game_id"]) == [g for g in gbl["game_id"] for _ in range(2)]
    assert list(first["side"]) == ["home", "away"] * len(gbl)


@pytest.mark.parametrize("days", [0.0, 8.0])
def test_schema_rejects_out_of_range_days(days: float) -> None:
    good = rest_features(_games([("G1", "2026-10-04 17:00", "A", "B", False)]), None, {})
    bad = good.assign(days_rest=days)
    with pytest.raises(pandera.errors.SchemaError):
        REST_SCHEMA.validate(bad)
