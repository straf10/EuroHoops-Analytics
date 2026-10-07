"""A small repository tree for the API tests: the layout ``eurohoops.config`` names, under a
temporary root, built from the committed fixtures.

- EuroLeague: the cached schedules and box scores of ``stats_raw`` (rounds 1-6 of 2024-25 and
  round 1 of 2010-11) plus a synthetic live season 2026 on six of their clubs (round 1 played).
- GBL: a synthetic double round-robin (2025 played, 2026 not) on ``AAA``-``FFF`` plus one real
  game, ``box_8FC479F6.html``, as 2025 game ``GBL2025_X1`` with its cached box page and
  ``team_games`` rows hand-read from the page's totals row.
- ``player_xwalk`` maps the EuroLeague player P013382 and the GBL player FC6A957C to one person.
- ``tests/fixtures/api``: the reports and prediction logs (ungated simulation optional).
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from eurohoops.config import EUROLEAGUE, GBL, MART_PATH
from eurohoops.ingest.cache import write_atomic
from eurohoops.marts import write_tables
from eurohoops.parse.games import conform
from eurohoops.parse.team_box import TEAM_GAMES_SCHEMA
from eurohoops.stats.export import load_cached_games
from tests.conftest import FIXTURES, TEAMS, esake_fixture, make_games, teams_table

NOW = datetime(2026, 10, 8, 8, 0, tzinfo=UTC)
API_FIXTURES = FIXTURES / "api"
EL_CLUBS = dict(zip(TEAMS, ("BAR", "MAD", "OLY", "PAN", "ULK", "PAM"), strict=True))
GBL_REAL_GAME = "GBL2025_X1"
GBL_REAL_CODE = int("8FC479F6", 16)
EL_PERSON = "P:P013382"  # Levi Randolph: EuroLeague id P013382, GBL id FC6A957C
GBL_ONLY_PERSON = "G:02211B0A"


def _euroleague_games(cached: pd.DataFrame) -> pd.DataFrame:
    """The cached 2010 and 2024 games plus the live season on six clubs; round 1 is played."""
    live = make_games({2026: False})
    live = live.assign(home=live["home"].map(EL_CLUBS), away=live["away"].map(EL_CLUBS))
    first = live["round"] == 1
    live.loc[first, ["home_score", "away_score", "played"]] = [90, 80, True]
    return conform(pd.concat([cached, live], ignore_index=True))


def _gbl_games() -> pd.DataFrame:
    real = {
        "game_id": GBL_REAL_GAME,
        "season": 2025,
        "game_code": GBL_REAL_CODE,
        "phase": "RS",
        "round": 1,
        "round_label": "Round 1",
        "tipoff_utc": pd.Timestamp("2025-11-02T18:00:00Z"),
        "home": "AAAA0001",
        "away": "AAAA0002",
        "home_score": 93,
        "away_score": 101,
        "played": True,
        "forfeit": False,
        "neutral": False,
        "confirmed_date": True,
    }
    synthetic = make_games({2025: True, 2026: False}, gbl_like=True)
    return conform(pd.concat([synthetic, pd.DataFrame([real])], ignore_index=True))


def _gbl_team_games() -> pd.DataFrame:
    """The real game's two ``team_games`` rows: counts from the page's totals row (ΣΥΝΟΛΟ)."""
    rows = []
    for team, opponent, home, pts, fga, fta, oreb, dreb, tov in (
        ("AAAA0001", "AAAA0002", True, 93, 74, 27, 18, 27, 18),
        ("AAAA0002", "AAAA0001", False, 101, 71, 28, 11, 25, 15),
    ):
        rows.append(
            {
                "competition": "gbl",
                "season": 2025,
                "game_id": GBL_REAL_GAME,
                "team": team,
                "opponent": opponent,
                "home": home,
                "points": pts,
                "fga": fga,
                "fta": fta,
                "oreb": oreb,
                "dreb": dreb,
                "tov": tov,
                "minutes": 45.0,
                "poss_raw": fga - oreb + tov + 0.42 * fta,
                "source": "esake_box",
            }
        )
    frame = pd.DataFrame(rows)
    frame["poss_game"] = frame["poss_raw"].mean()
    return TEAM_GAMES_SCHEMA.validate(frame[list(TEAM_GAMES_SCHEMA.columns)])


def build_tree(root: Path, *, ungated: bool = True) -> None:
    """Write the tree under ``root`` (the marts, the raw caches, reports, prediction logs)."""
    raw = root / EUROLEAGUE.raw_dir
    shutil.copytree(FIXTURES / "stats_raw", raw)
    write_atomic(
        root / GBL.raw_dir / "boxscore" / "2025" / "8FC479F6.html.gz",
        esake_fixture("box_8FC479F6.html").encode(),
    )
    for source in ("reports", "predictions"):
        shutil.copytree(API_FIXTURES / source, root / source)
    if not ungated:
        (root / "reports" / "sim_ungated_euroleague.json").unlink()

    cached, cached_teams = load_cached_games(raw)
    el_games, gbl_games = _euroleague_games(cached), _gbl_games()
    el_teams = pd.concat([cached_teams, teams_table(el_games)]).drop_duplicates("team")
    teams = pd.concat(
        [
            el_teams.assign(competition="euroleague"),
            teams_table(gbl_games).assign(competition="gbl"),
        ],
        ignore_index=True,
    )
    xwalk = pd.DataFrame(
        {
            "person_id": [EL_PERSON, EL_PERSON],
            "competition": ["euroleague", "gbl"],
            "source_id": ["P013382", "FC6A957C"],
            "method": ["none", "auto"],
            "score": [None, 0.95],
        }
    )
    names = pd.DataFrame(
        {
            "competition": ["gbl"],
            "source_id": ["02211B0A"],
            "season": [2025],
            "team": ["AAAA0001"],
            "jersey": ["7"],
            "surname_raw": ["ΠΑΠΑΣ"],
            "first_raw": ["ΝΙΚΟΣ"],
            "games": [1],
            "minutes": [39.0],
        }
    )
    mart = root / MART_PATH
    mart.parent.mkdir(parents=True)
    write_tables(
        mart,
        {
            "games": pd.concat(
                [el_games.assign(competition="euroleague"), gbl_games.assign(competition="gbl")],
                ignore_index=True,
            ),
            "teams": teams,
            "team_games": _gbl_team_games(),
            "player_xwalk": xwalk,
            "player_names": names,
        },
    )
