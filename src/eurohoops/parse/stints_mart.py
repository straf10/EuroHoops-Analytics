"""EuroLeague stints mart (2011-12 on): 5-on-5 stints with points and possessions, and game checks.

A stint here is a stretch of one period in which neither team's five changes (a substitution
by either team closes it). Points and possession ends count for the stint open when their row
is logged; possession ends come from the play-by-play counter in ``parse.possessions``.

Every game with cached play-by-play and box score gets a ``stint_game_checks`` row: the spike's
four checks (``parse.stints.validate``) plus ``possessions`` (the play-by-play possession count
is within ``POSSESSION_TOLERANCE`` of the box formula for both teams). Failing games keep
their stints and are listed with reasons; nothing is dropped. The stints of games that pass
the four checks are the ones a lineup model may use.

2007-08 to 2010-11 are out: their substitution clocks are whole minutes (docs/spikes/stints.md).
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.possessions import FT_WEIGHT, PossessionEnd, box_possessions, possession_ends
from eurohoops.parse.stints import (
    CHECKS,
    POINTS,
    QUARTERS,
    Event,
    box_facts,
    build_stints,
    events,
    period_length,
    period_start,
    validate,
)
from eurohoops.parse.team_box import euroleague_lines

FIRST_SEASON = 2011
POSSESSION_TOLERANCE = 5.0  # per team; the ±2 agreement is reported in reports/possessions.json
GAME_CHECKS = (*CHECKS, "possessions")
STINT_COLUMNS = (
    "game_id",
    "season",
    "period",
    "start_s",
    "end_s",
    "home",
    "away",
    "home_players",
    "away_players",
    "home_points",
    "away_points",
    "home_poss",
    "away_poss",
)


@dataclass(frozen=True)
class GameStint:
    period: int
    start: int
    end: int
    players: tuple[tuple[str, ...], tuple[str, ...]]  # home five, away five (sorted ids)
    points: tuple[int, int]
    possessions: tuple[int, int]


@dataclass
class _OpenStint:
    period: int
    start: int
    teams: tuple[str, str]
    points: dict[str, int] = field(default_factory=dict)
    poss: dict[str, int] = field(default_factory=dict)

    def close(self, at: int, lineups: dict[str, set[str]], out: list[GameStint]) -> None:
        home, away = self.teams
        out.append(
            GameStint(
                self.period,
                self.start,
                at,
                (tuple(sorted(lineups[home])), tuple(sorted(lineups[away]))),
                (self.points.get(home, 0), self.points.get(away, 0)),
                (self.poss.get(home, 0), self.poss.get(away, 0)),
            )
        )
        self.start, self.points, self.poss = at, {}, {}


def build_game_stints(
    game_events: Sequence[Event],
    starters: dict[str, set[str]],
    n_overtimes: int,
    teams: tuple[str, str],
    ends: Sequence[PossessionEnd],
) -> list[GameStint]:
    """5-on-5 stints in time order; ``teams`` = (home, away), ``ends`` index ``game_events``."""
    ended: dict[int, list[str]] = {}
    for end in ends:
        ended.setdefault(end.index, []).append(end.team)
    lineups = {team: set(starters.get(team, set())) for team in teams}
    by_period: dict[int, list[int]] = {}
    for index, event in enumerate(game_events):
        by_period.setdefault(event.period, []).append(index)
    out: list[GameStint] = []
    for period in range(1, len(QUARTERS) + n_overtimes + 1):
        stint = _OpenStint(period, period_start(period), teams)
        for index in by_period.get(period, []):
            event = game_events[index]
            if event.team in lineups and event.kind in {"IN", "OUT"} and event.elapsed is not None:
                if event.elapsed > stint.start:
                    stint.close(event.elapsed, lineups, out)
                if event.kind == "IN":
                    lineups[event.team].add(event.player)
                else:
                    lineups[event.team].discard(event.player)
            elif event.team in lineups and event.kind in POINTS:
                stint.points[event.team] = stint.points.get(event.team, 0) + POINTS[event.kind]
            for team in ended.get(index, []):
                stint.poss[team] = stint.poss.get(team, 0) + 1
        stint.close(period_start(period) + period_length(period), lineups, out)
    return out


@dataclass(frozen=True)
class GameResult:
    game_id: str
    season: int
    stints: list[GameStint]
    teams: tuple[str, str]
    reasons: dict[str, list[str]]  # check -> failure reasons (empty = pass)


def game_stints(
    game_id: str, season: int, pbp: dict[str, Any], box: dict[str, Any], teams: tuple[str, str]
) -> GameResult:
    """Stints and check results of one game (``teams``: home, away from the schedule)."""
    parsed = euroleague_lines(box)
    if isinstance(parsed, str):
        return GameResult(game_id, season, [], teams, {c: [parsed] for c in GAME_CHECKS})
    lines, _ = parsed
    facts = box_facts(box)
    game_events = events(pbp)
    reasons = validate(build_stints(game_events, facts.starters, facts.overtimes), facts)
    ends = possession_ends(game_events, teams)
    stints = build_game_stints(game_events, facts.starters, facts.overtimes, teams, ends)
    reasons["possessions"] = []
    for side, team in enumerate(teams):
        line = lines.get(team)
        if line is None:
            reasons["possessions"].append(f"{team}: not in the box score")
            continue
        counted = sum(s.possessions[side] for s in stints)
        box_poss = box_possessions(line["fga"], line["oreb"], line["tov"], line["fta"])
        if abs(counted - box_poss) > POSSESSION_TOLERANCE:
            reasons["possessions"].append(f"{team}: {counted} counted, {box_poss:.2f} box formula")
    return GameResult(game_id, season, stints, teams, reasons)


def _games(raw_dir: Path, games: pd.DataFrame) -> list[tuple[str, int, int, tuple[str, str]]]:
    rated = games[games["played"] & ~games["forfeit"] & (games["season"] >= FIRST_SEASON)]
    return [
        (str(game_id), int(season), int(code), (str(home), str(away)))
        for game_id, season, code, home, away in zip(
            rated["game_id"],
            rated["season"],
            rated["game_code"],
            rated["home"],
            rated["away"],
            strict=True,
        )
    ]


@dataclass(frozen=True)
class StintsMart:
    stints: pd.DataFrame
    checks: pd.DataFrame
    not_cached: list[str]  # rated games without cached play-by-play or box score


def build_stints_mart(raw_dir: Path, games: pd.DataFrame) -> StintsMart:
    """Stints and per-game checks of every rated EuroLeague game from 2011-12 with both files."""
    stint_rows: list[dict[str, Any]] = []
    check_rows: list[dict[str, Any]] = []
    not_cached: list[str] = []
    for game_id, season, code, teams in _games(raw_dir, games):
        name = f"{code}.json.gz"
        pbp_path = raw_dir / "playbyplay" / f"E{season}" / name
        box_path = raw_dir / "boxscore" / f"E{season}" / name
        if not (pbp_path.exists() and box_path.exists()):
            not_cached.append(game_id)
            continue
        result = game_stints(
            game_id,
            season,
            json.loads(read_cached(pbp_path)),
            json.loads(read_cached(box_path)),
            teams,
        )
        for stint in result.stints:
            stint_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "period": stint.period,
                    "start_s": stint.start,
                    "end_s": stint.end,
                    "home": teams[0],
                    "away": teams[1],
                    "home_players": list(stint.players[0]),
                    "away_players": list(stint.players[1]),
                    "home_points": stint.points[0],
                    "away_points": stint.points[1],
                    "home_poss": stint.possessions[0],
                    "away_poss": stint.possessions[1],
                }
            )
        four_pass = not any(result.reasons[c] for c in CHECKS)
        check_rows.append(
            {
                "game_id": game_id,
                "season": season,
                **{c: not result.reasons[c] for c in GAME_CHECKS},
                "passed": four_pass,
                "reasons": "; ".join(
                    f"{c}: {why}" for c in GAME_CHECKS for why in result.reasons[c]
                ),
            }
        )
    stints = pd.DataFrame(stint_rows, columns=list(STINT_COLUMNS))
    checks = pd.DataFrame(
        check_rows, columns=["game_id", "season", *GAME_CHECKS, "passed", "reasons"]
    )
    return StintsMart(
        stints.astype({"season": "int64", "period": "int64", "start_s": "int64"}),
        checks.astype({"season": "int64"}),
        not_cached,
    )


def digest(frame: pd.DataFrame) -> str:
    """sha256 of a table's canonical CSV (row order as stored), for the two-build check."""
    text = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def _possession_agreement(
    mart: StintsMart, team_games: pd.DataFrame
) -> dict[str, dict[str, Any] | None]:
    """Share of team-games whose play-by-play count is within 2 of the box formula, per span."""
    counted = mart.stints.groupby(["game_id", "season"], as_index=False).agg(
        home=("home", "first"),
        away=("away", "first"),
        home_poss=("home_poss", "sum"),
        away_poss=("away_poss", "sum"),
    )
    sides = pd.concat(
        [
            counted[["game_id", "season", "home", "home_poss"]].set_axis(
                ["game_id", "season", "team", "pbp"], axis=1
            ),
            counted[["game_id", "season", "away", "away_poss"]].set_axis(
                ["game_id", "season", "team", "pbp"], axis=1
            ),
        ]
    )
    joined = sides.merge(team_games[["game_id", "team", "poss_raw", "fta"]], on=["game_id", "team"])
    gap = joined["pbp"] - joined["poss_raw"]
    out = {}
    for label, first, last in (
        ("2011_2014", 2011, 2014),
        ("2015_on", 2015, 9999),
        ("all", 0, 9999),
    ):
        span = joined["season"].between(first, last)
        out[label] = (
            {
                "team_games": int(span.sum()),
                "within_2_share": round(float((gap[span].abs() <= 2.0).mean()), 4),
                "mean_gap_pbp_minus_box": round(float(gap[span].mean()), 4),
                # The weight that zeroes the mean gap: FT_WEIGHT + sum(gap) / sum(FTA).
                "ft_weight_matching_pbp": round(
                    FT_WEIGHT + float(gap[span].sum()) / float(joined["fta"][span].sum()), 4
                ),
            }
            if span.any()
            else None
        )
    return out


def mart_report(mart: StintsMart, games: pd.DataFrame, team_games: pd.DataFrame) -> dict[str, Any]:
    """Pass rates per season (four spike checks; possessions reported separately), the
    stint-points-vs-final-score check for every passing game, and PBP vs box possessions."""
    checks = mart.checks
    sums = mart.stints.groupby("game_id")[["home_points", "away_points"]].sum()
    finals = games.set_index("game_id")[["home_score", "away_score"]]
    joined = sums.join(finals, how="left")
    off = joined[
        (joined["home_points"] != joined["home_score"])
        | (joined["away_points"] != joined["away_score"])
    ]
    passing = set(checks.loc[checks["passed"], "game_id"])
    points_off = sorted(str(game_id) for game_id in off.index if game_id in passing)
    seasons: dict[str, Any] = {}
    for season, rows in checks.groupby("season"):
        seasons[str(season)] = {
            "games": len(rows),
            "passed": int(rows["passed"].sum()),
            "pass_rate": round(float(rows["passed"].mean()), 4),
            "possessions_pass_rate": round(float(rows["possessions"].mean()), 4),
            "failures_by_check": {c: int((~rows[c]).sum()) for c in GAME_CHECKS},
            "failed_games": {
                str(r["game_id"]): str(r["reasons"])
                for r in rows[~rows["passed"]].to_dict("records")
            },
        }

    def rate(first: int, last: int) -> float | None:
        span = checks[checks["season"].between(first, last)]
        return None if span.empty else round(float(span["passed"].mean()), 4)

    return {
        "first_season": FIRST_SEASON,
        "possession_tolerance": POSSESSION_TOLERANCE,
        "pass_rate_2011_2014": rate(2011, 2014),
        "pass_rate_2015_on": rate(2015, 9999),
        "games": len(checks),
        "stints": len(mart.stints),
        "not_cached": mart.not_cached,
        "passing_games_where_stint_points_miss_the_final": points_off,
        "pbp_vs_box_possessions": _possession_agreement(mart, team_games),
        "seasons": seasons,
        "table_sha256": {"stints": digest(mart.stints), "stint_game_checks": digest(checks)},
    }
