"""E1 validation of ``team_games``: coverage, points against the results, and possessions.

- Coverage: per competition and season, rated games with two team rows vs listed as missing.
- Points: every team row's points equal the ``games`` score (build refuses others; re-checked).
- EuroLeague: box-formula possessions vs the play-by-play count on the stint sample (seed
  20260925), per team; the target is agreement within 2 possessions (PLAN §4.4).
- GBL: play-by-play counts vs ESAKE totals on the 2018-20 games that have both.

JSON-ready and free of timestamps, so reruns are byte-identical.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eurohoops.ingest.cache import read_cached
from eurohoops.parse.esake import parse_box_score
from eurohoops.parse.gbl_pbp import TEAM_COUNTS, team_counts
from eurohoops.parse.possessions import FT_WEIGHT, box_possessions, count_possessions
from eurohoops.parse.stints import SAMPLE_SEED, events, sample_games
from eurohoops.parse.team_box import euroleague_lines

AGREEMENT = 2.0  # possessions per team per game (PLAN §4.4)


def _round(value: float) -> float:
    return round(float(value), 4)


def coverage(
    team_games: pd.DataFrame, missing: pd.DataFrame, games: pd.DataFrame
) -> dict[str, Any]:
    """``games``: both competitions' marts rows with a ``competition`` column."""
    rated = games[games["played"] & ~games["forfeit"]]
    out: dict[str, Any] = {}
    for (competition, season), rows in rated.groupby(["competition", "season"]):
        have = team_games[
            (team_games["competition"] == competition) & (team_games["season"] == season)
        ]
        gaps = missing[(missing["competition"] == competition) & (missing["season"] == season)]
        out.setdefault(str(competition), {})[str(season)] = {
            "rated_games": len(rows),
            "with_rows": int(have["game_id"].nunique()),
            "missing": len(gaps),
        }
    return out


def points_mismatches(team_games: pd.DataFrame, games: pd.DataFrame) -> list[str]:
    """Game ids whose team-row points differ from the ``games`` score (should be none)."""
    sides = pd.concat(
        [
            games[["game_id", "home", "home_score"]].set_axis(["game_id", "team", "score"], axis=1),
            games[["game_id", "away", "away_score"]].set_axis(["game_id", "team", "score"], axis=1),
        ]
    )
    joined = team_games.merge(sides, on=["game_id", "team"], how="left")
    bad = joined[joined["score"].isna() | (joined["points"] != joined["score"])]
    return sorted(set(bad["game_id"]))


def euroleague_sample(raw_dir: Path) -> dict[str, Any]:
    """Box-formula vs play-by-play possessions per team on the stint sample."""
    rows = []
    excluded: dict[str, str] = {}
    for season, code in sample_games(raw_dir):
        name = f"{code}.json.gz"
        box = json.loads(read_cached(raw_dir / "boxscore" / f"E{season}" / name))
        parsed = euroleague_lines(box)
        if isinstance(parsed, str):
            excluded[f"E{season}_{code}"] = parsed
            continue
        lines, _ = parsed
        pbp = events(json.loads(read_cached(raw_dir / "playbyplay" / f"E{season}" / name)))
        counted = count_possessions(pbp, list(lines))
        for team, line in lines.items():
            rows.append(
                {
                    "game_id": f"E{season}_{code}",
                    "team": team,
                    "fta": line["fta"],
                    "box": box_possessions(line["fga"], line["oreb"], line["tov"], line["fta"]),
                    "pbp": counted[team],
                }
            )
    scored = pd.DataFrame(rows)
    gap = (scored["pbp"] - scored["box"]).to_numpy(dtype=np.float64)
    fta = scored["fta"].to_numpy(dtype=np.float64)
    within = np.abs(gap) <= AGREEMENT
    per_game = pd.Series(within).groupby(scored["game_id"].to_numpy()).all()
    # The free-throw weight that would make the mean gap zero on this sample (in-sample).
    weight = FT_WEIGHT + float(gap.mean()) / float(fta.mean())
    refit_gap = gap - (weight - FT_WEIGHT) * fta
    return {
        "seed": SAMPLE_SEED,
        "games": int(scored["game_id"].nunique()),
        "excluded": excluded,
        "team_games": len(scored),
        "tolerance": AGREEMENT,
        "within_tolerance_share_of_teams": _round(within.mean()),
        "within_tolerance_share_of_games": _round(per_game.mean()),
        "mean_gap_pbp_minus_box": _round(gap.mean()),
        "sd_gap": _round(gap.std()),
        "max_abs_gap": _round(np.abs(gap).max()),
        "ft_weight_matching_pbp": _round(weight),
        "within_tolerance_share_of_teams_at_that_weight": _round(
            (np.abs(refit_gap) <= AGREEMENT).mean()
        ),
        "per_team": [
            {
                "game_id": r["game_id"],
                "team": r["team"],
                "box": _round(r["box"]),
                "pbp": int(r["pbp"]),
            }
            for r in scored.to_dict("records")
        ],
    }


def gbl_pbp_vs_esake(raw_dir: Path, gbl_games: pd.DataFrame, pbp: pd.DataFrame) -> dict[str, Any]:
    """Play-by-play team counts vs ESAKE totals on games that have both."""
    counts = team_counts(pbp)
    where = {
        str(game_id): (int(season), int(code))
        for game_id, season, code in zip(
            gbl_games["game_id"], gbl_games["season"], gbl_games["game_code"], strict=True
        )
    }
    gaps: dict[str, list[int]] = {column: [] for column in TEAM_COUNTS}
    games = 0
    for game_id, rows in counts.groupby("game_id"):
        season, code = where[str(game_id)]
        path = raw_dir / "boxscore" / str(season) / f"{code:08X}.html.gz"
        boxes = parse_box_score(read_cached(path).decode()) if path.exists() else None
        if boxes is None:
            continue
        games += 1
        by_side = {str(r["side"]): r for r in rows.to_dict("records")}
        for side, box in zip(("home", "away"), boxes, strict=True):
            t = box.totals
            esake = {
                "points": t.points,
                "fga": t.fg2a + t.fg3a,
                "fta": t.fta,
                "oreb": t.oreb,
                "dreb": t.dreb,
                "tov": t.tov,
            }
            for column in TEAM_COUNTS:
                gaps[column].append(int(by_side[side][column]) - esake[column])
    return {
        "games": games,
        "exact_share": {c: _round(np.mean(np.array(v) == 0)) for c, v in gaps.items()},
        "mean_gap_pbp_minus_esake": {c: _round(np.mean(v)) for c, v in gaps.items()},
    }


def possession_report(
    team_games: pd.DataFrame,
    *,
    missing: pd.DataFrame,
    games: pd.DataFrame,
    euroleague_raw: Path,
    gbl_raw: Path,
    gbl_pbp: pd.DataFrame | None,
) -> dict[str, Any]:
    has_sample = any((euroleague_raw / "playbyplay").glob("E*/*.json.gz"))
    return {
        "coverage": coverage(team_games, missing, games),
        "missing_games": missing.to_dict("records"),
        "points_mismatches": points_mismatches(team_games, games),
        "euroleague_pbp_sample": euroleague_sample(euroleague_raw) if has_sample else None,
        "gbl_pbp_vs_esake": (
            None
            if gbl_pbp is None
            else gbl_pbp_vs_esake(gbl_raw, games[games["competition"] == "gbl"], gbl_pbp)
        ),
    }
