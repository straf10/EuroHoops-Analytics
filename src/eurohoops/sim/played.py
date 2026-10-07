"""Played regular-season games as ``standings.Result``, with regulation scores for overtime games.

Overtime points count for no tie-break (EuroLeague Bylaws Art. 19.4), so a EuroLeague game that
went to overtime carries its score after 40 minutes, read from ``EndOfQuarter`` in the cached
box score (the home team first). The GBL's overtime rule is unverified and its quarter scores
are not parsed, so GBL results use the final score (``GBL_2026.unverified``).
"""

import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from eurohoops.ingest.cache import read_cached
from eurohoops.standings import Result


def regulation_score(box: Mapping[str, object]) -> tuple[int, int] | None:
    """(home, away) after the fourth quarter of a box score, or None without overtime."""
    end = box["EndOfQuarter"]
    assert isinstance(end, list)
    home, away = end
    extra = [k for side in end for k in side if k.startswith("Extra") and side[k]]
    return (int(home["Quarter4"]), int(away["Quarter4"])) if extra else None


def regulation_scores(raw_dir: Path, games: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """``game_id`` -> regulation score of every played EuroLeague game in ``games`` that went
    to overtime, from the cached box scores under ``raw_dir`` (never fetched here)."""
    out: dict[str, tuple[int, int]] = {}
    for game in games[games["played"]].itertuples():
        path = raw_dir / "boxscore" / f"E{game.season}" / f"{game.game_code}.json.gz"
        score = regulation_score(json.loads(read_cached(path)))
        if score is not None:
            out[str(game.game_id)] = score
    return out


def season_results(
    games: pd.DataFrame, regulation: Mapping[str, tuple[int, int]] | None = None
) -> list[Result]:
    """Played games of ``games`` in tip-off order, as ``Result``. A forfeit counts as the win
    it was awarded (its 20-0 score), as in the official table."""
    regulation = regulation or {}
    played = games[games["played"]].sort_values(["tipoff_utc", "game_id"])
    columns = ("game_id", "home", "away", "home_score", "away_score")
    return [
        Result(str(home), str(away), int(hs), int(as_), regulation.get(str(gid)))
        for gid, home, away, hs, as_ in zip(*(played[c] for c in columns), strict=True)
    ]
