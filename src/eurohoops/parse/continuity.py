"""Team continuity across seasons: new, returning and departed club codes, and name changes.

Ratings follow a club code from season to season, so a code that a different club took over
would silently hand one club's history to another. The report lists every season's changes
per competition; a name that shares no word with the code's previous name is flagged for
review (a sponsor rename such as "Tau Ceramica" -> "Caja Laboral" also shares none, so a
flag means "confirm it is the same club", not "wrong"). Informational: the build never fails
on it. JSON-ready and free of timestamps, so reruns are byte-identical.
"""

import re
from typing import Any

import pandas as pd

MIN_WORD = 3  # shorter tokens ("bc", "kk", "tt") say nothing about the club


def _words(name: str) -> set[str]:
    return {w for w in re.findall(r"[^\W\d_]+", name.lower()) if len(w) >= MIN_WORD}


def _shares_a_word(old: str, new: str) -> bool:
    return bool(_words(old) & _words(new))


def continuity(team_seasons: pd.DataFrame) -> dict[str, Any]:
    """``team_seasons``: one competition's (season, team, name) rows."""
    by_season: dict[int, dict[str, str]] = {}
    for season, team, name in zip(
        team_seasons["season"], team_seasons["team"], team_seasons["name"], strict=True
    ):
        by_season.setdefault(int(season), {})[str(team)] = str(name)
    seasons = sorted(by_season)
    out: dict[str, Any] = {}
    review: list[dict[str, Any]] = []
    last_seen: dict[str, tuple[int, str]] = {}  # team -> (last season, name then)
    for i, season in enumerate(seasons):
        teams = by_season[season]
        if i == 0:
            out[str(season)] = {"teams": len(teams), "first_season_in_data": True}
        else:
            previous = by_season[seasons[i - 1]]
            new, returning, renamed = [], [], []
            for team in sorted(teams):
                name = teams[team]
                if team not in last_seen:
                    new.append({"team": team, "name": name})
                    continue
                last, name_then = last_seen[team]
                if team not in previous:
                    returning.append(
                        {"team": team, "name": name, "last_season": last, "name_then": name_then}
                    )
                if name != name_then:
                    renamed.append({"team": team, "from": name_then, "to": name})
                    if not _shares_a_word(name_then, name):
                        review.append(
                            {"season": season, "team": team, "from": name_then, "to": name}
                        )
            out[str(season)] = {
                "teams": len(teams),
                "new": new,
                "returning": returning,
                "left": [
                    {"team": t, "name": previous[t]} for t in sorted(set(previous) - set(teams))
                ],
                "renamed": renamed,
            }
        for team, name in teams.items():
            last_seen[team] = (season, name)
    return {"seasons": out, "review_name_changes": review}


def continuity_report(team_seasons: pd.DataFrame) -> dict[str, Any]:
    """``team_seasons`` with a ``competition`` column -> one block per competition."""
    return {
        str(competition): continuity(rows)
        for competition, rows in sorted(team_seasons.groupby("competition"))
    }
