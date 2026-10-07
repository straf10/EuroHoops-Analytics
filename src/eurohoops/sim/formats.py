"""Regular-season format of every scored EuroLeague and GBL season, with its source.

Past formats come from the K0 verification (reports/m7_progress.md, section 3, decisions D3 and
D5): team count and rounds from the games mart, the final table and the playoff/play-in field
from the API games feed, checked by scripts/checks/m7_facts.py. 2026-27 is ``standings``'
``EUROLEAGUE_2026`` / ``GBL_2026``, unchanged.
"""

from eurohoops.standings import EUROLEAGUE_2026, GBL_2026, Format, Series

_BYLAWS_2025 = (
    "https://ftpserver.euroleague.net/general/2025_26_EuroLeague_Bylaws.pdf Art. 18-19"
    " (cached data/raw/regulations/, accessed 2026-10-07)"
)
_NO_EARLIER_BYLAWS = (
    "series lengths and playoff home order from the games feed (games 1-2 at the higher seed),"
    " no bylaws for this season (earlier bylaws are not published at the ftpserver path, 404)"
)

_GBL_UNVERIFIED = (
    "full table order below the cut (only the top-8 set is checked)",
    "series lengths and home order (2025-26: QF/SF best of 3, final best of 5 per the ESAKE"
    " round codes; earlier seasons assumed)",
    "tie-breaks: EuroLeague-style procedure assumed",
    "relegation",
)

_EUROLEAGUE_SEASONS = (2016, 2017, 2018, 2020, 2022, 2023, 2024, 2025)
_GBL_SEASONS = (2020, 2021, 2022, 2025)

_EXCLUDED: dict[tuple[str, int], str] = {
    ("euroleague", 2015): "2015-16 had a group stage and a Top 16, not a single league table",
    ("euroleague", 2019): "2019-20 was cancelled (COVID-19)",
    ("euroleague", 2021): "2021-22 had Russian clubs removed mid-season",
    ("gbl", 2023): (
        "2023-24: the quarterfinal field is not the table's top 8 (cause unverified;"
        " see reports/m7_progress.md)"
    ),
    ("gbl", 2024): (
        "2024-25: the quarterfinal field is not the table's top 8 (cause unverified;"
        " see reports/m7_progress.md)"
    ),
}


def _label(season: int) -> str:
    return f"{season}-{(season + 1) % 100:02d}"


def _euroleague(season: int) -> Format:
    teams = 20 if season >= 2025 else 18 if season >= 2020 else 16
    rounds = 2 * (teams - 1)
    play_in = season >= 2023
    direct = tuple(range(1, 7 if play_in else 9))
    in_play = (7, 8, 9, 10) if play_in else ()
    sources = [
        f"official final table https://api-live.euroleague.net/v1/standings?seasonCode=E{season}"
        f"&roundNumber={rounds} (accessed 2026-10-07) and the playoff/play-in field in the API"
        " games feed, checked by scripts/checks/m7_facts.py"
    ]
    unverified: tuple[str, ...] = ()
    if season == 2025:
        sources.insert(0, _BYLAWS_2025)
    else:
        unverified = (_NO_EARLIER_BYLAWS,)
    series: tuple[Series, ...] = (Series("playoffs", 5), Series("final four", 1))
    if play_in:
        series = (Series("play-in", 1), *series)
    return Format(
        competition="euroleague",
        season=_label(season),
        teams=teams,
        regular_season_rounds=rounds,
        playoffs_direct=direct,
        play_in=in_play,
        eliminated=tuple(range(1 + len(direct) + len(in_play), teams + 1)),
        relegated=(),
        series=series,
        sources=tuple(sources),
        unverified=unverified,
        # Panathinaikos had 2 wins deducted in 2022-23; Format stores points at 2 per win.
        points_deducted=(("PAN", 4),) if season == 2022 else (),
    )


def _gbl(season: int) -> Format:
    teams, rounds = (13, 26) if season in (2021, 2025) else (12, 22)
    bye = " (a bye per round, 24 games per team)" if teams == 13 else ""
    return Format(
        competition="gbl",
        season=_label(season),
        teams=teams,
        regular_season_rounds=rounds,
        playoffs_direct=tuple(range(1, 9)),
        play_in=(),
        eliminated=tuple(range(9, teams + 1)),
        relegated=(),
        series=(Series("quarterfinals", 3), Series("semifinals", 3), Series("final", 5)),
        sources=(
            f"ESAKE results pages (games mart): {teams} teams, {rounds} rounds{bye};"
            " quarterfinal field = top 8 of the table, checked by scripts/checks/m7_facts.py",
        ),
        unverified=_GBL_UNVERIFIED,
    )


def season_format(competition: str, season: int) -> Format:
    """Format of ``competition`` ("euroleague" or "gbl") in the season starting in ``season``.

    2026 returns ``EUROLEAGUE_2026`` / ``GBL_2026`` unchanged; scored past seasons come from the
    K0 table; anything else raises ValueError.
    """
    if competition not in ("euroleague", "gbl"):
        raise ValueError(f"unknown competition {competition!r}")
    if season == 2026:
        return EUROLEAGUE_2026 if competition == "euroleague" else GBL_2026
    if (competition, season) in _EXCLUDED:
        raise ValueError(
            f"{competition} {_label(season)} is not scored: {_EXCLUDED[competition, season]}"
        )
    if competition == "euroleague" and season in _EUROLEAGUE_SEASONS:
        return _euroleague(season)
    if competition == "gbl" and season in _GBL_SEASONS:
        return _gbl(season)
    raise ValueError(f"no verified format for {competition} {_label(season)}")


def cut_lines(fmt: Format) -> dict[str, tuple[int, ...]]:
    """Positions the backtest scores against: "direct" playoffs, plus "play_in" and "top10"
    (direct + play-in) when the format has a play-in."""
    lines = {"direct": fmt.playoffs_direct}
    if fmt.play_in:
        lines["play_in"] = fmt.play_in
        lines["top10"] = fmt.playoffs_direct + fmt.play_in
    return lines
