import json
from pathlib import Path

import pytest

from eurohoops.sim.formats import cut_lines, season_format
from eurohoops.standings import EUROLEAGUE_2026, GBL_2026

SUMMARY = json.loads((Path(__file__).parent / "fixtures" / "m7_seasons.json").read_text())


def _scored() -> list[tuple[str, int]]:
    scored = []
    for key in SUMMARY:
        competition, season = key.rsplit("_", 1)
        try:
            season_format(competition, int(season))
        except ValueError:
            continue
        scored.append((competition, int(season)))
    return scored


def test_every_fixture_season_is_scored_or_raises():
    keys = {f"{c}_{s}" for c, s in _scored()}
    for key in SUMMARY:
        competition, season = key.rsplit("_", 1)
        if key not in keys:
            with pytest.raises(ValueError):
                season_format(competition, int(season))
    assert set(SUMMARY) - keys == {"gbl_2023", "gbl_2024"}


@pytest.mark.parametrize("competition,season", _scored())
def test_format_matches_games_mart(competition, season):
    fmt = season_format(competition, season)
    mart = SUMMARY[f"{competition}_{season}"]
    assert fmt.teams == mart["teams"]
    assert fmt.regular_season_rounds == mart["rounds"]
    assert mart["games_per_team"] == [2 * (fmt.teams - 1)]
    assert fmt.competition == competition
    assert fmt.season == f"{season}-{(season + 1) % 100:02d}"


@pytest.mark.parametrize("competition,season", _scored())
def test_sources_and_partition(competition, season):
    fmt = season_format(competition, season)
    assert len(fmt.sources) >= 1
    positions = [*fmt.playoffs_direct, *fmt.play_in, *fmt.eliminated, *fmt.relegated]
    assert sorted(positions) == list(range(1, fmt.teams + 1))


def test_2026_is_unchanged():
    assert season_format("euroleague", 2026) is EUROLEAGUE_2026
    assert season_format("gbl", 2026) is GBL_2026


def test_deduction_only_in_euroleague_2022():
    assert season_format("euroleague", 2022).deducted_wins() == {"PAN": 2}
    for competition, season in _scored():
        if (competition, season) != ("euroleague", 2022):
            assert season_format(competition, season).points_deducted == ()


def test_cut_lines_with_and_without_play_in():
    assert cut_lines(season_format("euroleague", 2025)) == {
        "direct": (1, 2, 3, 4, 5, 6),
        "play_in": (7, 8, 9, 10),
        "top10": tuple(range(1, 11)),
    }
    assert cut_lines(season_format("euroleague", 2018)) == {"direct": tuple(range(1, 9))}
    assert cut_lines(season_format("gbl", 2021)) == {"direct": tuple(range(1, 9))}


def test_series_follow_the_play_in():
    assert season_format("euroleague", 2023).series[0].name == "play-in"
    assert [s.name for s in season_format("euroleague", 2022).series] == ["playoffs", "final four"]


@pytest.mark.parametrize(
    "competition,season",
    [
        ("euroleague", 2015),
        ("euroleague", 2019),
        ("euroleague", 2021),
        ("euroleague", 2014),
        ("euroleague", 2027),
        ("gbl", 2018),
        ("gbl", 2019),
        ("gbl", 2023),
        ("gbl", 2024),
        ("nba", 2025),
    ],
)
def test_unscored_seasons_raise(competition, season):
    with pytest.raises(ValueError):
        season_format(competition, season)


def test_gbl_excluded_message_names_the_reason():
    with pytest.raises(ValueError, match="quarterfinal field"):
        season_format("gbl", 2023)
