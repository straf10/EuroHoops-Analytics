import numpy as np
import pandas as pd
import pytest

from eurohoops.models.player_seasons import (
    COUNT_COLUMNS,
    PLAYER_SEASONS_SCHEMA,
    build_player_seasons,
    person_ids,
    player_ages,
    rate_table,
)

XWALK = pd.DataFrame(
    {
        "person_id": ["P:PA", "P:PA", "P:PB"],
        "competition": ["euroleague", "gbl", "euroleague"],
        "source_id": ["PA", "G1", "PB"],
        "method": ["none", "auto", "none"],
        "score": [None, 0.9, None],
    }
)


def _line(season: int, game: str, team: str, pid: str, sec: int, **stats: int) -> dict[str, object]:
    row: dict[str, object] = {
        "season": season,
        "game_id": game,
        "team": team,
        "player_id": pid,
        "sec": sec,
    }
    row.update({c: stats.get(c, 0) for c in COUNT_COLUMNS})
    row["poss"] = sec / 60.0 * 1.8  # any exposure works for the sums below
    return row


def _games() -> dict[str, pd.DataFrame]:
    el = pd.DataFrame(
        [
            _line(2020, "E1", "BAR", "PA", 600, pts=10, fg2a=6, fg2m=3, fta=4, ftm=4),
            _line(2020, "E2", "MAD", "PA", 1200, pts=4, fg3a=3, fg3m=1, ftm=1, fta=2),
            _line(2020, "E2", "MAD", "PB", 0, pts=0),  # DNP: dropped
            _line(2021, "E3", "BAR", "PX", 300, ast=2),  # not in the crosswalk
        ]
    )
    gbl = pd.DataFrame([_line(2019, "G9", "00000001", "G1", 900, dreb=5)])
    return {"euroleague": el, "gbl": gbl}


def test_sums_teams_debut_and_ids() -> None:
    table = build_player_seasons(_games(), XWALK)
    PLAYER_SEASONS_SCHEMA.validate(table)
    a20 = table[(table["person_id"] == "P:PA") & (table["season"] == 2020)].iloc[0]
    # 600 + 1200 seconds = 30 minutes; MAD has the most seconds; 10 + 4 points; 2 games.
    assert a20["minutes"] == pytest.approx(30.0)
    assert a20["team"] == "MAD"
    assert (a20["pts"], a20["fta"], a20["ftm"], a20["fg3a"], a20["games"]) == (14, 6, 5, 3, 2)
    assert a20["poss"] == pytest.approx(30.0 * 1.8)
    # The GBL 2019 season is the same person's first: debut 2019 on both rows.
    assert set(table.loc[table["person_id"] == "P:PA", "debut_season"]) == {2019}
    assert "P:PB" not in set(table["person_id"])  # no seconds played
    x = table[table["person_id"] == "P:PX"].iloc[0]
    assert not x["mapped"] and x["debut_season"] == 2021
    assert not table["partial"].any()


def test_partial_flag_marks_only_the_cut_season() -> None:
    table = build_player_seasons(_games(), XWALK, partial={"euroleague": 2021})
    assert table.loc[table["partial"], "season"].tolist() == [2021]
    assert table.loc[table["partial"], "competition"].tolist() == ["euroleague"]


def test_unmapped_ids_keep_the_crosswalk_scheme() -> None:
    ids = person_ids(pd.Series(["PA", "Q7"]), "euroleague", XWALK)
    assert ids["person_id"].tolist() == ["P:PA", "P:Q7"]
    assert ids["mapped"].tolist() == [True, False]
    assert person_ids(pd.Series(["Z"]), "gbl", XWALK)["person_id"].tolist() == ["G:Z"]


def test_ages_at_first_october() -> None:
    bios = pd.DataFrame(
        {
            "competition": ["euroleague", "gbl", "euroleague"],
            "source_id": ["PA", "G1", "PB"],
            "birth_date": [
                pd.Timestamp("2000-10-01").date(),
                pd.Timestamp("1990-01-01").date(),
                None,
            ],
            "country": ["GRE", "GRE", None],
        }
    )
    seasons = pd.DataFrame({"person_id": ["P:PA", "P:PA", "P:PB"], "season": [2020, 2021, 2020]})
    ages = player_ages(bios, XWALK, seasons)
    # Born 1 Oct 2000: exactly 20 on 1 Oct 2020 (7,305 days / 365.2425); the EuroLeague date
    # wins over the GBL one; PB has no date and is left out.
    assert ages["person_id"].tolist() == ["P:PA", "P:PA"]
    assert ages["age"].tolist() == pytest.approx([7305 / 365.2425, 7670 / 365.2425])


def test_rate_table_by_hand() -> None:
    table = build_player_seasons(_games(), XWALK)
    rates = rate_table(table)
    a = rates[(rates["person_id"] == "P:PA") & (rates["season"] == 2020)].iloc[0]
    poss = 30.0 * 1.8  # 54 possessions
    assert a["pts"] == pytest.approx(100 * 14 / poss)
    assert a["pts_n"] == pytest.approx(poss)
    # tsa = 6 + 3 + 0.44 * 6 = 11.64; ts = 14 / (2 * 11.64)
    assert a["ts_n"] == pytest.approx(11.64)
    assert a["ts"] == pytest.approx(14 / (2 * 11.64))
    assert (a["fg3"], a["fg3_n"], a["ft"], a["ft_n"]) == pytest.approx((1 / 3, 3, 5 / 6, 6))
    x = rates[rates["person_id"] == "P:PX"].iloc[0]
    assert np.isnan(x["fg3"]) and x["fg3_n"] == 0  # no attempt: no percentage


def test_a_birth_date_outside_the_age_range_is_dropped_for_the_person() -> None:
    bios = pd.DataFrame(
        {
            "competition": ["euroleague", "euroleague"],
            "source_id": ["PA", "PB"],
            # PA: a placeholder date in the season it played (age ~0); PB: plausible.
            "birth_date": [pd.Timestamp("2020-09-01").date(), pd.Timestamp("1995-06-01").date()],
            "country": [None, None],
        }
    )
    seasons = pd.DataFrame({"person_id": ["P:PA", "P:PA", "P:PB"], "season": [2020, 2041, 2020]})
    ages = player_ages(bios, XWALK, seasons)
    # PA is 20 in 2041 but ~0 in 2020: the date is wrong, so PA has no ages at all.
    assert ages["person_id"].tolist() == ["P:PB"]
