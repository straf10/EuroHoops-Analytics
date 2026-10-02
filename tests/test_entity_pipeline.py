from datetime import date
from pathlib import Path

import pandas as pd

from eurohoops.entity.match import MatchParams
from eurohoops.entity.pipeline import (
    OVERRIDE_COLUMNS,
    draft_labels,
    label_metrics,
    read_overrides,
    run_entity,
    same_name,
    silver_metrics,
    silver_set,
    tune,
)

CLUBS = {"00000001": "PAN"}


def row(  # noqa: PLR0917 -- compact name-row builder
    competition: str,
    source_id: str,
    season: int,
    team: str,
    surname: str,
    first: str,
    jersey: str | None = None,
    minutes: float = 400.0,
) -> dict[str, object]:
    return {
        "competition": competition,
        "source_id": source_id,
        "season": season,
        "team": team,
        "jersey": jersey,
        "surname_raw": surname,
        "first_raw": first,
        "games": 20,
        "minutes": minutes,
    }


NAMES = pd.DataFrame(
    [
        row("gbl", "0000000A", 2023, "00000001", "ΣΛΟΥΚAΣ", "ΚΩΣΤAΣ", "11"),
        row("gbl", "0000000B", 2023, "00000001", "ΠAΠAΠΕΤΡΟΥ", "ΙΩAΝΝΗΣ", "4"),
        row("gbl", "0000000C", 2023, "00000009", "ΑΛΛΟΣ", "ΚΑΠΟΙΟΣ"),
        row("gbl", "pbp:4:Ioannis Papapetrou", 2023, "00000001", "PAPAPETROU", "IOANNIS", "4"),
        row("euroleague", "P001926", 2023, "PAN", "SLOUKAS", "KOSTAS", "11"),
        row("euroleague", "P005161", 2023, "PAN", "PAPAPETROU", "IOANNIS", "4"),
        row("euroleague", "P009999", 2023, "MAD", "LLULL", "SERGIO", "23"),
    ]
)
BIOS = pd.DataFrame(
    [
        ("gbl", "0000000A", date(1990, 1, 1), None),
        ("gbl", "0000000B", date(1994, 5, 5), None),
        ("euroleague", "P001926", date(1990, 1, 1), None),
        ("euroleague", "P005161", date(1994, 5, 5), None),
        ("euroleague", "P009999", date(1987, 11, 15), None),
    ],
    columns=["competition", "source_id", "birth_date", "country"],
)
NO_OVERRIDES = pd.DataFrame(columns=OVERRIDE_COLUMNS, dtype=str)


def test_silver_set_uses_club_season_and_equal_birth_dates_only() -> None:
    pos, neg = silver_set(NAMES, BIOS, CLUBS)
    assert pos == {("0000000A", "P001926"), ("0000000B", "P005161")}
    assert neg == {("0000000A", "P005161"), ("0000000B", "P001926")}


def test_run_links_greek_to_latin_and_the_pbp_key() -> None:
    run = run_entity(NAMES, BIOS, NO_OVERRIDES, CLUBS)
    person = run.xwalk.set_index("source_id")["person_id"]
    assert person["0000000A"] == person["P001926"] == "P:P001926"
    assert person["0000000B"] == person["pbp:4:Ioannis Papapetrou"] == "P:P005161"
    assert person["0000000C"] == "G:0000000C"
    assert silver_metrics(run.matches, *silver_set(NAMES, BIOS, CLUBS))["precision"] == 1.0
    assert run.report["pbp_links"] == {"keys": 1, "linked": 1, "unlinked": {}}


def test_same_name_across_scripts() -> None:
    assert same_name("PAPAPETROU IOANNIS", "ΠAΠAΠΕΤΡΟΥ ΙΩAΝΝΗΣ")
    assert not same_name("PAPAPETROU IOANNIS", "ΣΛΟΥΚAΣ ΚΩΣΤAΣ")


def test_label_sheet_and_metrics_as_frozen_and_after_overrides() -> None:
    sheet = draft_labels(NAMES, BIOS, NO_OVERRIDES, CLUBS)
    assert sorted(sheet["gbl_id"]) == ["0000000A", "0000000B", "0000000C"]
    assert set(sheet["stratum"]) == {"greek/matched", "greek/unmatched"}
    assert sheet.loc[sheet["gbl_id"] == "0000000A", "cand1_el_id"].item() == "P001926"
    labels = sheet[["gbl_id", "stratum", "stratum_size"]].assign(
        el_id_true=["P001926", "P005161", "P009999"]
    )
    overrides = pd.DataFrame(
        [["gbl", "0000000C", "euroleague", "P009999", "match", "label: owner", "2026-10-01"]],
        columns=OVERRIDE_COLUMNS,
    )
    run = run_entity(NAMES, BIOS, overrides, CLUBS)
    metrics = label_metrics(run, overrides, labels.astype(str))
    frozen, after = metrics["as_frozen"], metrics["after_overrides"]
    assert (frozen["precision"], frozen["recall"]) == (1.0, round(2 / 3, 6))
    assert (after["precision"], after["recall"]) == (1.0, 1.0)
    assert frozen["strata"]["greek/unmatched"]["correct"] == 0
    # the M4 input crosswalk ignores label overrides: C stays its own person there
    persons = lambda x: x.set_index(["competition", "source_id"])["person_id"]  # noqa: E731
    assert persons(run.xwalk)[("gbl", "0000000C")] == persons(run.xwalk)[("euroleague", "P009999")]
    assert (
        persons(run.xwalk_frozen)[("gbl", "0000000C")]
        != persons(run.xwalk_frozen)[("euroleague", "P009999")]
    )


def test_read_overrides_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_overrides(tmp_path / "none.csv").empty


def test_tune_picks_a_grid_point_and_keeps_the_birth_date_thresholds() -> None:
    grid = {"w_surname": (0.6,), "t_nodob": (0.9, 0.95), "b_club": (0.0, 0.05), "b_jersey": (0.0,)}
    result = tune(NAMES, BIOS, CLUBS, grid)
    assert result["silver"] == {"positives": 2, "negatives": 2}
    assert result["chosen_silver_hidden_dates"]["precision"] == 1.0
    assert result["chosen_silver_visible_dates"]["precision"] == 1.0
    chosen = MatchParams(**result["chosen"])
    assert (chosen.t_dob, chosen.t_near) == (MatchParams().t_dob, MatchParams().t_near)
    assert len(result["table"]) == 4


def test_near_birth_dates_are_silver_positives() -> None:
    bios = BIOS.copy()
    bios.loc[bios["source_id"] == "P001926", "birth_date"] = date(1990, 1, 3)  # 2 days off
    pos, _ = silver_set(NAMES, bios, CLUBS)
    assert ("0000000A", "P001926") in pos
