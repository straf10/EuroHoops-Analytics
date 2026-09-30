"""Synthetic tests for ``entity.match`` (D3)."""

from __future__ import annotations

import difflib
from datetime import date

import pandas as pd
import pytest

from eurohoops.entity.match import (
    MatchParams,
    assign,
    candidate_pairs,
    careers,
    dob_status,
    score_pairs,
)

CLUBS = {"00000001": "PAN", "00000002": "OLY"}

# Greek surname/first → Latin variants (tokens already sorted where multi-token).
_VARIANT_MAP: dict[str, tuple[str, ...]] = {
    "ΠΑΠΑΔΟΠΟΥΛΟΣ": ("PAPADOPOULOS",),
    "ΝΙΚΟΣ": ("NIKOS",),
    "ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΣ": ("NIKOS PAPADOPOULOS",),
    "ΣΛΟΥΚΑΣ": ("SLOUKAS",),
    "ΚΩΣΤΑΣ": ("KOSTAS",),
    "ΣΛΟΥΚΑΣ ΚΩΣΤΑΣ": ("KOSTAS SLOUKAS",),
}


def latin_key(s: str) -> str:
    return " ".join(sorted(part for part in s.upper().split() if part))


def variants(s: str) -> tuple[str, ...]:
    if s in _VARIANT_MAP:
        return _VARIANT_MAP[s]
    return (latin_key(s),)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _names(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _bios(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _row(  # noqa: PLR0917 -- compact fixture row builder
    competition: str,
    source_id: str,
    season: int,
    team: str,
    surname: str,
    first: str,
    *,
    jersey: str | None = None,
    games: int = 10,
    minutes: float = 200.0,
) -> dict[str, object]:
    return {
        "competition": competition,
        "source_id": source_id,
        "season": season,
        "team": team,
        "jersey": jersey,
        "surname_raw": surname,
        "first_raw": first,
        "games": games,
        "minutes": minutes,
    }


def _run(
    names: pd.DataFrame,
    bios: pd.DataFrame,
    params: MatchParams | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = params or MatchParams()
    pairs = candidate_pairs(names, bios, CLUBS, variants, latin_key, params)
    scored = score_pairs(pairs, names, variants, latin_key, similarity, params)
    matched = assign(scored, pd.DataFrame())
    return pairs, scored, matched


def test_careers_picks_most_frequent_spelling() -> None:
    names = _names(
        [
            _row("gbl", "0000AAAA", 2020, "00000001", "ALPHA", "A", games=5),
            _row("gbl", "0000AAAA", 2021, "00000001", "BETA", "B", games=20),
            _row("gbl", "0000AAAA", 2022, "00000001", "ALPHA", "A", games=5),
        ]
    )
    card = careers(names)
    assert len(card) == 1
    assert card.iloc[0]["surname"] == "BETA"
    assert card.iloc[0]["first"] == "B"
    assert card.iloc[0]["first_season"] == 2020
    assert card.iloc[0]["last_season"] == 2022


def test_planted_greek_and_latin_pairs_match() -> None:
    names = _names(
        [
            _row("gbl", "0000G001", 2022, "00000001", "ΠΑΠΑΔΟΠΟΥΛΟΣ", "ΝΙΚΟΣ", jersey="7"),
            _row("euroleague", "P000001", 2022, "PAN", "PAPADOPOULOS", "NIKOS", jersey="7"),
            _row("gbl", "0000L001", 2023, "00000002", "WALKER", "KEMBA", jersey="15"),
            _row("euroleague", "P000002", 2023, "OLY", "WALKER", "KEMBA", jersey="15"),
        ]
    )
    bios = _bios(
        [
            {
                "competition": "gbl",
                "source_id": "0000G001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P000001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
            {
                "competition": "gbl",
                "source_id": "0000L001",
                "birth_date": date(1992, 5, 5),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P000002",
                "birth_date": date(1992, 5, 5),
                "country": None,
            },
        ]
    )
    _, scored, matched = _run(names, bios)
    assert set(zip(matched["gbl_id"], matched["el_id"], strict=True)) == {
        ("0000G001", "P000001"),
        ("0000L001", "P000002"),
    }
    assert scored.set_index(["gbl_id", "el_id"]).loc[("0000G001", "P000001"), "accepted"]


def test_same_name_different_people_not_merged() -> None:
    # Overlapping careers, identical names, different birth dates → cross pairs rejected.
    names = _names(
        [
            _row("gbl", "0000A001", 2020, "00000001", "SMITH", "JOHN", jersey="1"),
            _row("gbl", "0000A002", 2021, "00000002", "SMITH", "JOHN", jersey="99"),
            _row("euroleague", "P00A001", 2020, "PAN", "SMITH", "JOHN", jersey="1"),
            _row("euroleague", "P00A002", 2021, "OLY", "SMITH", "JOHN", jersey="99"),
        ]
    )
    bios = _bios(
        [
            {
                "competition": "gbl",
                "source_id": "0000A001",
                "birth_date": date(1985, 1, 1),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P00A001",
                "birth_date": date(1985, 1, 1),
                "country": None,
            },
            {
                "competition": "gbl",
                "source_id": "0000A002",
                "birth_date": date(2000, 6, 6),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P00A002",
                "birth_date": date(2000, 6, 6),
                "country": None,
            },
        ]
    )
    _, scored, matched = _run(names, bios)
    cross = scored[
        ((scored["gbl_id"] == "0000A001") & (scored["el_id"] == "P00A002"))
        | ((scored["gbl_id"] == "0000A002") & (scored["el_id"] == "P00A001"))
    ]
    assert not cross.empty
    assert not cross["accepted"].any()
    assert set(zip(matched["gbl_id"], matched["el_id"], strict=True)) == {
        ("0000A001", "P00A001"),
        ("0000A002", "P00A002"),
    }


def test_swapped_name_order_matches_via_full_sim() -> None:
    names = _names(
        [
            _row("gbl", "0000S001", 2021, "00000001", "ΠΑΠΑΔΟΠΟΥΛΟΣ", "ΝΙΚΟΣ"),
            # EL stores given name as surname field (swapped order); same club for blocking.
            _row("euroleague", "P00S001", 2021, "PAN", "NIKOS", "PAPADOPOULOS"),
        ]
    )
    bios = _bios([])
    params = MatchParams(t_nodob=0.90, b_club=0.0, b_jersey=0.0)
    _, scored, matched = _run(names, bios, params)
    row = scored.set_index(["gbl_id", "el_id"]).loc[("0000S001", "P00S001")]
    assert row["full_sim"] >= 0.99
    assert row["accepted"]
    assert list(zip(matched["gbl_id"], matched["el_id"], strict=True)) == [("0000S001", "P00S001")]


@pytest.mark.parametrize(
    ("a", "b", "status"),
    [
        (date(1990, 1, 1), date(1990, 1, 1), "equal"),
        (date(1986, 10, 24), date(1986, 10, 26), "near"),  # Gist: ESAKE and EL 2 days apart
        (date(2000, 10, 19), date(2000, 11, 19), "near"),  # Balcerowski: a month off
        (date(1990, 3, 4), date(1990, 4, 3), "near"),  # day and month swapped
        (date(1990, 3, 4), date(1991, 3, 4), "near"),  # year off by one
        (date(1990, 1, 1), date(1990, 3, 1), "different"),
        (None, date(1990, 1, 1), "unknown"),
    ],
)
def test_dob_status(a: date | None, b: date | None, status: str) -> None:
    assert dob_status(a, b, MatchParams().near_days) == status


def test_near_dates_need_the_stricter_name_threshold() -> None:
    names = _names(
        [
            _row("gbl", "0000N001", 2022, "00000001", "WALKER", "KEMBA"),
            _row("euroleague", "P00N001", 2022, "PAN", "WALKER", "KEMBA"),
            _row("gbl", "0000N002", 2022, "00000001", "WALKER", "KEMBO"),
            _row("euroleague", "P00N002", 2022, "PAN", "WALKER", "KEMBA"),
        ]
    )
    bios = _bios(
        [
            {"competition": c, "source_id": s, "birth_date": d, "country": None}
            for c, s, d in [
                ("gbl", "0000N001", date(1990, 1, 1)),
                ("euroleague", "P00N001", date(1990, 1, 4)),
                ("gbl", "0000N002", date(1980, 5, 5)),
                ("euroleague", "P00N002", date(1980, 5, 9)),
            ]
        ]
    )
    _, scored, matched = _run(names, bios)
    near = scored[scored["dob"] == "near"].set_index(["gbl_id", "el_id"])
    assert near.loc[("0000N001", "P00N001"), "accepted"]
    assert not near.loc[("0000N002", "P00N002"), "accepted"]
    assert near.loc[("0000N002", "P00N002"), "name_score"] >= MatchParams().t_dob
    assert list(zip(matched["gbl_id"], matched["el_id"], strict=True)) == [("0000N001", "P00N001")]


def test_dob_different_never_matches() -> None:
    names = _names(
        [
            _row("gbl", "0000D001", 2022, "00000001", "WALKER", "KEMBA", jersey="15"),
            _row("euroleague", "P00D001", 2022, "PAN", "WALKER", "KEMBA", jersey="15"),
        ]
    )
    bios = _bios(
        [
            {
                "competition": "gbl",
                "source_id": "0000D001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P00D001",
                "birth_date": date(1993, 6, 15),
                "country": None,
            },
        ]
    )
    pairs, scored, matched = _run(names, bios)
    assert pd.isna(scored.iloc[0]["name_score"])  # skipped: it can never be accepted
    assert not scored.iloc[0]["accepted"]
    full = score_pairs(
        pairs, names, variants, latin_key, similarity, MatchParams(), skip_different=False
    )
    assert full.iloc[0]["name_score"] >= 0.99
    assert not full.iloc[0]["accepted"]
    assert matched.empty


def test_dob_unknown_uses_stricter_threshold_and_bonuses() -> None:
    # Near-miss surname so bare name_score sits under t_nodob; club+jersey bonuses tip it.
    names = _names(
        [
            _row("gbl", "0000U001", 2022, "00000001", "SLOUKAS", "KOSTAS", jersey="10"),
            _row("euroleague", "P00U001", 2022, "PAN", "SLOUKASX", "KOSTAS", jersey="10"),
            # Same names, no club/jersey overlap → score stays under t_nodob.
            _row("gbl", "0000U002", 2022, "00000099", "SLOUKAS", "KOSTAS", jersey="1"),
            _row("euroleague", "P00U002", 2022, "MAD", "SLOUKASX", "KOSTAS", jersey="2"),
        ]
    )
    bios = _bios([])
    params = MatchParams(t_nodob=0.98, b_club=0.05, b_jersey=0.03, t_dob=0.70)
    pairs, scored, matched = _run(names, bios, params)
    s = scored.set_index(["gbl_id", "el_id"])
    u1 = s.loc[("0000U001", "P00U001")]
    assert u1["same_club_season"] and u1["same_jersey"]
    assert u1["name_score"] < params.t_nodob
    assert u1["score"] >= params.t_nodob
    assert u1["accepted"]
    u2 = s.loc[("0000U002", "P00U002")]
    assert not u2["same_club_season"] and not u2["same_jersey"]
    assert u2["name_score"] < params.t_nodob
    assert not u2["accepted"]
    assert ("0000U001", "P00U001") in set(zip(matched["gbl_id"], matched["el_id"], strict=True))
    assert pairs is not None


def test_no_match_override_removes_would_be_match() -> None:
    names = _names(
        [
            _row("gbl", "0000O001", 2022, "00000001", "WALKER", "KEMBA"),
            _row("euroleague", "P00O001", 2022, "PAN", "WALKER", "KEMBA"),
        ]
    )
    bios = _bios(
        [
            {
                "competition": "gbl",
                "source_id": "0000O001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P00O001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
        ]
    )
    _, scored, _ = _run(names, bios)
    assert scored.iloc[0]["accepted"]
    overrides = pd.DataFrame(
        [
            {
                "competition_a": "gbl",
                "source_a": "0000O001",
                "competition_b": "euroleague",
                "source_b": "P00O001",
                "decision": "no_match",
                "reason": "test",
                "date": "2026-01-01",
            }
        ]
    )
    assert assign(scored, overrides).empty


def test_one_to_one_higher_score_wins() -> None:
    names = _names(
        [
            _row("gbl", "0000T001", 2022, "00000001", "WALKER", "KEMBA", jersey="15"),
            _row("gbl", "0000T002", 2022, "00000001", "WALKER", "K", jersey="15"),
            _row("euroleague", "P00T001", 2022, "PAN", "WALKER", "KEMBA", jersey="15"),
        ]
    )
    bios = _bios(
        [
            {
                "competition": "gbl",
                "source_id": "0000T001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
            {
                "competition": "gbl",
                "source_id": "0000T002",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P00T001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
        ]
    )
    _, scored, matched = _run(names, bios)
    accepted = scored[scored["accepted"]]
    assert len(accepted) == 2
    assert list(zip(matched["gbl_id"], matched["el_id"], strict=True)) == [("0000T001", "P00T001")]
    assert matched.iloc[0]["score"] == accepted.set_index("gbl_id").loc["0000T001", "score"]


def test_initial_first_name_scores_one() -> None:
    names = _names(
        [
            _row("gbl", "0000I001", 2022, "00000001", "WALKER", "K"),
            _row("euroleague", "P00I001", 2022, "PAN", "WALKER", "KEMBA"),
        ]
    )
    bios = _bios(
        [
            {
                "competition": "gbl",
                "source_id": "0000I001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P00I001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
        ]
    )
    _, scored, _ = _run(names, bios)
    assert scored.iloc[0]["first_sim"] == 1.0


def test_determinism_two_runs_identical() -> None:
    names = _names(
        [
            _row("gbl", "0000G001", 2022, "00000001", "ΠΑΠΑΔΟΠΟΥΛΟΣ", "ΝΙΚΟΣ"),
            _row("euroleague", "P000001", 2022, "PAN", "PAPADOPOULOS", "NIKOS"),
            _row("gbl", "0000L001", 2023, "00000002", "WALKER", "KEMBA"),
            _row("euroleague", "P000002", 2023, "OLY", "WALKER", "KEMBA"),
        ]
    )
    bios = _bios(
        [
            {
                "competition": "gbl",
                "source_id": "0000G001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P000001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
        ]
    )
    a = _run(names, bios)
    b = _run(names, bios)
    for left, right in zip(a, b, strict=True):
        assert left.to_csv(index=False) == right.to_csv(index=False)
