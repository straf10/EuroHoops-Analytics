"""Synthetic tests for ``entity.xwalk``."""

from __future__ import annotations

import difflib
from datetime import date

import pandas as pd
import pytest

from eurohoops.entity.match import MatchParams, assign, candidate_pairs, score_pairs
from eurohoops.entity.xwalk import (
    XWALK_SCHEMA,
    build_xwalk,
    entity_report,
    pair_metrics,
    wilson,
)

CLUBS = {"00000001": "PAN", "00000002": "OLY"}

_VARIANT_MAP: dict[str, tuple[str, ...]] = {
    "ΠΑΠΑΔΟΠΟΥΛΟΣ": ("PAPADOPOULOS",),
    "ΝΙΚΟΣ": ("NIKOS",),
    "ΠΑΠΑΔΟΠΟΥΛΟΣ ΝΙΚΟΣ": ("NIKOS PAPADOPOULOS",),
}


def latin_key(s: str) -> str:
    return " ".join(sorted(part for part in s.upper().split() if part))


def variants(s: str) -> tuple[str, ...]:
    if s in _VARIANT_MAP:
        return _VARIANT_MAP[s]
    return (latin_key(s),)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


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


def test_match_override_links_rejected_pair_and_two_el_ids() -> None:
    names = pd.DataFrame(
        [
            _row("gbl", "0000X001", 2022, "00000001", "WALKER", "KEMBA"),
            _row("euroleague", "P00X001", 2022, "PAN", "WALKER", "KEMBA"),
            _row("euroleague", "P00X002", 2014, "MAD", "YURTSEVEN", "OMER"),
            _row("euroleague", "P00X003", 2015, "MAD", "YURTSEVEN", "OMER"),
            _row("gbl", "0000X099", 2019, "00000099", "SOLO", "PLAYER"),
        ]
    )
    bios = pd.DataFrame(
        [
            {
                "competition": "gbl",
                "source_id": "0000X001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P00X001",
                "birth_date": date(1993, 6, 15),
                "country": None,
            },
        ]
    )
    params = MatchParams()
    pairs = candidate_pairs(names, bios, CLUBS, variants, latin_key, params)
    scored = score_pairs(pairs, names, variants, latin_key, similarity, params)
    matched = assign(scored, pd.DataFrame())
    assert matched.empty  # different dob blocks the rule match
    overrides = pd.DataFrame(
        [
            {
                "competition_a": "gbl",
                "source_a": "0000X001",
                "competition_b": "euroleague",
                "source_b": "P00X001",
                "decision": "match",
                "reason": "known dual",
                "date": "2026-01-01",
            },
            {
                "competition_a": "euroleague",
                "source_a": "P00X002",
                "competition_b": "euroleague",
                "source_b": "P00X003",
                "decision": "match",
                "reason": "same person two ids",
                "date": "2026-01-01",
            },
        ]
    )
    xwalk = build_xwalk(names, matched, overrides, pd.DataFrame())
    XWALK_SCHEMA.validate(xwalk)
    by_id = xwalk.set_index(["competition", "source_id"])
    assert (
        by_id.loc[("gbl", "0000X001"), "person_id"]
        == by_id.loc[("euroleague", "P00X001"), "person_id"]
    )
    assert (
        by_id.loc[("euroleague", "P00X002"), "person_id"]
        == by_id.loc[("euroleague", "P00X003"), "person_id"]
    )
    assert by_id.loc[("gbl", "0000X001"), "method"] == "override"
    assert by_id.loc[("euroleague", "P00X002"), "method"] == "override"
    assert by_id.loc[("gbl", "0000X099"), "method"] == "none"
    assert by_id.loc[("gbl", "0000X099"), "person_id"] == "G:0000X099"


def test_links_attach_pbp_keys() -> None:
    names = pd.DataFrame(
        [
            _row("gbl", "0000P001", 2018, "00000001", "SLOUKAS", "KOSTAS"),
            _row("gbl", "pbp:10:Kostas Sloukas", 2018, "00000001", "Sloukas", "Kostas"),
            _row("euroleague", "P00P001", 2018, "PAN", "SLOUKAS", "KOSTAS"),
        ]
    )
    bios = pd.DataFrame(
        [
            {
                "competition": "gbl",
                "source_id": "0000P001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
            {
                "competition": "euroleague",
                "source_id": "P00P001",
                "birth_date": date(1990, 1, 1),
                "country": None,
            },
        ]
    )
    params = MatchParams()
    pairs = candidate_pairs(names, bios, CLUBS, variants, latin_key, params)
    scored = score_pairs(pairs, names, variants, latin_key, similarity, params)
    matched = assign(scored, pd.DataFrame())
    links = pd.DataFrame(
        [
            {
                "competition": "gbl",
                "source_id": "pbp:10:Kostas Sloukas",
                "linked_source_id": "0000P001",
                "method": "pbp",
            }
        ]
    )
    xwalk = build_xwalk(names, matched, pd.DataFrame(), links)
    by_id = xwalk.set_index(["competition", "source_id"])
    assert (
        by_id.loc[("gbl", "pbp:10:Kostas Sloukas"), "person_id"]
        == by_id.loc[("gbl", "0000P001"), "person_id"]
    )
    assert by_id.loc[("gbl", "pbp:10:Kostas Sloukas"), "method"] == "pbp"
    assert by_id.loc[("gbl", "0000P001"), "method"] == "pbp"  # also on a link edge
    assert by_id.loc[("euroleague", "P00P001"), "method"] == "rule"


def test_no_source_id_maps_to_two_persons() -> None:
    names = pd.DataFrame(
        [
            _row("gbl", "0000G001", 2022, "00000001", "ΠΑΠΑΔΟΠΟΥΛΟΣ", "ΝΙΚΟΣ"),
            _row("euroleague", "P000001", 2022, "PAN", "PAPADOPOULOS", "NIKOS"),
            _row("gbl", "0000L001", 2023, "00000002", "WALKER", "KEMBA"),
            _row("euroleague", "P000002", 2023, "OLY", "WALKER", "KEMBA"),
        ]
    )
    bios = pd.DataFrame(
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
    params = MatchParams()
    pairs = candidate_pairs(names, bios, CLUBS, variants, latin_key, params)
    scored = score_pairs(pairs, names, variants, latin_key, similarity, params)
    matched = assign(scored, pd.DataFrame())
    xwalk = build_xwalk(names, matched, pd.DataFrame(), pd.DataFrame())
    assert xwalk.duplicated(["competition", "source_id"]).sum() == 0
    # No person holds two ids of one competition without override/link.
    for _, group in xwalk.groupby("person_id"):
        counts = group.groupby("competition")["source_id"].nunique()
        assert (counts <= 1).all()


def test_build_xwalk_raises_on_same_competition_without_soft_edge() -> None:
    names = pd.DataFrame(
        [
            _row("gbl", "0000A", 2022, "00000001", "A", "A"),
            _row("gbl", "0000B", 2022, "00000001", "B", "B"),
            _row("euroleague", "P00A", 2022, "PAN", "A", "A"),
        ]
    )
    # Fake matches that would put two GBL ids on one EL id (bypassing assign).
    matches = pd.DataFrame(
        [
            {"gbl_id": "0000A", "el_id": "P00A", "score": 1.5},
            {"gbl_id": "0000B", "el_id": "P00A", "score": 1.4},
        ]
    )
    with pytest.raises(ValueError, match="multiple gbl"):
        build_xwalk(names, matches, pd.DataFrame(), pd.DataFrame())


def test_wilson_and_pair_metrics_hand_values() -> None:
    lo, hi = wilson(18, 20)
    assert 0.68 < lo < 0.75
    assert 0.95 < hi < 0.99
    assert wilson(0, 0) == (0.0, 1.0)

    metrics = pair_metrics(
        predicted={("g1", "e1"), ("g2", "e2"), ("g9", "e9")},
        truth_pos={("g1", "e1"), ("g3", "e3")},
        truth_neg={("g2", "e2"), ("g4", "e4")},
    )
    assert metrics == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_entity_report_and_determinism() -> None:
    names = pd.DataFrame(
        [
            _row("gbl", "0000G001", 2022, "00000001", "ΠΑΠΑΔΟΠΟΥΛΟΣ", "ΝΙΚΟΣ"),
            _row("euroleague", "P000001", 2022, "PAN", "PAPADOPOULOS", "NIKOS"),
            _row("gbl", "0000Z001", 2020, "00000099", "ALONE", "GUY"),
        ]
    )
    bios = pd.DataFrame(
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
    params = MatchParams()
    pairs = candidate_pairs(names, bios, CLUBS, variants, latin_key, params)
    scored = score_pairs(pairs, names, variants, latin_key, similarity, params)
    matched = assign(scored, pd.DataFrame())
    xwalk_a = build_xwalk(names, matched, pd.DataFrame(), pd.DataFrame())
    xwalk_b = build_xwalk(names, matched, pd.DataFrame(), pd.DataFrame())
    assert xwalk_a.to_csv(index=False) == xwalk_b.to_csv(index=False)
    report = entity_report(names, scored, xwalk_a, params)
    assert report["persons"] == 2
    assert report["matched_gbl_ids"] == 1
    assert report["by_method"]["rule"] == 2
    assert report["by_method"]["none"] == 1
    assert report["params"]["t_dob"] == params.t_dob
