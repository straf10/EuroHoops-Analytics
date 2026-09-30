"""Transliteration fold / variants / latin_key (entity I2)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eurohoops.entity.similarity import jaro_winkler
from eurohoops.entity.translit import fold, latin_key, variants

PAIRS = Path(__file__).parent / "fixtures" / "entity" / "translit_pairs.csv"


def _load_pairs() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    with PAIRS.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            el_surname = row["el_name"].split(",", 1)[0].strip()
            rows.append((row["gbl_id"], row["gbl_surname"], el_surname, row["el_name"]))
    return rows


@pytest.mark.parametrize(("gbl_id", "gbl_surname", "el_surname", "el_name"), _load_pairs())
def test_fixture_surname_matches(
    gbl_id: str, gbl_surname: str, el_surname: str, el_name: str
) -> None:
    del el_name  # evidence only
    target = latin_key(el_surname)
    cands = variants(gbl_surname)
    assert cands, f"{gbl_id}: no variants for {gbl_surname!r}"
    assert len(cands) <= 64
    if target in cands:
        return
    best = max(jaro_winkler(v, target) for v in cands)
    assert best >= 0.95, f"{gbl_id}: best={best:.4f} target={target!r} sample={cands[:8]}"


def test_variant_cap_on_long_name() -> None:
    name = "ΠAΠAΘAΝAΣΙΟΥ ΤΖΟΡΤΖ ΠΙΤΕΡΣ"
    out = variants(name)
    assert len(out) <= 64
    assert len(out) == len(set(out))


def test_fold_mixed_token_tyson() -> None:
    assert fold("ΤYSON") == "TYSON"
    assert fold("ΚΩΝΣΤAΝΤΙΝΟΣ") == "ΚΩΝΣΤΑΝΤΙΝΟΣ"


def test_latin_key_sorts_hyphenated() -> None:
    assert latin_key("SANT-ROOS, HOWARD") == "HOWARD ROOS SANT"
    assert latin_key("SANT-ROOS") == "ROOS SANT"


@pytest.mark.parametrize(
    ("raw", "expected_fold"),
    [
        ("μητόγλου", "ΜΗΤΟΓΛΟΥ"),
        ("ΜΗΤΌΓΛΟΥ", "ΜΗΤΟΓΛΟΥ"),
        ("ΚΩΝΣΤAΝΤΙΝΟΣ", "ΚΩΝΣΤΑΝΤΙΝΟΣ"),
        ("κωνσταντινος", "ΚΩΝΣΤΑΝΤΙΝΟΣ"),
        ("WARD", "WARD"),
        ("ΣAΝΤ-ΡΟΣ", "ΣΑΝΤ ΡΟΣ"),
    ],
)
def test_fold_case_accent_lookalike(raw: str, expected_fold: str) -> None:
    assert fold(raw) == expected_fold


def test_variants_independent_of_case_accent_lookalike() -> None:
    a = variants("ΚΩΝΣΤAΝΤΙΝΟΣ")
    b = variants("κωνσταντινος")
    c = variants("ΚΩΝΣΤΑΝΤΙΝΟΣ")
    assert a == b == c


def test_variants_deterministic() -> None:
    name = "ΓΚΕΪΜΠΡΙΕΛ"
    assert variants(name) == variants(name)
    assert variants(name) == variants(name.lower())


def test_latin_input_is_single_latin_key() -> None:
    assert variants("WARD") == ("WARD",)
    assert variants("SKORDILIS GAIOS") == ("GAIOS SKORDILIS",)


@pytest.mark.parametrize(
    ("gbl", "el", "min_jw"),
    [
        ("ΓΟΥΙΛΙΑΜΣ", "WILLIAMS", 0.95),
        ("ΓΟΥΙΛΙAΜΣ", "WILLIAMS", 0.95),  # Latin A look-alike inside
        ("ΟΥΑΪΤ", "WHITE", 0.90),
        ("ΟΥAΪΤ", "WHITE", 0.90),
        ("ΓΚΑΪ", "GUY", 0.85),
    ],
)
def test_gou_w_and_ai_diaeresis(gbl: str, el: str, min_jw: float) -> None:
    target = latin_key(el)
    cands = variants(gbl)
    assert cands and len(cands) <= 64
    best = max(jaro_winkler(v, target) for v in cands)
    assert best >= min_jw, f"{gbl!r}→{target!r} best={best:.4f} sample={cands[:12]}"
