"""Transliteration fold / variants / latin_key (entity I2)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from eurohoops.entity.similarity import jaro_winkler
from eurohoops.entity.translit import fold, latin_key, variants

PAIRS = Path(__file__).parent / "fixtures" / "entity" / "translit_pairs.csv"


def _load_pairs() -> list[dict[str, str]]:
    with PAIRS.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _surname_pass(gbl_surname: str, el_name: str) -> tuple[bool, float, str]:
    el_surname = el_name.split(",", 1)[0].strip()
    target = latin_key(el_surname)
    cands = variants(gbl_surname)
    if not cands:
        return False, 0.0, target
    if target in cands:
        return True, 1.0, target
    best = max(jaro_winkler(v, target) for v in cands)
    return best >= 0.95, best, target


def test_fixture_surnames_pass_at_least_38_of_40() -> None:
    rows = _load_pairs()
    assert len(rows) == 40
    scores: list[tuple[str, str, str, float, bool]] = []
    for row in rows:
        ok, best, target = _surname_pass(row["gbl_surname"], row["el_name"])
        scores.append((row["gbl_id"], row["gbl_surname"], target, best, ok))
        assert len(variants(row["gbl_surname"])) <= 64
    passed = sum(1 for *_, ok in scores if ok)
    assert passed >= 38, f"only {passed}/40; misses=" + ", ".join(
        f"{gid}:{sur}->{tgt}@{best:.3f}" for gid, sur, tgt, best, ok in scores if not ok
    )


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


def test_latin_key_drops_generational_suffix() -> None:
    assert latin_key("MACON JR") == "MACON"
    assert latin_key("SMITH SR III") == "SMITH"
    assert latin_key("JR") == "JR"
    assert latin_key("II") == "II"


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


@pytest.mark.parametrize(
    ("gbl", "el"),
    [
        ("ΡΑΪΣ", "RICE"),
        ("ΡAΪΣ", "RICE"),
        ("ΛΙ", "LEE"),
        ("ΜΕΪΚΟΝ", "MACON JR"),
    ],
)
def test_final_sigma_iota_and_jr_suffix(gbl: str, el: str) -> None:
    target = latin_key(el)
    cands = variants(gbl)
    assert target in cands or max(jaro_winkler(v, target) for v in cands) >= 0.95
