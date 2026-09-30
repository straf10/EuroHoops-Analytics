"""Jaro-Winkler similarity (entity I2)."""

from eurohoops.entity.similarity import jaro_winkler


def test_published_values() -> None:
    assert abs(jaro_winkler("MARTHA", "MARHTA") - 0.961) < 1e-3
    assert abs(jaro_winkler("DWAYNE", "DUANE") - 0.840) < 1e-3
    assert abs(jaro_winkler("DIXON", "DICKSONX") - 0.813) < 1e-3


def test_empty_and_identity() -> None:
    assert jaro_winkler("", "") == 1.0
    assert jaro_winkler("", "A") == 0.0
    assert jaro_winkler("A", "") == 0.0
    assert jaro_winkler("MITOGLOU", "MITOGLOU") == 1.0


def test_symmetry_and_bounds() -> None:
    pairs = [
        ("MARTHA", "MARHTA"),
        ("DWAYNE", "DUANE"),
        ("DIXON", "DICKSONX"),
        ("BROWN", "BRAWN"),
        ("A", "B"),
        ("ABC", "ABC"),
    ]
    for a, b in pairs:
        score = jaro_winkler(a, b)
        assert 0.0 <= score <= 1.0
        assert abs(score - jaro_winkler(b, a)) < 1e-12
