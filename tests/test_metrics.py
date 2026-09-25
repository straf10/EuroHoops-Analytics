import numpy as np
import pytest

from eurohoops.eval.metrics import ece, reliability, score

# Four games: bins 1 (two games), 7 and 9.
P = np.array([0.15, 0.18, 0.72, 0.95])
Y = np.array([0.0, 1.0, 1.0, 1.0])


def test_ece_on_a_hand_computed_toy_set() -> None:
    # bin 1: mean P .165 vs observed .5 (x2 games); bin 7: .72 vs 1; bin 9: .95 vs 1.
    expected = (2 * abs(0.165 - 0.5) + abs(0.72 - 1.0) + abs(0.95 - 1.0)) / 4
    assert expected == pytest.approx(0.25)
    assert ece(P, Y) == pytest.approx(expected, abs=1e-12)


def test_reliability_bins() -> None:
    bins = reliability(P, Y)
    assert len(bins) == 10
    assert [b["n"] for b in bins] == [0, 2, 0, 0, 0, 0, 0, 1, 0, 1]
    assert bins[1] == {"low": 0.1, "high": 0.2, "n": 2, "mean_p": 0.165, "observed": 0.5}
    assert bins[0]["mean_p"] is None
    assert bins[0]["observed"] is None


def test_bin_edges_are_left_closed_and_one_falls_in_the_last_bin() -> None:
    edges = np.array([0.0, 0.1, 0.5, 0.9999, 1.0])
    counts = [b["n"] for b in reliability(edges, np.ones(5))]
    assert counts == [1, 1, 0, 0, 0, 1, 0, 0, 0, 2]


def test_a_perfectly_calibrated_bin_has_zero_ece() -> None:
    p = np.full(4, 0.75)
    assert ece(p, np.array([1.0, 1.0, 1.0, 0.0])) == pytest.approx(0.0)


def test_score_carries_calibration() -> None:
    m = score(P, np.zeros(4), np.where(Y == 1.0, 5.0, -5.0))
    assert m.ece == pytest.approx(0.25)
    assert m.as_dict()["reliability"][7]["observed"] == 1.0


def test_empty_score_keeps_the_bins() -> None:
    empty = score(np.empty(0), np.empty(0), np.empty(0)).as_dict()
    assert empty["ece"] is None
    assert [b["n"] for b in empty["reliability"]] == [0] * 10
