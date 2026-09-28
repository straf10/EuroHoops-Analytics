import numpy as np
import pandas as pd
import pytest

from eurohoops.models import box_impact as bi
from eurohoops.models.box_impact import STAT_COLUMNS, BoxGrid, box_only_margins, pir_margins

# ---------------------------------------------------------------------------------------------
# Hand example: the decayed, shrunk per-100 rate at a cutoff (H2 "Done when" #2).
# ---------------------------------------------------------------------------------------------


def test_decayed_shrunk_rate_hand_example() -> None:
    """Two players, one stat, half-life 10 days, k = 5 minutes -- worked by hand."""
    acc = bi._Accumulator(n_players=2, n_stats=1, half_life_days=10.0)
    acc.advance(0.0)
    # player 0: 10 of the stat over 20 possessions and 25 minutes, at t = 0.
    acc.add(
        np.array([0]),
        np.array([[10.0]]),
        np.array([20.0]),
        np.array([25.0]),
        np.array([0.0]),
    )
    ten_days = 10 * bi.SECONDS_PER_DAY
    acc.advance(ten_days)  # exactly one half-life later: player 0's sums halve
    # player 1: 6 of the stat over 15 possessions and 10 minutes, added "now" (age 0).
    acc.add(
        np.array([1]),
        np.array([[6.0]]),
        np.array([15.0]),
        np.array([10.0]),
        np.array([ten_days]),
    )
    # Decayed sums: player 0 -> stat 5, poss 10, minutes 12.5; player 1 -> stat 6, poss 15,
    # minutes 10; league -> stat 11, poss 25, minutes 22.5.
    assert acc.stats[:, 0] == pytest.approx([5.0, 6.0])
    assert acc.poss == pytest.approx([10.0, 15.0])
    assert acc.minutes == pytest.approx([12.5, 10.0])
    assert acc.league_stats[0] == pytest.approx(11.0)
    assert acc.league_poss == pytest.approx(25.0)

    rate = bi._rate(acc.stats, acc.poss[:, None], 100.0)
    league_rate = bi._rate(acc.league_stats, np.array([acc.league_poss]), 100.0)
    assert rate[:, 0] == pytest.approx([50.0, 40.0])  # 100*5/10, 100*6/15
    assert league_rate[0] == pytest.approx(44.0)  # 100*11/25

    k = 5.0
    snapshot = bi._RoundSnapshot(
        games_idx=np.array([], dtype=np.int64),
        player_rate=rate,
        league_rate=league_rate,
        weight=acc.minutes.copy(),
    )
    feature = bi._features(snapshot, k)
    # weight/(weight+k) * (rate - league): player 0: 12.5/17.5 * (50-44) = 30/7; player 1:
    # 10/15 * (40-44) = -8/3.
    assert feature[:, 0] == pytest.approx([12.5 / 17.5 * 6.0, 10.0 / 15.0 * -4.0])
    assert feature[:, 0] == pytest.approx([30.0 / 7.0, -8.0 / 3.0])


def test_unseen_player_has_zero_weight_and_zero_feature() -> None:
    acc = bi._Accumulator(n_players=1, n_stats=1, half_life_days=100.0)
    acc.advance(0.0)  # nothing ever added
    rate = bi._rate(acc.stats, acc.poss[:, None], 100.0)
    league_rate = bi._rate(acc.league_stats, np.array([acc.league_poss]), 100.0)
    snapshot = bi._RoundSnapshot(
        np.array([], dtype=np.int64), rate, league_rate, acc.minutes.copy()
    )
    feature = bi._features(snapshot, 250.0)
    assert feature[:, 0] == pytest.approx([0.0])


# ---------------------------------------------------------------------------------------------
# Synthetic dataset shared by the recovery, leakage and determinism tests.
# ---------------------------------------------------------------------------------------------

TEAMS = ("TA", "TB", "TC", "TD")
PLAYERS = {t: [f"{t}1", f"{t}2"] for t in TEAMS}


def _pairs() -> list[tuple[str, str]]:
    return [(h, a) for h in TEAMS for a in TEAMS if h != a]  # 12 ordered pairs


def _dataset(
    seed: int, n_seasons: int = 2, first_season: int = 2020
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, dict[str, float]]:
    """A small league: 4 teams, 2 players each, one round-robin per season (one game a round).

    Returns ``(games, player_games, shares, possessions, pace)`` -- ``pace`` is the raw
    game_id -> possessions dict ``possessions`` is built from, for tests that need it directly.
    """
    rng = np.random.default_rng(seed)
    games_rows: list[dict[str, object]] = []
    player_rows: list[dict[str, object]] = []
    share_rows: list[dict[str, object]] = []
    pace: dict[str, float] = {}
    game_index = 0
    for season in range(first_season, first_season + n_seasons):
        start = pd.Timestamp(f"{season}-10-01T18:00:00Z")
        for rnd, (home, away) in enumerate(_pairs(), start=1):
            game_index += 1
            game_id = f"G{game_index}"
            games_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "phase": "RS",
                    "round": rnd,
                    "tipoff_utc": start + pd.Timedelta(days=rnd),
                    "home": home,
                    "away": away,
                    "home_score": np.nan,  # filled in by the caller
                    "away_score": np.nan,
                    "played": True,
                    "forfeit": False,
                    "neutral": False,
                }
            )
            pace[game_id] = float(rng.uniform(65.0, 85.0))
            for side, team in (("home", home), ("away", away)):
                for player in PLAYERS[team]:
                    stats = rng.integers(0, 12, size=len(STAT_COLUMNS)).astype(float)
                    row: dict[str, object] = {
                        "game_id": game_id,
                        "team": team,
                        "player_id": player,
                        "sec": float(rng.uniform(600.0, 2200.0)),
                        "poss": float(rng.uniform(50.0, 90.0)),
                        "pir": float(rng.integers(-5, 25)),
                    }
                    row.update(dict(zip(STAT_COLUMNS, stats.tolist(), strict=True)))
                    player_rows.append(row)
                    share_rows.append(
                        {
                            "game_id": game_id,
                            "side": side,
                            "team": team,
                            "player_id": player,
                            "share": float(rng.uniform(1.0, 3.0)),
                        }
                    )
    games = pd.DataFrame(games_rows)
    player_games = pd.DataFrame(player_rows)
    shares = pd.DataFrame(share_rows)
    possessions = pd.Series(pace, name="possessions")
    return games, player_games, shares, possessions, pace


def _set_margins(games: pd.DataFrame, margin: np.ndarray) -> pd.DataFrame:
    out = games.copy()
    out["home_score"] = 100.0 + margin / 2.0
    out["away_score"] = 100.0 - margin / 2.0
    return out


# ---------------------------------------------------------------------------------------------
# beta/h and a/c recovery: margins generated from a known linear model of the module's own
# features must be recovered by the ridge (box-only) and OLS (PIR) fits.
# ---------------------------------------------------------------------------------------------


def test_box_only_beta_and_h_recovery_on_synthetic_games() -> None:
    games, player_games, shares, possessions, _ = _dataset(seed=1)
    half_life, k = 365.0, 250.0
    player_index = bi._player_index(player_games, shares)
    rows = bi._rows(games, player_games, player_index, STAT_COLUMNS)
    lookup = bi._shares_lookup(shares, player_index)
    snapshots = bi._round_snapshots(
        games, rows, len(player_index), half_life, bi._RateSpec("poss", 100.0)
    )
    game_ids = games["game_id"].astype(str).to_numpy()
    z_true = bi._z_from_snapshots(snapshots, game_ids, lookup, k)
    p = possessions.reindex(games["game_id"]).to_numpy(dtype=np.float64)
    home_flag = bi._home_flag(games)

    beta_true = np.linspace(0.4, 2.6, len(STAT_COLUMNS))
    h_true = 3.5
    margin_true = z_true @ beta_true + h_true * (p / 100.0) * home_flag
    games = _set_margins(games, margin_true)

    tuning = np.ones(len(games), dtype=bool)
    grid = BoxGrid(k=(k,), half_life_days=(half_life,), ridge=(0.1,))
    result = box_only_margins(games, player_games, shares, possessions, tuning, grid)

    assert result.params["half_life_days"] == half_life
    assert result.params["k"] == k
    recovered_beta = np.array([result.params["beta"][s] for s in STAT_COLUMNS])
    assert recovered_beta == pytest.approx(beta_true, abs=0.05)
    assert result.params["h"] == pytest.approx(h_true, abs=0.05)
    assert result.margin == pytest.approx(margin_true, abs=0.5)


def test_pir_a_and_c_recovery_on_synthetic_games() -> None:
    games, player_games, shares, possessions, _ = _dataset(seed=2)
    player_index = bi._player_index(player_games, shares)
    rows = bi._rows(games, player_games, player_index, ("pir",))
    lookup = bi._shares_lookup(shares, player_index)
    snapshots = bi._round_snapshots(
        games, rows, len(player_index), bi.PIR_HALF_LIFE_DAYS, bi._RateSpec("minutes", 40.0)
    )
    game_ids = games["game_id"].astype(str).to_numpy()
    z_true = bi._z_from_snapshots(snapshots, game_ids, lookup, bi.PIR_K_MINUTES)
    team_diff_true = z_true[:, 0] / 5.0
    home_flag = bi._home_flag(games)

    a_true, c_true = 2.7, 1.4
    margin_true = a_true * home_flag + c_true * team_diff_true
    games = _set_margins(games, margin_true)

    tuning = np.ones(len(games), dtype=bool)
    result = pir_margins(games, player_games, shares, possessions, tuning)

    assert result.params["a"] == pytest.approx(a_true, abs=1e-4)
    assert result.params["c"] == pytest.approx(c_true, abs=1e-4)
    assert result.margin == pytest.approx(margin_true, abs=1e-4)


# ---------------------------------------------------------------------------------------------
# Leakage: editing the last game's box score must not change any earlier prediction, the fitted
# parameters, or even that game's own prediction (its features come from games strictly before
# it; the tuning games all precede it too).
# ---------------------------------------------------------------------------------------------


def _random_margins(games: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _set_margins(games, rng.normal(0.0, 10.0, size=len(games)))


def test_editing_the_last_games_box_score_changes_nothing_box_only() -> None:
    games, player_games, shares, possessions, _ = _dataset(seed=3)
    games = _random_margins(games, seed=30)
    tuning = (games["game_id"] != games["game_id"].iloc[-1]).to_numpy()  # every game but the last
    grid = BoxGrid(k=(250.0,), half_life_days=(365.0,), ridge=(10.0,))

    before = box_only_margins(games, player_games, shares, possessions, tuning, grid)

    edited = player_games.copy()
    last_game = games["game_id"].iloc[-1]
    edited.loc[edited["game_id"] == last_game, "pts"] += 50
    edited.loc[edited["game_id"] == last_game, "poss"] += 20

    after = box_only_margins(games, edited, shares, possessions, tuning, grid)

    np.testing.assert_array_equal(before.margin, after.margin)
    assert before.params["beta"] == after.params["beta"]
    assert before.params["h"] == after.params["h"]
    assert before.params["cv_grid"] == after.params["cv_grid"]


def test_editing_the_last_games_box_score_changes_nothing_pir() -> None:
    games, player_games, shares, possessions, _ = _dataset(seed=4)
    games = _random_margins(games, seed=40)
    tuning = (games["game_id"] != games["game_id"].iloc[-1]).to_numpy()

    before = pir_margins(games, player_games, shares, possessions, tuning)

    edited = player_games.copy()
    last_game = games["game_id"].iloc[-1]
    edited.loc[edited["game_id"] == last_game, "pir"] += 50

    after = pir_margins(games, edited, shares, possessions, tuning)

    np.testing.assert_array_equal(before.margin, after.margin)
    assert before.params == after.params


def test_editing_an_earlier_games_box_score_does_change_a_later_prediction() -> None:
    """Sanity check for the leakage tests above: an edit to an *earlier* game's box score is
    allowed to (and here does) change a later game's feature, since the later game's cutoff is
    after it. This is not leakage -- it shows the tests above are not vacuously true."""
    games, player_games, shares, possessions, _ = _dataset(seed=5)
    games = _random_margins(games, seed=50)
    tuning = (games["game_id"] != games["game_id"].iloc[-1]).to_numpy()
    grid = BoxGrid(k=(250.0,), half_life_days=(365.0,), ridge=(10.0,))

    before = box_only_margins(games, player_games, shares, possessions, tuning, grid)

    edited = player_games.copy()
    first_game = games["game_id"].iloc[0]
    edited.loc[edited["game_id"] == first_game, "pts"] += 500
    edited.loc[edited["game_id"] == first_game, "poss"] += 200

    after = box_only_margins(games, edited, shares, possessions, tuning, grid)
    assert not np.allclose(before.margin[1:], after.margin[1:])


# ---------------------------------------------------------------------------------------------
# Determinism: two runs on the same inputs give byte-identical outputs.
# ---------------------------------------------------------------------------------------------


def test_two_runs_of_box_only_margins_are_identical() -> None:
    games, player_games, shares, possessions, _ = _dataset(seed=6)
    games = _random_margins(games, seed=60)
    tuning = np.ones(len(games), dtype=bool)
    grid = BoxGrid()

    first = box_only_margins(games, player_games, shares, possessions, tuning, grid)
    second = box_only_margins(games, player_games, shares, possessions, tuning, grid)

    np.testing.assert_array_equal(first.margin, second.margin)
    assert first.params == second.params


def test_two_runs_of_pir_margins_are_identical() -> None:
    games, player_games, shares, possessions, _ = _dataset(seed=7)
    games = _random_margins(games, seed=70)
    tuning = np.ones(len(games), dtype=bool)

    first = pir_margins(games, player_games, shares, possessions, tuning)
    second = pir_margins(games, player_games, shares, possessions, tuning)

    np.testing.assert_array_equal(first.margin, second.margin)
    assert first.params == second.params


# ---------------------------------------------------------------------------------------------
# Grid mechanics: CV must actually discriminate between grid points, not just pick the first.
# ---------------------------------------------------------------------------------------------


def test_cv_prefers_the_ridge_that_matches_the_generating_model() -> None:
    games, player_games, shares, possessions, _ = _dataset(seed=8)
    half_life, k = 365.0, 250.0
    player_index = bi._player_index(player_games, shares)
    rows = bi._rows(games, player_games, player_index, STAT_COLUMNS)
    lookup = bi._shares_lookup(shares, player_index)
    snapshots = bi._round_snapshots(
        games, rows, len(player_index), half_life, bi._RateSpec("poss", 100.0)
    )
    game_ids = games["game_id"].astype(str).to_numpy()
    z_true = bi._z_from_snapshots(snapshots, game_ids, lookup, k)
    p = possessions.reindex(games["game_id"]).to_numpy(dtype=np.float64)
    home_flag = bi._home_flag(games)
    rng = np.random.default_rng(80)
    beta_true = np.linspace(0.4, 2.6, len(STAT_COLUMNS))
    noise = rng.normal(0.0, 1.0, size=len(games))
    margin_true = z_true @ beta_true + 3.0 * (p / 100.0) * home_flag + noise
    games = _set_margins(games, margin_true)

    tuning = np.ones(len(games), dtype=bool)
    # A very large ridge should score clearly worse (on CV) than a small one, for data generated
    # with almost no penalty.
    grid = BoxGrid(k=(k,), half_life_days=(half_life,), ridge=(0.1, 100000.0))
    result = box_only_margins(games, player_games, shares, possessions, tuning, grid)
    assert result.params["ridge"] == pytest.approx(0.1)
    small, large = result.params["cv_grid"]
    assert small["cv_rmse"] < large["cv_rmse"]
