from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from eurohoops.models.elo import FloatArray
from eurohoops.sim.season import (
    NoiseModel,
    PaceModel,
    SimOutput,
    StrengthSampler,
    _order_tables,
    _play_regular_season,
    simulate,
)
from eurohoops.standings import EUROLEAGUE_2026, GBL_2026, Format, Result, Series, rank
from tests.test_rank_by_wins import random_deductions, random_games

TINY = NoiseModel(scale=0.01, df=None)
NO_PLAY_IN = Format(
    competition="euroleague",
    season="test",
    teams=16,
    regular_season_rounds=30,
    playoffs_direct=tuple(range(1, 9)),
    play_in=(),
    eliminated=tuple(range(9, 17)),
    relegated=(),
    series=(Series("playoffs", 5), Series("final four", 1)),
    sources=(),
)
GBL_PLAIN = replace(GBL_2026, points_deducted=())


def small_format(n_teams: int, best_of: int = 1) -> Format:
    """GBL-shaped format on ``n_teams`` >= 8 teams, no deductions."""
    return replace(
        GBL_PLAIN,
        teams=n_teams,
        regular_season_rounds=2 * (n_teams - 1),
        eliminated=tuple(range(9, n_teams + 1)),
        relegated=(),
        series=(
            Series("quarterfinals", best_of),
            Series("semifinals", best_of),
            Series("final", best_of),
        ),
    )


def codes(n: int) -> list[str]:
    return [f"T{i:02d}" for i in range(n)]


def fixtures(pairs: list[tuple[str, str]], neutral: bool = False) -> pd.DataFrame:
    tipoff = pd.date_range("2026-10-01", periods=len(pairs), freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "home": [h for h, _ in pairs],
            "away": [a for _, a in pairs],
            "neutral": [neutral] * len(pairs),
            "tipoff_utc": tipoff,
        }
    )


def double_round_robin(teams: list[str]) -> list[tuple[str, str]]:
    return [(h, a) for h in teams for a in teams if h != a]


def fixed_sampler(off: FloatArray, home: float = 0.0) -> StrengthSampler:
    def sampler(n_sims: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
        return (
            np.tile(off, (n_sims, 1)),
            np.zeros((n_sims, len(off))),
            np.full(n_sims, home),
        )

    return sampler


def strength_ladder(n: int, gap: float = 4.0) -> FloatArray:
    """Team i is stronger than team i + 1 by ``gap`` rating points (margin ~ 0.7 · gap · levels)."""
    return 120.0 - gap * np.arange(n)


def pace_for(n: int) -> PaceModel:
    return PaceModel(mu=70.0, team=np.zeros(n))


def run(
    teams: list[str],
    fmt: Format,
    sampler: StrengthSampler,
    *,
    remaining: pd.DataFrame | None = None,
    played: list[Result] | None = None,
    noise: NoiseModel = TINY,
    n_sims: int = 50,
    seed: int = 1,
) -> SimOutput:
    if remaining is None:
        remaining = fixtures(double_round_robin(teams))
    return simulate(
        teams=teams,
        played=played or [],
        remaining=remaining,
        sampler=sampler,
        pace=pace_for(len(teams)),
        noise=noise,
        fmt=fmt,
        n_sims=n_sims,
        seed=seed,
    )


def probability(out: SimOutput, field: str) -> dict[str, float]:
    return dict(zip(out.teams, getattr(out, field).tolist(), strict=True))


def ones(out: SimOutput, field: str) -> set[str]:
    return {team for team, p in probability(out, field).items() if p == 1.0}


def assert_deterministic_table(out: SimOutput) -> None:
    """Team i finishes i + 1 in every simulation (teams are listed best first)."""
    assert np.array_equal(out.rank_counts, out.n_sims * np.eye(len(out.teams), dtype=np.int64))


def test_el_2026_bracket_with_huge_gaps() -> None:
    teams = codes(20)
    out = run(teams, EUROLEAGUE_2026, fixed_sampler(strength_ladder(20)))
    assert_deterministic_table(out)
    assert np.array_equal(out.wins, 2.0 * (19 - np.arange(20)))
    assert ones(out, "p_direct") == set(teams[:6])
    assert ones(out, "p_play_in") == set(teams[6:10])
    assert ones(out, "p_top10") == set(teams[:10])
    # play-in A (7v8): T06; B (9v10): T08; C: T07 (loser A) beats T08 (winner B)
    assert ones(out, "p_playoffs") == set(teams[:8])
    assert ones(out, "p_semis") == set(teams[:4])  # 1v8, 4v5, 3v6, 2v7 won by the higher seed
    assert ones(out, "p_final") == {teams[0], teams[1]}
    assert ones(out, "p_title") == {teams[0]}


def test_el_2026_play_in_winners_become_seeds_seven_and_eight() -> None:
    """Make the 10th team the best of the play-in group: it must beat 9th and then lose
    nothing before the playoffs, entering as the 8th seed against the 1st."""
    teams = codes(20)
    # table fixed by results already played (unique wins), strengths decide only the play-in
    played = [Result(teams[i], teams[j], 80, 70) for i in range(20) for j in range(i + 1, 20)]
    off = np.full(20, 100.0)
    off[[6, 7, 8, 9]] = [101.0, 100.0, 130.0, 160.0]  # T09 > T08 > T06 > T07 on strength
    empty = fixtures([])
    out = run(teams, EUROLEAGUE_2026, fixed_sampler(off), remaining=empty, played=played)
    assert_deterministic_table(out)
    # A: T06 beats T07; B: T09 beats T08; C: T07 (loser A) v T09 (winner B) -> T09 is seed 8
    assert ones(out, "p_playoffs") == {*teams[:6], teams[6], teams[9]}
    # T09 (160) beats the 1st seed in a best of 5 even away: it is the strongest team
    assert "T09" in ones(out, "p_semis")


def test_el_without_play_in_bracket() -> None:
    teams = codes(16)
    out = run(teams, NO_PLAY_IN, fixed_sampler(strength_ladder(16)))
    assert_deterministic_table(out)
    assert ones(out, "p_direct") == set(teams[:8])
    assert ones(out, "p_play_in") == set()
    assert ones(out, "p_top10") == set(teams[:10])
    assert ones(out, "p_playoffs") == set(teams[:8])
    assert ones(out, "p_semis") == set(teams[:4])
    assert ones(out, "p_final") == {teams[0], teams[1]}
    assert ones(out, "p_title") == {teams[0]}


def test_gbl_bracket_with_deductions_and_huge_gaps() -> None:
    teams = codes(14)
    teams[4], teams[9] = "00000002", "00000001"  # sanctioned teams (-1 win each)
    out = run(teams, GBL_2026, fixed_sampler(strength_ladder(14)))
    assert_deterministic_table(out)  # wins differ by 2, so a lost win never reorders
    assert ones(out, "p_direct") == set(teams[:8])
    assert ones(out, "p_play_in") == set()
    assert ones(out, "p_top10") == set(teams[:10])
    assert ones(out, "p_playoffs") == set(teams[:8])
    assert ones(out, "p_semis") == set(teams[:4])
    assert ones(out, "p_final") == {teams[0], teams[1]}
    assert ones(out, "p_title") == {teams[0]}
    assert out.wins[4] == 2 * (13 - 4)  # wins are reported before deductions


def test_hand_built_ties_are_ranked_as_rank_does() -> None:
    """Head-to-head with a 3-way tie, and a restart of the tied pair, inside an 8-team league
    whose remaining games (all among the bottom teams) are decided by strength."""
    scenarios = {
        "three-way": (
            ["A", "B", "C", "D"],
            [
                ("A", "B", 80, 70),
                ("B", "A", 75, 70),
                ("A", "C", 70, 75),
                ("C", "A", 70, 80),
                ("B", "C", 80, 70),
                ("C", "B", 85, 70),
                *[(t, "D", 90, 60) for t in "ABC"],
                *[("D", t, 60, 90) for t in "ABC"],
            ],
        ),
        "restart": (
            ["A", "B", "C"],
            [
                ("A", "B", 80, 78),
                ("B", "A", 80, 75),
                ("A", "C", 90, 80),
                ("C", "A", 82, 80),
                ("B", "C", 80, 70),
                ("C", "B", 78, 70),
            ],
        ),
    }
    for name, (top, games) in scenarios.items():
        tail = [f"X{i}" for i in range(8 - len(top))]
        played = [Result(*row) for row in games]
        for s, t in product(top, tail):  # every top team beats every tail team twice
            played += [Result(s, t, 90, 60), Result(t, s, 60, 90)]
        # tail teams play each other for the rest of the season, X0 the strongest
        pairs = double_round_robin(tail)
        stronger = {t: i for i, t in enumerate(tail)}
        decided = [
            Result(h, a, 100, 50) if stronger[h] < stronger[a] else Result(h, a, 50, 100)
            for h, a in pairs
        ]
        expected = rank(played + decided)
        teams = sorted([*top, *tail], reverse=True)  # unrelated to the table order
        off = np.array([100.0 - 4 * stronger.get(t, 0) for t in teams])
        out = run(
            teams,
            small_format(8),
            fixed_sampler(off),
            remaining=fixtures(pairs),
            played=played,
            n_sims=20,
        )
        place = out.rank_counts.argmax(axis=1) + 1
        assert (out.rank_counts.max(axis=1) == 20).all(), name
        assert [t for _, t in sorted(zip(place, teams, strict=True))] == expected, name


def test_equal_teams_give_a_flat_rank_distribution() -> None:
    teams = codes(10)
    n_sims = 6000

    def sampler(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
        return np.full((n, 10), 100.0), np.zeros((n, 10)), np.full(n, 3.0)

    out = run(teams, small_format(10), sampler, noise=NoiseModel(10.0, 5.0), n_sims=n_sims, seed=11)
    assert out.rank_counts.sum(axis=0).tolist() == [n_sims] * 10
    assert out.rank_counts.sum(axis=1).tolist() == [n_sims] * 10
    # each cell is Binomial(n_sims, 1/10); 5 standard errors (all 100 cells stay inside with
    # probability > 1 - 100 · 6e-7)
    bound = 5 * np.sqrt(0.1 * 0.9 / n_sims)
    assert np.abs(out.rank_counts / n_sims - 0.1).max() < bound
    assert np.abs(out.p_title - 1 / 10).max() < bound
    assert np.abs(out.wins - 9.0).max() < 0.2  # 18 games per team, mean 9 by symmetry


def series_probability(per_game: list[float]) -> float:
    """P(win a series from independent per-game win probabilities): every outcome of the
    games, summed where the favourite wins a majority."""
    need = len(per_game) // 2 + 1
    total = 0.0
    for outcome in product([True, False], repeat=len(per_game)):
        if sum(outcome) >= need:
            total += float(
                np.prod([p if won else 1 - p for p, won in zip(per_game, outcome, strict=True)])
            )
    return total


def series_sampler(p_home: float, p_away: float, noise: NoiseModel, n: int) -> StrengthSampler:
    """Team 0 beats team n - 1 with probability ``p_home`` at home and ``p_away`` away.

    P(home team wins) = F(mean / scale) with F the t cdf, mean = pace/100 · (Δ ± 2h); so
    Δ + 2h and Δ - 2h are scale · F⁻¹(p) · 100 / pace for p_home and p_away."""
    assert noise.df is not None
    pace = 70.0
    at_home = noise.scale * stats.t.ppf(p_home, noise.df) * 100 / pace
    away = noise.scale * stats.t.ppf(p_away, noise.df) * 100 / pace
    delta, hf = (at_home + away) / 2, (at_home - away) / 4
    off = np.full(n, 100.0)
    off[0] += delta
    return fixed_sampler(off, home=hf)


def ladder_results(teams: list[str]) -> list[Result]:
    """Each team beats every later team once: unique wins, fixed table."""
    return [
        Result(teams[i], teams[j], 80, 70)
        for i in range(len(teams))
        for j in range(i + 1, len(teams))
    ]


def test_best_of_five_matches_the_closed_form() -> None:
    teams = codes(16)
    p_home, p_away, n_sims = 0.72, 0.41, 40000
    noise = NoiseModel(scale=9.0, df=6.0)
    out = run(
        teams,
        NO_PLAY_IN,
        series_sampler(p_home, p_away, noise, 16),
        remaining=fixtures([]),
        played=ladder_results(teams),
        noise=noise,
        n_sims=n_sims,
        seed=5,
    )
    # games 1, 2, 5 at the higher seed (team 0), games 3, 4 at team 15
    exact = series_probability([p_home, p_home, p_away, p_away, p_home])
    # team 0 wins the series: P(reach the Final Four); 5 standard errors of a proportion
    assert out.p_semis[0] == pytest.approx(exact, abs=5 * np.sqrt(exact * (1 - exact) / n_sims))
    assert out.p_semis[7] == pytest.approx(1 - exact, abs=5 * np.sqrt(exact * (1 - exact) / n_sims))


def test_best_of_three_matches_the_closed_form() -> None:
    teams = codes(14)
    p_home, p_away, n_sims = 0.66, 0.38, 40000
    noise = NoiseModel(scale=9.0, df=6.0)
    out = run(
        teams,
        GBL_PLAIN,
        series_sampler(p_home, p_away, noise, 14),
        remaining=fixtures([]),
        played=ladder_results(teams),
        noise=noise,
        n_sims=n_sims,
        seed=6,
    )
    exact = series_probability([p_home, p_away, p_home])  # games 1 and 3 at the higher seed
    assert out.p_semis[0] == pytest.approx(exact, abs=5 * np.sqrt(exact * (1 - exact) / n_sims))


def random_sampler(n_teams: int) -> StrengthSampler:
    def sampler(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray, FloatArray]:
        return (
            rng.normal(100, 3, (n, n_teams)),
            rng.normal(0, 2, (n, n_teams)),
            rng.normal(3, 0.5, n),
        )

    return sampler


def random_run(seed: int, n_sims: int = 3000) -> SimOutput:
    teams = codes(10)
    pairs = double_round_robin(teams)
    played = [Result(h, a, 80, 75) for h, a in pairs[:20]]
    return run(
        teams,
        small_format(10, best_of=3),
        random_sampler(10),
        remaining=fixtures(pairs[20:]),
        played=played,
        noise=NoiseModel(10.0, 5.0),
        n_sims=n_sims,
        seed=seed,
    )


def test_same_seed_gives_identical_output() -> None:
    a, b = random_run(3, 600), random_run(3, 600)
    for field in SimOutput.__dataclass_fields__:
        x, y = getattr(a, field), getattr(b, field)
        assert np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y, field


def test_different_seeds_agree_within_monte_carlo_error() -> None:
    n_sims = 3000
    a, b = random_run(1, n_sims), random_run(2, n_sims)
    assert a.seed != b.seed
    # difference of two independent proportions: sd <= sqrt(2 · 0.25 / n); 5 sd
    bound = 5 * np.sqrt(2 * 0.25 / n_sims)
    for field in ("p_direct", "p_top10", "p_playoffs", "p_semis", "p_final", "p_title"):
        assert np.abs(getattr(a, field) - getattr(b, field)).max() < bound, field
    assert np.abs(a.rank_counts - b.rank_counts).max() / n_sims < bound
    # wins lie in [0, 18] (Popoviciu: sd <= 9), mean difference sd <= 9 · sqrt(2 / n)
    assert np.abs(a.wins - b.wins).max() < 5 * 9 * np.sqrt(2 / n_sims)


def test_probabilities_sum_to_the_format_counts() -> None:
    for fmt, n in ((EUROLEAGUE_2026, 20), (small_format(10), 10), (NO_PLAY_IN, 16)):
        teams = codes(n)
        out = run(teams, fmt, random_sampler(n), noise=NoiseModel(10.0, 5.0), n_sims=300)
        assert out.p_direct.sum() == pytest.approx(len(fmt.playoffs_direct))
        assert out.p_play_in.sum() == pytest.approx(len(fmt.play_in))
        assert out.p_playoffs.sum() == pytest.approx(8)
        assert out.p_semis.sum() == pytest.approx(4)
        assert out.p_final.sum() == pytest.approx(2)
        assert out.p_title.sum() == pytest.approx(1)
        assert out.p_top10.sum() == pytest.approx(min(10, n))
        assert out.rank_counts.sum(axis=0).tolist() == [300] * n
        assert out.rank_counts.sum(axis=1).tolist() == [300] * n


def test_gbl_2026_with_random_strengths_runs_with_deductions() -> None:
    teams = codes(14)
    teams[0], teams[1] = "00000002", "00000001"
    out = run(teams, GBL_2026, random_sampler(14), noise=NoiseModel(10.0, 5.0), n_sims=200)
    assert out.p_playoffs.sum() == pytest.approx(8)
    assert out.wins.sum() == pytest.approx(14 * 26 / 2)


def test_a_zero_margin_becomes_plus_or_minus_one() -> None:
    n_sims, teams = 2000, codes(10)
    pairs = double_round_robin(teams)
    index = {t: i for i, t in enumerate(teams)}
    home_points, away_points = _play_regular_season(
        off=np.full((n_sims, 10), 100.0),
        def_=np.zeros((n_sims, 10)),
        home=np.zeros(n_sims),
        home_idx=np.array([index[h] for h, _ in pairs]),
        away_idx=np.array([index[a] for _, a in pairs]),
        neutral=np.zeros(len(pairs), dtype=bool),
        pace=pace_for(10),
        noise=NoiseModel(scale=1e-6, df=None),  # every margin rounds to 0
        rng=np.random.default_rng(0),
    )
    assert (np.abs(home_points - away_points) == 1).all()
    n = home_points.size
    # coin flips: the home win share is Binomial(n, 1/2); 5 standard errors
    assert abs((home_points > away_points).mean() - 0.5) < 5 * np.sqrt(0.25 / n)


def test_fast_path_and_tie_breaks_equal_rank_on_1000_random_tables() -> None:
    rng = np.random.default_rng(21)
    checked = 0
    for _ in range(20):
        teams = [f"T{i}" for i in range(int(rng.integers(4, 8)))]
        games = random_games(rng, teams)
        n_played = int(rng.integers(0, len(games) // 2))
        played = games[:n_played]  # keep their regulation scores
        pairs = [(r.home, r.away) for r in games[n_played:]]
        index = {t: i for i, t in enumerate(teams)}
        home_idx = np.array([index[h] for h, _ in pairs], dtype=np.int64)
        away_idx = np.array([index[a] for _, a in pairs], dtype=np.int64)
        n_sims = 50
        hp = rng.integers(70, 74, (n_sims, len(pairs)))
        ap = hp + rng.choice([-3, -2, -1, 1, 2, 3], (n_sims, len(pairs)))
        deducted = random_deductions(rng, teams)
        wins = np.zeros((n_sims, len(teams)), dtype=np.int64)
        for r in played:
            wins[:, index[r.winner]] += 1
        for g, (h, a) in enumerate(pairs):
            wins[np.arange(n_sims), np.where(hp[:, g] > ap[:, g], index[h], index[a])] += 1
        order = _order_tables(
            wins=wins,
            deducted=np.array([deducted.get(t, 0) for t in teams], dtype=np.int64),
            teams=teams,
            played=played,
            home_idx=home_idx,
            away_idx=away_idx,
            home_points=hp,
            away_points=ap,
        )
        for s in range(n_sims):
            results = played + [
                Result(h, a, int(hp[s, g]), int(ap[s, g])) for g, (h, a) in enumerate(pairs)
            ]
            assert [teams[i] for i in order[s]] == rank(results, deducted)
            checked += 1
    assert checked == 1000


def test_unknown_team_and_unsupported_formats_raise() -> None:
    teams = codes(20)
    sampler = fixed_sampler(strength_ladder(20))
    with pytest.raises(ValueError, match="not in `teams`"):
        run(teams, EUROLEAGUE_2026, sampler, played=[Result("T00", "ZZZ", 80, 70)])
    with pytest.raises(ValueError, match="not in `teams`"):
        run(teams, EUROLEAGUE_2026, sampler, remaining=fixtures([("T00", "ZZZ")]))
    odd = replace(EUROLEAGUE_2026, series=(Series("playoffs", 5), Series("semifinal", 1)))
    with pytest.raises(ValueError, match="knockout structure"):
        run(teams, odd, sampler)
    longer = replace(NO_PLAY_IN, series=(Series("playoffs", 7), Series("final four", 1)))
    with pytest.raises(ValueError, match="best of 7"):
        run(codes(16), longer, fixed_sampler(strength_ladder(16)))
