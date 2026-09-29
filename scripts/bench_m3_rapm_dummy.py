"""H6 performance benchmark (week 9-12 subagent E): one ``rapm_dummy`` walk-forward pass on
real-size synthetic data, one core -- the ``rapm_dummy`` analogue of ``bench_m3_rapm.py``.

Same synthetic size as ``bench_m3_rapm.py`` (close to the real EuroLeague stints mart: 15
seasons, 18 teams/season, ~300 games/season, 33 stints/game -> ~148k stints, ~4,600
player-seasons; the real mart is 4,246 games, 141,625 stints, 1,727 players, 4,577
player-seasons, 2011-2025), then times one walk-forward pass of ``rapm_dummy``
(``fit_walk_forward_dummy``, one half-life, one ridge, one nonzero threshold) over every round
cutoff, and compares it against the same-size plain ``rapm`` pass. H6's own target: <= ~20 s for
one pass at this size (roughly 2x plain ``rapm``'s ~9.4 s on the real mart, since ``rapm_dummy``
re-aggregates the normal equations from the base Gram every round instead of keeping the
model-level Gram incrementally -- module docstring in ``models/rapm_dummy.py``). Data generation
is timed separately and excluded from the budget, as in ``bench_m3_rapm.py``.

Run: ``uv run python scripts/bench_m3_rapm_dummy.py`` (``--quick`` for a small smoke-test size).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tests/ is not a package

from eurohoops.models.rapm import (
    build_design_rows,
    build_minutes_rows,
    build_spell_index,
    fit_walk_forward,
    plain_player_columns,
)
from eurohoops.models.rapm_dummy import build_dummy_columns, fit_walk_forward_dummy
from tests.m3_synthetic import make_synthetic_m3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="a small size, for a smoke test")
    args = parser.parse_args()

    if args.quick:
        seasons, teams, games_per_season, stints_per_game, roster = 3, 8, 30, 20, 10
    else:
        seasons, teams, games_per_season, stints_per_game, roster = 15, 18, 300, 33, 17

    clock = time.perf_counter()
    data = make_synthetic_m3(
        list(range(2011, 2011 + seasons)),
        teams=teams,
        games_per_season=games_per_season,
        stints_per_game=stints_per_game,
        roster=roster,
        seed=20261001,
    )
    build_s = time.perf_counter() - clock
    n_players = len(data.true_o)

    clock = time.perf_counter()
    spell_index = build_spell_index(data.stints)
    rows = build_design_rows(data.stints, data.checks, data.games, spell_index)
    minutes_rows = build_minutes_rows(data.stints, data.checks, data.games, spell_index)
    plain_columns = plain_player_columns(spell_index)
    dummy_columns = build_dummy_columns(spell_index)
    design_s = time.perf_counter() - clock

    print(
        f"games={len(data.games)} stints={len(data.stints)} player_seasons={spell_index.n_spells} "
        f"players={n_players} model_columns(rapm)={plain_columns.n_model} "
        f"model_columns(rapm_dummy)={dummy_columns.columns.n_model} "
        f"team_seasons={dummy_columns.n_team_seasons}"
    )
    print(f"data generation: {build_s:.2f} s; design/minutes rows + spell index: {design_s:.2f} s")

    clock = time.perf_counter()
    plain_wf = fit_walk_forward(
        data.games, rows, plain_columns, half_life_days=730.0, ridge_o=1000.0, ridge_d=1000.0
    )
    plain_fit_s = time.perf_counter() - clock
    n_plain_cutoffs = sum(1 for lookup in plain_wf.lookups if lookup is not None)
    print(f"plain rapm walk-forward fit: {plain_fit_s:.2f} s ({n_plain_cutoffs} predicted)")

    clock = time.perf_counter()
    dummy_wf = fit_walk_forward_dummy(
        data.games,
        rows,
        minutes_rows,
        spell_index,
        dummy_columns,
        half_life_days=730.0,
        ridge_o=1000.0,
        ridge_d=1000.0,
        dummy_minutes=100.0,
    )
    dummy_fit_s = time.perf_counter() - clock
    n_dummy_cutoffs = sum(1 for lookup in dummy_wf.lookups if lookup is not None)

    print(f"rapm_dummy walk-forward fit: {dummy_fit_s:.2f} s ({n_dummy_cutoffs} predicted)")
    print(f"rapm_dummy / plain rapm: {dummy_fit_s / plain_fit_s:.2f}x")
    print(f"H6 target: <= ~20 s for one pass -> {'PASS' if dummy_fit_s <= 20.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
