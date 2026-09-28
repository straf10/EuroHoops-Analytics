"""H1 performance benchmark (week 9-12 H-j): one RAPM walk-forward pass on real-size synthetic
data, one core.

Builds a synthetic history close to the real EuroLeague stints mart's size (15 seasons, 18
teams/season, ~300 games/season, 33 stints/game -> ~148k stints, ~4,600 player-seasons -- the
real mart is 4,246 games, 141,625 stints, 1,727 players, 4,577 player-seasons, 2011-2025), then
times one walk-forward pass of the plain ``rapm`` variant (one half-life, one ridge) over every
round cutoff. Data generation is timed separately and excluded from the 30 s budget (H1's own
target: real tuning, which repeats this many times over a grid, is the orchestrator's job at H5
and has its own 1,800 s budget, H-j).

Run: ``uv run python scripts/bench_m3_rapm.py`` (``--quick`` for a small smoke-test size).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tests/ is not a package

from eurohoops.models.rapm import (
    build_design_rows,
    build_spell_index,
    fit_walk_forward,
    plain_player_columns,
)
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
    columns = plain_player_columns(spell_index)
    design_s = time.perf_counter() - clock

    print(
        f"games={len(data.games)} stints={len(data.stints)} player_seasons={spell_index.n_spells} "
        f"players={n_players} model_columns={columns.n_model}"
    )
    print(f"data generation: {build_s:.2f} s; design rows + spell index: {design_s:.2f} s")

    clock = time.perf_counter()
    wf = fit_walk_forward(
        data.games, rows, columns, half_life_days=730.0, ridge_o=1000.0, ridge_d=1000.0
    )
    fit_s = time.perf_counter() - clock
    n_cutoffs = sum(1 for lookup in wf.lookups if lookup is not None)

    print(f"walk-forward fit: {fit_s:.2f} s over {len(data.games)} games ({n_cutoffs} predicted)")
    print(f"H-j target: <= 30 s for one pass -> {'PASS' if fit_s <= 30.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
