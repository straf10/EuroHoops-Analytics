"""M3's input hash covers only the seasons a backtest scores (checklist items 35 and 37)."""

import pandas as pd

from eurohoops.config import M3Backtest, M3Grid
from eurohoops.eval.m3_backtest import prepare_data
from tests.m3_synthetic import make_synthetic_m3

SPEC = M3Backtest(
    report=None,  # type: ignore[arg-type]
    warmup=(2019,),
    tuning=(2020,),
    validation=(2021,),
    test=(2021,),
    grid=M3Grid(half_life_days=(365.0,), ridge=(500.0,)),
)


def test_the_input_hash_ignores_seasons_after_the_test() -> None:
    """A live season's games, added every round, do not move a committed report's hash; an edit
    inside the scored seasons does."""
    synth = make_synthetic_m3(
        [2019, 2020, 2021], teams=6, games_per_season=20, stints_per_game=8, roster=8, seed=7
    )
    tables = [synth.team_games, synth.player_games, synth.stints, synth.checks]
    base = prepare_data(synth.games, *tables, spec=SPEC).snapshot
    last = SPEC.test[-1]
    later = [
        pd.concat([t, t[t["season"] == last].assign(season=last + 1, game_id="L" + t["game_id"])])
        for t in tables
    ]
    assert prepare_data(synth.games, *later, spec=SPEC).snapshot == base
    edited = synth.stints.copy()
    edited.loc[edited.index[0], "home_points"] = int(edited["home_points"].iloc[0]) + 1
    tables[2] = edited
    assert prepare_data(synth.games, *tables, spec=SPEC).snapshot != base
