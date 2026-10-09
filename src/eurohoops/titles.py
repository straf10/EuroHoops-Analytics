"""EuroLeague titles, all-time since 1958: a frozen seed of the titles won before 2007-08 plus the
champions counted from our games since 2007-08.

A season's champion is the winner of its last played Final Four game by tip-off. Before 2025-26
a third-place game is played earlier the same weekend; from 2025-26 there is none. 2019-20 has
no Final Four, so it has no champion.

Codes are the source codes (the seed's ``code`` column and the games table); the payload is keyed
by the display codes the caller passes in (``publish.DISPLAY_CODES``).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from eurohoops.config import EL_TITLES_SEED

SINCE = 1958  # the first EuroLeague (Champions Cup) final, of the 1957-58 season
DATA_FROM = "2007-08"  # the first season in the games table


SEED_SCHEMA = pa.DataFrameSchema(
    {
        "club": pa.Column(str, pa.Check.str_length(min_value=1)),
        "code": pa.Column(str, nullable=True),  # blank when the club is not in our data
        "titles": pa.Column(int, pa.Check.ge(0)),
    }
)


def load_seed(path: Path = EL_TITLES_SEED) -> pd.DataFrame:
    """The seed rows; a blank ``code`` becomes missing."""
    seed = pd.read_csv(path, dtype={"club": str, "code": str, "titles": int}, keep_default_na=False)
    seed["code"] = seed["code"].str.strip().mask(lambda c: c == "")
    return SEED_SCHEMA.validate(seed)


def champions(games: pd.DataFrame) -> pd.DataFrame:
    """One row per season with a Final Four final: its winner's source code (``champion``).

    ``games`` is one competition's games table. The final is the last played FF game by tip-off.
    """
    if games.empty:
        return pd.DataFrame({"season": pd.Series(dtype=int), "champion": pd.Series(dtype=str)})
    ff = games[(games["phase"] == "FF") & games["played"].astype(bool)]
    last = ff.sort_values(["season", "tipoff_utc"], kind="stable").groupby("season").tail(1)
    home_won = last["home_score"] > last["away_score"]
    return pd.DataFrame(
        {
            "season": last["season"].astype(int).to_numpy(),
            "champion": np.where(home_won, last["home"], last["away"]),
        }
    )


def all_time(seed: pd.DataFrame, champs: pd.DataFrame) -> pd.Series:
    """Source code -> all-time titles (seed plus champions) for every club with a code."""
    seeded = seed.dropna(subset=["code"]).groupby("code")["titles"].sum()
    won = champs["champion"].value_counts()
    return seeded.add(won, fill_value=0).astype(int).sort_index()


def titles_payload(
    games: pd.DataFrame | None, display: Mapping[str, str], seed: pd.DataFrame | None = None
) -> dict[str, Any]:
    """``site.json``'s ``el_titles``: clubs with at least one title, by display code."""
    table = load_seed() if seed is None else seed
    champs = champions(games if games is not None else pd.DataFrame())
    totals = all_time(table, champs)
    return {
        "since": SINCE,
        "data_from": DATA_FROM,
        "titles": {
            display.get(str(code), str(code)): int(n) for code, n in totals.items() if n >= 1
        },
    }
