"""Competitions, their seasons, backtest splits and file locations (relative to the repo root)."""

from dataclasses import dataclass
from pathlib import Path

from eurohoops.models.elo import EloParams

LIVE_SEASON = 2026
MART_PATH = Path("data/marts/eurohoops.duckdb")
SQL_DIR = Path("sql")
BOX_INVARIANTS_REPORT = Path("reports/gbl_box_invariants.json")
SITE_DIR = Path("site")


@dataclass(frozen=True)
class Grid:
    k: tuple[float, ...]
    hca: tuple[float, ...]
    reversion: tuple[float, ...]


DEFAULT_GRID = Grid(
    k=(15.0, 20.0, 30.0, 40.0),
    hca=(50.0, 70.0, 90.0, 110.0, 130.0),
    reversion=(0.25, 0.5, 0.75),
)


@dataclass(frozen=True)
class Backtest:
    """Seasons are replayed from the first warm-up season; only tuning and test are scored.

    With ``frozen`` set, the grid is still searched and reported, but the report's tuned
    parameters (which the live log reads) stay ``frozen``.
    """

    report: Path
    warmup: tuple[int, ...]
    tuning: tuple[int, ...]
    test: tuple[int, ...]
    grid: Grid = DEFAULT_GRID
    frozen: EloParams | None = None


@dataclass(frozen=True)
class Competition:
    name: str
    default_seasons: tuple[int, ...]
    live_backtest: Backtest  # its report holds the parameters the live log uses
    history_backtests: tuple[Backtest, ...]  # informational only, never used live
    prediction_log: Path
    scorecard: Path

    @property
    def raw_dir(self) -> Path:
        return Path("data/raw") / self.name

    @property
    def staging_games(self) -> Path:
        return Path("data/staging") / f"{self.name}_games.parquet"

    @property
    def staging_teams(self) -> Path:
        return Path("data/staging") / f"{self.name}_teams.parquet"


def _seasons(first: int, last: int) -> tuple[int, ...]:
    return tuple(range(first, last + 1))


EUROLEAGUE = Competition(
    name="euroleague",
    default_seasons=_seasons(2023, LIVE_SEASON),
    live_backtest=Backtest(Path("reports/backtest_elo.json"), (2023,), (2024,), (2025,)),
    history_backtests=(
        Backtest(
            Path("reports/backtest_elo_history.json"),
            _seasons(2007, 2014),
            _seasons(2015, 2023),
            (2024, 2025),
        ),
    ),
    prediction_log=Path("predictions/euroleague_2026-27.csv"),
    scorecard=Path("reports/live_scorecard.json"),
)

GBL = Competition(
    name="gbl",
    default_seasons=_seasons(2018, LIVE_SEASON),
    # Frozen for 2026-27 (2026-09-25): the wide grid's best (K50/HCA170/rev0) beat these on
    # tuning but not on test (95% CI of the log-loss gap spans 0); see reports/week3_closeout.md.
    live_backtest=Backtest(
        Path("reports/backtest_elo_gbl.json"),
        _seasons(2018, 2021),
        (2022, 2023),
        (2024, 2025),
        grid=Grid(
            k=(10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0),
            hca=(0.0, 50.0, 90.0, 110.0, 130.0, 150.0, 170.0, 200.0, 230.0, 260.0),
            reversion=(0.0, 0.1, 0.25, 0.5, 0.75),
        ),
        frozen=EloParams(k=40.0, hca=130.0, reversion=0.25),
    ),
    history_backtests=(),
    prediction_log=Path("predictions/gbl_2026-27.csv"),
    scorecard=Path("reports/live_scorecard_gbl.json"),
)

COMPETITIONS = {c.name: c for c in (EUROLEAGUE, GBL)}
GBL_PLAYER_BOX = Path("data/staging/gbl_player_box.parquet")
GBL_TEAM_BOX = Path("data/staging/gbl_team_box.parquet")
