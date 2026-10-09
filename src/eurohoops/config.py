"""Competitions, their seasons, backtest splits and file locations (relative to the repo root)."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eurohoops.models.elo import EloParams

LIVE_SEASON = 2026
MART_PATH = Path("data/marts/eurohoops.duckdb")
SQL_DIR = Path(__file__).parent / "sql"  # shipped inside the package, found from any cwd
BOX_INVARIANTS_REPORT = Path("reports/gbl_box_invariants.json")
STINT_REPORT = Path("reports/stint_validation.json")
POSSESSION_REPORT = Path("reports/possessions.json")
STINTS_MART_REPORT = Path("reports/stints_mart.json")
TEAM_CONTINUITY_REPORT = Path("reports/team_continuity.json")
SHOTS_REPORT = Path("reports/shots.json")
SITE_DATA = Path("web/src/data/site.json")  # read by the Astro build in web/


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
class DecayGrid:
    """M1 ridge hyperparameters: time-decay half-life, season carry-over and ridge penalty."""

    half_life_days: tuple[float, ...]
    carry: tuple[float, ...]
    ridge: tuple[float, ...]


# v1 pre-declared in reports/week5-7_progress.md; v2 widened post-hoc (reports/m1-v2_progress.md)
# after v1 tuned values sat on the grid edges.
M1_RATING_GRID = DecayGrid(
    half_life_days=(60.0, 120.0, 240.0, 480.0, 960.0, 1920.0),
    carry=(0.25, 0.5, 0.75, 1.0),
    ridge=(62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0),  # possessions of evidence
)
M1_PACE_GRID = DecayGrid(
    half_life_days=(30.0, 60.0, 120.0, 240.0, 480.0, 960.0),
    carry=(0.25, 0.5, 0.75, 1.0),
    ridge=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0),  # games of evidence
)


@dataclass(frozen=True)
class M1Backtest:
    """M1 vs a comparison Elo re-tuned on the same tuning seasons (E-a, E-b).

    Training always uses every game before the prediction point from the first warm-up season;
    the seasons only say where tuning, validation and test scores are computed.
    """

    report: Path
    warmup: tuple[int, ...]
    tuning: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    elo_grid: Grid = DEFAULT_GRID
    rating_grid: DecayGrid = M1_RATING_GRID
    pace_grid: DecayGrid = M1_PACE_GRID


@dataclass(frozen=True)
class Competition:
    name: str
    default_seasons: tuple[int, ...]
    live_backtest: Backtest  # its report holds the parameters the live log uses
    history_backtests: tuple[Backtest, ...]  # informational only, never used live
    prediction_log: Path
    scorecard: Path
    # GitHub push times of log rows committed by hand before the daily workflow took over.
    # A row first became public at the earliest such push at or after its stamp; later rows
    # are committed by the workflow in the run that stamps them.
    manual_pushes: tuple[datetime, ...] = ()
    odds_log: Path | None = None  # forward-recorded market consensus (EuroLeague only)
    m1: M1Backtest | None = None
    m1_prediction_log: Path | None = None  # live M1 log (E-g), separate from the Elo log
    m5_prediction_log: Path | None = None  # live M5 shadow log (EuroLeague only; not gated for GBL)

    @property
    def raw_dir(self) -> Path:
        return Path("data/raw") / self.name

    @property
    def staging_games(self) -> Path:
        return Path("data/staging") / f"{self.name}_games.parquet"

    @property
    def staging_teams(self) -> Path:
        return Path("data/staging") / f"{self.name}_teams.parquet"

    @property
    def staging_team_seasons(self) -> Path:
        return Path("data/staging") / f"{self.name}_team_seasons.parquet"


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
    # From the repository's activity feed (gh api repos/straf10/EuroHoops-Analytics/activity).
    manual_pushes=(
        datetime(2026, 9, 24, 16, 24, 29, tzinfo=UTC),  # branch creation, round-1 rows v0.1.0
        datetime(2026, 9, 24, 20, 29, 32, tzinfo=UTC),  # round-1 rows v0.2.0
    ),
    odds_log=Path("odds/euroleague_2026-27.csv"),
    m1=M1Backtest(
        Path("reports/backtest_m1.json"),
        warmup=_seasons(2007, 2014),
        tuning=_seasons(2015, 2022),
        validation=(2023,),
        test=(2024, 2025),
    ),
    m1_prediction_log=Path("predictions/euroleague_m1_2026-27.csv"),
    m5_prediction_log=Path("predictions/euroleague_m5_2026-27.csv"),
)

GBL_ELO_GRID = Grid(
    k=(10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0),
    hca=(0.0, 50.0, 90.0, 110.0, 130.0, 150.0, 170.0, 200.0, 230.0, 260.0),
    reversion=(0.0, 0.1, 0.25, 0.5, 0.75),
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
        grid=GBL_ELO_GRID,
        frozen=EloParams(k=40.0, hca=130.0, reversion=0.25),
    ),
    history_backtests=(),
    prediction_log=Path("predictions/gbl_2026-27.csv"),
    scorecard=Path("reports/live_scorecard_gbl.json"),
    m1=M1Backtest(
        Path("reports/backtest_m1_gbl.json"),
        warmup=_seasons(2018, 2020),
        tuning=(2021, 2022),
        validation=(2023,),
        test=(2024, 2025),
        elo_grid=GBL_ELO_GRID,
    ),
    m1_prediction_log=Path("predictions/gbl_m1_2026-27.csv"),
)

COMPETITIONS = {c.name: c for c in (EUROLEAGUE, GBL)}


@dataclass(frozen=True)
class M2Seasons:
    """Shot model M2 (F-a): leave-one-season-out CV on development, then validation, then test
    (scored once, after the verdict). 2007-10 shots are kept in the table and never used."""

    development: tuple[int, ...] = _seasons(2011, 2022)
    validation: tuple[int, ...] = (2023,)
    test: tuple[int, ...] = (2024, 2025)


M2_SEASONS = M2Seasons()
FREE_THROWS_REPORT = Path("reports/free_throws.json")
M2_REPORT = Path("reports/backtest_m2.json")
M2_TEAMS_REPORT = Path("reports/m2_teams.json")
M2_PLAYERS_REPORT = Path("reports/m2_players.json")
M2_CHARTS_DIR = Path("docs/models/m2")
# F8 subjects, named in reports/week7-10_progress.md before drawing: three teams (source codes),
# and the three players with the most validation-season FGA.
M2_CHART_TEAMS = ("PAN", "OLY", "MAD")
M2_CHART_PLAYERS = 3


@dataclass(frozen=True)
class M3Grid:
    """M3 hyperparameters, all chosen on the tuning seasons by future-margin RMSE (H-b, H-f).

    ``ridge`` is in possessions of evidence (the penalty added to each player column of the
    possession-weighted normal equations). The search: every (half-life, shared ridge) pair,
    then one-dimensional searches of a separate offense and defense ridge at the best
    half-life; ``dummy_minutes`` (0 = off) is searched with the chosen half-life and ridge.
    """

    # half-life widened once on 2026-09-28 before the verdict (tuning only): the first tuning
    # run chose 1,460 days, the grid edge (reports/week9-12_progress.md, iteration 7).
    half_life_days: tuple[float, ...] = (182.0, 365.0, 730.0, 1460.0, 2920.0, 5840.0)
    ridge: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
    dummy_minutes: tuple[float, ...] = (0.0, 50.0, 100.0, 200.0, 400.0)


@dataclass(frozen=True)
class M3Backtest:
    """M3 player-impact backtest (week 9-12 §0): walk forward by round, projected minutes."""

    report: Path
    warmup: tuple[int, ...]
    tuning: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    grid: M3Grid = M3Grid()
    projection_games: int = 5  # H-c: minutes share over the team's previous 5 games
    bootstrap_resamples: int = 1000  # H-d
    bootstrap_seed: int = 20261001


M3 = M3Backtest(
    Path("reports/backtest_m3.json"),
    warmup=_seasons(2011, 2014),
    tuning=_seasons(2015, 2022),
    validation=(2023,),
    test=(2024, 2025),
)
M3_GBL = M3Backtest(
    Path("reports/backtest_m3_gbl.json"),
    warmup=_seasons(2018, 2020),
    tuning=(2021, 2022),
    validation=(2023,),
    test=(2024, 2025),
)
M3_PLAYERS_REPORT = Path("reports/m3_players.json")
GBL_STINTS_REPORT = Path("reports/gbl_stints.json")
GBL_PLAYER_BOX = Path("data/staging/gbl_player_box.parquet")
GBL_TEAM_BOX = Path("data/staging/gbl_team_box.parquet")
GBL_BOX_FILL = Path("data/staging/gbl_box_fill.parquet")
GBL_PBP = Path("data/staging/gbl_pbp.parquet")
# weeks 12-14 I-c: birth dates for matching only; staging stays in data/ (never committed)
PLAYER_BIOS = Path("data/staging/player_bios.parquet")
BIO_EL_SEASONS = tuple(range(2007, 2026))
# GBL team id -> EuroLeague code of the same club (the only GBL clubs in the EuroLeague 2007-2025)
GREEK_EL_CLUBS = {"00000001": "PAN", "00000002": "OLY"}
GBL_PLAYER_NAMES = Path("data/staging/gbl_player_names.parquet")
EL_PLAYER_NAMES = Path("data/staging/euroleague_player_names.parquet")
ENTITY_SEASONS = tuple(range(2007, 2026))  # I-a: the live season is not touched
ENTITY_OVERRIDES = Path("entity/overrides.csv")
ENTITY_LABELS = Path("entity/labels.csv")
ENTITY_LABELS_TODO = Path("data/entity/labels_todo.csv")  # shows birth dates: never committed
ENTITY_REPORT = Path("reports/entity_resolution.json")
# The player_xwalk mart's ids (competition, source_id, person_id; no names, no dates), committed
# so the daily workflow can run `eurohoops project` without the local `entity` step (owner,
# 2026-10-08, weeks 16-18 D25). `entity` rewrites it with the mart.
PLAYER_XWALK_FILE = Path("entity/player_xwalk.csv")
# EuroLeague titles won before 2007-08 (frozen; later seasons come from the games table)
EL_TITLES_SEED = Path("entity/el_titles_before_2007.csv")
ENTITY_TUNING_REPORT = Path("reports/entity_tuning.json")


@dataclass(frozen=True)
class M4Backtest:
    """M4 league translation (weeks 12-14 I-h): walk forward by EuroLeague target season."""

    report: Path = Path("reports/backtest_m4.json")
    translation_report: Path = Path("reports/m4_translation.json")
    tuning: tuple[int, ...] = _seasons(2019, 2022)
    validation: tuple[int, ...] = (2023,)
    test: tuple[int, ...] = (2024, 2025)
    min_validation_movers: int = 20  # below it the gate pools 2019-2023 with `translate` declared
    shrink_minutes: float = 250.0  # `shrunk_same_stats` prior weight, fixed (not tuned)
    sd_min_minutes: float = 300.0  # EL player-seasons that set each stat's sd
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 20261015


M4 = M4Backtest()
GBL_BOX_GAPS = Path("reports/gbl_box_gaps.csv")
ODDS_CALLS = Path("odds/api_calls.csv")
ODDS_TEAMS = Path("odds/euroleague_teams.csv")
ODDS_RAW_DIR = Path("data/raw/odds/euroleague")
INJURY_LOG = Path(f"injuries/euroleague_{LIVE_SEASON}-{(LIVE_SEASON + 1) % 100}.csv")
INJURY_RAW_DIR = Path("data/raw/basketnews")


@dataclass(frozen=True)
class M5Grid:
    """M5 grids (week 14-16 J-c, J-d, J-e), every value chosen on tuning only."""

    # Both widened once on 2026-10-05 before the verdict (tuning only): the first tuning run's
    # best residual half-life was 180 days and the best candidate overall used 2 games, each the
    # lower grid edge (reports/week14-16_progress.md, iteration 7).
    half_life_games: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    residual_half_life_days: tuple[float, ...] = (60.0, 120.0, 180.0, 365.0, 730.0)
    residual_ridge: tuple[float, ...] = (10.0, 40.0, 160.0)
    rest_ridge: tuple[float, ...] = (25.0, 100.0)


@dataclass(frozen=True)
class M5Backtest:
    """M5 roster-aware game predictor backtest (week 14-16 J4): walk forward by round."""

    report: Path
    games_report: Path
    warmup: tuple[int, ...]
    tuning: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    grid: M5Grid = M5Grid()
    projection_games: int = 5  # proj_hc: minutes share over the team's previous 5 games
    absent_games: int = 2  # proj_avail: absent from each of the last 2 games drops a player
    margin_variant: str = "student_t_const"
    tie_tolerance: float = 0.0005  # tuning log-loss difference below which the simpler wins
    total_rest_ridge: float = 100.0  # ridge penalty of the totals rest regression
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 20261020


M5 = M5Backtest(
    Path("reports/backtest_m5.json"),
    Path("reports/backtest_m5_games.csv"),
    warmup=_seasons(2011, 2014),
    tuning=_seasons(2015, 2022),
    validation=(2023,),
    test=(2024, 2025),
)
M5_REST_REPORT = Path("reports/m5_rest.json")
M5_GBL = M5Backtest(
    Path("reports/backtest_m5_gbl.json"),
    Path("reports/backtest_m5_gbl_games.csv"),
    warmup=_seasons(2018, 2020),
    tuning=(2021, 2022),
    validation=(2023,),
    test=(2024, 2025),
)


@dataclass(frozen=True)
class M7Backtest:
    """M7 season simulator backtest (week 14-16 K4): replay past seasons at checkpoints."""

    report: Path
    teams_report: Path
    tuning: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    checkpoints: tuple[float, ...] = (0.25, 0.5, 0.75)  # fractions of the regular-season rounds
    n_sims: int = 4000
    seed: int = 20261101  # + the checkpoint's index in the run
    inflate: tuple[float, ...] = (1.5, 2.0)  # strength covariance multipliers of sim_inflate
    tie_tolerance: float = 0.002  # tuning mean RPS difference below which the simpler wins
    noise_floor: float = 0.5  # sim_net's noise scale is at least this share of M1's scale
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 20261102


# EuroLeague scored seasons are the single-table regular seasons only (reports/m7_progress.md, K-a).
M7 = M7Backtest(
    Path("reports/backtest_m7.json"),
    Path("reports/backtest_m7_teams.csv"),
    tuning=(2016, 2017, 2018, 2020, 2022),
    validation=(2023,),
    test=(2024, 2025),
)
# GBL: the seasons whose format section 3 can verify; reported only (it scores the EuroLeague
# verdict as `fixed`).
M7_GBL = M7Backtest(
    Path("reports/backtest_m7_gbl.json"),
    Path("reports/backtest_m7_gbl_teams.csv"),
    tuning=(2020, 2021, 2022),
    validation=(),
    test=(2025,),
)
# Live M7 (K8): one append-only simulation log per competition, plus the latest run's report.
SIM_LOGS = {
    name: Path(f"predictions/{name}_sim_{LIVE_SEASON}-{(LIVE_SEASON + 1) % 100}.csv")
    for name in ("euroleague", "gbl")
}
SIM_LATEST = {name: Path(f"reports/sim_latest_{name}.json") for name in ("euroleague", "gbl")}


@dataclass(frozen=True)
class M6Backtest:
    """M6 player projections backtest (weeks 16-18 L6): next-season and rest-of-season targets."""

    report: Path
    players_report: Path
    tuning: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    checkpoints: tuple[float, ...] = (0.25, 0.5, 0.75)  # fractions of the regular-season rounds
    min_poss: float = 500.0  # scored set: possessions in the target season (L-g)
    half_lives: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0)  # season decay (L-f; 0.5 added once, D20)
    interval: float = 0.8
    coverage_band: tuple[float, float] = (0.75, 0.85)  # pooled tuning + validation (L-h)
    tie_tolerance: float = 0.005  # relative tuning-loss difference below which the simpler wins
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 20261201


# Target seasons need a prior senior season in either league (L-a, L-g). EuroLeague 2019-20
# passes the possessions floor (128 scored players, reports/week16-18_progress.md) and stays.
M6 = M6Backtest(
    Path("reports/backtest_m6.json"),
    Path("reports/backtest_m6_players.csv"),
    tuning=_seasons(2016, 2022),
    validation=(2023,),
    test=(2024, 2025),
)
# GBL box scores start in 2018-19, so the first target with a prior GBL season is 2019-20;
# reported only (it scores the EuroLeague verdict as `fixed`).
M6_GBL = M6Backtest(
    Path("reports/backtest_m6_gbl.json"),
    Path("reports/backtest_m6_gbl_players.csv"),
    tuning=_seasons(2019, 2022),
    validation=(2023,),
    test=(2024, 2025),
)
# Live M6 (L9) and the ungated standings report (owner, 2026-10-07: decision D1).
M6_PROJECTIONS = Path("reports/m6_projections.json")
M6_BOARD = Path("reports/m6_board.json")
M6_SIMILARITY = Path("reports/m6_similarity.json")
SIM_UNGATED = {name: Path(f"reports/sim_ungated_{name}.json") for name in ("euroleague", "gbl")}
