"""Project-wide paths and constants. Paths are relative to the repository root (the CWD)."""

from pathlib import Path

RAW_DIR = Path("data/raw/euroleague")
GAMES_PATH = Path("data/staging/euroleague_games.parquet")
BACKTEST_REPORT = Path("reports/backtest_elo.json")
SCORECARD_REPORT = Path("reports/live_scorecard.json")

DEFAULT_SEASONS = (2023, 2024, 2025, 2026)
LIVE_SEASON = 2026
PREDICTION_LOG = Path(f"predictions/euroleague_{LIVE_SEASON}-{(LIVE_SEASON + 1) % 100:02d}.csv")

WARMUP_SEASON = 2023
TUNING_SEASON = 2024
TEST_SEASON = 2025
