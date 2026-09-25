"""Checklist item 17: live M1 dry run on the real marts with a fixed clock and a temporary log.

A second run must add nothing, and the Elo log must stay byte-identical.
"""

import hashlib
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from eurohoops.config import COMPETITIONS, LIVE_SEASON, MART_PATH
from eurohoops.live_m1 import load_m1, predict_upcoming_m1
from eurohoops.marts import read_games, read_table

ok = True
for comp in COMPETITIONS.values():
    if comp.m1 is None:
        continue
    model = load_m1(comp.m1.report)
    if model is None or not model.gate_passed:
        print(f"{comp.name}: SKIPPED (gate failed)")
        continue
    games = read_games(MART_PATH, comp.name)
    team_games = read_table(MART_PATH, "team_games", comp.name)
    ahead = games[(games["season"] == LIVE_SEASON) & ~games["played"] & games["confirmed_date"]]
    clock = (ahead["tipoff_utc"].min() - timedelta(hours=10)).to_pydatetime()
    elo_before = hashlib.sha256(comp.prediction_log.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / f"{comp.name}_m1.csv"
        args = (games, team_games, model)
        kwargs = {
            "log_path": log,
            "season": LIVE_SEASON,
            "replay_from": comp.live_backtest.warmup[0],
            "window": timedelta(hours=36),
            "clock": lambda at=clock: at,
        }
        n1 = predict_upcoming_m1(*args, **kwargs)  # type: ignore[arg-type]
        content = log.read_bytes() if log.exists() else b""
        n2 = predict_upcoming_m1(*args, **kwargs)  # type: ignore[arg-type]
        same = (log.read_bytes() if log.exists() else b"") == content
        lines = content.decode().splitlines()
    elo_after = hashlib.sha256(comp.prediction_log.read_bytes()).hexdigest()
    print(
        f"{comp.name}: clock {clock:%Y-%m-%dT%H:%MZ}, first run {n1} M1 rows, second run {n2}; "
        f"log unchanged by run 2: {same}; Elo log byte-identical: {elo_before == elo_after}"
    )
    for line in lines[:3]:
        print("   ", line)
    ok &= n1 > 0 and n2 == 0 and same and elo_before == elo_after
sys.exit(0 if ok else 1)
