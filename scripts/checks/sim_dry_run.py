"""Checklist item 52: live M7 simulate on the real marts (K8).

For each competition: `run_live_sim` with the committed verdict; the probability sums hold
(Σ P(direct) = places in the direct cut, Σ P(play-in) = play-in places, Σ P(title) = 1, Σ P(Final
Four / semifinals) = 4); with the committed (failed or passed) gate, `write_live_sim` into a
temporary log adds rows only if the gate passed; with the gate forced open, a second write adds
nothing (idempotent); predictions/ and reports/ are untouched; and `daily.yml` runs `simulate`
only if the EuroLeague gate passed (K-j). Prints `RUNTIME simulate N` (the slower competition).
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eurohoops.config import COMPETITIONS, LIVE_SEASON, M7, M7_GBL, MART_PATH
from eurohoops.live_sim import load_sim, run_live_sim, write_live_sim
from eurohoops.marts import read_games, read_table
from eurohoops.sim.formats import season_format


def tree_state() -> str:
    """Hash of the tracked and untracked files under predictions/ and reports/."""
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", "predictions", "reports"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--", "predictions", "reports"], capture_output=True, text=True, check=True
    ).stdout
    return hashlib.sha256((out + diff).encode()).hexdigest()


ok = True
model = load_sim(M7.report)
assert model is not None, "no committed M7 report"
before = tree_state()
slowest = 0.0
for comp in COMPETITIONS.values():
    assert comp.m1 is not None
    tuned = json.loads(comp.m1.report.read_text(encoding="utf-8"))["tuned"]
    fmt = season_format(comp.name, LIVE_SEASON)
    started = time.perf_counter()
    games = read_games(MART_PATH, comp.name)
    team_games = read_table(MART_PATH, "team_games", comp.name)
    assert team_games is not None
    run = run_live_sim(
        games[games["season"] <= LIVE_SEASON].reset_index(drop=True),
        team_games,
        tuned,
        model,
        fmt=fmt,
        season=LIVE_SEASON,
        spec=M7 if comp.name == "euroleague" else M7_GBL,
    )
    slowest = max(slowest, time.perf_counter() - started)
    if run is None:
        print(f"{comp.name}: regular season over, nothing to simulate")
        continue
    out = run.out
    sums = (
        np.isclose(out.p_direct.sum(), len(fmt.playoffs_direct))
        and np.isclose(out.p_play_in.sum(), len(fmt.play_in))
        and np.isclose(out.p_title.sum(), 1.0)
        and np.isclose(out.p_semis.sum(), 4.0)
    )
    with tempfile.TemporaryDirectory() as tmp:
        log, report = Path(tmp) / "sim.csv", Path(tmp) / "latest.json"
        at = datetime.now(UTC)
        kwargs = {
            "fmt": fmt,
            "season": LIVE_SEASON,
            "at": at,
            "log_path": log,
            "report_path": report,
        }
        committed = write_live_sim(run, model, **kwargs)  # type: ignore[arg-type]
        gated_ok = (committed > 0) == model.gate_passed
        forced = replace(model, gate_passed=True)
        first = write_live_sim(run, forced, **kwargs)  # type: ignore[arg-type]
        second = write_live_sim(run, forced, **kwargs)  # type: ignore[arg-type]
        idempotent = (committed or first) == len(out.teams) and second == 0
    print(
        f"{comp.name}: after round {run.after_round}, {len(out.teams)} teams, sums hold {sums}, "
        f"gate passed {model.gate_passed} -> rows with the committed gate {committed}, "
        f"forced gate idempotent {idempotent}"
    )
    ok &= bool(sums) and gated_ok and idempotent

untouched = tree_state() == before
workflow = Path(".github/workflows/daily.yml").read_text(encoding="utf-8")
scheduled = "eurohoops simulate" in workflow
print(f"predictions/ and reports/ untouched: {untouched}")
print(f"daily.yml runs simulate: {scheduled} (gate passed: {model.gate_passed})")
ok &= untouched and scheduled == model.gate_passed
print(f"RUNTIME simulate {slowest:.0f}")
print("simulate dry run:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
