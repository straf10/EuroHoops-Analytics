#!/usr/bin/env bash
# Checklist item 31 (weeks 7-10b G5): the season-level variants' report fields present and
# labelled post-hoc, item 24's two runs identical on them, the level leakage tests green, the
# declaration commit before the first run, and the weeks 7-10 numbers unchanged.
cd "$ROOT" || exit 1
uv run pytest -q -p no:cacheprovider tests/test_season_level.py tests/test_m2_leakage.py 2>&1 | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
uv run python "$ROOT/scripts/checks/m2_level.py"
