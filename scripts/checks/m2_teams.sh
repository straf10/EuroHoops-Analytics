#!/usr/bin/env bash
# Checklist item 25 (F6): league-season xPTS within 0.5% of actual FG points in every
# development season (calibration in the large), from out-of-fold xPTS; the team and player
# reports rebuild to the committed ones.
cd "$ROOT" || exit 1
uv run eurohoops shot-quality || exit 1
git diff --quiet -- reports/m2_teams.json reports/m2_players.json \
  && echo "team and player reports equal the committed ones" || { echo "reports differ"; exit 1; }
uv run python -c "
import json, sys
r = json.load(open('reports/m2_teams.json'))
for season, block in r['calibration_in_the_large'].items():
    print(season, block['ratio'], 'ok' if block['within_tolerance'] else 'OUTSIDE 0.5%')
sys.exit(0 if r['calibration_in_the_large_ok'] else 1)"
