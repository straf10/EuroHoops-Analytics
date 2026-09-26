#!/usr/bin/env bash
# Checklist item 22 (F2): with rates fitted leave-one-season-out, expected FT points reconcile
# with actual FT points in every development season (mean gap within +-0.1 point per team-game)
# and on validation (within +-0.2).
cd "$ROOT" || exit 1
uv run eurohoops free-throws || exit 1
uv run python -c "
import json, sys
r = json.load(open('reports/free_throws.json'))
print('between-season sd of FT points per team-game (development):', r['season_ft_points_sd_development'])
sys.exit(0 if r['all_within_tolerance'] else 1)"
