#!/usr/bin/env bash
# Checklist item 22 (F2, re-specified by the user 2026-09-26, weeks 7-10b decision 2): with rates
# fitted leave-one-season-out, the within-season shares reconcile in every development season
# and on validation: each distance band's share of the and-one FT points, and the teams' shares
# of the league's FT points (mean |gap| and Pearson r), each within 2 x its game-level bootstrap
# SE; band flags gate by the count rule (the user's decision 2026-09-26: at most the 95th
# percentile of Binomial(band checks, 0.0455), the flags a right model makes by chance), team
# checks in every season. The per-season mean gap of FT points per team-game (the first F2 check, +-0.1) is printed
# as a limitation, "season level, not gated".
cd "$ROOT" || exit 1
uv run eurohoops free-throws || exit 1
uv run python -c "
import json, sys
r = json.load(open('reports/free_throws.json'))
g = r['share_check']
print('gate:', g['gate'])
print('band flags', g['band_flags'], 'of', g['band_checks'], 'allowed', g['band_flags_allowed'],
      '| team checks pass every season:', g['team_checks_pass_every_season'], '| passed:', g['passed'])
print('between-season sd of FT points per team-game (development):', r['season_ft_points_sd_development'])
print('season-level gaps within +-0.1/0.2 (not gated):', r['all_within_tolerance'])
sys.exit(0 if r['share_checks_pass'] else 1)"
