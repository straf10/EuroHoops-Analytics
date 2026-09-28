#!/usr/bin/env bash
# Checklist item 33: M3 unit tests (synthetic recovery, hand examples, posterior = ridge,
# 90% interval coverage, dummy and SPM variants, projected minutes) and the §3 facts on the
# local data (scripts/checks/m3_facts.py).
cd "$ROOT" || exit 1
uv run pytest -q -p no:cacheprovider tests/test_rapm.py tests/test_rapm_posterior.py \
  tests/test_minutes.py tests/test_rapm_dummy.py tests/test_spm.py tests/test_box_impact.py \
  tests/test_gbl_box_lines.py tests/test_gbl_stints.py 2>&1 | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
PYTHONIOENCODING=utf-8 uv run python "$ROOT/scripts/checks/m3_facts.py"
