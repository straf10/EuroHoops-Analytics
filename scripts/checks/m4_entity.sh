#!/usr/bin/env bash
# Checklist item 38: entity unit tests (names, bios, transliteration, similarity, matcher,
# crosswalk, pipeline); two `eurohoops entity` builds give identical marts and a report equal to
# the committed one, within the I-l runtime (300 s); matcher < labels < label overrides in git.
cd "$ROOT" || exit 1
uv run pytest -q -p no:cacheprovider tests/test_player_names.py tests/test_bios.py \
  tests/test_translit.py tests/test_similarity.py tests/test_entity_match.py \
  tests/test_entity_xwalk.py tests/test_entity_pipeline.py 2>&1 | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
f=reports/entity_resolution.json
PYTHONIOENCODING=utf-8 uv run eurohoops entity > "$SCRATCH/entity_run1.txt" 2>&1 || { tail -5 "$SCRATCH/entity_run1.txt"; exit 1; }
cp "$f" "$SCRATCH/entity_run1.json"
uv run python "$ROOT/scripts/checks/entity_tables.py" > "$SCRATCH/entity_tables1.txt" || exit 1
PYTHONIOENCODING=utf-8 uv run eurohoops entity > "$SCRATCH/entity_run2.txt" 2>&1 || exit 1
uv run python "$ROOT/scripts/checks/entity_tables.py" > "$SCRATCH/entity_tables2.txt" || exit 1
cmp -s "$f" "$SCRATCH/entity_run1.json" && echo "$f: two builds byte-identical" || { echo "$f: builds differ"; exit 1; }
cmp -s "$SCRATCH/entity_tables1.txt" "$SCRATCH/entity_tables2.txt" && echo "marts identical:" || { echo "marts differ"; exit 1; }
cat "$SCRATCH/entity_tables1.txt"
git diff --quiet -- "$f" && echo "$f: equals the committed report" || { echo "$f differs from the committed report"; exit 1; }
secs=$(grep -oE '^RUNTIME [0-9]+' "$SCRATCH/entity_run1.txt" | cut -d' ' -f2)
echo "entity RUNTIME ${secs:-none} s (limit 300 s)"
[ -n "$secs" ] && [ "$secs" -lt 300 ] || exit 1
uv run python "$ROOT/scripts/checks/entity_order.py"
