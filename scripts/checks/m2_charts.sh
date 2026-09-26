#!/usr/bin/env bash
# Checklist item 27 (F8): the seven charts exist and are referenced by the model card, and the
# court-geometry test passes.
cd "$ROOT" || exit 1
uv run pytest -q -p no:cacheprovider tests/test_shot_quality.py -k "geometry or charts_render" 2>&1 | tail -1
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
n=$(ls docs/models/m2/*.png 2>/dev/null | wc -l)
echo "$n charts in docs/models/m2"
[ "$n" -ge 7 ] || exit 1
for f in docs/models/m2/*.png; do
  grep -q "m2/$(basename "$f")" docs/models/m2.md || { echo "not in the card: $f"; exit 1; }
done
echo "every chart is referenced from docs/models/m2.md"
