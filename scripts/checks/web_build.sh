#!/usr/bin/env bash
# Build web/ from the test fixture in a copy ($SCRATCH/web/web -> $SCRATCH/web/site): running
# astro dev/preview servers lock web/node_modules, so `npm ci` cannot run in place.
# The stats fixture stands in only when the copy has no stats of its own (real stats win). The
# site.json and API fixtures always replace the copy's, so the page checks (items 20 and 60) do not
# depend on whether `eurohoops publish` ran; WEB_REAL_DATA=1 keeps the real ones for a review.
set -e
dest="$SCRATCH/web"
rm -rf "$dest" && mkdir -p "$dest/web"
(cd "$ROOT/web" && tar --exclude=node_modules --exclude=.astro -cf - .) | (cd "$dest/web" && tar -xf -)
if [ -z "$WEB_REAL_DATA" ] || [ ! -f "$dest/web/src/data/site.json" ]; then
  cp "$ROOT/tests/fixtures/site.json" "$dest/web/src/data/site.json"
fi
test -d "$dest/web/src/data/stats" || cp -r "$ROOT/tests/fixtures/stats" "$dest/web/src/data/stats"
if [ -z "$WEB_REAL_DATA" ] || [ ! -d "$dest/web/src/data/api" ]; then
  rm -rf "$dest/web/src/data/api" && cp -r "$ROOT/tests/fixtures/web_api/src/data/api" "$dest/web/src/data/api"
fi
cd "$dest/web"
npm ci --no-audit --no-fund 2>&1 | tail -1
npm run build 2>&1 | tail -2
test -f "$dest/site/index.html" && echo "built $(find "$dest/site" -maxdepth 1 | tail -n +2 | wc -l) entries into site/"
