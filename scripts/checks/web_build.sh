#!/usr/bin/env bash
# Build web/ from the test fixture in a copy ($SCRATCH/web/web -> $SCRATCH/web/site): running
# astro dev/preview servers lock web/node_modules, so `npm ci` cannot run in place.
set -e
dest="$SCRATCH/web"
rm -rf "$dest" && mkdir -p "$dest/web"
(cd "$ROOT/web" && tar --exclude=node_modules --exclude=.astro -cf - .) | (cd "$dest/web" && tar -xf -)
cp "$ROOT/tests/fixtures/site.json" "$dest/web/src/data/site.json"
cd "$dest/web"
npm ci --no-audit --no-fund 2>&1 | tail -1
npm run build 2>&1 | tail -2
test -f "$dest/site/index.html" && echo "built $(find "$dest/site" -maxdepth 1 | tail -n +2 | wc -l) entries into site/"
