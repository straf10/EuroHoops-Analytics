"""Checklist item 60 (L11, part J): screenshots of the Standings, Methodology, Teams and Forecasts
pages at 1440 and 390 px, light and dark, with three checks per page: no horizontal overflow, no
console error, and every `[data-num]` figure sits in a `[data-figure]` that shows a visible
`[data-interval]` or `[data-sample]` label.
Usage: screenshots_m6_pages.py <built site dir> <out dir>.
Needs a build with the stats fixture (a team page) and the API fixture (tests/fixtures/web_api).
"""

import functools
import http.server
import shutil
import sys
import tempfile
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

site, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

# name, path below the base, competition to switch to (the Forecasts page shows one at a time),
# number of [data-num] the page must show at least, text it must show
PAGES = [
    ("standings", "standings/", None, 20, "Not gated"),
    ("methodology", "methodology/", None, 0, "Gate failed"),
    ("team_fbt", "teams/fbt/", None, 4, "Four factors"),
    ("team_asv", "teams/asv/", None, 0, "no box-score games"),
    ("forecasts", "", None, 10, "Elo beside the team models"),
    ("forecasts_gbl", "", "gbl", 5, "Elo beside the team models"),
]

# The numbers on show, and those without a visible sample or interval label inside their
# [data-figure].
NUMBERS = """
() => {
  const shown = (el) => {
    const box = el.getBoundingClientRect();
    return box.width > 0 && box.height > 0;
  };
  const numbers = [...document.querySelectorAll('[data-num]')].filter(shown);
  const unlabelled = numbers.filter((el) => {
    const figure = el.closest('[data-figure]');
    if (!figure) return true;
    return ![...figure.querySelectorAll('[data-interval], [data-sample]')].some(
      (label) => shown(label) && label.textContent.trim() !== ''
    );
  });
  return [numbers.length, unlabelled.map((el) => el.textContent.trim().slice(0, 30))];
}
"""

root = Path(tempfile.mkdtemp())
shutil.copytree(site, root, dirs_exist_ok=True)  # the site is served at the root


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


handler = functools.partial(Quiet, directory=str(root))
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_address[1]}/"
ok = True
with sync_playwright() as p:
    browser = p.chromium.launch(channel="msedge")
    for name, path, competition, minimum, text in PAGES:
        for scheme in ("light", "dark"):
            for width in (1440, 390):
                page = browser.new_page(
                    viewport={"width": width, "height": 900},
                    color_scheme=scheme,  # type: ignore[arg-type]
                )
                errors: list[str] = []
                page.on("console", lambda m, e=errors: m.type == "error" and e.append(m.text))
                page.on("pageerror", lambda x, e=errors: e.append(str(x)))
                page.goto(base + path)
                page.wait_for_load_state("networkidle")
                if competition:
                    page.click(f"label[for=comp-{competition}]")
                overflow = page.evaluate(
                    "document.documentElement.scrollWidth - document.documentElement.clientWidth"
                )
                numbers, unlabelled = page.evaluate(NUMBERS)
                present = text in page.inner_text("body")
                page.screenshot(path=str(out / f"m6_{name}_{width}_{scheme}.png"), full_page=True)
                page.close()
                good = (
                    overflow <= 0
                    and not errors
                    and not unlabelled
                    and numbers >= minimum
                    and present
                )
                ok &= good
                print(
                    f"{'ok  ' if good else 'FAIL'} {name} {width} {scheme}: "
                    f"overflow {overflow} px, "
                    f"{numbers} numbers, unlabelled {unlabelled}, console errors {errors}, "
                    f"text {text!r} {'shown' if present else 'MISSING'}"
                )
    browser.close()
server.shutdown()
shutil.rmtree(root, ignore_errors=True)
sys.exit(0 if ok else 1)
